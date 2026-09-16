"""GPU-accelerated objective delta mask computation for PlacementEnv.

Replaces the triple Python loop in PlacementEnv.objective_delta_mask with
vectorized PyTorch operations on GPU, eliminating the W¡ÁH inner loop entirely.

Rotation-independent penalties (conn, align, group, anchor, boundary_group,
pitch, interior, density, line_neatness) are computed once and broadcast.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .env import PlacementEnv


ACTION_CONDITIONED_ACTION_FEAT_DIM = 42
ACTION_FEATURE_NAMES = [
    "x_norm", "y_norm", "rot_sin", "rot_cos",
    "dx_to_anchor", "dy_to_anchor", "dist_to_anchor", "has_anchor_xy",
    "dx_to_conn_centroid", "dy_to_conn_centroid", "dist_to_conn_centroid", "has_conn_centroid",
    "dx_to_module_center", "dy_to_module_center", "dist_to_module_center", "has_module_center",
    "inside_module_bbox", "dist_to_module_bbox", "module_region_confidence",
    "nearest_same_module_dist", "nearest_same_functional_group_dist",
    "nearest_critical_neighbor_dist", "placed_same_module_count_norm",
    "local_density", "soft_spacing_penalty_approx", "min_gap_to_placed", "overlap_risk_approx",
    "left_clearance", "right_clearance", "top_clearance", "bottom_clearance",
    "is_edge_band", "is_corner_band",
    "dist_to_preferred_edge", "on_preferred_edge_band", "boundary_axis_norm", "has_preferred_edge",
    "prior_heatmap_value_at_action", "prior_heatmap_logprob_at_action",
    "dist_to_prior_peak_norm", "inside_prior_mass_50", "module_prior_confidence",
]


def _grid_cache_key(env: PlacementEnv, device: torch.device) -> Tuple[Any, ...]:
    dev = torch.device(device)
    return (
        f"{dev.type}:{-1 if dev.index is None else int(dev.index)}",
        tuple(env.grid_shape()),
        tuple(float(v) for v in env.task.bbox_mm),
        float(env.task.grid_mm),
    )


@torch.no_grad()
def _cached_grid_maps_cuda(env: PlacementEnv, device: torch.device) -> Dict[str, torch.Tensor]:
    """Cache board grid tensors reused by mask/objective/features kernels.

    The previous hot path recreated arange/meshgrid tensors separately in
    action_mask_and_bias_cuda(), objective_delta_mask_cuda(), action features,
    and dynamic priors.  A PlacementEnv survives for many suffix steps, so a tiny
    per-env/per-device cache avoids repeated allocation and keeps more work on
    the GPU.
    """
    key = _grid_cache_key(env, device)
    cache = getattr(env, "_cuda_grid_maps_cache", None)
    if isinstance(cache, dict) and key in cache:
        return cache[key]

    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    w_cells, h_cells = env.grid_shape()
    grid_mm = float(env.task.grid_mm)
    bw = max(1.0e-6, xmax - xmin)
    bh = max(1.0e-6, ymax - ymin)

    ix = torch.arange(w_cells, device=device, dtype=torch.float32).view(w_cells, 1)
    iy = torch.arange(h_cells, device=device, dtype=torch.float32).view(1, h_cells)
    x_col = xmin + (ix + 0.5) * grid_mm
    y_row = ymin + (iy + 0.5) * grid_mm
    x_full = x_col.expand(w_cells, h_cells)
    y_full = y_row.expand(w_cells, h_cells)
    out = {
        "x_col": x_col,
        "y_row": y_row,
        "x": x_full,
        "y": y_full,
        "x_norm": (x_full - xmin) / bw,
        "y_norm": (y_full - ymin) / bh,
    }
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_cuda_grid_maps_cache", cache)
    cache[key] = out
    return out


def _coerce_wh_map(t: torch.Tensor, w_cells: int, h_cells: int, name: str) -> torch.Tensor:
    """Return *t* as a contiguous [W, H] tensor.

    The CUDA objective code mixes two grid representations: compact/skinny
    coordinate tensors ([W,1] and [1,H]) for cheap broadcasting, and full
    coordinate tensors ([W,H]) for kernels that need both axes.  A few legacy
    sub-kernels can therefore produce [H,W], [W,1], [1,H], or even a leading
    singleton/rotation dimension depending on the active semantic path.
    Normalize every sub-map before it enters the objective sum.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)

    target = (int(w_cells), int(h_cells))
    shape = tuple(int(v) for v in t.shape)
    if shape == target:
        return t.contiguous()

    if t.dim() == 0:
        return t.reshape(1, 1).expand(target).contiguous()

    if t.dim() == 1:
        if shape == (w_cells,):
            return t.view(w_cells, 1).expand(target).contiguous()
        if shape == (h_cells,):
            return t.view(1, h_cells).expand(target).contiguous()

    if t.dim() == 2:
        if shape == (h_cells, w_cells):
            return t.transpose(0, 1).contiguous()
        if shape == (w_cells, 1):
            return t.expand(target).contiguous()
        if shape == (1, h_cells):
            return t.expand(target).contiguous()
        if shape == (1, 1):
            return t.expand(target).contiguous()

    if t.dim() == 3:
        # Allow callers to pass a singleton or already-reduced rotation plane.
        if shape[0] == 1:
            return _coerce_wh_map(t[0], w_cells, h_cells, name).contiguous()
        if shape[-1] == 1 and shape[:2] == target:
            return t[..., 0].contiguous()

    try:
        return t.expand(target).contiguous()
    except RuntimeError as exc:
        raise RuntimeError(
            f"{name} map has shape {tuple(t.shape)}, expected broadcastable [W,H]={target}"
        ) from exc


def _coerce_rwh_map(t: torch.Tensor, R: int, w_cells: int, h_cells: int, name: str) -> torch.Tensor:
    """Return *t* as a contiguous [R, W, H] tensor."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)

    target = (int(R), int(w_cells), int(h_cells))
    shape = tuple(int(v) for v in t.shape)
    if shape == target:
        return t.contiguous()

    if t.dim() <= 2:
        return _coerce_wh_map(t, w_cells, h_cells, name).unsqueeze(0).expand(target).contiguous()

    if t.dim() == 3:
        if shape[0] == R:
            if shape[1:] == (h_cells, w_cells):
                return t.transpose(1, 2).contiguous()
            if shape[1:] == (w_cells, 1):
                return t.expand(target).contiguous()
            if shape[1:] == (1, h_cells):
                return t.expand(target).contiguous()
            if shape[1:] == (1, 1):
                return t.expand(target).contiguous()
        if shape[0] == 1:
            return _coerce_wh_map(t[0], w_cells, h_cells, name).unsqueeze(0).expand(target).contiguous()

    try:
        return t.expand(target).contiguous()
    except RuntimeError as exc:
        raise RuntimeError(
            f"{name} map has shape {tuple(t.shape)}, expected broadcastable [R,W,H]={target}"
        ) from exc


# ---------------------------------------------------------------------------
# Precompute placed-component data as GPU tensors
# ---------------------------------------------------------------------------

def _precompute_placed_data(
    env: PlacementEnv,
    ref: str,
    placed_order: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    n = len(placed_order)
    if n == 0:
        return {}

    px = torch.tensor([float(env.placed[r][0]) for r in placed_order], device=device)
    py = torch.tensor([float(env.placed[r][1]) for r in placed_order], device=device)

    bbs = torch.tensor(
        [env._ref_bbox(r, *env.placed[r]) for r in placed_order],
        device=device, dtype=torch.float32,
    )

    conn_w = torch.tensor(
        [env._pair_conn_weight(ref, r) for r in placed_order],
        device=device, dtype=torch.float32,
    )

    is_reg = torch.tensor(
        [env._is_regularized_non_interface(env.comp_by_ref[r]) for r in placed_order],
        device=device, dtype=torch.bool,
    )

    diags = torch.tensor(
        [math.hypot(float(env.comp_by_ref[r].size_mm[0]),
                     float(env.comp_by_ref[r].size_mm[1]))
         for r in placed_order],
        device=device, dtype=torch.float32,
    )

    density_scale = torch.tensor(
        [env._density_scale(ref, r) for r in placed_order],
        device=device, dtype=torch.float32,
    )

    soft_spacing_scale = torch.tensor(
        [env._soft_spacing_scale(ref, r) for r in placed_order],
        device=device, dtype=torch.float32,
    )

    pref_gaps = torch.tensor(
        [env._preferred_gap_mm(ref, r) for r in placed_order],
        device=device, dtype=torch.float32,
    )

    return {
        'px': px, 'py': py,
        'bbs': bbs,
        'conn_w': conn_w,
        'is_reg': is_reg,
        'diags': diags,
        'density_scale': density_scale,
        'soft_spacing_scale': soft_spacing_scale,
        'pref_gaps': pref_gaps,
    }


def _placed_data_cache_key(
    env: PlacementEnv,
    ref: str,
    placed_order: List[str],
    device: torch.device,
) -> Tuple[Any, ...]:
    dev = torch.device(device)
    return (
        f"{dev.type}:{-1 if dev.index is None else int(dev.index)}",
        str(ref),
        tuple(
            (
                str(r),
                float(env.placed[r][0]),
                float(env.placed[r][1]),
                int(env.placed[r][2]),
            )
            for r in placed_order
            if r in env.placed
        ),
    )


def _cached_placed_data_cuda(
    env: PlacementEnv,
    ref: str,
    placed_order: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Cache dynamic placed-component tensors for one env state/ref/device.

    objective_delta_mask_cuda() and action_features_cuda() are called back to
    back for the same current state in the training hot path.  Both need the
    same placed centers, boxes, pair weights, density scales, and spacing hints.
    This cache avoids rebuilding and re-uploading those tensors twice.
    """
    if not placed_order:
        return {}
    key = _placed_data_cache_key(env, ref, placed_order, device)
    cache = getattr(env, "_cuda_placed_data_cache", None)
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    out = _precompute_placed_data(env, ref, placed_order, device)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_cuda_placed_data_cache", cache)
    # Keep the cache bounded because env clones share shallow-copied attrs and
    # replay can visit many dynamic states over a long run.
    if len(cache) > 512:
        cache.clear()
    cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Rotation-independent penalties  ¡ú  [W, H]
# ---------------------------------------------------------------------------

def _compute_conn(
    Xc: torch.Tensor, Yc: torch.Tensor,
    pd: Dict[str, torch.Tensor],
    board_diag: float,
) -> torch.Tensor:
    w = pd['conn_w']
    mask = w > 0
    if not mask.any():
        return torch.zeros_like(Xc)
    px, py, ww = pd['px'][mask], pd['py'][mask], w[mask]
    dx = torch.abs(Xc.unsqueeze(-1) - px)
    dy = torch.abs(Yc.unsqueeze(-1) - py)
    return (ww * (dx + dy) / board_diag).sum(dim=-1)


def _compute_interior(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
) -> torch.Tensor:
    comp = env.comp_by_ref[ref]
    if not env._is_regularized_non_interface(comp):
        return torch.zeros_like(Xc)
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
    mr = float(env.interior_margin_ratio)
    mx = min(0.45 * bw, max(float(env.task.grid_mm), bw * max(0.0, mr)))
    my = min(0.45 * bh, max(float(env.task.grid_mm), bh * max(0.0, mr)))
    dx = torch.clamp(xmin + mx - Xc, min=0) + torch.clamp(Xc - (xmax - mx), min=0)
    dy = torch.clamp(ymin + my - Yc, min=0) + torch.clamp(Yc - (ymax - my), min=0)
    base = (dx / max(1e-6, mx)) ** 2 + (dy / max(1e-6, my)) ** 2
    return max(0.0, float(env.interior_penalty)) * base


def _compute_density(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    pd: Dict[str, torch.Tensor],
) -> torch.Tensor:
    comp = env.comp_by_ref[ref]
    if not env._is_regularized_non_interface(comp):
        return torch.zeros_like(Xc)
    valid = pd['is_reg']
    if not valid.any():
        return torch.zeros_like(Xc)

    da = math.hypot(float(comp.size_mm[0]), float(comp.size_mm[1]))
    dr = float(env.density_radius_mm)

    db = pd['diags'][valid]
    soft_r = 0.5 * (da + db) + dr
    cx = (pd['bbs'][valid, 0] + pd['bbs'][valid, 2]) / 2
    cy = (pd['bbs'][valid, 1] + pd['bbs'][valid, 3]) / 2
    sc = pd['density_scale'][valid]

    dist = torch.sqrt((Xc.unsqueeze(-1) - cx) ** 2 + (Yc.unsqueeze(-1) - cy) ** 2)
    overflow = torch.clamp(soft_r - dist, min=0)
    u = overflow / soft_r.clamp(min=1e-6)
    return max(0.0, float(env.density_penalty)) * (sc * u * u * (soft_r > 1e-6).float()).sum(dim=-1)


def _compute_align_delta(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    bw: float, bh: float,
    device: torch.device,
) -> torch.Tensor:
    ag = env._align_groups.get(ref)
    if not ag:
        return torch.zeros_like(Xc)
    cur = list(env.placed_order)
    cur_xy = {r: (float(env.placed[r][0]), float(env.placed[r][1])) for r in cur}
    peers = [r for r in cur if r != ref and env._align_groups.get(r) == ag]
    if not peers:
        return torch.zeros_like(Xc)

    ppx = torch.tensor([cur_xy[r][0] for r in peers], device=device)
    ppy = torch.tensor([cur_xy[r][1] for r in peers], device=device)

    dx = torch.abs(Xc.unsqueeze(-1) - ppx) / bw
    dy = torch.abs(Yc.unsqueeze(-1) - ppy) / bh
    vals = torch.minimum(dx, dy)
    k = min(3, len(peers))
    top, _ = torch.topk(vals, k=k, dim=-1, largest=False)
    delta = top.mean(dim=-1)

    for pref in peers:
        px, py = cur_xy[pref]
        old_pen = float(env._align_penalty_for_layout(pref, px, py, cur, cur_xy))
        old_targets = [r for r in cur if r != pref and env._align_groups.get(r) == ag]
        new_ref_val = torch.minimum(
            torch.abs(torch.tensor(px, device=device) - Xc) / bw,
            torch.abs(torch.tensor(py, device=device) - Yc) / bh,
        )
        if old_targets:
            ov = torch.tensor(
                [min(abs(px - cur_xy[r][0]) / bw, abs(py - cur_xy[r][1]) / bh) for r in old_targets],
                device=device, dtype=torch.float32,
            )
            all_v = torch.cat([
                ov.view(1, 1, -1).expand(new_ref_val.shape[0], new_ref_val.shape[1], -1),
                new_ref_val.unsqueeze(-1),
            ], dim=-1)
            k2 = min(3, all_v.shape[-1])
            top2, _ = torch.topk(all_v, k=k2, dim=-1, largest=False)
            new_pen = top2.mean(dim=-1)
        else:
            new_pen = new_ref_val
        delta = delta + (new_pen - old_pen)
    return delta


def _compute_group_delta(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    my_g = env._explicit_functional_group.get(ref, '')
    cur = list(env.placed_order)
    cur_xy = {r: (float(env.placed[r][0]), float(env.placed[r][1])) for r in cur}
    if not my_g:
        return torch.zeros_like(Xc)
    grefs = [r for r in cur if env._explicit_functional_group.get(r, '') == my_g]

    if not grefs:
        return torch.zeros_like(Xc)
    old_bbs = {r: env._group_proxy_bbox(r, *cur_xy[r]) for r in grefs}
    old_pen = float(env._group_span_penalty_from_bbs(grefs, old_bbs))

    b0 = env.comp_by_ref[ref]
    w0 = max(float(env.task.grid_mm), float(b0.size_mm[0]))
    h0 = max(float(env.task.grid_mm), float(b0.size_mm[1]))
    cand_a = Xc - 0.5 * w0
    cand_b = Yc - 0.5 * h0
    cand_c = Xc + 0.5 * w0
    cand_d = Yc + 0.5 * h0

    xmins = torch.tensor([old_bbs[r][0] for r in grefs], device=device, dtype=torch.float32)
    ymins = torch.tensor([old_bbs[r][1] for r in grefs], device=device, dtype=torch.float32)
    xmaxs = torch.tensor([old_bbs[r][2] for r in grefs], device=device, dtype=torch.float32)
    ymaxs = torch.tensor([old_bbs[r][3] for r in grefs], device=device, dtype=torch.float32)

    span_x = torch.maximum(xmaxs.max(), cand_c) - torch.minimum(xmins.min(), cand_a)
    span_y = torch.maximum(ymaxs.max(), cand_d) - torch.minimum(ymins.min(), cand_b)
    area = torch.clamp(span_x * span_y, min=1e-6)

    tgt_min_x, tgt_max_x, tgt_min_y, tgt_max_y, tgt_min_area, tgt_max_area = env._group_span_targets(grefs + [ref])
    collapse_x = torch.clamp(tgt_min_x - span_x, min=0) / max(1e-6, tgt_min_x)
    collapse_y = torch.clamp(tgt_min_y - span_y, min=0) / max(1e-6, tgt_min_y)
    spread_x = torch.clamp(span_x - tgt_max_x, min=0) / max(1e-6, tgt_max_x)
    spread_y = torch.clamp(span_y - tgt_max_y, min=0) / max(1e-6, tgt_max_y)
    collapse_area = torch.clamp(tgt_min_area - area, min=0) / max(1e-6, tgt_min_area)
    spread_area = torch.clamp(area - tgt_max_area, min=0) / max(1e-6, tgt_max_area)
    new_pen = (
        0.28 * (collapse_x * collapse_x + collapse_y * collapse_y)
        + 0.12 * (collapse_area * collapse_area)
        + 0.18 * (spread_x * spread_x + spread_y * spread_y)
        + 0.06 * (spread_area * spread_area)
    )

    sep_delta = torch.zeros_like(Xc)
    my_group = env._explicit_functional_group.get(ref, '')
    other_groups = sorted({env._explicit_functional_group.get(r, '') for r in cur if env._explicit_functional_group.get(r, '') not in {'', my_group}})
    if other_groups:
        my_center_x = Xc
        my_center_y = Yc
        my_gap_bb = (cand_a, cand_b, cand_c, cand_d)
        for og in other_groups:
            members = [r for r in cur if env._explicit_functional_group.get(r, '') == og]
            if not members:
                continue
            peer_bbs = {r: env._group_proxy_bbox(r, *cur_xy[r]) for r in members}
            xmin_o = min(bb[0] for bb in peer_bbs.values())
            ymin_o = min(bb[1] for bb in peer_bbs.values())
            xmax_o = max(bb[2] for bb in peer_bbs.values())
            ymax_o = max(bb[3] for bb in peer_bbs.values())
            dx = torch.clamp(torch.maximum(torch.tensor(xmin_o, device=device) - cand_c, cand_a - torch.tensor(xmax_o, device=device)), min=0)
            dy = torch.clamp(torch.maximum(torch.tensor(ymin_o, device=device) - cand_d, cand_b - torch.tensor(ymax_o, device=device)), min=0)
            bbox_gap = torch.where(dx <= 0, dy, torch.where(dy <= 0, dx, torch.sqrt(dx * dx + dy * dy)))
            peer_cx = sum(cur_xy[r][0] for r in members) / len(members)
            peer_cy = sum(cur_xy[r][1] for r in members) / len(members)
            centroid_gap = torch.sqrt((my_center_x - peer_cx) ** 2 + (my_center_y - peer_cy) ** 2)
            min_bbox_gap = 1.2 * float(env.task.grid_mm)
            min_centroid_gap = max(2.0 * float(env.task.grid_mm), 0.05 * float(env.board_diag))
            weight = float(env._group_separation_pair_weight(my_group, og))
            sep_delta = sep_delta + weight * (torch.clamp(min_bbox_gap - bbox_gap, min=0) / max(1e-6, min_bbox_gap)) ** 2
            sep_delta = sep_delta + 0.6 * weight * (torch.clamp(min_centroid_gap - centroid_gap, min=0) / max(1e-6, min_centroid_gap)) ** 2
    n = float(len(grefs))
    strength = float(env._semantic_strength(ref))
    return strength * (n * (new_pen - old_pen) + new_pen) + strength * sep_delta


def _anchor_penalty_vec(
    env: PlacementEnv, comp_ref: str,
    comp_x: torch.Tensor, comp_y: torch.Tensor,
    anchor_x: torch.Tensor, anchor_y: torch.Tensor,
    board_diag: float,
) -> torch.Tensor:
    anchor_ref_name = env._anchor_refs.get(comp_ref)
    if not anchor_ref_name:
        return torch.zeros_like(comp_x)
    anchor_comp = env.comp_by_ref.get(anchor_ref_name)
    if not anchor_comp:
        return torch.zeros_like(comp_x)

    comp = env.comp_by_ref[comp_ref]
    cs = 0.5 * math.hypot(float(comp.size_mm[0]), float(comp.size_mm[1]))
    as_ = 0.5 * math.hypot(float(anchor_comp.size_mm[0]), float(anchor_comp.size_mm[1]))
    min_r = max(float(env.task.grid_mm), 0.35 * (cs + as_))
    max_r = max(4.0 * float(env.task.grid_mm), 1.35 * (cs + as_), 0.18 * board_diag)

    dx = comp_x - anchor_x
    dy = comp_y - anchor_y
    dist = torch.sqrt(dx * dx + dy * dy)
    bd2 = board_diag * board_diag

    pen = torch.clamp(min_r - dist, min=0) ** 2 / bd2
    pen = pen + torch.clamp(dist - max_r, min=0) ** 2 / bd2

    sub = env._subzones.get(comp_ref, 'free')
    if sub == 'left':
        pen = pen + 1.60 * torch.clamp(dx, min=0) ** 2 / bd2
        pen = pen + 0.45 * torch.clamp(torch.abs(dy) - torch.abs(dx), min=0) ** 2 / bd2
    elif sub == 'right':
        pen = pen + 1.60 * torch.clamp(-dx, min=0) ** 2 / bd2
        pen = pen + 0.45 * torch.clamp(torch.abs(dy) - torch.abs(dx), min=0) ** 2 / bd2
    elif sub == 'top':
        pen = pen + 1.60 * torch.clamp(-dy, min=0) ** 2 / bd2
        pen = pen + 0.45 * torch.clamp(torch.abs(dx) - torch.abs(dy), min=0) ** 2 / bd2
    elif sub == 'bottom':
        pen = pen + 1.60 * torch.clamp(dy, min=0) ** 2 / bd2
        pen = pen + 0.45 * torch.clamp(torch.abs(dx) - torch.abs(dy), min=0) ** 2 / bd2
    elif sub == 'around':
        target = 0.5 * (min_r + max_r)
        pen = pen + 0.10 * ((dist - target) / board_diag) ** 2
    return pen


def _compute_anchor_delta(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    board_diag: float,
    device: torch.device,
) -> torch.Tensor:
    strength = float(env._semantic_strength(ref))
    delta = torch.zeros_like(Xc)
    cur = list(env.placed_order)
    cur_xy = {r: (float(env.placed[r][0]), float(env.placed[r][1])) for r in cur}

    strength = float(env._semantic_strength(ref))
    anchor_ref = env._anchor_refs.get(ref)
    if anchor_ref and anchor_ref in cur_xy:
        ax, ay = cur_xy[anchor_ref]
        ax_t = torch.tensor(ax, device=device).expand_as(Xc)
        ay_t = torch.tensor(ay, device=device).expand_as(Yc)
        delta = delta + strength * _anchor_penalty_vec(env, ref, Xc, Yc, ax_t, ay_t, board_diag)

    for pref in cur:
        if env._anchor_refs.get(pref) != ref:
            continue
        px, py = cur_xy[pref]
        px_t = torch.full_like(Xc, px)
        py_t = torch.full_like(Yc, py)
        delta = delta + float(env._semantic_strength(pref)) * _anchor_penalty_vec(env, pref, px_t, py_t, Xc, Yc, board_diag)
    return delta


def _compute_boundary_group_delta(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    W = Xc.shape[0]
    H = Yc.shape[1]
    X = Xc.expand(W, H)
    Y = Yc.expand(W, H)

    group = env._same_side_groups.get(ref)
    if not group:
        return torch.zeros((W, H), device=device, dtype=Xc.dtype)

    my_side = env._side_preferences.get(ref, env._region_targets.get(ref, 'free'))
    if my_side not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
        return torch.zeros((W, H), device=device, dtype=Xc.dtype)

    my_order = env._boundary_orders.get(ref)
    strength = float(env._semantic_strength(ref))
    cur = list(env.placed_order)

    if my_side in {'edge_left', 'edge_right'}:
        ref_axis = Y
        span = max(1e-6, env.task.bbox_mm[3] - env.task.bbox_mm[1])
    else:
        ref_axis = X
        span = max(1e-6, env.task.bbox_mm[2] - env.task.bbox_mm[0])

    delta = torch.zeros((W, H), device=device, dtype=Xc.dtype)
    for pref in cur:
        if pref == ref or env._same_side_groups.get(pref) != group:
            continue

        peer_side = env._side_preferences.get(pref, env._region_targets.get(pref, 'free'))
        if peer_side != my_side:
            delta = delta + strength * 0.50
            continue

        peer_order = env._boundary_orders.get(pref)
        if my_order is None or peer_order is None or my_order == peer_order:
            continue

        px, py, _ = env.placed[pref]
        pa = float(py) if my_side in {'edge_left', 'edge_right'} else float(px)

        if my_order < peer_order:
            term = (torch.clamp(ref_axis - pa, min=0) / span) ** 2
        else:
            term = (torch.clamp(pa - ref_axis, min=0) / span) ** 2

        delta = delta + strength * 2.0 * term

    return delta



def _compute_pitch_delta(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    cur = list(env.placed_order)
    cur_xy = {r: (float(env.placed[r][0]), float(env.placed[r][1])) for r in cur}
    new_order = cur + [ref]
    refs_new = env._pitch_group_refs(ref, new_order)
    existing = [r for r in refs_new if r != ref]
    K = len(existing)

    w_cells, h_cells = env.grid_shape()
    if K + 1 < 3:
        return torch.zeros((w_cells, h_cells), device=device, dtype=Xc.dtype)

    # Xc/Yc may be full [W,H] maps (cached-grid fast path) or skinny
    # [W,1]/[1,H] maps (older kernels).  Pitch only varies along one axis, so
    # use the unique coordinate vector instead of flattening the whole 2-D map.
    x_axis = Xc[:, 0] if Xc.dim() == 2 and Xc.shape[1] > 0 else Xc.reshape(-1)
    y_axis = Yc[0, :] if Yc.dim() == 2 and Yc.shape[0] > 0 else Yc.reshape(-1)

    test_xy = dict(cur_xy)
    test_xy[ref] = (float(x_axis.mean().item()), float(y_axis.mean().item()))
    axis = env._pitch_axis_for_group(refs_new, test_xy)

    if axis == 'x':
        span = max(1e-6, env.task.bbox_mm[2] - env.task.bbox_mm[0])
        ref_axis = x_axis
    else:
        span = max(1e-6, env.task.bbox_mm[3] - env.task.bbox_mm[1])
        ref_axis = y_axis

    ev = sorted(float(cur_xy[r][0] if axis == 'x' else cur_xy[r][1]) for r in existing)

    if K >= 3:
        ev_t = torch.tensor(ev, device=device, dtype=torch.float32)
        og = ev_t[1:] - ev_t[:-1]
        om = og.mean()
        old_pen = ((og - om) ** 2).mean() / (span * span) if om > 1e-6 else torch.tensor(0.0, device=device)
    else:
        old_pen = torch.tensor(0.0, device=device)

    ev_t = torch.tensor(ev, device=device, dtype=torch.float32)
    rf = ref_axis.reshape(-1)
    all_v = torch.cat([ev_t.unsqueeze(0).expand(rf.shape[0], -1), rf.unsqueeze(-1)], dim=-1)
    sv, _ = torch.sort(all_v, dim=-1)
    gaps = sv[:, 1:] - sv[:, :-1]
    mg = gaps.mean(dim=-1, keepdim=True)
    ok = mg.squeeze(-1) > 1e-6
    new_pen = ((gaps - mg) ** 2).mean(dim=-1) / (span * span) * ok.float()
    delta = float(env._semantic_strength(ref)) * (K + 1) * (new_pen - old_pen)
    if axis == "x":
        return delta.view(w_cells, 1).expand(w_cells, h_cells).contiguous()
    return delta.view(1, h_cells).expand(w_cells, h_cells).contiguous()


def _compute_line_neatness(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    bw: float, bh: float,
    placed_order: List[str],
    device: torch.device,
) -> torch.Tensor:
    my_g = env._explicit_functional_group.get(ref, '')
    my_a = env._align_groups.get(ref)
    pxs, pys, wts = [], [], []
    for pref in placed_order:
        if pref == ref or pref not in env.placed:
            continue
        if env._hard_boundary_required(pref):
            continue
        px, py, _ = env.placed[pref]
        w = 1.0
        if my_a and env._align_groups.get(pref) == my_a:
            w = max(w, 2.0)
        if my_g and env._explicit_functional_group.get(pref, '') == my_g:
            w = max(w, 1.6)
        pxs.append(px); pys.append(py); wts.append(w)

    if not pxs:
        return torch.zeros_like(Xc)

    px_t = torch.tensor(pxs, device=device, dtype=torch.float32)
    py_t = torch.tensor(pys, device=device, dtype=torch.float32)
    w_t = torch.tensor(wts, device=device, dtype=torch.float32)

    dx = torch.abs(Xc.unsqueeze(-1) - px_t) / w_t
    dy = torch.abs(Yc.unsqueeze(-1) - py_t) / w_t
    nx = dx.min(dim=-1).values
    ny = dy.min(dim=-1).values
    return torch.minimum(nx / bw, ny / bh) ** 2


# ---------------------------------------------------------------------------
# Rotation-dependent penalties  ¡ú  [W, H] (called per rotation)
# ---------------------------------------------------------------------------

def _compute_region(
    env: PlacementEnv, ref: str,
    a: torch.Tensor, b: torch.Tensor,
    c: torch.Tensor, d: torch.Tensor,
    Xc: torch.Tensor, Yc: torch.Tensor,
    board_diag: float,
) -> torch.Tensor:
    W = a.shape[0]
    H = b.shape[1]
    A = a.expand(W, H)
    B = b.expand(W, H)
    C = c.expand(W, H)
    D = d.expand(W, H)
    X = Xc.expand(W, H)
    Y = Yc.expand(W, H)

    role = env._region_role(ref)
    side_pref = env._side_preferences.get(ref, 'free')
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    gm = float(env.task.grid_mm)
    bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
    strength = float(env._semantic_strength(ref))

    def _edge_dist(clearance: torch.Tensor, side: str) -> torch.Tensor:
        tc = float(env._edge_target_clearance_mm(ref, side))
        return torch.abs(clearance - tc)

    if role == 'edge_left':
        dist = _edge_dist(A - xmin, 'left')
    elif role == 'edge_right':
        dist = _edge_dist(xmax - C, 'right')
    elif role == 'edge_bottom':
        dist = _edge_dist(B - ymin, 'bottom')
    elif role == 'edge_top':
        dist = _edge_dist(ymax - D, 'top')
    elif role == 'core':
        mx = min(0.40 * bw, max(gm, 0.28 * bw))
        my = min(0.40 * bh, max(gm, 0.28 * bh))
        dx = torch.clamp(xmin + mx - X, min=0) + torch.clamp(X - (xmax - mx), min=0)
        dy = torch.clamp(ymin + my - Y, min=0) + torch.clamp(Y - (ymax - my), min=0)
        dist = torch.sqrt(dx * dx + dy * dy)
    elif side_pref in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
        s = side_pref.split('_')[1]
        clr = {'left': A - xmin, 'right': xmax - C, 'bottom': B - ymin, 'top': ymax - D}
        dist = 0.35 * _edge_dist(clr[s], s)
    else:
        return torch.zeros((W, H), device=Xc.device, dtype=Xc.dtype)

    return float(strength) * (dist / board_diag) ** 2



def _compute_edge_clearance(
    env: PlacementEnv, ref: str,
    a: torch.Tensor, b: torch.Tensor,
    c: torch.Tensor, d: torch.Tensor,
) -> torch.Tensor:
    if not env._is_regularized_non_interface(env.comp_by_ref[ref]):
        return torch.zeros_like(a)
    margin = max(1e-6, float(env.non_interface_edge_margin_mm))
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    clearance = torch.minimum(
        torch.minimum(a - xmin, xmax - c),
        torch.minimum(b - ymin, ymax - d),
    )
    u = torch.clamp(margin - clearance, min=0) / margin
    return max(0.0, float(env.non_interface_edge_penalty)) * u * u


def _compute_soft_spacing(
    env: PlacementEnv, ref: str,
    a: torch.Tensor, b: torch.Tensor,
    c: torch.Tensor, d: torch.Tensor,
    pd: Dict[str, torch.Tensor],
) -> torch.Tensor:
    comp = env.comp_by_ref[ref]
    if not env._is_regularized_non_interface(comp):
        return torch.zeros_like(a)
    valid = pd['is_reg']
    if not valid.any():
        return torch.zeros_like(a)

    bbs = pd['bbs'][valid]
    sc = pd['soft_spacing_scale'][valid]
    pg = pd['pref_gaps'][valid]
    ms = float(env.min_spacing)

    oa, ob, oc, od = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]

    dx_gap = torch.clamp(torch.maximum(oa - c.unsqueeze(-1), a.unsqueeze(-1) - oc), min=0)
    dy_gap = torch.clamp(torch.maximum(ob - d.unsqueeze(-1), b.unsqueeze(-1) - od), min=0)
    both = (dx_gap > 0) & (dy_gap > 0)
    gap = torch.where(both, torch.sqrt(dx_gap ** 2 + dy_gap ** 2), torch.maximum(dx_gap, dy_gap))

    active = (gap >= ms) & (gap < pg)
    denom = torch.clamp(pg - ms, min=1e-6)
    u = (pg - gap) / denom
    return (sc * u * u * active.float()).sum(dim=-1)


def _compute_whitespace_balance(
    env: PlacementEnv, ref: str,
    a: torch.Tensor, b: torch.Tensor,
    c: torch.Tensor, d: torch.Tensor,
    pd: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    sem = env._semantic_class.get(ref, '')
    role = getattr(env, '_placement_roles', {}).get(ref, 'member')
    large_like = (
        role in {'anchor_large', 'edge_anchor', 'main_anchor'}
        or sem in {
            'core', 'power', 'power_support', 'clock', 'support',
            'large', 'mechanical', 'core_active', 'power_active'
        }
    )
    rw = 1.0 if large_like else 0.35

    W = a.shape[0]
    H = b.shape[1]
    A = a.expand(W, H)
    B = b.expand(W, H)
    C = c.expand(W, H)
    D = d.expand(W, H)

    xmin, ymin, xmax, ymax = env.task.bbox_mm
    left = A - xmin
    right = xmax - C
    bottom = B - ymin
    top = ymax - D

    n = pd['bbs'].shape[0] if (pd and 'bbs' in pd) else 0
    if n > 0:
        bbs = pd['bbs']
        oa, ob, oc, od = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]

        INF = torch.tensor(float('inf'), device=device, dtype=a.dtype)
        ov_y = torch.minimum(D.unsqueeze(-1), od) - torch.maximum(B.unsqueeze(-1), ob)
        ov_x = torch.minimum(C.unsqueeze(-1), oc) - torch.maximum(A.unsqueeze(-1), oa)

        is_l = (ov_y > 0) & (oc <= A.unsqueeze(-1))
        ld = torch.where(is_l, torch.clamp(A.unsqueeze(-1) - oc, min=0), INF)
        left = torch.minimum(left, ld.min(dim=-1).values)

        is_r = (ov_y > 0) & (oa >= C.unsqueeze(-1))
        rd = torch.where(is_r, torch.clamp(oa - C.unsqueeze(-1), min=0), INF)
        right = torch.minimum(right, rd.min(dim=-1).values)

        is_b = (ov_x > 0) & (od <= B.unsqueeze(-1))
        bd = torch.where(is_b, torch.clamp(B.unsqueeze(-1) - od, min=0), INF)
        bottom = torch.minimum(bottom, bd.min(dim=-1).values)

        is_t = (ov_x > 0) & (ob >= D.unsqueeze(-1))
        td = torch.where(is_t, torch.clamp(ob - D.unsqueeze(-1), min=0), INF)
        top = torch.minimum(top, td.min(dim=-1).values)

    left = torch.clamp(left, min=0)
    right = torch.clamp(right, min=0)
    bottom = torch.clamp(bottom, min=0)
    top = torch.clamp(top, min=0)

    eps = 1e-6
    lr = ((left - right) / (left + right + eps)) ** 2
    tb = ((top - bottom) / (top + bottom + eps)) ** 2
    return rw * (lr + tb)




def _compute_whitespace_floor(
    env: PlacementEnv, ref: str,
    a: torch.Tensor, b: torch.Tensor,
    c: torch.Tensor, d: torch.Tensor,
    pd: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    target_mm, role_weight = env._whitespace_floor_profile(ref)
    W = a.shape[0]
    H = b.shape[1]

    if role_weight <= 0.0 or target_mm <= 0.0:
        return torch.zeros((W, H), device=device, dtype=a.dtype)

    A = a.expand(W, H)
    B = b.expand(W, H)
    C = c.expand(W, H)
    D = d.expand(W, H)

    xmin, ymin, xmax, ymax = env.task.bbox_mm
    left = A - xmin
    right = xmax - C
    bottom = B - ymin
    top = ymax - D

    if pd is not None and pd['bbs'].numel() > 0:
        bbs = pd['bbs']
        oa, ob, oc, od = [bbs[:, i] for i in range(4)]

        INF = torch.tensor(1e9, device=device, dtype=a.dtype)

        ov_y = torch.minimum(D.unsqueeze(-1), od) - torch.maximum(B.unsqueeze(-1), ob)
        ov_x = torch.minimum(C.unsqueeze(-1), oc) - torch.maximum(A.unsqueeze(-1), oa)

        is_l = (ov_y > 0) & (oc <= A.unsqueeze(-1))
        ld = torch.where(is_l, torch.clamp(A.unsqueeze(-1) - oc, min=0), INF)
        left = torch.minimum(left, ld.min(dim=-1).values)

        is_r = (ov_y > 0) & (oa >= C.unsqueeze(-1))
        rd = torch.where(is_r, torch.clamp(oa - C.unsqueeze(-1), min=0), INF)
        right = torch.minimum(right, rd.min(dim=-1).values)

        is_b = (ov_x > 0) & (od <= B.unsqueeze(-1))
        bd = torch.where(is_b, torch.clamp(B.unsqueeze(-1) - od, min=0), INF)
        bottom = torch.minimum(bottom, bd.min(dim=-1).values)

        is_t = (ov_x > 0) & (ob >= D.unsqueeze(-1))
        td = torch.where(is_t, torch.clamp(ob - D.unsqueeze(-1), min=0), INF)
        top = torch.minimum(top, td.min(dim=-1).values)

    vals = torch.stack([
        torch.clamp(left, min=0),
        torch.clamp(right, min=0),
        torch.clamp(top, min=0),
        torch.clamp(bottom, min=0),
    ], dim=-1)

    smallest, _ = torch.topk(vals, k=2, dim=-1, largest=False)
    deficits = torch.clamp(float(target_mm) - smallest, min=0) / max(1e-6, float(target_mm))
    return float(role_weight) * (deficits * deficits).mean(dim=-1)


def _compute_module_region(
    env: PlacementEnv,
    ref: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c_t: torch.Tensor,
    d_t: torch.Tensor,
    board_diag: float,
) -> torch.Tensor:
    """Confidence-scaled module-region hint for candidate component bboxes.

    A broad/overlapping bbox is only a coarse allowed-zone signal.  The
    optional module shape hint adds an anchor-relative / edge-corridor / support
    ring term so wide regions do not behave like hard global boxes.
    """
    zero = torch.zeros_like(a + b)
    diag = max(1e-6, float(board_diag))
    conf = 0.0
    if hasattr(env, '_module_region_confidence_for_ref'):
        conf = float(env._module_region_confidence_for_ref(ref))
    rb = env._module_region_bbox_for_ref(ref)
    bbox_penalty = zero
    if rb is not None and conf > 0.0:
        margin = float(env.module_region_margin_mm)
        x0, y0, x1, y1 = [float(v) for v in rb]
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin
        over_x = torch.clamp(float(x0) - a, min=0) + torch.clamp(c_t - float(x1), min=0)
        over_y = torch.clamp(float(y0) - b, min=0) + torch.clamp(d_t - float(y1), min=0)
        bbox_penalty = ((over_x + over_y) / diag) ** 2

    X = 0.5 * (a + c_t)
    Y = 0.5 * (b + d_t)
    hint = env._module_shape_hint_for_ref(ref) if hasattr(env, '_module_shape_hint_for_ref') else {}
    shape_penalty = zero
    if hint:
        center = hint.get('module_center_mm')
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
            shape_penalty = shape_penalty + 0.20 * (torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / diag) ** 2
        anchor_xy = hint.get('anchor_xy_mm')
        zone = str(hint.get('anchor_relative_zone') or 'around').strip().lower()
        if isinstance(anchor_xy, (list, tuple)) and len(anchor_xy) == 2:
            ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
            tol = max(2.0 * float(env.task.grid_mm), 1.0)
            if zone == 'left':
                shape_penalty = shape_penalty + 0.55 * (torch.clamp(X - (ax - tol), min=0.0) / diag) ** 2
            elif zone == 'right':
                shape_penalty = shape_penalty + 0.55 * (torch.clamp((ax + tol) - X, min=0.0) / diag) ** 2
            elif zone == 'bottom':
                shape_penalty = shape_penalty + 0.55 * (torch.clamp(Y - (ay - tol), min=0.0) / diag) ** 2
            elif zone == 'top':
                shape_penalty = shape_penalty + 0.55 * (torch.clamp((ay + tol) - Y, min=0.0) / diag) ** 2
            elif zone in {'around', 'edge_aligned'}:
                radius = max(float(hint.get('support_ring_radius_mm') or 0.0), 2.5 * float(env.task.grid_mm))
                dist = torch.sqrt((X - ax) ** 2 + (Y - ay) ** 2)
                shape_penalty = shape_penalty + 0.35 * (torch.clamp(dist - radius, min=0.0) / diag) ** 2
        edge = str(hint.get('edge_corridor') or '').strip().lower()
        if edge in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
            band = max(2.5 * float(env.task.grid_mm), 0.08 * min(max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)))
            if edge == 'edge_left':
                dist_edge = X - xmin
            elif edge == 'edge_right':
                dist_edge = xmax - X
            elif edge == 'edge_bottom':
                dist_edge = Y - ymin
            else:
                dist_edge = ymax - Y
            shape_penalty = shape_penalty + 0.40 * (torch.clamp(dist_edge - band, min=0.0) / diag) ** 2
        shape_penalty = torch.clamp(shape_penalty, min=0.0, max=1.0)
    shape_conf = max(conf, 0.35 if hint else 0.0)
    return conf * bbox_penalty + shape_conf * shape_penalty


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def objective_delta_mask_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """GPU-accelerated replacement for PlacementEnv.objective_delta_mask.

    Returns dict of torch.Tensor on *device*, each shaped [R, W, H].
    """
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    grid_mm = float(env.task.grid_mm)
    board_diag = float(env.board_diag)
    bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
    cfg = env.objective_cfg

    grid_maps = _cached_grid_maps_cuda(env, device)
    Xc = grid_maps["x"]
    Yc = grid_maps["y"]

    # Keep the wire/HPWL terms on GPU. The previous implementation called
    # env.wire_delta_mask(ref), which rebuilt the most expensive wire maps with
    # NumPy on CPU and then copied them back to the GPU every suffix step.
    wire = wire_delta_mask_cuda(env, ref, device)
    hpwl = _coerce_rwh_map(wire['hpwl'], R, w_cells, h_cells, 'hpwl')
    w_hpwl = _coerce_rwh_map(wire['w_hpwl'], R, w_cells, h_cells, 'w_hpwl')
    nslw = _coerce_rwh_map(wire['nslw'], R, w_cells, h_cells, 'nslw')

    comp = env.comp_by_ref[ref]
    w0, h0 = float(comp.size_mm[0]), float(comp.size_mm[1])

    placed_order = list(env.placed_order)
    n_placed = len(placed_order)
    pd = _cached_placed_data_cuda(env, ref, placed_order, device) if n_placed else {}

    # --- rotation-independent terms [W, H] -> broadcast to [R, W, H] ---
    # Normalize every 2-D sub-map here.  Some kernels still use skinny grids or
    # axis-only outputs internally for speed, and a subset of semantic paths only
    # appears after several components are placed.  Canonicalizing at the boundary
    # keeps the objective composition safe for highly anisotropic boards.
    zero_2d = torch.zeros((w_cells, h_cells), device=device, dtype=Xc.dtype)
    conn_2d = _coerce_wh_map(_compute_conn(Xc, Yc, pd, board_diag), w_cells, h_cells, 'conn') if n_placed else zero_2d
    align_2d = _coerce_wh_map(_compute_align_delta(env, ref, Xc, Yc, bw, bh, device), w_cells, h_cells, 'align')
    group_2d = _coerce_wh_map(_compute_group_delta(env, ref, Xc, Yc, device), w_cells, h_cells, 'group')
    anchor_2d = _coerce_wh_map(_compute_anchor_delta(env, ref, Xc, Yc, board_diag, device), w_cells, h_cells, 'anchor')
    bgroup_2d = _coerce_wh_map(_compute_boundary_group_delta(env, ref, Xc, Yc, device), w_cells, h_cells, 'boundary_group')
    pitch_2d = _coerce_wh_map(_compute_pitch_delta(env, ref, Xc, Yc, device), w_cells, h_cells, 'pitch')
    interior_2d = _coerce_wh_map(_compute_interior(env, ref, Xc, Yc), w_cells, h_cells, 'interior')
    density_2d = _coerce_wh_map(_compute_density(env, ref, Xc, Yc, pd), w_cells, h_cells, 'density') if n_placed else zero_2d
    line_neat_2d = _coerce_wh_map(_compute_line_neatness(env, ref, Xc, Yc, bw, bh, placed_order, device), w_cells, h_cells, 'line_neatness') if n_placed else zero_2d

    def _bcast(t: torch.Tensor, name: str) -> torch.Tensor:
        return _coerce_rwh_map(t, R, w_cells, h_cells, name)

    conn = _bcast(conn_2d, 'conn')
    align = _bcast(align_2d, 'align')
    group = _bcast(group_2d, 'group')
    anchor = _bcast(anchor_2d, 'anchor')
    boundary_group = _bcast(bgroup_2d, 'boundary_group')
    pitch = _bcast(pitch_2d, 'pitch')
    interior = _bcast(interior_2d, 'interior')
    density = _bcast(density_2d, 'density')

    # --- rotation-dependent terms ---
    region = torch.zeros(R, w_cells, h_cells, device=device)
    module_region = torch.zeros_like(region)
    edge_clearance = torch.zeros_like(region)
    soft_spacing = torch.zeros_like(region)
    neatness = torch.zeros_like(region)
    orientation = torch.zeros_like(region)

    for ri, rot in enumerate(env.rotations):
        wm, hm = (w0, h0) if rot % 180 == 0 else (h0, w0)
        a = Xc - wm / 2
        b_t = Yc - hm / 2
        c_t = Xc + wm / 2
        d_t = Yc + hm / 2

        region[ri] = _coerce_wh_map(_compute_region(env, ref, a, b_t, c_t, d_t, Xc, Yc, board_diag), w_cells, h_cells, f'region[{ri}]')
        module_region[ri] = _coerce_wh_map(_compute_module_region(env, ref, a, b_t, c_t, d_t, board_diag), w_cells, h_cells, f'module_region[{ri}]')
        edge_clearance[ri] = _coerce_wh_map(_compute_edge_clearance(env, ref, a, b_t, c_t, d_t), w_cells, h_cells, f'edge_clearance[{ri}]')
        orientation[ri] = float(env._orientation_delta_for_candidate(ref, int(rot)))

        if n_placed:
            soft_spacing[ri] = _coerce_wh_map(_compute_soft_spacing(env, ref, a, b_t, c_t, d_t, pd), w_cells, h_cells, f'soft_spacing[{ri}]')
        wb = _coerce_wh_map(_compute_whitespace_balance(env, ref, a, b_t, c_t, d_t, pd if n_placed else None, device), w_cells, h_cells, f'whitespace_balance[{ri}]')
        wf = _coerce_wh_map(_compute_whitespace_floor(env, ref, a, b_t, c_t, d_t, pd if n_placed else None, device), w_cells, h_cells, f'whitespace_floor[{ri}]')
        neatness[ri] = _coerce_wh_map(0.50 * line_neat_2d + 0.20 * wb + 0.30 * wf, w_cells, h_cells, f'neatness[{ri}]')

    # One final normalization pass before the objective sum catches any future
    # sub-kernel that accidentally returns a transposed or broadcast-only view.
    objective_terms = {
        'hpwl': hpwl, 'w_hpwl': w_hpwl, 'nslw': nslw,
        'region': region, 'module_region': module_region, 'conn': conn,
        'align': align, 'group': group, 'anchor': anchor,
        'boundary_group': boundary_group, 'pitch': pitch, 'orientation': orientation,
        'edge_clearance': edge_clearance, 'interior': interior,
        'density': density, 'soft_spacing': soft_spacing, 'neatness': neatness,
    }
    objective_terms = {
        name: _coerce_rwh_map(value, R, w_cells, h_cells, name)
        for name, value in objective_terms.items()
    }
    hpwl = objective_terms['hpwl']; w_hpwl = objective_terms['w_hpwl']; nslw = objective_terms['nslw']
    region = objective_terms['region']; module_region = objective_terms['module_region']; conn = objective_terms['conn']
    align = objective_terms['align']; group = objective_terms['group']; anchor = objective_terms['anchor']
    boundary_group = objective_terms['boundary_group']; pitch = objective_terms['pitch']; orientation = objective_terms['orientation']
    edge_clearance = objective_terms['edge_clearance']; interior = objective_terms['interior']
    density = objective_terms['density']; soft_spacing = objective_terms['soft_spacing']; neatness = objective_terms['neatness']

    duplicated_action_terms = (
        cfg.module_region_weight * module_region
        + cfg.conn_weight * conn
        + cfg.anchor_weight * anchor
        + cfg.density_weight * density
        + cfg.soft_spacing_weight * soft_spacing
    )
    total = (
        cfg.hpwl_weight * hpwl
        + cfg.w_hpwl_weight * w_hpwl
        - cfg.nslw_weight * nslw
        + cfg.region_weight * region
        + duplicated_action_terms
        + cfg.align_weight * align
        + cfg.group_weight * group
        + cfg.boundary_group_weight * boundary_group
        + cfg.pitch_weight * pitch
        + cfg.orientation_weight * orientation
        + cfg.edge_clearance_weight * edge_clearance
        + cfg.interior_weight * interior
        + cfg.neatness_weight * neatness
    )
    # Used by teacher/inference when action-conditioned priors explicitly model
    # conn/anchor/module/spacing.  This prevents counting the same relation twice.
    residual_total = total - duplicated_action_terms

    return {
        'hpwl': hpwl, 'w_hpwl': w_hpwl, 'nslw': nslw,
        'region': region, 'module_region': module_region, 'conn': conn,
        'align': align, 'group': group,
        'anchor': anchor, 'boundary_group': boundary_group,
        'pitch': pitch, 'orientation': orientation,
        'edge_clearance': edge_clearance, 'interior': interior,
        'density': density, 'soft_spacing': soft_spacing,
        'neatness': neatness, 'total': total, 'residual_total': residual_total,
    }


# ---------------------------------------------------------------------------
# GPU-friendly teacher distribution (avoids numpy¡útorch round-trip)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_teacher_distribution_cuda(
    mask_flat: torch.Tensor,
    objective_flat: torch.Tensor,
    cfg,
    device: torch.device,
    region_flat: Optional[torch.Tensor] = None,
    action_prior_flat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU version of train.build_teacher_distribution.

    All inputs and outputs are torch tensors on *device*.
    Returns (q, cand) where cand is a bool tensor.
    """
    energy = objective_flat.clone()
    if region_flat is not None:
        energy = energy - float(cfg.lambda_region_prior) * region_flat
    if action_prior_flat is not None:
        energy = energy - action_prior_flat.reshape(-1).to(device=device, dtype=energy.dtype)

    legal = mask_flat > 0.5
    cand = legal.clone()

    if cfg.objective_delta_max is not None:
        cand2 = cand & (objective_flat <= float(cfg.objective_delta_max))
        if cand2.any():
            cand = cand2

    topk = int(cfg.topk) if cfg.topk is not None else 0
    if topk > 0:
        ci = torch.where(cand)[0]
        if ci.numel() > topk:
            _, tk = torch.topk(energy[ci], k=topk, largest=False)
            new_cand = torch.zeros_like(cand)
            new_cand[ci[tk]] = True
            cand = new_cand

    tlog = -energy / max(1e-6, float(cfg.tau))
    tlog2 = torch.full_like(tlog, -1e9)
    tlog2[cand] = tlog[cand]
    q = torch.softmax(tlog2, dim=-1)
    return q, cand


# ---------------------------------------------------------------------------
# GPU-accelerated wire delta mask computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pin_positions_cuda(env: PlacementEnv, device: torch.device) -> Dict[str, torch.Tensor]:
    """GPU version of _pin_positions - returns dict of net -> tensor(pins) on device.
    Optimized version with better memory usage and fewer operations.
    """
    if not env.placed:
        return {}
    
    result = {}
    
    # Process all placed components
    for ref, (x, y, rot) in env.placed.items():
        c = env.comp_by_ref[ref]
        angle = math.radians(rot)
        ca = math.cos(angle)
        sa = math.sin(angle)
        
        # Process all pads for this component
        for net, (rx, ry) in c.pads:
            if env._wire_net_ignored(net):
                continue
            # Compute pad position
            ax = ca * rx - sa * ry
            ay = sa * rx + ca * ry
            net_key = str(net)
            
            if net_key not in result:
                result[net_key] = []
            result[net_key].append((x + ax, y + ay))
    
    # Convert lists to tensors once all data is collected
    for net in result:
        result[net] = torch.tensor(
            result[net], 
            device=device, 
            dtype=torch.float32
        )
    
    return result


@torch.no_grad()
def wire_delta_mask_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """GPU-accelerated replacement for PlacementEnv.wire_delta_mask.
    
    Returns dict of torch.Tensor on *device*, each shaped [R, W, H].
    Optimized version with better vectorization.
    """
    w_cells, h_cells = env.grid_shape()
    grid_maps = _cached_grid_maps_cuda(env, device)
    # Reuse cached skinny grid columns/rows for HPWL/NSLW math to keep the wire
    # kernel memory-light.  Outputs are normalized to canonical [R, W, H] below
    # before they are composed with the other objective maps.
    Xc = grid_maps["x_col"]
    Yc = grid_maps["y_row"]
    
    # Get placed pins
    pins = _pin_positions_cuda(env, device)
    
    # Build net_bbox and net_deg on GPU
    net_bbox = {}
    net_deg = {}
    for net, pts in pins.items():
        if pts.shape[0] == 0:
            continue
        xs = pts[:, 0]
        ys = pts[:, 1]
        net_bbox[net] = (xs.min(), xs.max(), ys.min(), ys.max())
        net_deg[net] = pts.shape[0]
    
    # Get component pads
    c = env.comp_by_ref[ref]
    pads_by_net = {}
    for net, (rx, ry) in c.pads:
        if env._wire_net_ignored(net):
            continue
        pads_by_net.setdefault(str(net), []).append((float(rx), float(ry)))
    
    R = len(env.rotations)
    hpwl_delta = torch.zeros(R, w_cells, h_cells, device=device, dtype=torch.float32)
    w_hpwl_delta = torch.zeros(R, w_cells, h_cells, device=device, dtype=torch.float32)
    nslw_delta = torch.zeros(R, w_cells, h_cells, device=device, dtype=torch.float32)
    
    # Precompute all rotations first
    rotation_info = []
    for rot in env.rotations:
        angle = math.radians(rot)
        ca = math.cos(angle)
        sa = math.sin(angle)
        rotation_info.append((ca, sa))
    
    # Process each rotation
    for ri, (ca, sa) in enumerate(rotation_info):
        if pads_by_net:
            for net, rels in pads_by_net.items():
                d0 = int(net_deg.get(net, 0))
                
                ex_minx = ex_maxx = ex_miny = ex_maxy = 0.0
                if d0 > 0 and net in net_bbox:
                    ex_minx, ex_maxx, ex_miny, ex_maxy = net_bbox[net]
                
                # Transform relative pad positions in one go
                dxs = []
                dys = []
                for rx, ry in rels:
                    dxs.append(ca * rx - sa * ry)
                    dys.append(sa * rx + ca * ry)
                
                # Compute component bounding box for all grid positions
                if dxs:
                    # Stack and compute min/max
                    px_stack = torch.stack([Xc + dx for dx in dxs], dim=0)
                    py_stack = torch.stack([Yc + dy for dy in dys], dim=0)
                    
                    comp_minx = px_stack.min(dim=0).values
                    comp_maxx = px_stack.max(dim=0).values
                    comp_miny = py_stack.min(dim=0).values
                    comp_maxy = py_stack.max(dim=0).values
                    
                    d_comp = len(rels)
                    d_new = d0 + d_comp
                    
                    # Compute old HPWL
                    hpwl_old = 0.0
                    w_hpwl_old = 0.0
                    if d0 > 1:
                        hpwl_old = (ex_maxx - ex_minx) + (ex_maxy - ex_miny)
                        w_hpwl_old = hpwl_old * math.log(1.0 + d0)
                    
                    # Compute new HPWL
                    if d0 == 0:
                        new_minx = comp_minx
                        new_maxx = comp_maxx
                        new_miny = comp_miny
                        new_maxy = comp_maxy
                    else:
                        new_minx = torch.min(comp_minx, ex_minx)
                        new_maxx = torch.max(comp_maxx, ex_maxx)
                        new_miny = torch.min(comp_miny, ex_miny)
                        new_maxy = torch.max(comp_maxy, ex_maxy)
                    
                    hpwl_new = 0.0
                    w_hpwl_new = 0.0
                    if d_new > 1:
                        hpwl_new = (new_maxx - new_minx) + (new_maxy - new_miny)
                        w_hpwl_new = hpwl_new * math.log(1.0 + d_new)
                    
                    # Update deltas in place
                    hpwl_delta[ri].add_(hpwl_new - hpwl_old)
                    w_hpwl_delta[ri].add_(w_hpwl_new - w_hpwl_old)
        
        # NSLW computation
        if env.objective_cfg.nslw_weight != 0.0 and env.placed_order:
            nslw_delta[ri] = _compute_nslw_delta_cuda(env, ref, Xc, Yc, env.rotations[ri], device)
    
    hpwl_delta = _coerce_rwh_map(hpwl_delta, R, w_cells, h_cells, 'wire.hpwl')
    w_hpwl_delta = _coerce_rwh_map(w_hpwl_delta, R, w_cells, h_cells, 'wire.w_hpwl')
    nslw_delta = _coerce_rwh_map(nslw_delta, R, w_cells, h_cells, 'wire.nslw')
    total = (
        env.objective_cfg.hpwl_weight * hpwl_delta
        + env.objective_cfg.w_hpwl_weight * w_hpwl_delta
        - env.objective_cfg.nslw_weight * nslw_delta
    )
    
    return {
        'hpwl': hpwl_delta,
        'w_hpwl': w_hpwl_delta,
        'nslw': nslw_delta,
        'total': total,
    }


@torch.no_grad()
def _compute_nslw_delta_cuda(
    env: PlacementEnv,
    ref: str,
    Xc: torch.Tensor,
    Yc: torch.Tensor,
    rot: int,
    device: torch.device,
) -> torch.Tensor:
    """True PCBAgent-style NSLW delta for candidate placements.

    This mirrors ``PlacementEnv._is_surface_layer_wire_between`` instead of the
    previous approximate alignment score.  A candidate-to-placed pad pair counts
    as one surface-layer wire only when all PCBAgent-style conditions hold:

    1. the two pads are horizontally or vertically aligned within tolerance;
    2. the candidate fan-out direction uses one of the candidate pad's two
       nearest component edges;
    3. the placed-pad fan-out direction uses one of that pad's two nearest
       component edges;
    4. the straight axis-aligned segment is not blocked by any other placed
       component rectangle.

    The returned tensor is a true count per candidate grid cell; it is not
    clamped to [0, 1].
    """
    w_cells, h_cells = env.grid_shape()
    result = torch.zeros(w_cells, h_cells, device=device, dtype=torch.float32)

    if not env.placed_order:
        return result

    c = env.comp_by_ref[ref]
    pin_records = env._placed_pin_records()
    if not pin_records:
        return result

    angle = math.radians(int(rot))
    ca, sa = math.cos(angle), math.sin(angle)

    c_w, c_h = [float(v) for v in c.size_mm]
    if int(rot) % 180 != 0:
        c_w, c_h = c_h, c_w

    align_tol = max(0.05, 0.25 * float(env.task.grid_mm))
    eps = 1e-6

    edge_bits = {"left": 1, "right": 2, "bottom": 4, "top": 8}

    def _edge_mask(edges: Any) -> int:
        mask = 0
        for edge in edges:
            mask |= edge_bits[str(edge)]
        return int(mask)

    def _candidate_edge_maps(dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mathematically the nearest edges are translation-invariant, but the CPU
        # exact path evaluates absolute pad/bbox coordinates in float32 grid
        # coordinates.  At exact ties this can change the selected two edges.
        # Compute the edge choice per grid cell from the same absolute values.
        X64 = Xc.to(torch.float64).expand(w_cells, h_cells)
        Y64 = Yc.to(torch.float64).expand(w_cells, h_cells)
        cand_x64 = X64 + float(dx)
        cand_y64 = Y64 + float(dy)
        left = X64 - c_w / 2.0
        right = X64 + c_w / 2.0
        bottom = Y64 - c_h / 2.0
        top = Y64 + c_h / 2.0
        dists = torch.stack([
            torch.abs(cand_x64 - left),
            torch.abs(cand_x64 - right),
            torch.abs(cand_y64 - bottom),
            torch.abs(cand_y64 - top),
        ], dim=0)
        # Stable CPU sort order is left, right, bottom, top for exact ties.
        order_eps = torch.tensor([0.0, 1e-15, 2e-15, 3e-15], device=device, dtype=torch.float64).view(4, 1, 1)
        order = torch.argsort(dists + order_eps, dim=0)
        top2 = order[:2]
        return (
            (top2 == 0).any(dim=0),
            (top2 == 1).any(dim=0),
            (top2 == 2).any(dim=0),
            (top2 == 3).any(dim=0),
        )

    # Placed component rectangles are constant for this candidate step.  They
    # are also the blockers used by the CPU exact implementation, excluding the
    # target endpoint component for each candidate-to-placed segment.
    placed_refs = list(env.placed_order)
    placed_bbs_cpu = {pref: env._ref_bbox(pref, *env.placed[pref]) for pref in placed_refs}
    placed_bbs_t = torch.tensor(
        [placed_bbs_cpu[pref] for pref in placed_refs],
        device=device,
        dtype=torch.float32,
    )
    placed_ref_to_idx = {pref: i for i, pref in enumerate(placed_refs)}
    n_blockers = int(placed_bbs_t.shape[0])

    # Candidate pads grouped by net, with their rotation-applied offsets and
    # precomputed nearest-edge masks.
    cand_pads_by_net: Dict[str, List[Tuple[float, float]]] = {}
    for net, (rx, ry) in c.pads:
        if env._wire_net_ignored(net):
            continue
        net_key = str(net)
        if net_key not in pin_records:
            continue
        dx = ca * float(rx) - sa * float(ry)
        dy = sa * float(rx) + ca * float(ry)
        cand_pads_by_net.setdefault(net_key, []).append((dx, dy))

    if not cand_pads_by_net:
        return result

    # Cache placed-pad nearest-edge masks.  Using the CPU helper here is
    # intentional: it keeps the exact same tie-breaking and edge semantics as
    # PlacementEnv._is_surface_layer_wire_between.
    placed_pin_cache: Dict[Tuple[str, float, float], int] = {}
    for recs in pin_records.values():
        for pref, qx, qy in recs:
            key = (str(pref), float(qx), float(qy))
            if key not in placed_pin_cache and pref in placed_bbs_cpu:
                placed_pin_cache[key] = _edge_mask(env._closest_pad_edges((float(qx), float(qy)), placed_bbs_cpu[pref]))

    # Evaluate true NSLW over same-net candidate pad / placed pin pairs.  The
    # Python loop is over sparse net-connected pad pairs only; each pair still
    # evaluates the whole candidate grid and all blockers with tensor ops.
    for net_key, cand_pads in cand_pads_by_net.items():
        placed_pins = pin_records.get(net_key, [])
        if not placed_pins:
            continue

        for dx, dy in cand_pads:
            cand_x = Xc + float(dx)
            cand_y = Yc + float(dy)
            cand_has_left, cand_has_right, cand_has_bottom, cand_has_top = _candidate_edge_maps(float(dx), float(dy))

            for pref, qx, qy in placed_pins:
                pref = str(pref)
                if pref not in placed_ref_to_idx:
                    continue
                placed_edge_mask = placed_pin_cache.get((pref, float(qx), float(qy)), 0)
                if placed_edge_mask == 0:
                    continue

                qx_f = float(qx)
                qy_f = float(qy)
                target_idx = placed_ref_to_idx[pref]

                pair_ok = torch.zeros_like(result, dtype=torch.bool)

                # Horizontal surface-layer wire candidate.
                if placed_edge_mask & (edge_bits["left"] | edge_bits["right"]):
                    aligned_x_axis = torch.abs(cand_y - qy_f) <= align_tol
                    cand_edge_ok = ((qx_f >= cand_x) & cand_has_right) | ((qx_f < cand_x) & cand_has_left)

                    placed_edge_ok = torch.zeros_like(result, dtype=torch.bool)
                    if placed_edge_mask & edge_bits["right"]:
                        placed_edge_ok |= cand_x >= qx_f
                    if placed_edge_mask & edge_bits["left"]:
                        placed_edge_ok |= cand_x < qx_f

                    ok_x = aligned_x_axis & cand_edge_ok & placed_edge_ok
                    if bool(ok_x.any().item()):
                        if n_blockers > 1:
                            bb = placed_bbs_t
                            blocker_mask = torch.ones(n_blockers, device=device, dtype=torch.bool)
                            blocker_mask[target_idx] = False
                            bb = bb[blocker_mask]
                            if bb.numel() > 0:
                                a = bb[:, 0].view(-1, 1, 1)
                                b = bb[:, 1].view(-1, 1, 1)
                                cbb = bb[:, 2].view(-1, 1, 1)
                                d = bb[:, 3].view(-1, 1, 1)
                                y_mid = 0.5 * (cand_y + qy_f)
                                lo = torch.minimum(cand_x, torch.tensor(qx_f, device=device, dtype=torch.float32))
                                hi = torch.maximum(cand_x, torch.tensor(qx_f, device=device, dtype=torch.float32))
                                hits = (
                                    (b + eps < y_mid.unsqueeze(0))
                                    & (y_mid.unsqueeze(0) < d - eps)
                                    & (torch.maximum(lo.unsqueeze(0), a) + eps < torch.minimum(hi.unsqueeze(0), cbb) - eps)
                                )
                                ok_x &= ~hits.any(dim=0)
                        pair_ok |= ok_x

                # Vertical surface-layer wire candidate.
                if placed_edge_mask & (edge_bits["bottom"] | edge_bits["top"]):
                    aligned_y_axis = torch.abs(cand_x - qx_f) <= align_tol
                    cand_edge_ok = ((qy_f >= cand_y) & cand_has_top) | ((qy_f < cand_y) & cand_has_bottom)

                    placed_edge_ok = torch.zeros_like(result, dtype=torch.bool)
                    if placed_edge_mask & edge_bits["top"]:
                        placed_edge_ok |= cand_y >= qy_f
                    if placed_edge_mask & edge_bits["bottom"]:
                        placed_edge_ok |= cand_y < qy_f

                    ok_y = aligned_y_axis & cand_edge_ok & placed_edge_ok
                    if bool(ok_y.any().item()):
                        if n_blockers > 1:
                            bb = placed_bbs_t
                            blocker_mask = torch.ones(n_blockers, device=device, dtype=torch.bool)
                            blocker_mask[target_idx] = False
                            bb = bb[blocker_mask]
                            if bb.numel() > 0:
                                a = bb[:, 0].view(-1, 1, 1)
                                b = bb[:, 1].view(-1, 1, 1)
                                cbb = bb[:, 2].view(-1, 1, 1)
                                d = bb[:, 3].view(-1, 1, 1)
                                x_mid = 0.5 * (cand_x + qx_f)
                                lo = torch.minimum(cand_y, torch.tensor(qy_f, device=device, dtype=torch.float32))
                                hi = torch.maximum(cand_y, torch.tensor(qy_f, device=device, dtype=torch.float32))
                                hits = (
                                    (a + eps < x_mid.unsqueeze(0))
                                    & (x_mid.unsqueeze(0) < cbb - eps)
                                    & (torch.maximum(lo.unsqueeze(0), b) + eps < torch.minimum(hi.unsqueeze(0), d) - eps)
                                )
                                ok_y &= ~hits.any(dim=0)
                        pair_ok |= ok_y

                result.add_(pair_ok.to(torch.float32))

    return result


# ---------------------------------------------------------------------------
# GPU-accelerated action features computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _static_action_features_cuda(env: PlacementEnv, device: torch.device) -> torch.Tensor:
    """Board-static action grid features [A,4]. Safe to cache per board."""
    grid_maps = _cached_grid_maps_cuda(env, device)
    w, h = env.grid_shape()
    R = len(env.rotations)
    Xn = grid_maps["x_norm"]
    Yn = grid_maps["y_norm"]

    feats = torch.zeros(R, w, h, 4, device=device, dtype=torch.float32)
    for ri, rot in enumerate(env.rotations):
        rr = float(rot) * math.pi / 180.0
        feats[ri, :, :, 0] = Xn
        feats[ri, :, :, 1] = Yn
        feats[ri, :, :, 2] = math.sin(rr)
        feats[ri, :, :, 3] = math.cos(rr)
    return feats.reshape(-1, 4)


def _masked_zscore_flat(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    x = x.reshape(-1).to(dtype=torch.float32)
    if mask is None:
        valid = torch.isfinite(x)
    else:
        valid = (mask.reshape(-1) > 0.5) & torch.isfinite(x)
    if not bool(valid.any().item()):
        return torch.zeros_like(x)
    vals = x[valid]
    mu = vals.mean()
    sigma = vals.std(unbiased=False).clamp(min=1e-6)
    z = (x - mu) / sigma
    return torch.clamp(z, min=-3.0, max=3.0).masked_fill(~torch.isfinite(z), 0.0)


def _module_record_for_ref(env: PlacementEnv, ref: str) -> Dict[str, Any]:
    comp = env.comp_by_ref.get(ref)
    mid = str(getattr(comp, 'module_id', '') or '') if comp is not None else ''
    if not mid:
        return {}
    for m in (getattr(env.task, 'modules', None) or []):
        if isinstance(m, dict) and str(m.get('module_id', '') or '') == mid:
            return m
    return {}


def _prior_heatmap_payload_for_ref(env: PlacementEnv, ref: str) -> Dict[str, Any]:
    m = _module_record_for_ref(env, ref)
    payload = m.get('prior_region_heatmap') if isinstance(m.get('prior_region_heatmap'), dict) else None
    if isinstance(payload, dict):
        return payload
    comp = env.comp_by_ref.get(ref)
    module = getattr(comp, 'module', None) if comp is not None else None
    if isinstance(module, dict) and isinstance(module.get('prior_region_heatmap'), dict):
        return module.get('prior_region_heatmap')
    return {}


def _prior_heatmap_maps_cuda(env: PlacementEnv, ref: str, X: torch.Tensor, Y: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.zeros_like(X)
    payload = _prior_heatmap_payload_for_ref(env, ref)
    values = payload.get('values') if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return {
            'prior_heatmap_value_at_action': zero,
            'prior_heatmap_logprob_at_action': zero,
            'dist_to_prior_peak_norm': torch.ones_like(X),
            'inside_prior_mass_50': zero,
            'module_prior_confidence': zero,
            'prior_heatmap_raw_logprob': zero,
        }
    gx = int(payload.get('grid_x', 0) or 0)
    gy = int(payload.get('grid_y', 0) or 0)
    if gx <= 0 or gy <= 0:
        # Accept flattened values as a square-ish fallback only when metadata is absent.
        flat_len = len(values) if isinstance(values, list) else 0
        gx = int(round(math.sqrt(max(1, flat_len))))
        gy = max(1, flat_len // max(1, gx))
    try:
        heat = torch.tensor(values, device=device, dtype=torch.float32).reshape(gx * gy)
    except Exception:
        return {
            'prior_heatmap_value_at_action': zero,
            'prior_heatmap_logprob_at_action': zero,
            'dist_to_prior_peak_norm': torch.ones_like(X),
            'inside_prior_mass_50': zero,
            'module_prior_confidence': zero,
            'prior_heatmap_raw_logprob': zero,
        }
    heat = torch.clamp(heat, min=0.0)
    heat = heat / heat.sum().clamp(min=1e-8)
    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    tx = (X - xmin) / max(1e-6, xmax - xmin)
    ty = (Y - ymin) / max(1e-6, ymax - ymin)
    ix = torch.clamp((tx * gx).floor().long(), 0, gx - 1)
    iy = torch.clamp((ty * gy).floor().long(), 0, gy - 1)
    idx = ix * gy + iy
    val = heat[idx]
    # Raw log-prob is used by teacher action prior.  Feature log-prob is centered
    # by log(num_bins) and clipped to a compact numeric range.
    raw_logp = torch.log(val.clamp(min=1e-8))
    logp_feat = torch.clamp(raw_logp + math.log(float(max(1, gx * gy))), min=-6.0, max=6.0) / 6.0
    peak_idx = int(torch.argmax(heat).item())
    peak_xi = peak_idx // gy
    peak_yi = peak_idx % gy
    cell_w = max(1e-6, (xmax - xmin) / float(gx))
    cell_h = max(1e-6, (ymax - ymin) / float(gy))
    px = xmin + (float(peak_xi) + 0.5) * cell_w
    py = ymin + (float(peak_yi) + 0.5) * cell_h
    diag = max(1e-6, float(getattr(env, 'board_diag', math.hypot(xmax - xmin, ymax - ymin))))
    dist_peak = torch.sqrt((X - px) ** 2 + (Y - py) ** 2) / diag
    sorted_vals, sorted_idx = torch.sort(heat, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    top_mask_flat = torch.zeros_like(heat, dtype=torch.bool)
    top_mask_flat[sorted_idx[cumsum <= 0.50]] = True
    if not bool(top_mask_flat.any().item()):
        top_mask_flat[peak_idx] = True
    inside50 = top_mask_flat[idx].to(torch.float32)
    conf = float(env._module_region_confidence_for_ref(ref)) if hasattr(env, '_module_region_confidence_for_ref') else 0.0
    conf_map = torch.full_like(X, max(0.0, min(1.0, conf)))
    return {
        'prior_heatmap_value_at_action': val,
        'prior_heatmap_logprob_at_action': logp_feat,
        'dist_to_prior_peak_norm': dist_peak.clamp(0.0, 1.0),
        'inside_prior_mass_50': inside50,
        'module_prior_confidence': conf_map,
        'prior_heatmap_raw_logprob': raw_logp,
    }


def _edge_sides_for_ref(env: PlacementEnv, ref: str) -> List[str]:
    sides: List[str] = []
    try:
        sides = list(env._edge_band_sides(ref) or [])
    except Exception:
        sides = []
    if not sides:
        for label in (getattr(env, '_side_preferences', {}).get(ref, ''), getattr(env, '_region_targets', {}).get(ref, '')):
            label = str(label or '').strip().lower()
            if label in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
                sides.append(label.split('_', 1)[1])
    return list(dict.fromkeys([s for s in sides if s in {'left','right','top','bottom'}]))





def _action_dynamic_maps_cache_key(env: PlacementEnv, ref: str, device: torch.device) -> Tuple[Any, ...]:
    dev = torch.device(device)
    return (
        f"{dev.type}:{-1 if dev.index is None else int(dev.index)}",
        str(ref),
        tuple(env.grid_shape()),
        tuple(env.rotations),
        tuple(
            (
                str(r),
                float(env.placed[r][0]),
                float(env.placed[r][1]),
                int(env.placed[r][2]),
            )
            for r in getattr(env, 'placed_order', [])
            if r in getattr(env, 'placed', {})
        ),
    )


@torch.no_grad()
def _action_dynamic_maps_cuda(env: PlacementEnv, ref: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Cache full ref/state action-dynamic maps for back-to-back feature/prior calls.

    Training builds action features and action priors consecutively for the same
    env/ref/state.  Both used to rebuild anchor/net/module/spacing/prior maps.
    A tiny per-env cache keyed by placed state lets the second call reuse the
    already-built GPU tensors while automatically missing after env.step().
    """
    key = _action_dynamic_maps_cache_key(env, ref, device)
    cache = getattr(env, '_cuda_action_dynamic_maps_cache', None)
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    out = _compute_action_dynamic_maps_cuda(env, ref, device)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, '_cuda_action_dynamic_maps_cache', cache)
    if len(cache) >= 2:
        cache.clear()
    cache[key] = out
    return out


@torch.no_grad()
def _compute_action_dynamic_maps_cuda(env: PlacementEnv, ref: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Return ref/state-conditioned feature maps shaped [R,W,H] or [W,H]."""
    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    bw = max(1e-6, xmax - xmin)
    bh = max(1e-6, ymax - ymin)
    diag = max(1e-6, float(env.board_diag))
    grid = float(env.task.grid_mm)
    w, h = env.grid_shape()
    R = len(env.rotations)

    grid_maps = _cached_grid_maps_cuda(env, device)
    X = grid_maps["x"]
    Y = grid_maps["y"]
    zero2 = torch.zeros_like(X)

    out: Dict[str, torch.Tensor] = {}

    # Anchor relation: valid only when the effective anchor has already been placed.
    anchor_ref = str(getattr(env, '_anchor_refs', {}).get(ref, '') or '').strip()
    if anchor_ref and anchor_ref in env.placed:
        ax, ay, _ = env.placed[anchor_ref]
        dx = (X - float(ax)) / diag
        dy = (Y - float(ay)) / diag
        dist = torch.sqrt(dx * dx + dy * dy)
        out['dx_to_anchor'] = dx
        out['dy_to_anchor'] = dy
        out['dist_to_anchor'] = dist
        out['has_anchor_xy'] = torch.ones_like(X)
        anchor_prior = -dist
    else:
        out['dx_to_anchor'] = zero2
        out['dy_to_anchor'] = zero2
        out['dist_to_anchor'] = zero2
        out['has_anchor_xy'] = zero2
        anchor_prior = zero2

    placed = list(env.placed_order)
    pd = _cached_placed_data_cuda(env, ref, placed, device) if placed else {}

    # Connected centroid over already placed, non-ignored net-related components.
    conn_w = pd.get('conn_w')
    if conn_w is not None and bool((conn_w > 1e-9).any().item()):
        m = conn_w > 1e-9
        ww = conn_w[m].clamp(min=0.0)
        denom = ww.sum().clamp(min=1e-6)
        cx = (pd['px'][m] * ww).sum() / denom
        cy = (pd['py'][m] * ww).sum() / denom
        dx = (X - cx) / diag
        dy = (Y - cy) / diag
        dist = torch.sqrt(dx * dx + dy * dy)
        out['dx_to_conn_centroid'] = dx
        out['dy_to_conn_centroid'] = dy
        out['dist_to_conn_centroid'] = dist
        out['has_conn_centroid'] = torch.ones_like(X)
        conn_prior = -dist
    else:
        out['dx_to_conn_centroid'] = zero2
        out['dy_to_conn_centroid'] = zero2
        out['dist_to_conn_centroid'] = zero2
        out['has_conn_centroid'] = zero2
        conn_prior = zero2

    # Module center / bbox confidence. Prefer explicit shape hint center.
    hint = env._module_shape_hint_for_ref(ref) if hasattr(env, '_module_shape_hint_for_ref') else {}
    center = hint.get('module_center_mm') if isinstance(hint, dict) else None
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        rb = env._module_region_bbox_for_ref(ref) if hasattr(env, '_module_region_bbox_for_ref') else None
        center = [(float(rb[0]) + float(rb[2])) * 0.5, (float(rb[1]) + float(rb[3])) * 0.5] if rb and len(rb) == 4 else None
    if isinstance(center, (list, tuple)) and len(center) == 2:
        mcx, mcy = float(center[0]), float(center[1])
        dx = (X - mcx) / diag
        dy = (Y - mcy) / diag
        dist = torch.sqrt(dx * dx + dy * dy)
        out['dx_to_module_center'] = dx
        out['dy_to_module_center'] = dy
        out['dist_to_module_center'] = dist
        out['has_module_center'] = torch.ones_like(X)
        module_center_prior = -dist
    else:
        out['dx_to_module_center'] = zero2
        out['dy_to_module_center'] = zero2
        out['dist_to_module_center'] = zero2
        out['has_module_center'] = zero2
        module_center_prior = zero2

    rb = env._module_region_bbox_for_ref(ref) if hasattr(env, '_module_region_bbox_for_ref') else None
    conf = float(env._module_region_confidence_for_ref(ref)) if hasattr(env, '_module_region_confidence_for_ref') else 0.0
    if rb and len(rb) == 4:
        x0, y0, x1, y1 = [float(v) for v in rb]
        margin = float(getattr(env, 'module_region_margin_mm', 0.0))
        x0 -= margin; y0 -= margin; x1 += margin; y1 += margin
        dx_out = torch.clamp(x0 - X, min=0.0) + torch.clamp(X - x1, min=0.0)
        dy_out = torch.clamp(y0 - Y, min=0.0) + torch.clamp(Y - y1, min=0.0)
        dist_bbox = torch.sqrt(dx_out * dx_out + dy_out * dy_out) / diag
        inside = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)).to(torch.float32)
        out['inside_module_bbox'] = inside
        out['dist_to_module_bbox'] = dist_bbox
        module_bbox_prior = -dist_bbox
    else:
        out['inside_module_bbox'] = zero2
        out['dist_to_module_bbox'] = zero2
        module_bbox_prior = zero2
    out['module_region_confidence'] = torch.full_like(X, max(0.0, min(1.0, conf)))
    out['module_prior_raw'] = conf * module_bbox_prior + max(conf, 0.35 if hint else 0.0) * module_center_prior

    # Placed relation distances.
    px = pd.get('px')
    py = pd.get('py')
    if px is not None and px.numel() > 0:
        dx_all = X.unsqueeze(-1) - px
        dy_all = Y.unsqueeze(-1) - py
        dist_all_mm = torch.sqrt(dx_all * dx_all + dy_all * dy_all).clamp(min=0.0)
        dist_all = dist_all_mm / diag
        min_all = dist_all.min(dim=-1).values
        out['min_gap_to_placed'] = min_all
        radius = max(float(getattr(env, 'density_radius_mm', 4.0)), 2.0 * grid)
        out['local_density'] = torch.exp(-((dist_all_mm / max(1e-6, radius)) ** 2)).sum(dim=-1) / max(1.0, float(px.numel()))

        same_module_mask = torch.tensor([env._pair_primary_relation(ref, r) == 'module' for r in placed], device=device, dtype=torch.bool)
        same_fg_mask = torch.tensor([env._pair_primary_relation(ref, r) == 'functional_group' for r in placed], device=device, dtype=torch.bool)
        critical_mask = torch.tensor([env._pair_primary_relation(ref, r) == 'critical' for r in placed], device=device, dtype=torch.bool)
        def _nearest(mask: torch.Tensor) -> torch.Tensor:
            if mask.numel() == 0 or not bool(mask.any().item()):
                return torch.ones_like(X)
            return dist_all[..., mask].min(dim=-1).values.clamp(max=1.0)
        out['nearest_same_module_dist'] = _nearest(same_module_mask)
        out['nearest_same_functional_group_dist'] = _nearest(same_fg_mask)
        out['nearest_critical_neighbor_dist'] = _nearest(critical_mask)
        out['placed_same_module_count_norm'] = torch.full_like(X, float(same_module_mask.sum().item()) / max(1.0, float(len(placed))))
    else:
        out['min_gap_to_placed'] = torch.ones_like(X)
        out['local_density'] = zero2
        out['nearest_same_module_dist'] = torch.ones_like(X)
        out['nearest_same_functional_group_dist'] = torch.ones_like(X)
        out['nearest_critical_neighbor_dist'] = torch.ones_like(X)
        out['placed_same_module_count_norm'] = zero2

    # Rotation-dependent box features.
    comp = env.comp_by_ref[ref]
    w0, h0 = float(comp.size_mm[0]), float(comp.size_mm[1])
    left_clear = torch.zeros(R, w, h, device=device)
    right_clear = torch.zeros_like(left_clear)
    top_clear = torch.zeros_like(left_clear)
    bottom_clear = torch.zeros_like(left_clear)
    soft_pen = torch.zeros_like(left_clear)
    overlap = torch.zeros_like(left_clear)
    min_rect_gap = torch.ones_like(left_clear)
    for ri, rot in enumerate(env.rotations):
        wm, hm = (w0, h0) if int(rot) % 180 == 0 else (h0, w0)
        a = X - wm / 2.0
        b = Y - hm / 2.0
        c = X + wm / 2.0
        d = Y + hm / 2.0
        left_clear[ri] = torch.clamp(a - xmin, min=0.0) / bw
        right_clear[ri] = torch.clamp(xmax - c, min=0.0) / bw
        bottom_clear[ri] = torch.clamp(b - ymin, min=0.0) / bh
        top_clear[ri] = torch.clamp(ymax - d, min=0.0) / bh
        if pd:
            ss = _compute_soft_spacing(env, ref, a, b, c, d, pd)
            soft_pen[ri] = ss.clamp(min=0.0, max=3.0) / 3.0
            bbs = pd['bbs']
            oa, ob, oc, od = [bbs[:, j] for j in range(4)]
            sep_x = torch.clamp(torch.maximum(oa - c.unsqueeze(-1), a.unsqueeze(-1) - oc), min=0.0)
            sep_y = torch.clamp(torch.maximum(ob - d.unsqueeze(-1), b.unsqueeze(-1) - od), min=0.0)
            rgap = torch.sqrt(sep_x * sep_x + sep_y * sep_y) / diag
            min_rect_gap[ri] = rgap.min(dim=-1).values.clamp(max=1.0)
            ovx = torch.minimum(c.unsqueeze(-1), oc) - torch.maximum(a.unsqueeze(-1), oa)
            ovy = torch.minimum(d.unsqueeze(-1), od) - torch.maximum(b.unsqueeze(-1), ob)
            overlap[ri] = ((ovx > 0) & (ovy > 0)).any(dim=-1).to(torch.float32)

    out['left_clearance'] = left_clear
    out['right_clearance'] = right_clear
    out['top_clearance'] = top_clear
    out['bottom_clearance'] = bottom_clear
    out['soft_spacing_penalty_approx'] = soft_pen
    out['overlap_risk_approx'] = overlap
    out['min_gap_to_placed_rot'] = min_rect_gap

    band = max(2.0 * grid, float(getattr(env, 'edge_band_ratio', 0.12)) * min(bw, bh))
    is_edge = ((X - xmin <= band) | (xmax - X <= band) | (Y - ymin <= band) | (ymax - Y <= band)).to(torch.float32)
    corner = (((X - xmin <= band) | (xmax - X <= band)) & ((Y - ymin <= band) | (ymax - Y <= band))).to(torch.float32)
    out['is_edge_band'] = is_edge
    out['is_corner_band'] = corner

    sides = _edge_sides_for_ref(env, ref)
    if sides:
        dists = []
        axes = []
        for s in sides:
            if s == 'left':
                dists.append((X - xmin) / diag); axes.append((Y - ymin) / bh)
            elif s == 'right':
                dists.append((xmax - X) / diag); axes.append((Y - ymin) / bh)
            elif s == 'bottom':
                dists.append((Y - ymin) / diag); axes.append((X - xmin) / bw)
            elif s == 'top':
                dists.append((ymax - Y) / diag); axes.append((X - xmin) / bw)
        dist_edge = torch.stack(dists, dim=0).min(dim=0).values
        axis_norm = torch.stack(axes, dim=0).mean(dim=0).clamp(0.0, 1.0)
        out['dist_to_preferred_edge'] = dist_edge
        out['on_preferred_edge_band'] = (dist_edge * diag <= band).to(torch.float32)
        out['boundary_axis_norm'] = axis_norm
        out['has_preferred_edge'] = torch.ones_like(X)
        edge_prior = -dist_edge
    else:
        out['dist_to_preferred_edge'] = zero2
        out['on_preferred_edge_band'] = zero2
        out['boundary_axis_norm'] = zero2
        out['has_preferred_edge'] = zero2
        edge_prior = zero2

    prior_maps = _prior_heatmap_maps_cuda(env, ref, X, Y, device)
    out.update({k: v for k, v in prior_maps.items() if k != 'prior_heatmap_raw_logprob'})
    out['prior_heatmap_prior_raw'] = prior_maps.get('prior_heatmap_raw_logprob', zero2) * out['module_prior_confidence']

    out['conn_prior_raw'] = conn_prior
    out['anchor_prior_raw'] = anchor_prior
    out['spacing_prior_raw'] = -(soft_pen.mean(dim=0) + 0.5 * out['local_density'])
    out['edge_prior_raw'] = edge_prior
    return out


@torch.no_grad()
def action_features_cuda(
    env: PlacementEnv,
    device: torch.device,
    ref: Optional[str] = None,
    static_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GPU action features.

    With ref=None this returns the legacy board-static [A,4] grid features.
    With ref set, this returns [A,42] ref/state-specific features.  The first
    four columns are the static grid features, so old geometry helpers keep
    working while the policy key can see anchor/net/module/spacing context.
    """
    static = static_features if static_features is not None else _static_action_features_cuda(env, device)
    if ref is None:
        return static

    w, h = env.grid_shape()
    R = len(env.rotations)
    maps = _action_dynamic_maps_cuda(env, ref, device)
    dyn_names = ACTION_FEATURE_NAMES[4:]
    dyn = torch.zeros(R, w, h, len(dyn_names), device=device, dtype=torch.float32)

    for j, name in enumerate(dyn_names):
        key = 'min_gap_to_placed_rot' if name == 'min_gap_to_placed' else name
        val = maps.get(key)
        if val is None:
            continue
        if val.dim() == 2:
            dyn[..., j] = val.unsqueeze(0).expand(R, -1, -1)
        else:
            dyn[..., j] = val
    return torch.cat([static.reshape(R, w, h, 4), dyn], dim=-1).reshape(-1, ACTION_CONDITIONED_ACTION_FEAT_DIM)


@torch.no_grad()
def action_priors_cuda(env: PlacementEnv, ref: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Action-conditioned prior maps as flattened [A] tensors.

    Values are higher-is-better and are not weighted here.  Call
    action_prior_total_cuda() to z-score/clip and combine them.
    """
    w, h = env.grid_shape()
    R = len(env.rotations)
    maps = _action_dynamic_maps_cuda(env, ref, device)
    def _flat2(name: str) -> torch.Tensor:
        t = maps.get(name)
        if t is None:
            return torch.zeros(R * w * h, device=device)
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(R, -1, -1)
        return t.reshape(-1)
    return {
        'conn_centroid': _flat2('conn_prior_raw'),
        'anchor': _flat2('anchor_prior_raw'),
        'module': _flat2('module_prior_raw'),
        'spacing': _flat2('spacing_prior_raw'),
        'edge': _flat2('edge_prior_raw'),
        'density': _flat2('local_density') * -1.0,
        'prior_region_heatmap': _flat2('prior_heatmap_prior_raw'),
    }


@torch.no_grad()
def action_prior_total_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
    legal_mask_flat: Optional[torch.Tensor] = None,
    *,
    lambda_conn: float = 0.15,
    lambda_anchor: float = 0.20,
    lambda_module: float = 0.15,
    lambda_spacing: float = 0.08,
    lambda_edge: float = 0.05,
    lambda_prior_region_heatmap: float = 0.10,
) -> torch.Tensor:
    priors = action_priors_cuda(env, ref, device)
    total = torch.zeros_like(next(iter(priors.values())))
    weights = {
        'conn_centroid': float(lambda_conn),
        'anchor': float(lambda_anchor),
        'module': float(lambda_module),
        'spacing': float(lambda_spacing),
        'edge': float(lambda_edge),
        'prior_region_heatmap': float(lambda_prior_region_heatmap),
    }
    for key, weight in weights.items():
        if weight == 0.0:
            continue
        total = total + weight * _masked_zscore_flat(priors[key], legal_mask_flat)
    return total


# ---------------------------------------------------------------------------
# GPU-accelerated context tokens computation
# ---------------------------------------------------------------------------

# Type and module type constants (mirrored from train.py)
_CONTEXT_TYPES = ["interface", "mechanical", "chip", "capacitor", "resistor", "misc"]
_CONTEXT_MODULE_TYPES = [
    "none", "interface", "core_ic", "power_or_driver", "clock_local", 
    "mixed", "passive_cluster", "misc"
]
_CONTEXT_MODULE_TOKEN_EXTRA_DIM = len(_CONTEXT_MODULE_TYPES) + 5
_CONTEXT_OBS_TOKEN_DIM = len(_CONTEXT_TYPES) + 10 + _CONTEXT_MODULE_TOKEN_EXTRA_DIM


def _module_type_index_cuda(name: str) -> int:
    name = str(name or "none")
    if name not in _CONTEXT_MODULE_TYPES:
        name = "misc"
    return _CONTEXT_MODULE_TYPES.index(name)


def _coarse_type_index_cuda(fine_type: str) -> int:
    from .utils import coarse_type_from_fine
    t0 = coarse_type_from_fine(fine_type)
    return _CONTEXT_TYPES.index(t0) if t0 in _CONTEXT_TYPES else _CONTEXT_TYPES.index("misc")


@torch.no_grad()
def build_action_region_indices_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
    grid_x: int = 6,
    grid_y: int = 6,
    legacy_zone_edge_ratio: float = 0.12,
    legacy_zone_core_ratio: float = 0.28,
) -> torch.Tensor:
    """Build flattened action->heatmap-cell ids directly on GPU.

    Each action location is assigned to one region-heatmap bin from the
    configured (grid_x, grid_y) board partition. Rotations share the same
    spatial bin assignment; legality remains controlled by the legal mask.
    """
    del ref, legacy_zone_edge_ratio, legacy_zone_core_ratio
    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    grid_mm = float(env.task.grid_mm)
    grid_maps = _cached_grid_maps_cuda(env, device)
    tx = grid_maps["x_norm"]
    ty = grid_maps["y_norm"]
    gx = torch.clamp((tx * max(1, int(grid_x))).floor().long(), 0, max(1, int(grid_x)) - 1)
    gy = torch.clamp((ty * max(1, int(grid_y))).floor().long(), 0, max(1, int(grid_y)) - 1)
    base = gx * max(1, int(grid_y)) + gy
    out = base.unsqueeze(0).expand(R, w_cells, h_cells).contiguous()
    return out.reshape(-1)


@torch.no_grad()
def action_region_prior_from_predictions_cuda(

    region_heatmap_logits: torch.Tensor,
    action_heatmap_cell_idx_flat: torch.Tensor,
    heatmap_action_prior_weight: float,
) -> torch.Tensor:
    """GPU equivalent of action_region_prior_from_predictions()."""
    region_logp = torch.log_softmax(region_heatmap_logits.detach(), dim=-1).reshape(-1)
    prior = float(heatmap_action_prior_weight) * region_logp[action_heatmap_cell_idx_flat.long()]
    return prior - prior.max()


@torch.no_grad()
def action_mask_and_bias_cuda(env: PlacementEnv, ref: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU version of PlacementEnv.action_mask_and_bias().

    Returns (mask, bias) shaped [R, W, H] on *device*. Python still controls the
    tiny rotation/occupied loops, while every W×H map operation runs in torch.
    """
    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    grid_mm = float(env.task.grid_mm)
    grid_maps = _cached_grid_maps_cuda(env, device)
    X = grid_maps["x"]
    Y = grid_maps["y"]

    c = env.comp_by_ref[ref]
    mask = torch.zeros((R, w_cells, h_cells), device=device, dtype=torch.float32)
    bias = torch.zeros_like(mask)

    align_score = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
    current_align_group = env._align_groups.get(ref)
    if current_align_group:
        align_ix, align_iy = [], []
        for pref in env.placed_order:
            if env._align_groups.get(pref) == current_align_group:
                px, py, _ = env.placed[pref]
                align_ix.append(int(math.floor((float(px) - xmin) / grid_mm)))
                align_iy.append(int(math.floor((float(py) - ymin) / grid_mm)))
        valid_x = [i for i in sorted(set(align_ix)) if 0 <= i < w_cells]
        valid_y = [j for j in sorted(set(align_iy)) if 0 <= j < h_cells]
        if valid_x:
            align_score[torch.tensor(valid_x, device=device, dtype=torch.long), :] += 1.0
        if valid_y:
            align_score[:, torch.tensor(valid_y, device=device, dtype=torch.long)] += 1.0

    hard_boundary = bool(env.enforce_interface_on_boundary and env._hard_boundary_required(ref))
    edge_band_sides = env._edge_band_sides(ref)
    soft_edge_sides = env._soft_edge_preference_sides(ref)
    semantic_strength = max(0.0, float(env._semantic_strength(ref)))
    eps = float(env.edge_eps_mm)
    semantic_region = env._region_targets.get(ref, 'free')
    semantic_side = env._side_preferences.get(ref, 'free')
    board_diag = max(1e-6, float(env.board_diag))
    zero2 = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
    w0, h0 = float(c.size_mm[0]), float(c.size_mm[1])
    occ_t = (
        torch.as_tensor(env.occupied, device=device, dtype=torch.float32)
        if env.occupied
        else None
    )
    min_spacing = float(env.min_spacing)

    for ri, rot in enumerate(env.rotations):
        wm, hm = (w0, h0) if int(rot) % 180 == 0 else (h0, w0)
        a = X - wm / 2.0
        b = Y - hm / 2.0
        cc = X + wm / 2.0
        d = Y + hm / 2.0
        ok = (a >= xmin) & (b >= ymin) & (cc <= xmax) & (d <= ymax)
        if occ_t is not None:
            eps_sp = 1e-6
            # Vectorize occupied-overlap checks in chunks instead of launching one
            # W x H map operation per already-placed component.  The occupied
            # rectangles are converted to a device tensor once per mask call
            # instead of once per rotation.
            for occ_chunk in occ_t.split(128):
                oa2 = (occ_chunk[:, 0] - min_spacing).view(-1, 1, 1)
                ob2 = (occ_chunk[:, 1] - min_spacing).view(-1, 1, 1)
                oc2 = (occ_chunk[:, 2] + min_spacing).view(-1, 1, 1)
                od2 = (occ_chunk[:, 3] + min_spacing).view(-1, 1, 1)
                overlap = ~(
                    (cc.unsqueeze(0) <= oa2 + eps_sp)
                    | (a.unsqueeze(0) >= oc2 - eps_sp)
                    | (d.unsqueeze(0) <= ob2 + eps_sp)
                    | (b.unsqueeze(0) >= od2 - eps_sp)
                )
                ok &= ~overlap.any(dim=0)

        left = torch.abs(a - xmin) <= eps
        right = torch.abs(cc - xmax) <= eps
        bottom = torch.abs(b - ymin) <= eps
        top = torch.abs(d - ymax) <= eps
        touch_any = left | right | bottom | top
        left_clear = a - xmin
        right_clear = xmax - cc
        bottom_clear = b - ymin
        top_clear = ymax - d
        if hard_boundary:
            if edge_band_sides:
                sset = {str(x).lower() for x in edge_band_sides}
                touch = torch.zeros_like(touch_any)
                if 'left' in sset: touch |= left
                if 'right' in sset: touch |= right
                if 'bottom' in sset: touch |= bottom
                if 'top' in sset: touch |= top
                ok &= touch
            else:
                ok &= touch_any
        mask[ri] = ok.to(torch.float32)

        region_pen = zero2
        if semantic_region == 'edge_left':
            region_pen = torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left')))
        elif semantic_region == 'edge_right':
            region_pen = torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right')))
        elif semantic_region == 'edge_bottom':
            region_pen = torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom')))
        elif semantic_region == 'edge_top':
            region_pen = torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top')))
        elif semantic_region == 'core':
            bw, bh = xmax - xmin, ymax - ymin
            mx = min(0.40 * bw, max(grid_mm, 0.28 * bw))
            my = min(0.40 * bh, max(grid_mm, 0.28 * bh))
            dx = torch.maximum(torch.clamp((xmin + mx) - X, min=0.0), torch.clamp(X - (xmax - mx), min=0.0))
            dy = torch.maximum(torch.clamp((ymin + my) - Y, min=0.0), torch.clamp(Y - (ymax - my), min=0.0))
            region_pen = torch.sqrt(dx * dx + dy * dy)
        elif semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'}:
            if semantic_side == 'edge_left':
                region_pen = 0.35 * torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left')))
            elif semantic_side == 'edge_right':
                region_pen = 0.35 * torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right')))
            elif semantic_side == 'edge_bottom':
                region_pen = 0.35 * torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom')))
            else:
                region_pen = 0.35 * torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top')))
        region_bonus = -semantic_strength * float(env.objective_cfg.region_weight) * (region_pen / board_diag) ** 2

        soft_edge_bonus = zero2
        if (not hard_boundary) and soft_edge_sides:
            pref_terms = []
            if 'left' in soft_edge_sides:
                pref_terms.append(torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left'))))
            if 'right' in soft_edge_sides:
                pref_terms.append(torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right'))))
            if 'bottom' in soft_edge_sides:
                pref_terms.append(torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom'))))
            if 'top' in soft_edge_sides:
                pref_terms.append(torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top'))))
            if pref_terms:
                pref_pen = pref_terms[0]
                for term in pref_terms[1:]:
                    pref_pen = torch.minimum(pref_pen, term)
                soft_edge_bonus = -0.25 * semantic_strength * float(env.objective_cfg.region_weight) * (pref_pen / board_diag) ** 2

        edge_bonus_map = torch.zeros_like(zero2)
        edge_bonus_sides = list(edge_band_sides or soft_edge_sides or [])
        if edge_bonus_sides:
            edge_terms = []
            if 'left' in edge_bonus_sides:
                edge_terms.append(torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left'))))
            if 'right' in edge_bonus_sides:
                edge_terms.append(torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right'))))
            if 'bottom' in edge_bonus_sides:
                edge_terms.append(torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom'))))
            if 'top' in edge_bonus_sides:
                edge_terms.append(torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top'))))
            if edge_terms:
                edge_pen = edge_terms[0]
                for term in edge_terms[1:]:
                    edge_pen = torch.minimum(edge_pen, term)
                denom = max(float(env.task.grid_mm), float(env._edge_band_width_mm()), 1e-6)
                edge_bonus_map = float(env.edge_bonus) * semantic_strength * torch.exp(-((edge_pen / denom) ** 2))

        anchor_bonus = torch.zeros_like(zero2)
        anchor_ref = env._anchor_refs.get(ref)
        subzone = env._subzones.get(ref, 'free')
        if anchor_ref and anchor_ref in env.placed:
            ax0, ay0, _ = env.placed[anchor_ref]
            dx, dy = X - float(ax0), Y - float(ay0)
            anchor_bonus += -semantic_strength * float(env.objective_cfg.anchor_weight) * 0.20 * ((dx * dx + dy * dy) / (board_diag * board_diag))
            denom = max(grid_mm, 1e-6)
            if subzone == 'left': anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(-dx, min=0.0) / denom)
            elif subzone == 'right': anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(dx, min=0.0) / denom)
            elif subzone == 'top': anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(dy, min=0.0) / denom)
            elif subzone == 'bottom': anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(-dy, min=0.0) / denom)

        boundary_order_bonus = torch.zeros_like(zero2)
        same_side_group = env._same_side_groups.get(ref)
        my_order = env._boundary_orders.get(ref, None)
        if same_side_group and semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'} and my_order is not None:
            peers = [pref for pref in env.placed_order if env._same_side_groups.get(pref) == same_side_group and env._boundary_orders.get(pref, None) is not None]
            if peers:
                axis_grid = Y if semantic_side in {'edge_left', 'edge_right'} else X
                lower_vals, upper_vals = [], []
                for pref in peers:
                    peer_order = env._boundary_orders.get(pref, None)
                    if peer_order is None: continue
                    px, py, _ = env.placed[pref]
                    peer_axis = float(py) if semantic_side in {'edge_left', 'edge_right'} else float(px)
                    if peer_order < my_order: lower_vals.append(peer_axis)
                    elif peer_order > my_order: upper_vals.append(peer_axis)
                denom = max(grid_mm, 1e-6)
                if lower_vals: boundary_order_bonus += semantic_strength * 0.14 * torch.tanh((axis_grid - max(lower_vals)) / denom)
                if upper_vals: boundary_order_bonus += semantic_strength * 0.14 * torch.tanh((min(upper_vals) - axis_grid) / denom)

        module_region_penalty = _compute_module_region(env, ref, a, b, cc, d, board_diag)
        module_region_bonus = -float(env.module_region_bias) * module_region_penalty
        bias[ri] = (
            region_bonus
            + soft_edge_bonus
            + edge_bonus_map
            + float(env.alignment_bonus) * align_score
            + anchor_bonus
            + boundary_order_bonus
            + module_region_bonus
        ).to(torch.float32)
    return mask, bias


def _comp_nets_for_context(env: PlacementEnv, ref: str) -> set[str]:
    cache = getattr(env, "_context_nets_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_context_nets_cache", cache)
    ref_key = str(ref)
    if ref_key not in cache:
        c = env.comp_by_ref[ref]
        cache[ref_key] = {str(n) for (n, _) in c.pads if n and str(n).upper() not in ('', 'GND', 'GROUND')}
    return cache[ref_key]


def _context_ref_order(env: PlacementEnv, ref: str, max_tokens: int = 128) -> List[str]:
    cur_nets = _comp_nets_for_context(env, ref)
    shared, other = [], []
    for r in reversed(list(env.placed_order)):
        sh = len(cur_nets.intersection(_comp_nets_for_context(env, r)))
        (shared if sh > 0 else other).append((r, sh))
    ordered = [r for (r, _) in sorted(shared, key=lambda x: (-x[1],))] + [r for (r, _) in other]
    keep = ordered[:max(0, max_tokens - 1)]
    return list(reversed(keep))


def _module_type_index_context(name: str) -> int:
    name = str(name or 'none')
    if name not in _CONTEXT_MODULE_TYPES:
        name = 'misc'
    return _CONTEXT_MODULE_TYPES.index(name)


def _write_module_features_context(env: PlacementEnv, row: torch.Tensor, comp, x: Optional[float], y: Optional[float]) -> None:
    module_map = getattr(env, "_context_module_map_cache", None)
    if module_map is None:
        module_map = {str(m.get('module_id')): m for m in (getattr(env.task, 'modules', None) or []) if m.get('module_id')}
        setattr(env, "_context_module_map_cache", module_map)
    mid = str(getattr(comp, 'module_id', '') or '')
    module = module_map.get(mid)
    mtype = str(module.get('module_type') if module else 'none')
    base = len(_CONTEXT_TYPES) + 10
    row[base + _module_type_index_context(mtype)] = 1.0
    module_count = max(1, len(module_map))
    module_order = int(getattr(comp, 'module_order', module.get('module_order', 0) if module else 0))
    local_order = int(getattr(comp, 'module_local_order', 0))
    local_den = max(1, len(module.get('members', [])) - 1) if module else 1
    extra = base + len(_CONTEXT_MODULE_TYPES)
    row[extra + 0] = float(module_order) / max(1, module_count - 1)
    row[extra + 1] = float(local_order) / local_den
    module_anchor_ref = str(getattr(comp, 'module_anchor_ref', '') or (module.get('anchor_ref', '') if module else '')).strip()
    row[extra + 2] = 1.0 if module_anchor_ref and comp.ref == module_anchor_ref else 0.0
    if x is None or y is None:
        return
    # Use the same effective anchor relation as PlacementEnv: explicit
    # anchor_ref first, then module_anchor_ref fallback, but never self-anchor.
    anchor_ref = ''
    anchor_map = getattr(env, '_anchor_refs', None)
    if isinstance(anchor_map, dict):
        anchor_ref = str(anchor_map.get(comp.ref, '') or '').strip()
    if not anchor_ref:
        explicit_anchor = str(getattr(comp, 'anchor_ref', '') or '').strip()
        if explicit_anchor and explicit_anchor != comp.ref:
            anchor_ref = explicit_anchor
    if not anchor_ref and module_anchor_ref and module_anchor_ref != comp.ref:
        anchor_ref = module_anchor_ref
    anchor_xy = None
    if anchor_ref and anchor_ref in env.placed:
        ax, ay, _ = env.placed[anchor_ref]
        anchor_xy = (float(ax), float(ay))
    # Never fall back to expert_xy for unplaced anchors.  During inference this
    # would expose future/answer locations; during training it would create a
    # train/infer feature mismatch.
    if anchor_xy is not None:
        row[extra + 3] = min(1.0, math.hypot(float(x) - anchor_xy[0], float(y) - anchor_xy[1]) / max(1e-6, float(env.board_diag)))
    rb = env._module_region_bbox_for_ref(comp.ref) if hasattr(env, '_module_region_bbox_for_ref') else getattr(comp, 'module_region_bbox', None)
    if rb and len(rb) == 4:
        x0, y0, x1, y1 = [float(v) for v in rb]
        margin = float(getattr(env, 'module_region_margin_mm', 0.0))
        row[extra + 4] = 1.0 if (x0 - margin <= float(x) <= x1 + margin and y0 - margin <= float(y) <= y1 + margin) else 0.0


@torch.no_grad()
def build_context_tokens_cuda(env: PlacementEnv, ref: str, device: torch.device, max_tokens: int = 128) -> torch.Tensor:
    """Build the runtime context tokens consumed by MaskedPolicy."""
    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
    cur = env.comp_by_ref[ref]
    cur_nets = _comp_nets_for_context(env, ref)
    refs = _context_ref_order(env, ref, max_tokens=max_tokens)[::-1]
    placed_ratio = float(env.t) / max(1, len(env.sequence))
    tokens = torch.zeros((len(refs) + 1, _CONTEXT_OBS_TOKEN_DIM), device=device, dtype=torch.float32)
    off = len(_CONTEXT_TYPES)
    for i, r in enumerate(refs):
        c = env.comp_by_ref[r]
        tokens[i, _coarse_type_index_cuda(c.type)] = 1.0
        x, y, rot = env.placed[r]
        w_comp, h_comp = c.size_mm
        tokens[i, off + 0] = float(w_comp) / bw
        tokens[i, off + 1] = float(h_comp) / bh
        tokens[i, off + 2] = (float(x) - xmin) / bw
        tokens[i, off + 3] = (float(y) - ymin) / bh
        tokens[i, off + 4] = 1.0
        rr = math.radians(float(rot))
        tokens[i, off + 5] = math.sin(rr)
        tokens[i, off + 6] = math.cos(rr)
        tokens[i, off + 7] = min(1.0, float(len(cur_nets.intersection(_comp_nets_for_context(env, r)))) / 8.0)
        tokens[i, off + 8] = placed_ratio
        tokens[i, off + 9] = 0.0
        _write_module_features_context(env, tokens[i], c, float(x), float(y))
    i = len(refs)
    tokens[i, _coarse_type_index_cuda(cur.type)] = 1.0
    w_comp, h_comp = cur.size_mm
    tokens[i, off + 0] = float(w_comp) / bw
    tokens[i, off + 1] = float(h_comp) / bh
    tokens[i, off + 2] = 0.0
    tokens[i, off + 3] = 0.0
    tokens[i, off + 4] = 0.0
    tokens[i, off + 5] = 0.0
    tokens[i, off + 6] = 1.0
    tokens[i, off + 7] = 0.0
    tokens[i, off + 8] = placed_ratio
    tokens[i, off + 9] = 1.0
    # Current-ref token must not depend on expert_xy; candidate-position relations
    # are now carried by action_features_cuda(env, ref, ...).
    _write_module_features_context(env, tokens[i], cur, None, None)
    return tokens
