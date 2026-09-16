"""GPU-accelerated objective delta mask computation for PlacementEnv.

Replaces the triple Python loop in PlacementEnv.objective_delta_mask with
vectorized PyTorch operations on GPU, eliminating the W¡ÁH inner loop entirely.

Rotation-independent penalties (conn, align, group, anchor, boundary_group,
pitch, interior, density, line_neatness) are computed once and broadcast.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .env import PlacementEnv


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
    return (dx / max(1e-6, mx)) ** 2 + (dy / max(1e-6, my)) ** 2


def _compute_density(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    pd: Dict[str, torch.Tensor],
    board_diag: float,
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
    return (sc * u * u * (soft_r > 1e-6).float()).sum(dim=-1)


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
    board_diag: float,
    device: torch.device,
) -> torch.Tensor:
    my_g = env._functional_group.get(ref, 'misc')
    cur = list(env.placed_order)
    cur_xy = {r: (float(env.placed[r][0]), float(env.placed[r][1])) for r in cur}
    grefs = [r for r in cur if env._functional_group.get(r, 'misc') == my_g]

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
    my_group = env._functional_group.get(ref, 'misc')
    other_groups = sorted({env._functional_group.get(r, 'misc') for r in cur if env._functional_group.get(r, 'misc') not in {'', 'misc', my_group}})
    if other_groups:
        my_center_x = Xc
        my_center_y = Yc
        my_gap_bb = (cand_a, cand_b, cand_c, cand_d)
        for og in other_groups:
            members = [r for r in cur if env._functional_group.get(r, 'misc') == og]
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

    if K + 1 < 3:
        return torch.zeros_like(Xc)

    test_xy = dict(cur_xy)
    test_xy[ref] = (float(Xc.mean().item()), float(Yc.mean().item()))
    axis = env._pitch_axis_for_group(refs_new, test_xy)

    if axis == 'x':
        span = max(1e-6, env.task.bbox_mm[2] - env.task.bbox_mm[0])
        ref_axis = Xc
    else:
        span = max(1e-6, env.task.bbox_mm[3] - env.task.bbox_mm[1])
        ref_axis = Yc

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
    delta = (K + 1) * (new_pen - old_pen)
    strength = float(env._semantic_strength(ref))
    delta = strength * delta
    if axis == "x":
        return delta.view(-1, 1).expand(-1, Yc.shape[1])
    else:
        return delta.view(1, -1).expand(Xc.shape[0], -1)


def _compute_line_neatness(
    env: PlacementEnv, ref: str,
    Xc: torch.Tensor, Yc: torch.Tensor,
    bw: float, bh: float,
    placed_order: List[str],
    device: torch.device,
) -> torch.Tensor:
    my_g = env._functional_group.get(ref, 'misc')
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
        if env._functional_group.get(pref, 'misc') == my_g:
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
    return u * u


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
    large_like = sem in {
        'core', 'power', 'power_support', 'clock', 'support',
        'large', 'mechanical', 'core_active', 'power_active'
    }
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



# ---------------------------------------------------------------------------
# GPU legal-action mask and wire delta for fast inference
# ---------------------------------------------------------------------------

def _grid_centers_torch(env: PlacementEnv, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    w_cells, h_cells = env.grid_shape()
    xmin, ymin, _, _ = env.task.bbox_mm
    grid_mm = float(env.task.grid_mm)
    ix = torch.arange(w_cells, device=device, dtype=torch.float32).unsqueeze(1)
    iy = torch.arange(h_cells, device=device, dtype=torch.float32).unsqueeze(0)
    Xc = xmin + (ix + 0.5) * grid_mm
    Yc = ymin + (iy + 0.5) * grid_mm
    return Xc, Yc


@torch.no_grad()
def action_mask_cuda(env: PlacementEnv, ref: str, device: torch.device) -> torch.Tensor:
    """Return legal action mask as a bool tensor [R, W, H] on *device*.

    This mirrors the hard legality part of PlacementEnv.action_mask_and_bias(),
    but intentionally skips the heuristic bias map. In inference the bias is not
    used for argmax selection, while computing it on CPU is one of the major
    rollout bottlenecks.
    """
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    Xc, Yc = _grid_centers_torch(env, device)

    comp = env.comp_by_ref[ref]
    w0, h0 = float(comp.size_mm[0]), float(comp.size_mm[1])
    eps = float(env.edge_eps_mm)
    hard_boundary = bool(env.enforce_interface_on_boundary and env._hard_boundary_required(ref))
    edge_band_sides = env._edge_band_sides(ref)
    edge_band_sides_l = {str(x).lower() for x in edge_band_sides} if edge_band_sides else set()

    if env.occupied:
        occ = torch.tensor(env.occupied, device=device, dtype=torch.float32)
        oa = occ[:, 0] - float(env.min_spacing)
        ob = occ[:, 1] - float(env.min_spacing)
        oc = occ[:, 2] + float(env.min_spacing)
        od = occ[:, 3] + float(env.min_spacing)
    else:
        oa = ob = oc = od = None

    mask = torch.zeros((R, w_cells, h_cells), device=device, dtype=torch.bool)
    eps_sp = 1e-6
    for ri, rot in enumerate(env.rotations):
        wm, hm = (w0, h0) if int(rot) % 180 == 0 else (h0, w0)
        a = Xc - wm / 2
        b = Yc - hm / 2
        c = Xc + wm / 2
        d = Yc + hm / 2

        ok = (a >= xmin) & (b >= ymin) & (c <= xmax) & (d <= ymax)

        if oa is not None:
            overlap = ~(
                (c.unsqueeze(-1) <= oa + eps_sp)
                | (a.unsqueeze(-1) >= oc - eps_sp)
                | (d.unsqueeze(-1) <= ob + eps_sp)
                | (b.unsqueeze(-1) >= od - eps_sp)
            )
            ok = ok & (~overlap.any(dim=-1))

        if hard_boundary:
            left = torch.abs(a - xmin) <= eps
            right = torch.abs(c - xmax) <= eps
            bottom = torch.abs(b - ymin) <= eps
            top = torch.abs(d - ymax) <= eps
            if edge_band_sides_l:
                touch = torch.zeros_like(ok)
                if "left" in edge_band_sides_l:
                    touch = touch | left
                if "right" in edge_band_sides_l:
                    touch = touch | right
                if "bottom" in edge_band_sides_l:
                    touch = touch | bottom
                if "top" in edge_band_sides_l:
                    touch = touch | top
                ok = ok & touch
            else:
                ok = ok & (left | right | bottom | top)

        mask[ri] = ok
    return mask


def _is_non_ground_net(net: object) -> bool:
    nn = str(net or "").strip()
    return bool(nn) and nn.upper() not in ("GND", "GROUND")


@torch.no_grad()
def wire_delta_mask_cuda(env: PlacementEnv, ref: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """GPU version of PlacementEnv.wire_delta_mask().

    The small net/pad metadata is still gathered in Python, but all grid-sized
    HPWL/NSLW maps stay on the target device and no NumPy -> torch transfer is
    needed inside objective_delta_mask_cuda().
    """
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    Xc, Yc = _grid_centers_torch(env, device)
    full_grid = Xc + Yc

    # Existing net bounding boxes from already placed pins.
    pins = env._pin_positions()
    net_bbox: Dict[str, Tuple[float, float, float, float]] = {}
    net_deg: Dict[str, int] = {}
    for net, pts in pins.items():
        if not _is_non_ground_net(net) or len(pts) == 0:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        nn = str(net)
        net_bbox[nn] = (min(xs), max(xs), min(ys), max(ys))
        net_deg[nn] = int(len(pts))

    comp = env.comp_by_ref[ref]
    pads_by_net: Dict[str, List[Tuple[float, float]]] = {}
    for net, (rx, ry) in comp.pads:
        if not _is_non_ground_net(net):
            continue
        pads_by_net.setdefault(str(net), []).append((float(rx), float(ry)))

    hpwl_delta = torch.zeros((R, w_cells, h_cells), device=device, dtype=torch.float32)
    nslw_delta = torch.zeros_like(hpwl_delta)

    if not pads_by_net:
        total = env.objective_cfg.hpwl_weight * hpwl_delta + env.objective_cfg.nslw_weight * nslw_delta
        return {"hpwl": hpwl_delta, "nslw": nslw_delta, "total": total}

    for ri, rot in enumerate(env.rotations):
        angle = math.radians(float(rot))
        ca, sa = math.cos(angle), math.sin(angle)
        hpwl_total = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
        nslw_total = torch.zeros_like(hpwl_total)

        for net, rels in pads_by_net.items():
            rel = torch.tensor(rels, device=device, dtype=torch.float32)
            rx = rel[:, 0]
            ry = rel[:, 1]
            dx = ca * rx - sa * ry
            dy = sa * rx + ca * ry

            px_stack = Xc.unsqueeze(0) + dx.view(-1, 1, 1)
            py_stack = Yc.unsqueeze(0) + dy.view(-1, 1, 1)
            comp_minx = px_stack.min(dim=0).values
            comp_maxx = px_stack.max(dim=0).values
            comp_miny = py_stack.min(dim=0).values
            comp_maxy = py_stack.max(dim=0).values

            d0 = int(net_deg.get(net, 0))
            d_comp = int(len(rels))
            d_new = d0 + d_comp
            if d0 > 0 and net in net_bbox:
                ex_minx, ex_maxx, ex_miny, ex_maxy = net_bbox[net]
            else:
                ex_minx = ex_maxx = ex_miny = ex_maxy = 0.0

            if d0 <= 1:
                hpwl_old = 0.0
                nslw_old = 0.0
            else:
                hpwl_old = (ex_maxx - ex_minx) + (ex_maxy - ex_miny)
                nslw_old = hpwl_old * math.log(1.0 + d0)

            if d0 == 0:
                new_minx, new_maxx = comp_minx, comp_maxx
                new_miny, new_maxy = comp_miny, comp_maxy
            else:
                new_minx = torch.minimum(comp_minx, torch.tensor(ex_minx, device=device))
                new_maxx = torch.maximum(comp_maxx, torch.tensor(ex_maxx, device=device))
                new_miny = torch.minimum(comp_miny, torch.tensor(ex_miny, device=device))
                new_maxy = torch.maximum(comp_maxy, torch.tensor(ex_maxy, device=device))

            if d_new <= 1:
                hpwl_new = torch.zeros_like(full_grid)
                nslw_new = torch.zeros_like(full_grid)
            else:
                hpwl_new = (new_maxx - new_minx) + (new_maxy - new_miny)
                nslw_new = hpwl_new * math.log(1.0 + d_new)

            hpwl_total = hpwl_total + (hpwl_new - float(hpwl_old))
            nslw_total = nslw_total + (nslw_new - float(nslw_old))

        hpwl_delta[ri] = hpwl_total
        nslw_delta[ri] = nslw_total

    total = env.objective_cfg.hpwl_weight * hpwl_delta + env.objective_cfg.nslw_weight * nslw_delta
    return {"hpwl": hpwl_delta, "nslw": nslw_delta, "total": total}


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

    ix = torch.arange(w_cells, device=device, dtype=torch.float32).unsqueeze(1)
    iy = torch.arange(h_cells, device=device, dtype=torch.float32).unsqueeze(0)
    Xc = xmin + (ix + 0.5) * grid_mm
    Yc = ymin + (iy + 0.5) * grid_mm

    wire = wire_delta_mask_cuda(env, ref, device)
    hpwl = wire['hpwl']
    nslw = wire['nslw']

    comp = env.comp_by_ref[ref]
    w0, h0 = float(comp.size_mm[0]), float(comp.size_mm[1])

    placed_order = list(env.placed_order)
    n_placed = len(placed_order)
    pd = _precompute_placed_data(env, ref, placed_order, device) if n_placed else {}

    # --- rotation-independent terms [W, H] ¡ú broadcast to [R, W, H] ---
    conn_2d = _compute_conn(Xc, Yc, pd, board_diag) if n_placed else torch.zeros_like(Xc)
    align_2d = _compute_align_delta(env, ref, Xc, Yc, bw, bh, device)
    group_2d = _compute_group_delta(env, ref, Xc, Yc, board_diag, device)
    anchor_2d = _compute_anchor_delta(env, ref, Xc, Yc, board_diag, device)
    bgroup_2d = _compute_boundary_group_delta(env, ref, Xc, Yc, device)
    pitch_2d = _compute_pitch_delta(env, ref, Xc, Yc, device)
    interior_2d = _compute_interior(env, ref, Xc, Yc)
    density_2d = _compute_density(env, ref, Xc, Yc, pd, board_diag) if n_placed else torch.zeros_like(Xc)
    line_neat_2d = _compute_line_neatness(env, ref, Xc, Yc, bw, bh, placed_order, device) if n_placed else torch.zeros_like(Xc)

    def _bcast(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(R, -1, -1)

    conn = _bcast(conn_2d)
    align = _bcast(align_2d)
    group = _bcast(group_2d)
    anchor = _bcast(anchor_2d)
    boundary_group = _bcast(bgroup_2d)
    pitch = _bcast(pitch_2d)
    interior = _bcast(interior_2d)
    density = _bcast(density_2d)

    # --- rotation-dependent terms ---
    region = torch.zeros(R, w_cells, h_cells, device=device)
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

        region[ri] = _compute_region(env, ref, a, b_t, c_t, d_t, Xc, Yc, board_diag)
        edge_clearance[ri] = _compute_edge_clearance(env, ref, a, b_t, c_t, d_t)
        orientation[ri] = env._orientation_delta_for_candidate(ref, int(rot))

        if n_placed:
            soft_spacing[ri] = _compute_soft_spacing(env, ref, a, b_t, c_t, d_t, pd)
        wb = _compute_whitespace_balance(env, ref, a, b_t, c_t, d_t, pd if n_placed else None, device)
        wf = _compute_whitespace_floor(env, ref, a, b_t, c_t, d_t, pd if n_placed else None, device)
        neatness[ri] = 0.50 * line_neat_2d + 0.20 * wb + 0.30 * wf

    total = (
        cfg.hpwl_weight * hpwl
        + cfg.nslw_weight * nslw
        + cfg.region_weight * region
        + cfg.conn_weight * conn
        + cfg.align_weight * align
        + cfg.group_weight * group
        + cfg.anchor_weight * anchor
        + cfg.boundary_group_weight * boundary_group
        + cfg.pitch_weight * pitch
        + cfg.orientation_weight * orientation
        + cfg.edge_clearance_weight * edge_clearance
        + cfg.interior_weight * interior
        + cfg.density_weight * density
        + cfg.soft_spacing_weight * soft_spacing
        + cfg.neatness_weight * neatness
    )

    return {
        'hpwl': hpwl, 'nslw': nslw,
        'region': region, 'conn': conn,
        'align': align, 'group': group,
        'anchor': anchor, 'boundary_group': boundary_group,
        'pitch': pitch, 'orientation': orientation,
        'edge_clearance': edge_clearance, 'interior': interior,
        'density': density, 'soft_spacing': soft_spacing,
        'neatness': neatness, 'total': total,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU version of train.build_teacher_distribution.

    All inputs and outputs are torch tensors on *device*.
    Returns (q, cand) where cand is a bool tensor.
    """
    energy = objective_flat.clone()
    if region_flat is not None:
        energy = energy - float(cfg.lambda_region_prior) * region_flat

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
# Extra GPU helpers for inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def action_mask_and_bias_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch/GPU version of PlacementEnv.action_mask_and_bias.

    Returns (mask, bias), both shaped [R, W, H] on *device*.  It mirrors the CPU
    implementation but keeps the dense grid work in torch, so infer.py no longer
    has to call env.observe()/env.step() just to build masks.
    """
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    w_cells, h_cells = env.grid_shape()
    R = len(env.rotations)
    grid_mm = float(env.task.grid_mm)

    ix = torch.arange(w_cells, device=device, dtype=torch.float32).unsqueeze(1)
    iy = torch.arange(h_cells, device=device, dtype=torch.float32).unsqueeze(0)
    X = xmin + (ix + 0.5) * grid_mm
    Y = ymin + (iy + 0.5) * grid_mm
    I = torch.arange(w_cells, device=device, dtype=torch.int32).unsqueeze(1)
    J = torch.arange(h_cells, device=device, dtype=torch.int32).unsqueeze(0)

    comp = env.comp_by_ref[ref]
    mask = torch.zeros((R, w_cells, h_cells), device=device, dtype=torch.float32)
    bias = torch.zeros_like(mask)

    current_align_group = env._align_groups.get(ref)
    align_refs = [pref for pref in env.placed_order if current_align_group and env._align_groups.get(pref) == current_align_group]
    align_ix, align_iy = [], []
    for pref in align_refs:
        px, py, _ = env.placed[pref]
        align_ix.append(int(math.floor((px - xmin) / grid_mm)))
        align_iy.append(int(math.floor((py - ymin) / grid_mm)))
    if align_ix:
        ax_vals = torch.tensor(sorted(set(align_ix)), device=device, dtype=torch.int32)
        ax = torch.isin(I, ax_vals).to(torch.float32)
    else:
        ax = torch.zeros((w_cells, 1), device=device, dtype=torch.float32)
    if align_iy:
        ay_vals = torch.tensor(sorted(set(align_iy)), device=device, dtype=torch.int32)
        ay = torch.isin(J, ay_vals).to(torch.float32)
    else:
        ay = torch.zeros((1, h_cells), device=device, dtype=torch.float32)
    align_score = ax + ay

    hard_boundary = env.enforce_interface_on_boundary and env._hard_boundary_required(ref)
    edge_band_sides = env._edge_band_sides(ref)
    soft_edge_sides = env._soft_edge_preference_sides(ref)
    semantic_strength = max(0.0, float(env._semantic_strength(ref)))
    eps = float(env.edge_eps_mm)
    semantic_region = env._region_targets.get(ref, 'free')
    semantic_side = env._side_preferences.get(ref, 'free')

    if env.occupied:
        occ = torch.tensor(env.occupied, device=device, dtype=torch.float32)
        oa2 = occ[:, 0] - float(env.min_spacing)
        ob2 = occ[:, 1] - float(env.min_spacing)
        oc2 = occ[:, 2] + float(env.min_spacing)
        od2 = occ[:, 3] + float(env.min_spacing)
    else:
        occ = None
        oa2 = ob2 = oc2 = od2 = None

    for ri, rot in enumerate(env.rotations):
        w_mm, h_mm = float(comp.size_mm[0]), float(comp.size_mm[1])
        if int(rot) % 180 != 0:
            w_mm, h_mm = h_mm, w_mm

        a = X - w_mm / 2.0
        b = Y - h_mm / 2.0
        c = X + w_mm / 2.0
        d = Y + h_mm / 2.0

        inside = (a >= xmin) & (b >= ymin) & (c <= xmax) & (d <= ymax)
        ok = inside.expand(w_cells, h_cells).clone()

        if occ is not None and occ.numel() > 0:
            eps_sp = 1e-6
            A = a.expand(w_cells, h_cells).unsqueeze(-1)
            B = b.expand(w_cells, h_cells).unsqueeze(-1)
            C = c.expand(w_cells, h_cells).unsqueeze(-1)
            D = d.expand(w_cells, h_cells).unsqueeze(-1)
            overlap = ~((C <= oa2 + eps_sp) | (A >= oc2 - eps_sp) | (D <= ob2 + eps_sp) | (B >= od2 - eps_sp))
            ok &= ~overlap.any(dim=-1)

        left = torch.abs(a - xmin) <= eps
        right = torch.abs(c - xmax) <= eps
        bottom = torch.abs(b - ymin) <= eps
        top = torch.abs(d - ymax) <= eps
        touch_any = (left | right | bottom | top).expand(w_cells, h_cells)
        left_clear = a - xmin
        right_clear = xmax - c
        bottom_clear = b - ymin
        top_clear = ymax - d

        if hard_boundary:
            if edge_band_sides:
                sides = {str(x).lower() for x in edge_band_sides}
                touch = torch.zeros((w_cells, h_cells), device=device, dtype=torch.bool)
                if "left" in sides:
                    touch |= left.expand(w_cells, h_cells)
                if "right" in sides:
                    touch |= right.expand(w_cells, h_cells)
                if "bottom" in sides:
                    touch |= bottom.expand(w_cells, h_cells)
                if "top" in sides:
                    touch |= top.expand(w_cells, h_cells)
                ok &= touch
            else:
                ok &= touch_any

        mask[ri] = ok.to(torch.float32)

        region_pen = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
        if semantic_region == 'edge_left':
            region_pen = torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left'))).expand(w_cells, h_cells)
        elif semantic_region == 'edge_right':
            region_pen = torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right'))).expand(w_cells, h_cells)
        elif semantic_region == 'edge_bottom':
            region_pen = torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom'))).expand(w_cells, h_cells)
        elif semantic_region == 'edge_top':
            region_pen = torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top'))).expand(w_cells, h_cells)
        elif semantic_region == 'core':
            mx = min(0.40 * (xmax - xmin), max(grid_mm, 0.28 * (xmax - xmin)))
            my = min(0.40 * (ymax - ymin), max(grid_mm, 0.28 * (ymax - ymin)))
            core_x0 = xmin + mx
            core_x1 = xmax - mx
            core_y0 = ymin + my
            core_y1 = ymax - my
            dx = torch.maximum(torch.clamp(core_x0 - X, min=0.0), torch.clamp(X - core_x1, min=0.0))
            dy = torch.maximum(torch.clamp(core_y0 - Y, min=0.0), torch.clamp(Y - core_y1, min=0.0))
            region_pen = torch.sqrt(dx * dx + dy * dy).expand(w_cells, h_cells)
        elif semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'}:
            if semantic_side == 'edge_left':
                region_pen = 0.35 * torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left'))).expand(w_cells, h_cells)
            elif semantic_side == 'edge_right':
                region_pen = 0.35 * torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right'))).expand(w_cells, h_cells)
            elif semantic_side == 'edge_bottom':
                region_pen = 0.35 * torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom'))).expand(w_cells, h_cells)
            else:
                region_pen = 0.35 * torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top'))).expand(w_cells, h_cells)
        region_bonus = -semantic_strength * float(env.objective_cfg.region_weight) * (region_pen / float(env.board_diag)) ** 2

        soft_edge_bonus = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
        if (not hard_boundary) and soft_edge_sides:
            pref_terms = []
            if 'left' in soft_edge_sides:
                pref_terms.append(torch.abs(left_clear - float(env._edge_target_clearance_mm(ref, 'left'))).expand(w_cells, h_cells))
            if 'right' in soft_edge_sides:
                pref_terms.append(torch.abs(right_clear - float(env._edge_target_clearance_mm(ref, 'right'))).expand(w_cells, h_cells))
            if 'bottom' in soft_edge_sides:
                pref_terms.append(torch.abs(bottom_clear - float(env._edge_target_clearance_mm(ref, 'bottom'))).expand(w_cells, h_cells))
            if 'top' in soft_edge_sides:
                pref_terms.append(torch.abs(top_clear - float(env._edge_target_clearance_mm(ref, 'top'))).expand(w_cells, h_cells))
            if pref_terms:
                pref_pen = pref_terms[0]
                for term in pref_terms[1:]:
                    pref_pen = torch.minimum(pref_pen, term)
                soft_edge_bonus = -0.25 * semantic_strength * float(env.objective_cfg.region_weight) * (pref_pen / float(env.board_diag)) ** 2

        anchor_bonus = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
        anchor_ref = env._anchor_refs.get(ref)
        subzone = env._subzones.get(ref, 'free')
        if anchor_ref and anchor_ref in env.placed:
            ax0, ay0, _ = env.placed[anchor_ref]
            dx = X - float(ax0)
            dy = Y - float(ay0)
            dist2 = dx * dx + dy * dy
            anchor_bonus += (-semantic_strength * float(env.objective_cfg.anchor_weight) * 0.20 * (dist2 / max(1e-6, float(env.board_diag) ** 2))).expand(w_cells, h_cells)
            if subzone == 'left':
                anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(-dx, min=0.0) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)
            elif subzone == 'right':
                anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(dx, min=0.0) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)
            elif subzone == 'top':
                anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(dy, min=0.0) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)
            elif subzone == 'bottom':
                anchor_bonus += semantic_strength * 0.18 * torch.tanh(torch.clamp(-dy, min=0.0) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)

        boundary_order_bonus = torch.zeros((w_cells, h_cells), device=device, dtype=torch.float32)
        same_side_group = env._same_side_groups.get(ref)
        my_order = env._boundary_orders.get(ref, None)
        if same_side_group and (semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'}) and my_order is not None:
            peers = [pref for pref in env.placed_order if env._same_side_groups.get(pref) == same_side_group and env._boundary_orders.get(pref, None) is not None]
            if peers:
                axis_grid = Y if semantic_side in {'edge_left', 'edge_right'} else X
                lower_vals, upper_vals = [], []
                for pref in peers:
                    peer_order = env._boundary_orders.get(pref, None)
                    if peer_order is None:
                        continue
                    px, py, _ = env.placed[pref]
                    peer_axis = float(py) if semantic_side in {'edge_left', 'edge_right'} else float(px)
                    if peer_order < my_order:
                        lower_vals.append(peer_axis)
                    elif peer_order > my_order:
                        upper_vals.append(peer_axis)
                if lower_vals:
                    boundary_order_bonus += semantic_strength * 0.14 * torch.tanh((axis_grid - max(lower_vals)) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)
                if upper_vals:
                    boundary_order_bonus += semantic_strength * 0.14 * torch.tanh((min(upper_vals) - axis_grid) / max(grid_mm, 1e-6)).expand(w_cells, h_cells)

        bias[ri] = (region_bonus + soft_edge_bonus + float(env.alignment_bonus) * align_score + anchor_bonus + boundary_order_bonus).to(torch.float32)

    return mask, bias


@torch.no_grad()
def build_action_region_indices_cuda(
    env: PlacementEnv,
    ref: str,
    device: torch.device,
    zone_edge_ratio: float,
    zone_core_ratio: float,
) -> torch.Tensor:
    """GPU equivalent of region_prior.build_action_region_indices."""
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    w_cells, h_cells = env.grid_shape()
    grid_mm = float(env.task.grid_mm)
    R = len(env.rotations)
    comp = env.comp_by_ref[ref]

    ix = torch.arange(w_cells, device=device, dtype=torch.float32).unsqueeze(1)
    iy = torch.arange(h_cells, device=device, dtype=torch.float32).unsqueeze(0)
    X = xmin + (ix + 0.5) * grid_mm
    Y = ymin + (iy + 0.5) * grid_mm
    # REGION_TYPE_NAMES order: edge_top, edge_bottom, edge_left, edge_right, core, free
    names = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    region_idx = torch.empty((R, w_cells, h_cells), device=device, dtype=torch.long)
    span = max(1e-6, min(xmax - xmin, ymax - ymin))

    for ri, rot in enumerate(env.rotations):
        w_mm, h_mm = float(comp.size_mm[0]), float(comp.size_mm[1])
        if int(rot) % 180 != 0:
            w_mm, h_mm = h_mm, w_mm
        a = X - 0.5 * w_mm
        b = Y - 0.5 * h_mm
        c = X + 0.5 * w_mm
        d = Y + 0.5 * h_mm
        clearance = torch.minimum(torch.minimum(a - xmin, xmax - c), torch.minimum(b - ymin, ymax - d)).expand(w_cells, h_cells)
        ratio = clearance / span
        dstack = torch.stack([
            (ymax - d).expand(w_cells, h_cells),
            (b - ymin).expand(w_cells, h_cells),
            (a - xmin).expand(w_cells, h_cells),
            (xmax - c).expand(w_cells, h_cells),
        ], dim=0)
        nearest = torch.argmin(dstack, dim=0)
        region = torch.full((w_cells, h_cells), 5, device=device, dtype=torch.long)
        edge_mask = ratio <= float(zone_edge_ratio)
        core_mask = ratio >= float(zone_core_ratio)
        region[edge_mask] = names[nearest[edge_mask]]
        region[core_mask] = 4
        region_idx[ri] = region

    return region_idx.reshape(-1)


@torch.no_grad()
def action_region_prior_from_predictions_cuda(
    region_type_logits: torch.Tensor,
    region_type_idx_flat: torch.Tensor,
    zone_prior_weight: float,
) -> torch.Tensor:
    region_logp = torch.log_softmax(region_type_logits.detach(), dim=-1).reshape(-1)
    prior = float(zone_prior_weight) * region_logp.gather(0, region_type_idx_flat.long())
    prior = prior - prior.max()
    return prior.to(torch.float32)
