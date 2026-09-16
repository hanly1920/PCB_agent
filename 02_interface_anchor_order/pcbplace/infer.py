from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset import task_from_json
from .env import PlacementEnv
from .model import MaskedPolicy, ModelConfig
from .train import (
    TeacherConfig,
    _policy_outputs_with_region,
    build_context_tokens,
    build_teacher_distribution,
    _unflatten_action,
)
from .env_cuda import (
    objective_delta_mask_cuda,
    build_teacher_distribution_cuda,
    action_mask_cuda,
    action_mask_and_bias_cuda,
    build_action_region_indices_cuda,
    action_region_prior_from_predictions_cuda,
)
from .region_prior import RegionPriorConfig, REGION_TYPE_NAMES, SEMANTIC_CLASS_NAMES, SIDE_PREFERENCE_NAMES, SUBZONE_NAMES, PAIRWISE_RELATION_NAMES


def _layout_objective(task, placed: Dict[str, Tuple[float, float, int]], env_kwargs: Dict[str, Any]) -> float:
    env = PlacementEnv(task, **env_kwargs)
    env.placed = dict(placed)
    env.placed_order = [ref for ref in task.sequence if ref in env.placed]
    env.occupied = [env._ref_bbox(ref, *env.placed[ref]) for ref in env.placed_order]
    env.t = len(env.placed_order)
    env.prev_obj = env._objective()
    return float(env.prev_obj)


def _is_placement_legal(env: PlacementEnv, ref: str, placement: Tuple[float, float, int], placed: Dict[str, Tuple[float, float, int]]) -> bool:
    bb = env._ref_bbox(ref, *placement)
    if not env._inside(bb):
        return False
    occupied = []
    for pref, p in placed.items():
        if pref == ref:
            continue
        occupied.append(env._ref_bbox(pref, *p))
    old_occ = env.occupied
    env.occupied = occupied
    try:
        if env._violates_spacing(bb):
            return False
    finally:
        env.occupied = old_occ
    if env.enforce_interface_on_boundary:
        band_sides = env._edge_band_sides(ref)
        if env._hard_boundary_required(ref):
            if band_sides:
                if not env._touch_sides(bb, band_sides):
                    return False
            elif not env._touch_sides(bb, ['left', 'right', 'top', 'bottom']):
                return False
        elif band_sides:
            if not env._within_edge_band(bb, band_sides):
                return False
    return True


def _try_update(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]], ref: str, cand: Tuple[float, float, int], max_move_mm: float = 1.5) -> bool:
    old = placed.get(ref)
    if old is None:
        return False
    dx = float(cand[0] - old[0])
    dy = float(cand[1] - old[1])
    if (dx * dx + dy * dy) ** 0.5 > max_move_mm + 1e-6:
        return False
    if not _is_placement_legal(env, ref, cand, placed):
        return False
    placed[ref] = cand
    return True


def _declutter_move_limit(env: PlacementEnv, ref: str) -> float:
    comp = env.comp_by_ref[ref]
    sem = env._semantic_class.get(ref, '')
    if env._is_large_component(ref) or env._anchor_refs.get(ref) or sem in {'core', 'interface', 'power', 'power_support'} or env._edge_band_sides(ref):
        return 3.6
    return 2.4


def _refresh_env_state(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> None:
    env.placed = dict(placed)
    env.placed_order = [ref for ref in env.task.sequence if ref in env.placed]
    env.occupied = [env._ref_bbox(ref, *env.placed[ref]) for ref in env.placed_order]
    env.t = len(env.placed_order)
    env.prev_obj = env._objective() if env.placed else 0.0


def _preferred_edge_side(env: PlacementEnv, ref: str) -> Optional[str]:
    sides = list(env._edge_band_sides(ref) or [])
    if not sides:
        return None
    if len(sides) == 1:
        return sides[0]
    for semantic_side in (
        env._side_preferences.get(ref, 'free'),
        env._region_targets.get(ref, 'free'),
    ):
        if semantic_side in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            s = semantic_side.split('_', 1)[1]
            if s in sides:
                return s
    return sides[0]


def _anchor_reference_count(env: PlacementEnv, ref: str) -> int:
    return sum(1 for r in env.refs if env._anchor_refs.get(r) == ref)


def _frozen_anchor_refs(env: PlacementEnv) -> set[str]:
    frozen = set()
    for ref in env.refs:
        sem = env._semantic_class.get(ref, '')
        if env._anchor_refs.get(ref) is None and _anchor_reference_count(env, ref) >= 2 and (env._is_large_component(ref) or sem in {"core", "interface", "power", "mechanical"}):
            frozen.add(ref)
    return frozen


def _placement_issue_score(env: PlacementEnv, ref: str, placed: Dict[str, Tuple[float, float, int]]) -> float:
    if ref not in placed:
        return 0.0

    _refresh_env_state(env, placed)
    x, y, rot = placed[ref]
    bb = env._ref_bbox(ref, x, y, rot)

    density_pen = 0.0
    for other_ref, (ox, oy, orot) in env.placed.items():
        if other_ref == ref:
            continue
        obb = env._ref_bbox(other_ref, ox, oy, orot)
        density_pen += env._density_pair_penalty(ref, bb, other_ref, obb)

    return float(
        env.objective_cfg.edge_clearance_weight * env._edge_penalty_from_bbox(ref, bb)
        + env.objective_cfg.density_weight * density_pen
        + env.objective_cfg.soft_spacing_weight * env._soft_spacing_penalty_for_candidate(ref, bb)
        + env.objective_cfg.neatness_weight * env._neatness_penalty_for_candidate(ref, x, y, bb)
    )



def _decrowd_candidates(
    env: PlacementEnv,
    ref: str,
    placed: Dict[str, Tuple[float, float, int]],
) -> List[Tuple[float, float, int]]:
    if ref not in placed:
        return []
    x, y, rot = placed[ref]
    step = float(env.task.grid_mm)
    side = _preferred_edge_side(env, ref)
    others = [(r, p) for r, p in placed.items() if r != ref]
    near = sorted(others, key=lambda item: (item[1][0] - x) ** 2 + (item[1][1] - y) ** 2)[:4]
    if near:
        cx = sum(float(p[0]) for _, p in near) / len(near)
        cy = sum(float(p[1]) for _, p in near) / len(near)
        away_x = x - cx
        away_y = y - cy
    else:
        xmin, ymin, xmax, ymax = env.task.bbox_mm
        away_x = x - 0.5 * (xmin + xmax)
        away_y = y - 0.5 * (ymin + ymax)

    max_radius = 3 if (env._is_large_component(ref) or env._anchor_refs.get(ref) or env._edge_band_sides(ref) or env._semantic_class.get(ref, '') in {'core', 'interface', 'power'}) else 2
    offsets: List[Tuple[float, float, float]] = []
    for radius in range(1, max_radius + 1):
        for ix in range(-radius, radius + 1):
            for iy in range(-radius, radius + 1):
                if max(abs(ix), abs(iy)) != radius:
                    continue
                if ix == 0 and iy == 0:
                    continue
                dx = float(ix) * step
                dy = float(iy) * step
                norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                score = (dx * away_x + dy * away_y) / norm
                if side in {'left', 'right'}:
                    score += 0.18 * abs(dy) - 0.06 * abs(dx)
                elif side in {'top', 'bottom'}:
                    score += 0.18 * abs(dx) - 0.06 * abs(dy)
                offsets.append((-score, dx, dy))

    offsets.sort(key=lambda item: (item[0], abs(item[1]) + abs(item[2]), abs(item[1]), abs(item[2])))
    seen = set()
    out: List[Tuple[float, float, int]] = []
    for _, dx, dy in offsets:
        cand = (float(x + dx), float(y + dy), int(rot))
        key = (round(cand[0], 6), round(cand[1], 6), int(cand[2]))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _local_polish_candidates(env: PlacementEnv, ref: str, placed: Dict[str, Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    if ref not in placed:
        return []
    x, y, rot = placed[ref]
    step = float(env.task.grid_mm)
    deltas = [
        (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
        (step, step), (step, -step), (-step, step), (-step, -step),
    ]
    out = []
    seen = set()
    for dx, dy in deltas:
        cand = (float(x + dx), float(y + dy), int(rot))
        key = (round(cand[0], 6), round(cand[1], 6), int(cand[2]))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _postprocess_layout(task, placed: Dict[str, Tuple[float, float, int]], env_kwargs: Dict[str, Any]) -> Dict[str, Tuple[float, float, int]]:
    env = PlacementEnv(task, **env_kwargs)
    placed = dict(placed)
    xmin, ymin, xmax, ymax = task.bbox_mm

    _refresh_env_state(env, placed)
    frozen_anchors = _frozen_anchor_refs(env)

    # 1) edge / anchor regularization
    groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        g = env._same_side_groups.get(ref)
        if g and ref in placed:
            groups.setdefault(g, []).append(ref)

    for refs in groups.values():
        side_votes = [_preferred_edge_side(env, r) for r in refs]
        side_votes = [s for s in side_votes if s in {'left', 'right', 'top', 'bottom'}]
        if not side_votes:
            continue
        side = max(side_votes, key=side_votes.count)
        refs.sort(key=lambda r: (env._boundary_orders.get(r, 10**9), r))
        axis_vals = []
        for r in refs:
            x, y, rot = placed[r]
            axis_vals.append(y if side in {'left', 'right'} else x)
        lo = min(axis_vals)
        hi = max(axis_vals)
        if len(refs) == 1:
            targets = axis_vals
        else:
            step = (hi - lo) / max(1, len(refs) - 1)
            targets = [lo + i * step for i in range(len(refs))]
        for r, axis_t in zip(refs, targets):
            if r in frozen_anchors:
                continue
            x, y, rot = placed[r]
            c = env.comp_by_ref[r]
            w_mm, h_mm = c.size_mm
            if rot % 180 != 0:
                w_mm, h_mm = h_mm, w_mm
            target_clear = 0.0 if env._hard_boundary_required(r) else env._edge_target_clearance_mm(r, side)
            if side == 'left':
                cand = (xmin + w_mm / 2 + target_clear, axis_t, rot)
            elif side == 'right':
                cand = (xmax - w_mm / 2 - target_clear, axis_t, rot)
            elif side == 'bottom':
                cand = (axis_t, ymin + h_mm / 2 + target_clear, rot)
            else:
                cand = (axis_t, ymax - h_mm / 2 - target_clear, rot)
            _try_update(env, placed, r, cand, max_move_mm=1.8)

    for ref in task.sequence:
        if ref not in placed or env._same_side_groups.get(ref):
            continue
        side = _preferred_edge_side(env, ref)
        if side not in {'left', 'right', 'top', 'bottom'}:
            continue
        if ref in frozen_anchors:
            continue
        x, y, rot = placed[ref]
        c = env.comp_by_ref[ref]
        w_mm, h_mm = c.size_mm
        if rot % 180 != 0:
            w_mm, h_mm = h_mm, w_mm
        target_clear = 0.0 if env._hard_boundary_required(ref) else env._edge_target_clearance_mm(ref, side)
        if side == 'left':
            cand = (xmin + w_mm / 2 + target_clear, y, rot)
        elif side == 'right':
            cand = (xmax - w_mm / 2 - target_clear, y, rot)
        elif side == 'bottom':
            cand = (x, ymin + h_mm / 2 + target_clear, rot)
        else:
            cand = (x, ymax - h_mm / 2 - target_clear, rot)
        _try_update(env, placed, ref, cand, max_move_mm=1.8)

    _refresh_env_state(env, placed)

    # 2) anchor / align / pitch / orientation keep the existing structure, now starting from band-centered placements
    for ref in task.sequence:
        if ref not in placed:
            continue
        if ref in frozen_anchors:
            continue
        anchor_ref = env._anchor_refs.get(ref)
        if not anchor_ref or anchor_ref not in placed or anchor_ref == ref:
            continue
        x, y, rot = placed[ref]
        ax, ay, _ = placed[anchor_ref]
        sub = env._subzones.get(ref, 'free')
        tx, ty = x, y
        step = min(1.0, 1.5 * task.grid_mm)
        if sub == 'left':
            tx = min(x, ax - step)
        elif sub == 'right':
            tx = max(x, ax + step)
        elif sub == 'top':
            ty = max(y, ay + step)
        elif sub == 'bottom':
            ty = min(y, ay - step)
        elif sub == 'around':
            tx = x + 0.25 * (ax - x)
            ty = y + 0.25 * (ay - y)
        _try_update(env, placed, ref, (tx, ty, rot), max_move_mm=1.4)

    _refresh_env_state(env, placed)

    align_groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        g = env._align_groups.get(ref)
        if g and ref in placed:
            align_groups.setdefault(g, []).append(ref)
    for refs in align_groups.values():
        if len(refs) < 2:
            continue
        xs = [placed[r][0] for r in refs]
        ys = [placed[r][1] for r in refs]
        mx = sorted(xs)[len(xs) // 2]
        my = sorted(ys)[len(ys) // 2]
        dx = sum(abs(v - mx) for v in xs)
        dy = sum(abs(v - my) for v in ys)
        use_x = dx <= dy
        for r in refs:
            if r in frozen_anchors:
                continue
            x, y, rot = placed[r]
            cand = (mx, y, rot) if use_x else (x, my, rot)
            _try_update(env, placed, r, cand, max_move_mm=1.0)

    _refresh_env_state(env, placed)

    pitch_groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        if ref not in placed:
            continue
        g = env._same_side_groups.get(ref) or env._align_groups.get(ref)
        if g:
            pitch_groups.setdefault(str(g), []).append(ref)
    for refs in pitch_groups.values():
        if len(refs) < 3:
            continue
        side_votes = [_preferred_edge_side(env, r) for r in refs]
        side_votes = [s for s in side_votes if s in {'left', 'right', 'top', 'bottom'}]
        if side_votes:
            side = max(side_votes, key=side_votes.count)
            axis = 'y' if side in {'left', 'right'} else 'x'
        else:
            xs = [placed[r][0] for r in refs]
            ys = [placed[r][1] for r in refs]
            axis = 'y' if (max(xs) - min(xs)) <= (max(ys) - min(ys)) else 'x'
        refs_sorted = sorted(refs, key=lambda r: (placed[r][1] if axis == 'y' else placed[r][0], r))
        vals = [placed[r][1] if axis == 'y' else placed[r][0] for r in refs_sorted]
        lo, hi = min(vals), max(vals)
        step = (hi - lo) / max(1, len(refs_sorted) - 1)
        targets = [lo + i * step for i in range(len(refs_sorted))]
        for r, axis_t in zip(refs_sorted, targets):
            if r in frozen_anchors:
                continue
            x, y, rot = placed[r]
            cand = (x, axis_t, rot) if axis == 'y' else (axis_t, y, rot)
            _try_update(env, placed, r, cand, max_move_mm=1.1)

    _refresh_env_state(env, placed)

    ori_groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        if ref not in placed or (not env._orientation_sensitive(ref)):
            continue
        g = env._orientation_group_name(ref)
        if g:
            ori_groups.setdefault(str(g), []).append(ref)
    for refs in ori_groups.values():
        if len(refs) < 2:
            continue
        local_xy = {r: (placed[r][0], placed[r][1]) for r in refs if r in placed}
        local_rot = {r: int(placed[r][2]) for r in refs if r in placed}
        target_rot = int(env._target_orientation_for_group(refs, local_xy, local_rot))
        for r in refs:
            if r in frozen_anchors:
                continue
            x, y, rot = placed[r]
            if int(rot) % 360 == target_rot:
                continue
            _try_update(env, placed, r, (x, y, target_rot), max_move_mm=0.0)

    # 2) declutter / inflation search
    _refresh_env_state(env, placed)
    base_obj = _layout_objective(task, placed, env_kwargs)
    for _round in range(3):
        _refresh_env_state(env, placed)
        scored = []
        for ref in task.sequence:
            if ref not in placed:
                continue
            score = _placement_issue_score(env, ref, placed)
            if score > 1e-9:
                scored.append((score, ref))
        scored.sort(key=lambda t: (-t[0], t[1]))
        changed = False
        for _score, ref in scored:
            if ref in frozen_anchors:
                continue
            current = placed[ref]
            max_move = _declutter_move_limit(env, ref)
            for cand in _decrowd_candidates(env, ref, placed):
                if cand == current:
                    continue
                if not _is_placement_legal(env, ref, cand, placed):
                    continue
                trial = dict(placed)
                trial[ref] = cand
                if ((cand[0] - current[0]) ** 2 + (cand[1] - current[1]) ** 2) ** 0.5 > max_move + 1e-6:
                    continue
                trial_obj = _layout_objective(task, trial, env_kwargs)
                if trial_obj + 1e-9 < base_obj:
                    placed = trial
                    base_obj = trial_obj
                    changed = True
                    _refresh_env_state(env, placed)
                    break
        if not changed:
            break

    # 3) final local polish
    _refresh_env_state(env, placed)
    base_obj = _layout_objective(task, placed, env_kwargs)
    for _round in range(2):
        changed = False
        _refresh_env_state(env, placed)
        scored = []
        for ref in task.sequence:
            if ref in placed:
                score = _placement_issue_score(env, ref, placed)
                if score > 1e-9:
                    scored.append((score, ref))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for _score, ref in scored:
            if ref in frozen_anchors:
                continue
            current = placed[ref]
            for cand in _local_polish_candidates(env, ref, placed):
                if cand == current or not _is_placement_legal(env, ref, cand, placed):
                    continue
                trial = dict(placed)
                trial[ref] = cand
                trial_obj = _layout_objective(task, trial, env_kwargs)
                if trial_obj + 1e-9 < base_obj:
                    placed = trial
                    base_obj = trial_obj
                    changed = True
                    _refresh_env_state(env, placed)
                    break
        if not changed:
            break

    return placed


def _action_features(env: PlacementEnv) -> Any:
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw = max(1e-6, (xmax - xmin))
    bh = max(1e-6, (ymax - ymin))
    grid = float(env.task.grid_mm)
    w, h = env.grid_shape()
    R = len(env.rotations)

    import math

    xs = xmin + (np.arange(w, dtype=np.float32) + 0.5) * grid
    ys = ymin + (np.arange(h, dtype=np.float32) + 0.5) * grid
    xn = (xs - xmin) / bw
    yn = (ys - ymin) / bh
    Xn, Yn = np.meshgrid(xn, yn, indexing="ij")

    feats = np.zeros((R, w, h, 4), dtype=np.float32)
    for ri, rot in enumerate(env.rotations):
        rr = float(rot) * math.pi / 180.0
        feats[ri, :, :, 0] = Xn
        feats[ri, :, :, 1] = Yn
        feats[ri, :, :, 2] = math.sin(rr)
        feats[ri, :, :, 3] = math.cos(rr)
    return feats.reshape(-1, 4)


def _teacher_from_ckpt(ckpt: Dict[str, Any]) -> TeacherConfig:
    t = ckpt.get("teacher", {}) or {}
    return TeacherConfig(
        tau=float(t.get("tau", 0.5)),
        lambda_region_prior=float(t.get("lambda_region_prior", t.get("lambda_region", 0.12))),
        metric_weight=float(t.get("metric_weight", 0.15)),
        topk=int(t.get("topk", 256)),
        objective_delta_max=(
            None
            if t.get("objective_delta_max", t.get("hpwl_delta_max", None)) is None
            else float(t.get("objective_delta_max", t.get("hpwl_delta_max")))
        ),
        gate_rollout=bool(t.get("gate_rollout", True)),
    )


def _env_kwargs_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    e = ckpt.get("env_config", {}) or {}

    def _get_float(*keys: str, default: float) -> float:
        for k in keys:
            if k in e and e.get(k) is not None:
                return float(e.get(k))
        return float(default)

    return {
        "alignment_bonus": _get_float("alignment_bonus", default=0.05),
        "edge_bonus": _get_float("edge_bonus", default=0.15),
        "edge_eps_mm": _get_float("edge_eps_mm", default=1.5),
        "non_interface_edge_penalty": _get_float("non_interface_edge_penalty", default=10.0),
        "non_interface_edge_margin_mm": _get_float("non_interface_edge_margin_mm", default=2.5),
        "density_penalty": _get_float("density_penalty", default=3.0),
        "density_radius_mm": _get_float("density_radius_mm", default=4.0),
        "interior_penalty": _get_float("interior_penalty", default=1.0),
        "interior_margin_ratio": _get_float("interior_margin_ratio", default=0.18),
        "nslw_weight": _get_float("nslw_weight", default=0.2),
        "region_weight": _get_float("region_weight", default=0.55),
        "conn_weight": _get_float("conn_weight", default=0.50),
        "objective_align_weight": _get_float("objective_align_weight", "align_weight", default=0.28),
        "group_weight": _get_float("group_weight", default=0.12),
        "anchor_weight": _get_float("anchor_weight", default=0.18),
        "boundary_group_weight": _get_float("boundary_group_weight", default=0.22),
        "pitch_weight": _get_float("pitch_weight", default=0.22),
        "orientation_weight": _get_float("orientation_weight", default=0.14),
        "edge_clearance_weight": _get_float("edge_clearance_weight", "objective_edge_clearance_weight", default=0.40),
        "interior_weight": _get_float("interior_weight", "objective_interior_weight", default=0.30),
        "density_weight": _get_float("density_weight", "objective_density_weight", default=0.45),
        "soft_spacing_weight": _get_float("soft_spacing_weight", "objective_soft_spacing_weight", default=0.32),
        "neatness_weight": _get_float("neatness_weight", "objective_neatness_weight", default=0.12),
        "edge_band_ratio": _get_float("edge_band_ratio", default=0.12),
        "edge_band_center_ratio": _get_float("edge_band_center_ratio", default=0.55),
        "soft_spacing_same_group_extra_mm": _get_float("soft_spacing_same_group_extra_mm", default=0.6),
        "soft_spacing_cross_group_extra_mm": _get_float("soft_spacing_cross_group_extra_mm", default=1.4),
        "soft_spacing_large_extra_mm": _get_float("soft_spacing_large_extra_mm", default=0.7),
        "same_group_density_scale": _get_float("same_group_density_scale", default=0.40),
        "critical_neighbor_density_scale": _get_float("critical_neighbor_density_scale", default=0.25),
        "anchor_group_density_scale": _get_float("anchor_group_density_scale", default=0.50),
        "large_pair_density_scale": _get_float("large_pair_density_scale", default=1.20),
    }


def _region_cfg_from_ckpt(ckpt: Dict[str, Any]) -> RegionPriorConfig:
    r = ckpt.get("region_prior", {}) or {}
    return RegionPriorConfig(
        enabled=bool(r.get("enabled", False)),
        grid_x=int(r.get("grid_x", 6)),
        grid_y=int(r.get("grid_y", 6)),
        heatmap_sigma_cells=float(r.get("heatmap_sigma_cells", 0.85)),
        zone_edge_ratio=float(r.get("zone_edge_ratio", 0.12)),
        zone_core_ratio=float(r.get("zone_core_ratio", 0.28)),
        zone_prior_weight=float(r.get("zone_prior_weight", 0.35)),
        aux_heatmap_weight=float(r.get("aux_heatmap_weight", 0.30)),
        aux_zone_weight=float(r.get("aux_zone_weight", 0.10)),
    )


def load_model(
    ckpt_path: str, device: str = "cpu"
) -> Tuple[MaskedPolicy, Dict[str, Any], TeacherConfig, RegionPriorConfig]:
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt.get("model_cfg", {}) or {}
    cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else ModelConfig()
    action_feat_dim = int(ckpt.get("action_feat_dim", 4))
    region_grid_shape = tuple(ckpt.get("region_grid_shape", [6, 6]))
    num_region_types = int(ckpt.get("num_region_types", ckpt.get("num_zones", len(REGION_TYPE_NAMES))))
    num_semantic_classes = int(ckpt.get("num_semantic_classes", len(SEMANTIC_CLASS_NAMES)))
    num_side_preferences = int(ckpt.get("num_side_preferences", len(SIDE_PREFERENCE_NAMES)))
    num_subzones = int(ckpt.get("num_subzones", len(SUBZONE_NAMES)))
    num_pairwise_relations = int(ckpt.get("num_pairwise_relations", len(PAIRWISE_RELATION_NAMES)))

    model = MaskedPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        cfg=cfg,
        action_feat_dim=action_feat_dim,
        region_grid_shape=(int(region_grid_shape[0]), int(region_grid_shape[1])),
        num_region_types=int(num_region_types),
        num_semantic_classes=int(num_semantic_classes),
    )
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    if getattr(incompatible, 'missing_keys', None) or getattr(incompatible, 'unexpected_keys', None):
       region_section = ckpt.get("region_prior", {}) or {}
       if region_section:
          print(f'[infer] state_dict missing_keys={list(incompatible.missing_keys)} unexpected_keys={list(incompatible.unexpected_keys)}')
    model.to(device)
    model.eval()



    teacher = _teacher_from_ckpt(ckpt)
    region_cfg = _region_cfg_from_ckpt(ckpt)
    return model, ckpt, teacher, region_cfg



def _resolve_device(device: str) -> torch.device:
    device_t = torch.device(device)
    if device_t.type == "cuda" and not torch.cuda.is_available():
        print("[infer] CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return device_t


def _apply_action_fast(env: PlacementEnv, action: Tuple[int, int, int]) -> Tuple[bool, Dict[str, Any]]:
    """Apply a selected legal inference action without recomputing masks/objective.

    env.step() is training-oriented: it recomputes action_mask_and_bias(), then
    recomputes the whole objective for reward, then returns observe() for the next
    component. During greedy inference all three are redundant because the caller
    already has the current mask and ignores the reward/next observation.
    """
    if env.done():
        return True, {}

    rix, ix, iy = int(action[0]), int(action[1]), int(action[2])
    ref = env.sequence[env.t]
    w, h = env.grid_shape()
    if rix < 0 or rix >= len(env.rotations) or ix < 0 or ix >= w or iy < 0 or iy >= h:
        env.terminated = True
        env.t = len(env.sequence)
        return False, {"illegal": True, "reason": "index_oob"}

    xmin, ymin, _, _ = env.task.bbox_mm
    x = xmin + (ix + 0.5) * env.task.grid_mm
    y = ymin + (iy + 0.5) * env.task.grid_mm
    rot = int(env.rotations[rix])
    bb = env._ref_bbox(ref, x, y, rot)

    post_hard_boundary = env.enforce_interface_on_boundary and env._hard_boundary_required(ref)
    post_band_sides = env._edge_band_sides(ref)
    post_boundary_fail = False
    if post_hard_boundary:
        if post_band_sides:
            post_boundary_fail = not env._touch_sides(bb, post_band_sides)
        else:
            post_boundary_fail = not env._touch_sides(bb, ['left', 'right', 'top', 'bottom'])

    if (not env._inside(bb)) or env._violates_spacing(bb) or post_boundary_fail:
        env.terminated = True
        env.t = len(env.sequence)
        return False, {"illegal": True, "reason": "postcheck"}

    env.placed[ref] = (float(x), float(y), int(rot))
    env.placed_order.append(ref)
    env.occupied.append(bb)
    env.t += 1
    return True, {"obj": None}


@torch.inference_mode()
def _policy_logits_and_region_prior_fast(
    model: MaskedPolicy,
    env: PlacementEnv,
    ref: str,
    tokens_t: torch.Tensor,
    feat_t: torch.Tensor,
    region_cfg: RegionPriorConfig,
    device_t: torch.device,
    region_idx_cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    enc = model.encode(tokens_t)
    q_last = model.q_proj(enc[:, -1, :])
    logits = model.action_logits_from_query(q_last, feat_t).squeeze(0)

    if not bool(region_cfg.enabled):
        return logits, None

    region_type_logits = model.region_type_logits(q_last).squeeze(0)
    region_idx = region_idx_cache.get(ref)
    if region_idx is None or region_idx.device != device_t:
        region_idx = build_action_region_indices_cuda(
            env,
            ref,
            device=device_t,
            zone_edge_ratio=float(region_cfg.zone_edge_ratio),
            zone_core_ratio=float(region_cfg.zone_core_ratio),
        )
        region_idx_cache[ref] = region_idx
    region_prior_t = action_region_prior_from_predictions_cuda(
        region_type_logits,
        region_type_idx_flat=region_idx,
        zone_prior_weight=float(region_cfg.zone_prior_weight),
    )
    return logits, region_prior_t


def infer_layout(
    task_json_path: str,
    ckpt_path: str,
    device: str = "cuda",
    *,
    fast_step: bool = True,
    postprocess: bool = True,
) -> Dict[str, Any]:
    task = task_from_json(task_json_path)
    device_t = _resolve_device(device)
    model, meta, teacher, region_cfg = load_model(ckpt_path, device=str(device_t))
    env_kwargs = _env_kwargs_from_ckpt(meta)
    env = PlacementEnv(task, **env_kwargs)

    w, h = env.grid_shape()
    R = len(env.rotations)

    # rollout_episode() 一样：action features 整局只算一次，并常驻 GPU/torch device。
    feat = _action_features(env)
    feat_t = torch.from_numpy(feat).to(device_t)
    region_idx_cache: Dict[str, torch.Tensor] = {}

    while not env.done():
        ref = env.current_ref()

        # Dense legal-mask work stays in torch/GPU. The old path also built a
        # heuristic bias map, but greedy inference never used it.
        mask_flat_t = action_mask_cuda(env, ref, device_t).reshape(-1)
        if not bool((mask_flat_t > 0.5).any().item()):
            env.terminated = True
            break

        obj_maps = objective_delta_mask_cuda(env, ref, device_t)
        objective_delta_t = obj_maps["total"].reshape(-1)

        tokens = build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device_t)

        with torch.inference_mode():
            logits, region_prior_t = _policy_logits_and_region_prior_fast(
                model, env, ref, tokens_t, feat_t, region_cfg, device_t, region_idx_cache
            )
            logits = logits.masked_fill(mask_flat_t < 0.5, -1e9)

            if teacher.gate_rollout:
                _q, cand = build_teacher_distribution_cuda(
                    mask_flat_t, objective_delta_t, teacher, device=device_t, region_flat=region_prior_t
                )
                logits = logits.masked_fill(~cand, -1e9)

            a = int(torch.argmax(logits).item())

        action = _unflatten_action(a, w, h, R)
        if fast_step:
            ok, info = _apply_action_fast(env, action)
            if not ok or info.get("illegal"):
                break
        else:
            _obs2, _r, _done, info = env.step(action)
            if info.get("illegal"):
                break

    raw_placed = dict(env.placed)
    raw_obj = _layout_objective(task, raw_placed, env_kwargs) if raw_placed else 0.0

    if postprocess:
        final_placed = _postprocess_layout(task, raw_placed, env_kwargs)
        final_obj = _layout_objective(task, final_placed, env_kwargs)
    else:
        final_placed = raw_placed
        final_obj = raw_obj

    return {
        "placed": final_placed,
        "placed_raw": raw_placed,
        "objective": float(final_obj),
        "objective_raw": float(raw_obj),
        "postprocess_applied": bool(postprocess),
        "terminated": bool(env.terminated),
    }
