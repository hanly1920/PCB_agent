from __future__ import annotations
from typing import Any, Dict, Tuple

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
from .region_prior import RegionPriorConfig, REGION_TYPE_NAMES, SEMANTIC_CLASS_NAMES


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
    c = env.comp_by_ref[ref]
    if env.enforce_interface_on_boundary and (c.allowed_sides or ('conn' in (c.type or '').lower())):
        if c.allowed_sides and (not env._touch_sides(bb, c.allowed_sides)):
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


def _postprocess_layout(task, placed: Dict[str, Tuple[float, float, int]], env_kwargs: Dict[str, Any]) -> Dict[str, Tuple[float, float, int]]:
    env = PlacementEnv(task, **env_kwargs)
    placed = dict(placed)
    xmin, ymin, xmax, ymax = task.bbox_mm
    band = max(task.grid_mm, 0.10 * min(xmax - xmin, ymax - ymin))
    # 1) edge-group snap + order smoothing
    groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        g = env._same_side_groups.get(ref)
        if g and ref in placed:
            groups.setdefault(g, []).append(ref)
    for refs in groups.values():
        side_votes = [env._side_preferences.get(r, env._region_targets.get(r, 'free')) for r in refs]
        side = max(side_votes, key=side_votes.count)
        if side not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            continue
        refs.sort(key=lambda r: (env._boundary_orders.get(r, 10**9), r))
        axis_vals = []
        for r in refs:
            x, y, rot = placed[r]
            axis_vals.append(y if side in {'edge_left', 'edge_right'} else x)
        lo = min(axis_vals)
        hi = max(axis_vals)
        if len(refs) == 1:
            targets = axis_vals
        else:
            step = (hi - lo) / max(1, len(refs) - 1)
            targets = [lo + i * step for i in range(len(refs))]
        for r, axis_t in zip(refs, targets):
            x, y, rot = placed[r]
            c = env.comp_by_ref[r]
            w_mm, h_mm = c.size_mm
            if rot % 180 != 0:
                w_mm, h_mm = h_mm, w_mm
            margin = min(0.35 * band, 1.5 * task.grid_mm)
            if side == 'edge_left':
                cand = (xmin + w_mm / 2 + margin, axis_t, rot)
            elif side == 'edge_right':
                cand = (xmax - w_mm / 2 - margin, axis_t, rot)
            elif side == 'edge_bottom':
                cand = (axis_t, ymin + h_mm / 2 + margin, rot)
            else:
                cand = (axis_t, ymax - h_mm / 2 - margin, rot)
            _try_update(env, placed, r, cand, max_move_mm=1.5)
    # 2) anchor/subzone gentle nudge
    for ref in task.sequence:
        if ref not in placed:
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
        _try_update(env, placed, ref, (tx, ty, rot), max_move_mm=1.0)
    # 3) align-group snap
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
        mx = sorted(xs)[len(xs)//2]
        my = sorted(ys)[len(ys)//2]
        dx = sum(abs(v - mx) for v in xs)
        dy = sum(abs(v - my) for v in ys)
        use_x = dx <= dy
        for r in refs:
            x, y, rot = placed[r]
            cand = (mx, y, rot) if use_x else (x, my, rot)
            _try_update(env, placed, r, cand, max_move_mm=0.8)
    # 4) uniform pitch smoothing for align / side groups
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
        side_votes = [env._side_preferences.get(r, env._region_targets.get(r, 'free')) for r in refs]
        side = max(side_votes, key=side_votes.count)
        if side in {'edge_left', 'edge_right'}:
            axis = 'y'
        elif side in {'edge_top', 'edge_bottom'}:
            axis = 'x'
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
            x, y, rot = placed[r]
            cand = (x, axis_t, rot) if axis == 'y' else (axis_t, y, rot)
            _try_update(env, placed, r, cand, max_move_mm=0.9)

    # 5) orientation consistency (absolute orientation, not only axis parity)
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
            x, y, rot = placed[r]
            if int(rot) % 360 == target_rot:
                continue
            _try_update(env, placed, r, (x, y, target_rot), max_move_mm=0.0)
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
        lambda_region_prior=float(t.get("lambda_region_prior", t.get("lambda_region", 0.35))),
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
    return {
        "alignment_bonus": float(e.get("alignment_bonus", 0.05)),
        "edge_bonus": float(e.get("edge_bonus", 0.15)),
        "edge_eps_mm": float(e.get("edge_eps_mm", 1.5)),
        "non_interface_edge_penalty": float(e.get("non_interface_edge_penalty", 10.0)),
        "non_interface_edge_margin_mm": float(e.get("non_interface_edge_margin_mm", 2.5)),
        "density_penalty": float(e.get("density_penalty", 3.0)),
        "density_radius_mm": float(e.get("density_radius_mm", 4.0)),
        "interior_penalty": float(e.get("interior_penalty", 1.0)),
        "interior_margin_ratio": float(e.get("interior_margin_ratio", 0.18)),
        "nslw_weight": float(e.get("nslw_weight", 0.2)),
        "region_weight": float(e.get("region_weight", 0.9)),
        "conn_weight": float(e.get("conn_weight", 0.7)),
        "objective_align_weight": float(e.get("objective_align_weight", 0.25)),
        "group_weight": float(e.get("group_weight", 0.20)),
        "anchor_weight": float(e.get("anchor_weight", 0.30)),
        "boundary_group_weight": float(e.get("boundary_group_weight", 0.28)),
        "pitch_weight": float(e.get("pitch_weight", 0.18)),
        "orientation_weight": float(e.get("orientation_weight", 0.12)),
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
    model.eval()

    teacher = _teacher_from_ckpt(ckpt)
    region_cfg = _region_cfg_from_ckpt(ckpt)
    return model, ckpt, teacher, region_cfg


def infer_layout(task_json_path: str, ckpt_path: str, device: str = "cuda") -> Dict[str, Any]:
    task = task_from_json(task_json_path)
    model, meta, teacher, region_cfg = load_model(ckpt_path, device=device)
    env_kwargs = _env_kwargs_from_ckpt(meta)
    env = PlacementEnv(task, **env_kwargs)
    device_t = torch.device(device)

    w, h = env.grid_shape()
    R = len(env.rotations)

    # rollout_episode() 一样：action features 整局只算一次
    feat = _action_features(env)
    feat_t = torch.from_numpy(feat).to(device_t)

    while not env.done():
        ref = env.current_ref()
        obs = env.observe()
        mask = obs["action_mask"].reshape(-1).astype(np.float32)
        bias = obs["action_bias"].reshape(-1).astype(np.float32)

        if float(mask.max()) < 0.5:
            env.terminated = True
            break

        objective_delta = env.objective_delta_mask(ref)["total"].reshape(-1).astype(np.float32)

        tokens = build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device_t)

        with torch.no_grad():
            logits, _region_type_logits, _semantic_class_logits, region_prior = _policy_outputs_with_region(
                model, env, ref, tokens_t, feat_t, region_cfg
            )

            m = torch.from_numpy(mask).to(device_t)
            logits = logits.masked_fill(m < 0.5, -1e9)

            if teacher.gate_rollout:
                _q, cand = build_teacher_distribution(
                    mask, objective_delta, teacher, device=device_t, region_flat=region_prior
                )
                cand_t = torch.from_numpy(cand.astype(np.bool_)).to(device_t)
                logits = logits.masked_fill(~cand_t, -1e9)

            a = int(torch.argmax(logits).item())

        _obs2, _r, _done, info = env.step(_unflatten_action(a, w, h, R))
        if info.get("illegal"):
            break

    raw_placed = dict(env.placed)
    final_placed = _postprocess_layout(task, raw_placed, env_kwargs)
    final_obj = _layout_objective(task, final_placed, env_kwargs)
    return {
        "placed": final_placed,
        "placed_raw": raw_placed,
        "objective": float(final_obj),
        "objective_raw": float(env.prev_obj),
        "postprocess_applied": True,
        "terminated": bool(env.terminated),
    }
