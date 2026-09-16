from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
import fnmatch
import json
import math
import os
import tempfile

import numpy as np
import torch

from .dataset import task_from_json, normalize_sequence_policy, strip_expert_fields_from_task
from .env import PlacementEnv
from .model import MaskedPolicy, ModelConfig
from .train import (
    TeacherConfig,
    validate_placement_env_kwargs,
    build_placement_env_kwargs,
    INCOMPLETE_OBJECTIVE_PENALTY,
)
from .policy_runtime import (
    policy_outputs_with_region,
    unflatten_action,
    score_actions,
    ACTION_SCORING_VERSION,
    DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    DEFAULT_ROLLOUT_REGION_ALPHA,
)
from .env_cuda import (
    objective_delta_mask_cuda,
    action_features_cuda,
    action_prior_total_cuda,
    build_context_tokens_cuda,
    action_mask_and_bias_cuda,
    _cached_grid_maps_cuda,
)
from .region_prior import RegionPriorConfig, REGION_TYPE_NAMES, SEMANTIC_CLASS_NAMES, SIDE_PREFERENCE_NAMES, SUBZONE_NAMES, PAIRWISE_RELATION_NAMES
from .semantic_metrics import evaluate_task_semantic_metrics
from .utils import drop_mounting_holes_from_task_json


LAYOUT_OBJECTIVE_PRESETS: Dict[str, Dict[str, float]] = {
    # Keep checkpoint/objective values unchanged.
    "checkpoint": {},
    "none": {},
    # Maximize electrical compactness; visual priors remain weak.
    "hpwl": {
        "hpwl_weight": 1.0,
        "w_hpwl_weight": 0.20,
        "nslw_weight": 0.05,
        "objective_align_weight": 0.20,
        "pitch_weight": 0.15,
        "orientation_weight": 0.10,
        "objective_soft_spacing_weight": 0.25,
        "objective_neatness_weight": 0.08,
    },
    # Recommended default for engineering-looking layouts.
    "balanced": {
        "hpwl_weight": 1.0,
        "w_hpwl_weight": 0.20,
        "nslw_weight": 0.08,
        "conn_weight": 0.50,
        "objective_align_weight": 0.35,
        "pitch_weight": 0.30,
        "orientation_weight": 0.20,
        "group_weight": 0.16,
        "anchor_weight": 0.20,
        "boundary_group_weight": 0.24,
        "module_floorplan_weight": 0.20,
        "module_floorplan_separation_mm": 2.0,
        "objective_soft_spacing_weight": 0.40,
        "objective_density_weight": 0.50,
        "objective_neatness_weight": 0.18,
    },
    # Stronger polish / row-column aesthetics.  May trade away more HPWL.
    "neat": {
        "hpwl_weight": 1.0,
        "w_hpwl_weight": 0.20,
        "nslw_weight": 0.08,
        "conn_weight": 0.50,
        "objective_align_weight": 0.45,
        "pitch_weight": 0.42,
        "orientation_weight": 0.30,
        "group_weight": 0.18,
        "anchor_weight": 0.20,
        "boundary_group_weight": 0.28,
        "module_floorplan_weight": 0.28,
        "module_floorplan_separation_mm": 2.4,
        "objective_soft_spacing_weight": 0.48,
        "objective_density_weight": 0.55,
        "objective_neatness_weight": 0.25,
    },
    # Makes boundary connector/order constraints more influential.
    "edge_strict": {
        "hpwl_weight": 1.0,
        "w_hpwl_weight": 0.20,
        "nslw_weight": 0.07,
        "region_weight": 0.70,
        "objective_align_weight": 0.36,
        "boundary_group_weight": 0.40,
        "module_floorplan_weight": 0.30,
        "module_floorplan_separation_mm": 2.2,
        "objective_edge_clearance_weight": 0.55,
        "pitch_weight": 0.34,
        "orientation_weight": 0.22,
        "objective_soft_spacing_weight": 0.40,
        "objective_neatness_weight": 0.18,
    },
    # Emphasize module-level floorplanning for routeability: clear module bboxes,
    # soft channels between unrelated modules, and preserved connector/anchor priors.
    "routeaware": {
        "hpwl_weight": 0.92,
        "w_hpwl_weight": 0.18,
        "nslw_weight": 0.12,
        "region_weight": 0.60,
        "module_region_weight": 0.38,
        "module_floorplan_weight": 0.36,
        "module_floorplan_separation_mm": 2.5,
        "module_floorplan_overlap_scale": 1.10,
        "module_floorplan_channel_scale": 0.80,
        "module_floorplan_compact_scale": 0.24,
        "module_floorplan_region_scale": 0.70,
        "conn_weight": 0.65,
        "anchor_weight": 0.30,
        "boundary_group_weight": 0.38,
        "objective_edge_clearance_weight": 0.60,
        "objective_align_weight": 0.38,
        "pitch_weight": 0.28,
        "orientation_weight": 0.16,
        "objective_density_weight": 0.75,
        "objective_soft_spacing_weight": 0.55,
        "objective_neatness_weight": 0.22,
    },
}


def apply_layout_objective_preset(
    env_kwargs: Dict[str, Any],
    preset: str | None = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return env kwargs with an optional objective preset and explicit overrides.

    This is inference-safe: it changes only objective/mask knobs accepted by
    PlacementEnv and does not alter model weights or checkpoint structure.
    """
    out = dict(env_kwargs)
    name = str(preset or "checkpoint").strip().lower()
    if name not in LAYOUT_OBJECTIVE_PRESETS:
        raise ValueError(f"Unknown layout_preset={preset!r}; choose one of {sorted(LAYOUT_OBJECTIVE_PRESETS)}")
    out.update(LAYOUT_OBJECTIVE_PRESETS[name])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                out[str(key)] = float(value)
    return validate_placement_env_kwargs(out)



def _coerce_ref_patterns(value: Optional[Iterable[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [p.strip() for p in value.replace(";", ",").split(",")]
    else:
        raw = [str(p).strip() for p in value]
    return [p for p in raw if p]


def _component_has_fixed_xy_json(comp: Dict[str, Any]) -> bool:
    if comp.get("fixed_xy_mm") is not None or comp.get("locked_xy_mm") is not None:
        return True
    fixed = comp.get("fixed") if isinstance(comp.get("fixed"), dict) else {}
    placement = comp.get("placement") if isinstance(comp.get("placement"), dict) else {}
    layout = comp.get("layout") if isinstance(comp.get("layout"), dict) else {}
    for source in (fixed, placement, layout):
        for key in ("xy_mm", "center_mm", "position_mm", "at_mm", "fixed_xy_mm"):
            if source.get(key) is not None:
                return True
    return False


def _component_has_fixed_rot_json(comp: Dict[str, Any]) -> bool:
    for key in ("fixed_rot", "fixed_rot_deg", "locked_rot", "locked_rot_deg"):
        if comp.get(key) is not None:
            return True
    fixed = comp.get("fixed") if isinstance(comp.get("fixed"), dict) else {}
    placement = comp.get("placement") if isinstance(comp.get("placement"), dict) else {}
    layout = comp.get("layout") if isinstance(comp.get("layout"), dict) else {}
    for source in (fixed, placement, layout):
        for key in ("rot", "rot_deg", "rotation", "fixed_rot"):
            if source.get(key) is not None:
                return True
    return False


def _copy_runtime_current_pose_to_fixed(comp: Dict[str, Any]) -> None:
    """Promote a current/imported pose into a fixed pose when DSL asks to lock it."""
    if not _component_has_fixed_xy_json(comp):
        current_xy = comp.get("current_xy_mm") or comp.get("current_position_mm") or comp.get("current_pos_mm")
        if current_xy is None and isinstance(comp.get("current"), dict):
            current_xy = comp["current"].get("xy_mm") or comp["current"].get("center_mm") or comp["current"].get("position_mm")
        if current_xy is None and isinstance(comp.get("layout"), dict):
            current_xy = comp["layout"].get("current_xy_mm") or comp["layout"].get("xy_mm") or comp["layout"].get("center_mm")
        if current_xy is not None:
            comp["fixed_xy_mm"] = current_xy
    if not _component_has_fixed_rot_json(comp):
        current_rot = comp.get("current_rot", comp.get("current_rot_deg", comp.get("current_rotation")))
        if current_rot is None and isinstance(comp.get("current"), dict):
            current_rot = comp["current"].get("rot", comp["current"].get("rot_deg", comp["current"].get("rotation")))
        if current_rot is None and isinstance(comp.get("layout"), dict):
            current_rot = comp["layout"].get("current_rot", comp["layout"].get("rot", comp["layout"].get("rotation")))
        if current_rot is not None:
            comp["fixed_rot"] = current_rot


def materialize_fixed_refs_task_json(
    task_json_path: str,
    fixed_refs: Optional[Iterable[str] | str] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Return a task JSON path with runtime fixed_refs applied.

    The original file is left untouched. Patterns are exact refs or glob-style
    refs (e.g. ``J*``). If a matched component does not already have a fixed pose,
    ``current_xy_mm/current_rot`` imported from KiCad are promoted to
    ``fixed_xy_mm/fixed_rot``. ``PlacementEnv`` will raise a clear error if a
    fixed component still has no usable pose.
    """
    patterns = _coerce_ref_patterns(fixed_refs)
    if not patterns:
        return str(task_json_path), None, {"requested": [], "matched": [], "unmatched": []}
    with open(task_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    comps = data.get("components") or []
    refs = [str(c.get("ref") or "") for c in comps if isinstance(c, dict)]
    matched: set[str] = set()
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        ref = str(comp.get("ref") or "")
        if not ref:
            continue
        if any(fnmatch.fnmatchcase(ref, pat) for pat in patterns):
            matched.add(ref)
            comp["fixed"] = True
            comp.setdefault("fixed_source", "runtime_fixed_refs")
            _copy_runtime_current_pose_to_fixed(comp)
    unmatched = [pat for pat in patterns if not any(fnmatch.fnmatchcase(ref, pat) for ref in refs)]
    fd, temp_path = tempfile.mkstemp(prefix="pcbplace_fixed_", suffix=".json")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return temp_path, temp_path, {"requested": patterns, "matched": sorted(matched), "unmatched": unmatched}

def _env_fixed_refs(env: PlacementEnv) -> set[str]:
    return set(getattr(env, 'fixed_refs', []) or [])


def _env_dynamic_placed_count(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> int:
    return sum(1 for ref in getattr(env, 'sequence', []) if ref in placed)


def _env_expected_total_count(env: PlacementEnv) -> int:
    return int(len(getattr(env, 'refs', []) or []))


def _ordered_placed_refs(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for ref in list(getattr(env, 'fixed_refs', []) or []):
        if ref in placed and ref not in seen:
            order.append(ref); seen.add(ref)
    for ref in list(getattr(env, 'sequence', []) or []):
        if ref in placed and ref not in seen:
            order.append(ref); seen.add(ref)
    for ref in list(getattr(env, 'refs', []) or []):
        if ref in placed and ref not in seen:
            order.append(ref); seen.add(ref)
    for ref in placed:
        if ref not in seen:
            order.append(ref); seen.add(ref)
    return order


def _merge_fixed_placements(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> Dict[str, Tuple[float, float, int]]:
    out = dict(placed)
    fixed_map = env.fixed_placement_map() if hasattr(env, 'fixed_placement_map') else {}
    for ref, pos in fixed_map.items():
        out[ref] = pos
    return out


def _layout_objective(task, placed: Dict[str, Tuple[float, float, int]], env_kwargs: Dict[str, Any]) -> float:
    env = PlacementEnv(task, **env_kwargs)
    _refresh_env_state(env, placed)
    return float(env.prev_obj)


def _layout_guard_terms(task, placed: Dict[str, Tuple[float, float, int]], env_kwargs: Dict[str, Any]) -> Dict[str, float]:
    """Small set of protected terms used by polish guards.

    These are deliberately simple and stable:
    - hpwl / w_hpwl should not degrade too much;
    - nslw should not drop unless the caller explicitly allows it;
    - illegal_count must never increase.
    """
    env = PlacementEnv(task, **env_kwargs)
    _refresh_env_state(env, placed)
    pins = env._pin_positions()
    hpwl = 0.0
    w_hpwl = 0.0
    for net, pts in pins.items():
        if len(pts) <= 1 or env._wire_net_ignored(net):
            continue
        # imports kept local-compatible with env implementation
        from .utils import hpwl_from_pins, w_hpwl_from_pins
        hpwl += hpwl_from_pins(pts)
        w_hpwl += w_hpwl_from_pins(pts)
    try:
        nslw = float(env._nslw_count_for_current_layout())
    except Exception:
        nslw = 0.0
    illegal = float(_layout_illegal_count(env, placed))
    return {
        "hpwl": float(hpwl),
        "w_hpwl": float(w_hpwl),
        "nslw": float(nslw),
        "illegal_count": float(illegal),
    }


def _layout_illegal_count(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> int:
    n = 0
    _refresh_env_state(env, placed)
    for ref, p in placed.items():
        if not _is_placement_legal(env, ref, p, placed):
            n += 1
    return int(n)


def _protected_terms_accept(
    before_terms: Dict[str, float],
    after_terms: Dict[str, float],
    protect_terms: Optional[Dict[str, float]],
) -> bool:
    if not protect_terms:
        return True
    hpwl_rel = protect_terms.get("hpwl_rel")
    if hpwl_rel is not None:
        before = max(1e-6, float(before_terms.get("hpwl", 0.0)))
        after = float(after_terms.get("hpwl", 0.0))
        if after > before * (1.0 + float(hpwl_rel)) + 1e-9:
            return False
    w_hpwl_rel = protect_terms.get("w_hpwl_rel")
    if w_hpwl_rel is not None:
        before = max(1e-6, float(before_terms.get("w_hpwl", 0.0)))
        after = float(after_terms.get("w_hpwl", 0.0))
        if after > before * (1.0 + float(w_hpwl_rel)) + 1e-9:
            return False
    nslw_drop = protect_terms.get("nslw_drop")
    if nslw_drop is not None:
        if float(after_terms.get("nslw", 0.0)) < float(before_terms.get("nslw", 0.0)) - float(nslw_drop) - 1e-9:
            return False
    illegal_delta = protect_terms.get("illegal_delta")
    if illegal_delta is not None:
        if float(after_terms.get("illegal_count", 0.0)) > float(before_terms.get("illegal_count", 0.0)) + float(illegal_delta) + 1e-9:
            return False
    return True


def _is_placement_legal(env: PlacementEnv, ref: str, placement: Tuple[float, float, int], placed: Dict[str, Tuple[float, float, int]]) -> bool:
    if getattr(env, 'is_ref_fixed', lambda _r: False)(ref):
        fixed_pos = env.fixed_placement_for_ref(ref) if hasattr(env, 'fixed_placement_for_ref') else None
        if fixed_pos is None:
            return False
        return (
            abs(float(placement[0]) - float(fixed_pos[0])) <= 1e-6
            and abs(float(placement[1]) - float(fixed_pos[1])) <= 1e-6
            and int(placement[2]) % 360 == int(fixed_pos[2]) % 360
        )
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
        role = str(getattr(env, '_external_io_roles', {}).get(ref, 'none') or 'none').lower()
        level = str(getattr(env.comp_by_ref.get(ref), 'constraint_level', 'soft') or 'soft').lower()
        if role == 'edge_hard' and level == 'hard' and band_sides:
            if not env._within_edge_band(bb, band_sides, ref=ref):
                return False
        elif env._hard_boundary_required(ref):
            if band_sides:
                if not env._touch_sides(bb, band_sides):
                    return False
            elif not env._touch_sides(bb, ['left', 'right', 'top', 'bottom']):
                return False
        elif band_sides:
            if not env._within_edge_band(bb, band_sides, ref=ref):
                return False
    return True


def _try_update(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]], ref: str, cand: Tuple[float, float, int], max_move_mm: float = 1.5) -> bool:
    if getattr(env, 'is_ref_fixed', lambda _r: False)(ref):
        return False
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


def _objective_accepts(
    before_obj: float,
    after_obj: float,
    *,
    allow_worse_abs: float = 1e-9,
    allow_worse_rel: float = 0.0,
) -> bool:
    """Return True when a deterministic polish move does not hurt the layout beyond a tiny budget.

    The objective already includes HPWL / weighted HPWL / NSLW and engineering priors.
    This guard lets alignment/pitch/orientation cleanup improve visual regularity without
    silently destroying the objective that the model was trained to optimize.
    """
    budget = max(float(allow_worse_abs), float(allow_worse_rel) * max(1.0, abs(float(before_obj))))
    return float(after_obj) <= float(before_obj) + budget + 1e-12


def _try_update_guarded(
    task,
    env_kwargs: Dict[str, Any],
    env: PlacementEnv,
    placed: Dict[str, Tuple[float, float, int]],
    ref: str,
    cand: Tuple[float, float, int],
    *,
    max_move_mm: float = 1.5,
    allow_worse_abs: float = 1e-9,
    allow_worse_rel: float = 0.0,
    protect_terms: Optional[Dict[str, float]] = None,
) -> bool:
    """Legal, bounded, objective-aware update used by deterministic postprocess polish.

    In addition to total objective, protect_terms can prevent local polish from
    hiding HPWL/NSLW regressions behind other terms.
    """
    if getattr(env, 'is_ref_fixed', lambda _r: False)(ref):
        return False
    old = placed.get(ref)
    if old is None:
        return False

    dx = float(cand[0] - old[0])
    dy = float(cand[1] - old[1])
    if (dx * dx + dy * dy) ** 0.5 > float(max_move_mm) + 1e-6:
        return False
    if not _is_placement_legal(env, ref, cand, placed):
        return False

    before_obj = _layout_objective(task, placed, env_kwargs)
    before_terms = _layout_guard_terms(task, placed, env_kwargs) if protect_terms else None
    trial = dict(placed)
    trial[ref] = cand
    after_obj = _layout_objective(task, trial, env_kwargs)
    if not _objective_accepts(
        before_obj,
        after_obj,
        allow_worse_abs=float(allow_worse_abs),
        allow_worse_rel=float(allow_worse_rel),
    ):
        return False
    if protect_terms and before_terms is not None:
        after_terms = _layout_guard_terms(task, trial, env_kwargs)
        if not _protected_terms_accept(before_terms, after_terms, protect_terms):
            return False

    placed[ref] = cand
    _refresh_env_state(env, placed)
    return True



def _layout_has_illegal(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> bool:
    _refresh_env_state(env, placed)
    for ref, p in placed.items():
        if not _is_placement_legal(env, ref, p, placed):
            return True
    return False


def _declutter_move_limit(env: PlacementEnv, ref: str) -> float:
    comp = env.comp_by_ref[ref]
    sem = env._semantic_class.get(ref, '')
    if env._is_large_component(ref) or env._anchor_refs.get(ref) or sem in {'core', 'interface', 'power', 'power_support'} or env._edge_band_sides(ref):
        return 3.6
    return 2.4


def _refresh_env_state(env: PlacementEnv, placed: Dict[str, Tuple[float, float, int]]) -> None:
    merged = _merge_fixed_placements(env, placed)
    env.placed = dict(merged)
    env.placed_order = _ordered_placed_refs(env, env.placed)
    env.occupied = [env._ref_bbox(ref, *env.placed[ref]) for ref in env.placed_order]
    env.t = _env_dynamic_placed_count(env, env.placed)
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
    frozen = set(getattr(env, 'fixed_refs', []) or [])
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
        + env.objective_cfg.module_floorplan_weight * env._module_floorplan_penalty_for_candidate(ref, bb)
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
    original = dict(placed)
    placed = dict(placed)
    xmin, ymin, xmax, ymax = task.bbox_mm

    _refresh_env_state(env, placed)
    frozen_anchors = _frozen_anchor_refs(env)
    edge_protect = {"hpwl_rel": 0.06, "w_hpwl_rel": 0.06, "nslw_drop": 1.0, "illegal_delta": 0.0}
    soft_protect = {"hpwl_rel": 0.035, "w_hpwl_rel": 0.035, "nslw_drop": 0.0, "illegal_delta": 0.0}
    strict_protect = {"hpwl_rel": 0.015, "w_hpwl_rel": 0.015, "nslw_drop": 0.0, "illegal_delta": 0.0}

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
            _try_update_guarded(
                task,
                env_kwargs,
                env,
                placed,
                r,
                cand,
                max_move_mm=1.8,
                allow_worse_abs=0.05,
                allow_worse_rel=0.0025,
                protect_terms=edge_protect,
            )

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
        _try_update_guarded(
            task,
            env_kwargs,
            env,
            placed,
            ref,
            cand,
            max_move_mm=1.8,
            allow_worse_abs=0.05,
            allow_worse_rel=0.0025,
            protect_terms=edge_protect,
        )

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
        _try_update_guarded(
            task,
            env_kwargs,
            env,
            placed,
            ref,
            (tx, ty, rot),
            max_move_mm=1.4,
            allow_worse_abs=0.02,
            allow_worse_rel=0.0010,
            protect_terms=soft_protect,
        )

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
        axis_votes = [env._align_axes.get(r, 'auto') for r in refs if env._align_axes.get(r, 'auto') in {'x', 'y'}]
        if axis_votes:
            align_axis = max(sorted(set(axis_votes)), key=axis_votes.count)
            use_x = align_axis == 'x'
        else:
            dx = sum(abs(v - mx) for v in xs)
            dy = sum(abs(v - my) for v in ys)
            use_x = dx <= dy
        for r in refs:
            if r in frozen_anchors:
                continue
            x, y, rot = placed[r]
            cand = (mx, y, rot) if use_x else (x, my, rot)
            _try_update_guarded(
                task,
                env_kwargs,
                env,
                placed,
                r,
                cand,
                max_move_mm=1.0,
                allow_worse_abs=1e-9,
                allow_worse_rel=0.0,
                protect_terms=strict_protect,
            )

    _refresh_env_state(env, placed)

    pitch_groups: Dict[str, list[str]] = {}
    for ref in env.refs:
        if ref not in placed:
            continue
        g = env._pitch_group_name(ref)
        if g:
            pitch_groups.setdefault(str(g), []).append(ref)
    for group_name, refs in pitch_groups.items():
        if len(refs) < 3:
            continue
        # Preserve means: keep ordering constraint but do not force equal spacing
        # unless the user explicitly provided pitch_group/row_group.
        if not str(group_name).startswith('pitch::') and any(env._spacing_policies.get(r, 'equal') == 'preserve' for r in refs):
            continue
        axis = env._pitch_axis_for_group(refs, {r: (placed[r][0], placed[r][1]) for r in refs if r in placed})
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
            _try_update_guarded(
                task,
                env_kwargs,
                env,
                placed,
                r,
                cand,
                max_move_mm=1.1,
                allow_worse_abs=1e-9,
                allow_worse_rel=0.0,
                protect_terms=strict_protect,
            )

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
            _try_update_guarded(
                task,
                env_kwargs,
                env,
                placed,
                r,
                (x, y, target_rot),
                max_move_mm=0.0,
                allow_worse_abs=1e-9,
                allow_worse_rel=0.0,
                protect_terms=strict_protect,
            )

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
                if trial_obj + 1e-9 < base_obj and _protected_terms_accept(
                    _layout_guard_terms(task, placed, env_kwargs),
                    _layout_guard_terms(task, trial, env_kwargs),
                    soft_protect,
                ):
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
                if trial_obj + 1e-9 < base_obj and _protected_terms_accept(
                    _layout_guard_terms(task, placed, env_kwargs),
                    _layout_guard_terms(task, trial, env_kwargs),
                    soft_protect,
                ):
                    placed = trial
                    base_obj = trial_obj
                    changed = True
                    _refresh_env_state(env, placed)
                    break
        if not changed:
            break

    _refresh_env_state(env, placed)
    if _layout_has_illegal(env, placed):
        return original

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
        lambda_region_prior=float(t.get("lambda_region_prior", t.get("lambda_region", 0.15))),
        lambda_prior_region_heatmap=float(t.get("lambda_prior_region_heatmap", 0.10)),
        lambda_conn_prior=float(t.get("lambda_conn_prior", 0.15)),
        lambda_anchor_prior=float(t.get("lambda_anchor_prior", 0.20)),
        lambda_module_prior=float(t.get("lambda_module_prior", 0.15)),
        lambda_spacing_prior=float(t.get("lambda_spacing_prior", 0.08)),
        lambda_edge_prior=float(t.get("lambda_edge_prior", 0.05)),
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
    """Restore canonical PlacementEnv kwargs from checkpoint env_config.

    Training checkpoints store env_config in the same canonical PlacementEnv
    kwarg format returned by build_placement_env_kwargs().  The builder also
    accepts older CLI-style aliases, so infer and train cannot drift.
    """
    e = ckpt.get("env_config", {}) or {}
    if not isinstance(e, dict):
        e = {}
    return build_placement_env_kwargs(e)


def _region_cfg_from_ckpt(ckpt: Dict[str, Any]) -> RegionPriorConfig:
    r = ckpt.get("region_prior", {}) or {}
    return RegionPriorConfig(
        enabled=bool(r.get("enabled", False)),
        grid_x=int(r.get("grid_x", 6)),
        grid_y=int(r.get("grid_y", 6)),
        heatmap_sigma_cells=float(r.get("heatmap_sigma_cells", 0.85)),
        legacy_zone_edge_ratio=float(r.get("legacy_zone_edge_ratio", r.get("zone_edge_ratio", 0.12))),
        legacy_zone_core_ratio=float(r.get("legacy_zone_core_ratio", r.get("zone_core_ratio", 0.28))),
        heatmap_action_prior_weight=float(r.get("heatmap_action_prior_weight", r.get("zone_prior_weight", 0.35))),
        aux_heatmap_weight=float(r.get("aux_heatmap_weight", 0.30)),
        aux_prior_consistency_weight=float(r.get("aux_prior_consistency_weight", 0.03)),
        aux_semantic_weight=float(r.get("aux_semantic_weight", r.get("aux_zone_weight", 0.10))),
        aux_side_weight=float(r.get("aux_side_weight", 0.12)),
        aux_subzone_weight=float(r.get("aux_subzone_weight", 0.10)),
        aux_pairwise_weight=float(r.get("aux_pairwise_weight", 0.10)),
    )


_INFERENCE_STRIPPED_CHECKPOINT_KEYS = (
    "model_state",
    "optimizer_state",
    "replay_buffer",
)


def _strip_inference_checkpoint_payload(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Return inference metadata without training-only tensor payloads."""
    metadata = dict(checkpoint)
    for key in _INFERENCE_STRIPPED_CHECKPOINT_KEYS:
        metadata.pop(key, None)
    return metadata




_INFERENCE_MODEL_CACHE: Dict[Tuple[str, int, str, bool], Tuple[MaskedPolicy, Dict[str, Any], TeacherConfig, RegionPriorConfig]] = {}

def clear_inference_model_cache() -> None:
    """Release cached inference model bundles. Useful for long-running services."""
    _INFERENCE_MODEL_CACHE.clear()

def load_model_cached(
    ckpt_path: str,
    device: str = "cuda",
    *,
    strict_state_dict: bool = True,
) -> Tuple[MaskedPolicy, Dict[str, Any], TeacherConfig, RegionPriorConfig]:
    """Load an inference bundle once per process/checkpoint/device.

    ``scripts/step4_infer.py`` calls ``infer_layout`` once per board. Without
    this cache every board re-reads the checkpoint, rebuilds the transformer, and
    transfers weights to the GPU, which dominates runtime for batch inference.
    The key includes checkpoint mtime so replacing the file automatically reloads.
    """
    abs_path = os.path.abspath(str(ckpt_path))
    try:
        mtime_ns = int(os.stat(abs_path).st_mtime_ns)
    except OSError:
        # Let load_model raise the authoritative error.
        mtime_ns = -1
    key = (abs_path, mtime_ns, str(torch.device(device)), bool(strict_state_dict))
    cached = _INFERENCE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    bundle = load_model(abs_path, device=device, strict_state_dict=bool(strict_state_dict))
    # Avoid unbounded GPU memory growth if a caller cycles through many checkpoints.
    if len(_INFERENCE_MODEL_CACHE) >= 2:
        _INFERENCE_MODEL_CACHE.clear()
    _INFERENCE_MODEL_CACHE[key] = bundle
    return bundle

def load_model(
    ckpt_path: str,
    device: str = "cpu",
    *,
    strict_state_dict: bool = True,
) -> Tuple[MaskedPolicy, Dict[str, Any], TeacherConfig, RegionPriorConfig]:
    # Always deserialize on CPU. Loading a training checkpoint directly onto
    # CUDA would also move optimizer/replay tensors that inference never uses.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Checkpoint must contain a dict payload: {ckpt_path}"
        )
    if "model_state" not in ckpt:
        raise ValueError(
            f"Checkpoint is missing model_state: {ckpt_path}"
        )

    model_state = ckpt["model_state"]
    cfg_dict = ckpt.get("model_cfg", {}) or {}
    cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else ModelConfig()
    action_feat_dim = int(ckpt.get("action_feat_dim", 4))
    region_grid_shape = tuple(ckpt.get("region_grid_shape", [6, 6]))
    num_region_heatmap_bins = int(
        ckpt.get(
            "num_region_heatmap_bins",
            ckpt.get(
                "num_region_types",
                int(region_grid_shape[0]) * int(region_grid_shape[1]),
            ),
        )
    )
    num_semantic_classes = int(
        ckpt.get("num_semantic_classes", len(SEMANTIC_CLASS_NAMES))
    )
    num_side_preferences = int(
        ckpt.get("num_side_preferences", len(SIDE_PREFERENCE_NAMES))
    )
    num_subzones = int(
        ckpt.get("num_subzones", len(SUBZONE_NAMES))
    )
    num_pairwise_relations = int(
        ckpt.get("num_pairwise_relations", len(PAIRWISE_RELATION_NAMES))
    )

    model = MaskedPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        cfg=cfg,
        action_feat_dim=action_feat_dim,
        region_grid_shape=(
            int(region_grid_shape[0]),
            int(region_grid_shape[1]),
        ),
        num_region_heatmap_bins=int(num_region_heatmap_bins),
        num_semantic_classes=int(num_semantic_classes),
        num_side_preferences=int(num_side_preferences),
        num_subzones=int(num_subzones),
        num_pairwise_relations=int(num_pairwise_relations),
    )
    if bool(strict_state_dict):
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint model_state is incompatible with the current MaskedPolicy. "
                "Refusing to load partial weights because strict_state_dict=True. "
                "Use strict_state_dict=False only for explicit legacy migration."
            ) from exc
    else:
        incompatible = model.load_state_dict(model_state, strict=False)
        missing = list(
            getattr(incompatible, "missing_keys", []) or []
        )
        unexpected = list(
            getattr(incompatible, "unexpected_keys", []) or []
        )
        if missing or unexpected:
            print(
                "[infer] non-strict checkpoint load: "
                f"missing_keys={missing} unexpected_keys={unexpected}"
            )

    # Build all inference configuration while the full CPU checkpoint is still
    # available, then return only lightweight metadata.
    teacher = _teacher_from_ckpt(ckpt)
    region_cfg = _region_cfg_from_ckpt(ckpt)
    metadata = _strip_inference_checkpoint_payload(ckpt)

    # Drop the CPU checkpoint weight reference before moving the model to the
    # target device, preventing a second long-lived copy of model_state.
    del model_state
    del ckpt

    model.to(torch.device(device))
    model.eval()
    return model, metadata, teacher, region_cfg




def _apply_edge_hard_candidate_mask(
    env: PlacementEnv,
    ref: str,
    score: torch.Tensor,
    w: int,
    h: int,
    R: int,
    device: torch.device,
) -> torch.Tensor:
    """Approximate hard mask for true external interfaces during inference.

    ``edge_hard`` components are first limited to the edge band implied by their
    ``side_preference``.  If that band has no legal candidates, the original
    legal board-wide candidates are preserved as a fallback so inference does not
    fail merely because the preferred edge is temporarily blocked.
    """
    role = str(getattr(env, '_external_io_roles', {}).get(ref, 'none') or 'none').lower()
    level = str(getattr(env.comp_by_ref.get(ref), 'constraint_level', 'soft') or 'soft').lower()
    if role != 'edge_hard' or level != 'hard':
        return score
    side = str(getattr(env, '_side_preferences', {}).get(ref, 'free') or 'free').lower()

    xmin, ymin, xmax, ymax = [float(v) for v in env.task.bbox_mm]
    grid = float(env.task.grid_mm)
    band = float(env._component_edge_band_width_mm(ref) if hasattr(env, '_component_edge_band_width_mm') else max(grid, 2.5))
    # Reuse the per-env CUDA grid cache instead of allocating arange/meshgrid for
    # every component step. The edge hard mask is small, but infer calls it many
    # times and batch inference otherwise burns time on repeated allocations.
    maps = _cached_grid_maps_cuda(env, device)
    X = maps["x"]
    Y = maps["y"]
    if side == 'edge_left':
        band2 = X <= xmin + band
    elif side == 'edge_right':
        band2 = X >= xmax - band
    elif side == 'edge_bottom':
        band2 = Y <= ymin + band
    elif side == 'edge_top':
        band2 = Y >= ymax - band
    else:
        # No safe side_preference is available.  Keep a mechanical edge prior by
        # allowing any edge band, still with full-board fallback below.
        band2 = (X <= xmin + band) | (X >= xmax - band) | (Y <= ymin + band) | (Y >= ymax - band)
    band_flat = band2.unsqueeze(0).expand(R, -1, -1).reshape(-1)
    legal = torch.isfinite(score) & (score > -1e8)
    legal_in_band = legal & band_flat
    if bool(legal_in_band.any().item()):
        masked = score.clone()
        masked[legal & (~band_flat)] = -1e9
        return masked
    return score


def _infer_one_step_scores(
    model: MaskedPolicy,
    env: PlacementEnv,
    ref: str,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    device_t: torch.device,
    *,
    infer_objective_alpha: float,
    infer_region_alpha: float,
    max_tokens: int = 128,
) -> Tuple[torch.Tensor, int, int, int]:
    w, h = env.grid_shape()
    R = len(env.rotations)
    mask_map_t, bias_map_t = action_mask_and_bias_cuda(env, ref, device_t)
    mask_t = mask_map_t.reshape(-1)
    bias_t = bias_map_t.reshape(-1)
    if not bool((mask_t > 0.5).any().item()):
        return torch.full((R * w * h,), -1e9, device=device_t), w, h, R

    obj_maps = objective_delta_mask_cuda(env, ref, device_t)
    objective_residual_t = obj_maps.get("residual_total", obj_maps["total"]).reshape(-1)
    objective_total_t = obj_maps["total"].reshape(-1)
    feat_t = action_features_cuda(env, device_t, ref=ref)
    action_prior_t = action_prior_total_cuda(
        env, ref, device_t, legal_mask_flat=mask_t,
        lambda_conn=float(teacher.lambda_conn_prior),
        lambda_anchor=float(teacher.lambda_anchor_prior),
        lambda_module=float(teacher.lambda_module_prior),
        lambda_spacing=float(teacher.lambda_spacing_prior),
        lambda_edge=float(teacher.lambda_edge_prior),
        lambda_prior_region_heatmap=float(getattr(teacher, 'lambda_prior_region_heatmap', 0.10)),
    )
    tokens_t = build_context_tokens_cuda(env, ref, device_t, max_tokens=max_tokens)[None, :, :]

    with torch.no_grad():
        logits, _region_heatmap_logits, _semantic_class_logits, _side_logits, _subzone_logits, _pairwise_logits, region_prior, _enc_ctx, _token_refs = policy_outputs_with_region(
            model, env, ref, tokens_t, feat_t, region_cfg
        )
        score, _legal = score_actions(
            logits,
            bias_t,
            mask_t,
            objective_residual_t,
            objective_total_t,
            region_prior,
            action_prior_t,
            teacher=teacher,
            objective_alpha=float(infer_objective_alpha),
            region_alpha=float(infer_region_alpha),
        )
        score = _apply_edge_hard_candidate_mask(env, ref, score, w, h, R, device_t)
    return score, w, h, R


def _rollout_greedy_objective_aware(
    model: MaskedPolicy,
    env: PlacementEnv,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    device_t: torch.device,
    *,
    infer_objective_alpha: float,
    infer_region_alpha: float,
    max_tokens: int = 128,
) -> PlacementEnv:
    while not env.done():
        ref = env.current_ref()
        if ref is None:
            break
        score, w, h, R = _infer_one_step_scores(
            model,
            env,
            ref,
            teacher,
            region_cfg,
            device_t,
            infer_objective_alpha=float(infer_objective_alpha),
            infer_region_alpha=float(infer_region_alpha),
            max_tokens=max_tokens,
        )
        if not bool(torch.isfinite(score).any().item()) or float(torch.max(score).item()) <= -1e8:
            env.terminated = True
            break
        a = int(torch.argmax(score).item())
        _obs2, _r, _done, info = env.step(
            unflatten_action(a, w, h),
            assume_legal=True,
            return_observation=False,
            compute_objective=False,
        )
        if info.get("illegal"):
            env.terminated = True
            break
    return env


def _beam_env_is_complete(env: PlacementEnv) -> bool:
    """Return True only for a non-terminated, fully placed beam state."""
    expected_dynamic = int(len(env.sequence))
    placed_dynamic = _env_dynamic_placed_count(env, env.placed)
    return bool(
        not env.terminated
        and placed_dynamic == expected_dynamic
        and int(env.t) >= expected_dynamic
    )


def _select_final_beam(
    completed_beams: List[Tuple[float, PlacementEnv]],
    failed_beams: List[Tuple[float, PlacementEnv]],
    env0: PlacementEnv,
    env_kwargs: Dict[str, Any],
) -> PlacementEnv:
    """Select a beam without allowing a low-objective partial layout to win.

    Complete candidates always take precedence. If no complete candidate exists,
    the fallback favors the most-complete partial state before objective/score.
    """
    if completed_beams:
        def complete_key(item: Tuple[float, PlacementEnv]) -> Tuple[float, float]:
            beam_score, env = item
            objective = _layout_objective(
                env.task,
                dict(env.placed),
                env_kwargs,
            )
            return float(objective), -float(beam_score)

        selected = min(completed_beams, key=complete_key)[1]
        selected.beam_search_success = True
        selected.beam_failure_reason = None
        selected.beam_completed_candidate_count = int(len(completed_beams))
        selected.beam_failed_candidate_count = int(len(failed_beams))
        return selected

    candidates = list(failed_beams)
    if not candidates:
        candidates = [(float("-inf"), env0)]

    def partial_key(
        item: Tuple[float, PlacementEnv],
    ) -> Tuple[int, float, float]:
        beam_score, env = item
        placed_count = _env_dynamic_placed_count(env, env.placed)
        objective = _layout_objective(
            env.task,
            dict(env.placed),
            env_kwargs,
        )
        return -placed_count, float(objective), -float(beam_score)

    selected = min(candidates, key=partial_key)[1]
    selected.terminated = True
    selected.beam_search_success = False
    selected.beam_failure_reason = str(
        getattr(selected, "beam_failure_reason", "")
        or "no_complete_beam"
    )
    selected.beam_completed_candidate_count = 0
    selected.beam_failed_candidate_count = int(len(failed_beams))
    return selected


def _rollout_beam_objective_aware(
    model: MaskedPolicy,
    env0: PlacementEnv,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    device_t: torch.device,
    *,
    beam_width: int,
    beam_topk: int,
    infer_objective_alpha: float,
    infer_region_alpha: float,
    env_kwargs: Dict[str, Any],
    max_tokens: int = 128,
) -> PlacementEnv:
    """Small-width beam search over the same objective-aware action score.

    The search keeps only CUDA scoring in the inner loop.  Env copies are used as
    state containers and are stepped with compute_objective=False.
    """
    beams: List[Tuple[float, PlacementEnv]] = [(0.0, env0)]
    completed_beams: List[Tuple[float, PlacementEnv]] = []
    failed_beams: List[Tuple[float, PlacementEnv]] = []
    width = max(1, int(beam_width))
    topk = max(1, int(beam_topk))

    while beams:
        new_beams: List[Tuple[float, PlacementEnv]] = []
        for beam_score, env in beams:
            if _beam_env_is_complete(env):
                completed_beams.append((beam_score, env))
                continue
            if env.terminated or env.done():
                env.terminated = True
                env.beam_failure_reason = str(
                    getattr(env, "beam_failure_reason", "")
                    or "terminated_before_complete"
                )
                failed_beams.append((beam_score, env))
                continue

            ref = env.current_ref()
            if ref is None:
                env.terminated = True
                env.beam_failure_reason = "missing_current_ref"
                failed_beams.append((beam_score, env))
                continue

            score, w, h, R = _infer_one_step_scores(
                model,
                env,
                ref,
                teacher,
                region_cfg,
                device_t,
                infer_objective_alpha=float(infer_objective_alpha),
                infer_region_alpha=float(infer_region_alpha),
                max_tokens=max_tokens,
            )
            finite = torch.isfinite(score) & (score > -1e8)
            if not bool(finite.any().item()):
                env_bad = copy.deepcopy(env)
                env_bad.terminated = True
                env_bad.beam_failure_reason = "no_finite_legal_action"
                failed_beams.append((beam_score - 1e6, env_bad))
                continue

            k = min(topk, int(finite.sum().item()))
            vals, idxs = torch.topk(score, k=k)
            for val, idx in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
                child = copy.deepcopy(env)
                _obs2, _r, _done, info = child.step(
                    unflatten_action(int(idx), w, h),
                    assume_legal=True,
                    return_observation=False,
                    compute_objective=False,
                )
                if info.get("illegal"):
                    child.terminated = True
                    child.beam_failure_reason = str(
                        info.get("reason") or "illegal_action"
                    )
                    child_score = beam_score - 1e6
                    failed_beams.append((child_score, child))
                    continue

                child_score = beam_score + float(val)
                if _beam_env_is_complete(child):
                    completed_beams.append((child_score, child))
                elif child.terminated or child.done():
                    child.terminated = True
                    child.beam_failure_reason = str(
                        getattr(child, "beam_failure_reason", "")
                        or "terminated_before_complete"
                    )
                    failed_beams.append((child_score, child))
                else:
                    new_beams.append((child_score, child))

        # Keep active, unfinished beams by policy/objective-aware score.
        # Completed candidates are retained separately and can never be pruned.
        new_beams.sort(key=lambda item: item[0], reverse=True)
        beams = new_beams[:width]

    return _select_final_beam(
        completed_beams,
        failed_beams,
        env0,
        env_kwargs,
    )


def _metrics_for_layout(task_json_path: str, placed: Dict[str, Tuple[float, float, int]]) -> Dict[str, Any]:
    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = drop_mounting_holes_from_task_json(json.load(f))
        reference = {}
        for comp in task_data.get("components", []):
            ex = comp.get("expert") or {}
            if "xy_mm" in ex:
                xy = ex["xy_mm"]
                reference[str(comp["ref"])] = (float(xy[0]), float(xy[1]), int(round(float(ex.get("rot", 0)))) % 360)
        return evaluate_task_semantic_metrics(task_data, placed, reference_placed=reference)
    except Exception as exc:
        return {"metrics_error": str(exc)}

def _resolve_inference_sequence_policy(
    metadata: Dict[str, Any],
    requested_policy: Optional[str],
) -> Tuple[str, str]:
    """Resolve inference sequence policy and report where it came from.

    ``None``, an empty string, or ``"checkpoint"`` means: use the policy saved
    by training. Legacy checkpoints without the field fall back to ``rebuild``.
    Explicit ``rebuild``, ``stored``, or ``validate`` values override metadata.
    """
    requested = (
        None
        if requested_policy is None
        else str(requested_policy).strip().lower()
    )
    if requested in (None, "", "checkpoint"):
        saved = metadata.get("sequence_policy", "rebuild")
        effective = normalize_sequence_policy(
            "rebuild" if saved is None else str(saved)
        )
        source = (
            "checkpoint"
            if metadata.get("sequence_policy") is not None
            else "legacy_checkpoint_default"
        )
        return effective, source

    effective = normalize_sequence_policy(requested)
    return effective, "explicit_override"


def _resolve_inference_action_scoring(
    metadata: Dict[str, Any],
    requested_objective_alpha: Optional[float],
    requested_region_alpha: Optional[float],
) -> Tuple[float, float, Dict[str, str]]:
    """Resolve objective-aware inference scoring weights.

    ``None`` means restore the training-time values saved under checkpoint
    ``action_scoring`` metadata. Explicit numeric arguments override the
    checkpoint. Legacy checkpoints without metadata fall back to current
    runtime defaults.
    """
    raw_action_scoring = metadata.get("action_scoring")
    action_scoring = raw_action_scoring if isinstance(raw_action_scoring, dict) else {}

    def resolve_one(
        key: str,
        requested: Optional[float],
        fallback: float,
    ) -> Tuple[float, str]:
        if requested is not None:
            return float(requested), "explicit_override"
        saved = action_scoring.get(key)
        if saved is not None:
            return float(saved), "checkpoint"
        return float(fallback), "legacy_checkpoint_default"

    objective_alpha, objective_source = resolve_one(
        "objective_alpha",
        requested_objective_alpha,
        DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    )
    region_alpha, region_source = resolve_one(
        "region_alpha",
        requested_region_alpha,
        DEFAULT_ROLLOUT_REGION_ALPHA,
    )
    return objective_alpha, region_alpha, {
        "objective_alpha": objective_source,
        "region_alpha": region_source,
    }


def infer_layout(
    task_json_path: str,
    ckpt_path: str,
    device: str = "cuda",
    *,
    infer_objective_alpha: Optional[float] = None,
    infer_region_alpha: Optional[float] = None,
    beam_width: int = 1,
    beam_topk: int = 16,
    return_metrics: bool = True,
    max_tokens: Optional[int] = None,
    strict_state_dict: bool = True,
    postprocess: bool = True,
    layout_preset: str = "checkpoint",
    objective_overrides: Optional[Dict[str, Any]] = None,
    sequence_policy: Optional[str] = None,
    fixed_refs: Optional[Iterable[str] | str] = None,
) -> Dict[str, Any]:
    device_t = torch.device(device)
    if device_t.type != "cuda":
        raise ValueError(f"infer_layout() is CUDA-only; got device={device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("infer_layout() requires CUDA, but torch.cuda.is_available() is False.")

    model, meta, teacher, region_cfg = load_model_cached(
        ckpt_path,
        device=device,
        strict_state_dict=bool(strict_state_dict),
    )
    effective_sequence_policy, sequence_policy_source = (
        _resolve_inference_sequence_policy(
            meta,
            sequence_policy,
        )
    )
    infer_objective_alpha, infer_region_alpha, action_scoring_resolution = (
        _resolve_inference_action_scoring(
            meta,
            infer_objective_alpha,
            infer_region_alpha,
        )
    )
    runtime_task_json_path, _runtime_temp_task_json_path, runtime_fixed_ref_meta = materialize_fixed_refs_task_json(
        task_json_path,
        fixed_refs=fixed_refs,
    )
    try:
        task = task_from_json(
            runtime_task_json_path,
            sequence_policy=effective_sequence_policy,
            load_expert=False,
        )
    finally:
        if _runtime_temp_task_json_path:
            try:
                os.unlink(_runtime_temp_task_json_path)
            except FileNotFoundError:
                pass
    # Defense in depth: inference must be invariant to expert.xy_mm / expert.rot
    # even when callers pass raw training JSON by mistake.
    task = strip_expert_fields_from_task(task)
    env_kwargs = _env_kwargs_from_ckpt(meta)
    env_kwargs = apply_layout_objective_preset(env_kwargs, layout_preset, objective_overrides)
    max_tokens_eff = int(meta.get("max_tokens", 128) if max_tokens is None else max_tokens)
    env = PlacementEnv(task, **env_kwargs)

    if int(beam_width) > 1:
        env = _rollout_beam_objective_aware(
            model,
            env,
            teacher,
            region_cfg,
            device_t,
            beam_width=int(beam_width),
            beam_topk=int(beam_topk),
            infer_objective_alpha=float(infer_objective_alpha),
            infer_region_alpha=float(infer_region_alpha),
            env_kwargs=env_kwargs,
            max_tokens=max_tokens_eff,
        )
    else:
        env = _rollout_greedy_objective_aware(
            model,
            env,
            teacher,
            region_cfg,
            device_t,
            infer_objective_alpha=float(infer_objective_alpha),
            infer_region_alpha=float(infer_region_alpha),
            max_tokens=max_tokens_eff,
        )

    raw_placed = dict(env.placed)
    expected_dynamic_count = int(len(env.sequence))
    expected_count = _env_expected_total_count(env)
    fixed_refs = sorted(list(getattr(env, 'fixed_refs', []) or []))
    fixed_count = int(len(fixed_refs))
    raw_placed_count = int(len(raw_placed))
    raw_dynamic_placed_count = _env_dynamic_placed_count(env, raw_placed)
    raw_complete = bool(
        not bool(env.terminated)
        and raw_dynamic_placed_count == expected_dynamic_count
        and raw_placed_count == expected_count
        and int(getattr(env, "t", raw_dynamic_placed_count)) >= expected_dynamic_count
    )
    raw_obj_partial = _layout_objective(task, raw_placed, env_kwargs) if raw_placed_count > 0 else INCOMPLETE_OBJECTIVE_PENALTY
    if bool(postprocess):
        final_placed = _postprocess_layout(task, raw_placed, env_kwargs)
    else:
        final_placed = dict(raw_placed)
    placed_count = int(len(final_placed))
    final_dynamic_placed_count = _env_dynamic_placed_count(env, final_placed)
    complete = bool(
        raw_complete
        and placed_count == expected_count
        and final_dynamic_placed_count == expected_dynamic_count
    )
    final_obj_partial = _layout_objective(task, final_placed, env_kwargs) if placed_count > 0 else INCOMPLETE_OBJECTIVE_PENALTY
    raw_obj = float(raw_obj_partial) if raw_complete else INCOMPLETE_OBJECTIVE_PENALTY
    final_obj = float(final_obj_partial) if complete else INCOMPLETE_OBJECTIVE_PENALTY
    postprocess_delta = float(final_obj - raw_obj) if bool(complete and raw_complete) else 0.0
    failure_reason = None
    if not complete:
        if bool(env.terminated):
            failure_reason = str(
                getattr(env, "beam_failure_reason", "")
                or "terminated_before_complete"
            )
        elif placed_count != expected_count or final_dynamic_placed_count != expected_dynamic_count:
            failure_reason = "incomplete_layout"
        else:
            failure_reason = "invalid_incomplete_layout"

    changed_refs = sorted([ref for ref in final_placed if ref not in set(fixed_refs) and final_placed.get(ref) != raw_placed.get(ref)])

    result: Dict[str, Any] = {
        "placed": final_placed,
        "placed_raw": raw_placed,
        "objective": float(final_obj),
        "objective_raw": float(raw_obj),
        "objective_partial": float(final_obj_partial),
        "objective_partial_raw": float(raw_obj_partial),
        "objective_valid": bool(complete),
        "objective_raw_valid": bool(raw_complete),
        "postprocess_applied": bool(postprocess),
        "postprocess_objective_delta": float(postprocess_delta),
        "postprocess_objective_delta_valid": bool(complete and raw_complete),
        "postprocess_changed_count": int(len(changed_refs)),
        "postprocess_changed_refs": changed_refs,
        "complete": bool(complete),
        "failure_reason": failure_reason,
        "placed_count": int(placed_count),
        "raw_placed_count": int(raw_placed_count),
        "expected_count": int(expected_count),
        "dynamic_expected_count": int(expected_dynamic_count),
        "dynamic_placed_count": int(final_dynamic_placed_count),
        "raw_dynamic_placed_count": int(raw_dynamic_placed_count),
        "fixed_count": int(fixed_count),
        "fixed_refs": fixed_refs,
        "runtime_fixed_refs_requested": list(runtime_fixed_ref_meta.get("requested", [])),
        "runtime_fixed_refs_matched": list(runtime_fixed_ref_meta.get("matched", [])),
        "runtime_fixed_refs_unmatched": list(runtime_fixed_ref_meta.get("unmatched", [])),
        "layout_preset": str(layout_preset or "checkpoint"),
        "objective_overrides": dict(objective_overrides or {}),
        "sequence_policy": str(effective_sequence_policy),
        "sequence_policy_requested": (
            None if sequence_policy is None else str(sequence_policy)
        ),
        "sequence_policy_resolution": str(sequence_policy_source),
        "expert_fields_loaded": False,
        "expert_sanitized": True,
        "sequence_source": str(
            getattr(
                task,
                "sequence_source",
                effective_sequence_policy,
            )
        ),
        "terminated": bool(env.terminated or not complete),
        "infer_mode": "beam" if int(beam_width) > 1 else "objective_aware_greedy",
        "action_scoring_version": ACTION_SCORING_VERSION,
        "action_scoring_checkpoint_version": (
            meta.get("action_scoring", {}).get("version")
            if isinstance(meta.get("action_scoring"), dict)
            else None
        ),
        "action_scoring_resolution": dict(action_scoring_resolution),
        "beam_search_success": (
            bool(getattr(env, "beam_search_success", False))
            if int(beam_width) > 1
            else None
        ),
        "beam_failure_reason": (
            getattr(env, "beam_failure_reason", None)
            if int(beam_width) > 1
            else None
        ),
        "beam_completed_candidate_count": (
            int(getattr(env, "beam_completed_candidate_count", 0))
            if int(beam_width) > 1
            else None
        ),
        "beam_failed_candidate_count": (
            int(getattr(env, "beam_failed_candidate_count", 0))
            if int(beam_width) > 1
            else None
        ),
        "infer_objective_alpha": float(infer_objective_alpha),
        "infer_region_alpha": float(infer_region_alpha),
        "beam_width": int(beam_width),
        "beam_topk": int(beam_topk),
        "max_tokens": int(max_tokens_eff),
    }
    if bool(return_metrics):
        if bool(complete):
            result["metrics"] = _metrics_for_layout(task_json_path, final_placed)
        else:
            result["metrics"] = {
                "metrics_skipped": True,
                "reason": str(failure_reason or "incomplete_layout"),
            }
        if bool(raw_complete):
            result["metrics_raw"] = _metrics_for_layout(task_json_path, raw_placed)
        else:
            result["metrics_raw"] = {
                "metrics_skipped": True,
                "reason": str(failure_reason or "incomplete_layout"),
            }
    return result
