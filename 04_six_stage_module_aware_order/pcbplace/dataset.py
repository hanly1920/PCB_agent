from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .utils import load_json, drop_mounting_holes_from_task_json
from .env import Component, Task
from .module_partition import apply_modules_to_task
from .json_schema import validate_task_json


SEQUENCE_POLICIES: Tuple[str, ...] = ("rebuild", "stored", "validate")


def normalize_sequence_policy(sequence_policy: str) -> str:
    policy = str(sequence_policy or "rebuild").strip().lower()
    if policy not in SEQUENCE_POLICIES:
        raise ValueError(
            f"Unknown sequence_policy={sequence_policy!r}; "
            f"choose one of {list(SEQUENCE_POLICIES)}"
        )
    return policy


def _stored_graph_sequence(data: Dict[str, Any]) -> Optional[List[str]]:
    graph = data.get("graph")
    if not isinstance(graph, dict) or graph.get("sequence") is None:
        return None
    raw = graph.get("sequence")
    if not isinstance(raw, list):
        raise ValueError("Stored graph.sequence must be a JSON array of component refs.")
    return [str(ref) for ref in raw if str(ref or "").strip()]


def _validate_complete_sequence(
    sequence: List[str],
    component_refs: List[str],
    *,
    source: str,
) -> List[str]:
    refs = [str(ref) for ref in sequence]
    expected = [str(ref) for ref in component_refs]
    duplicate_refs = sorted({ref for ref in refs if refs.count(ref) > 1})
    missing = sorted(set(expected) - set(refs))
    unknown = sorted(set(refs) - set(expected))
    if duplicate_refs or missing or unknown or len(refs) != len(expected):
        raise ValueError(
            f"{source} must be a complete, duplicate-free permutation of components; "
            f"duplicates={duplicate_refs}, missing={missing}, unknown={unknown}, "
            f"stored_len={len(refs)}, component_len={len(expected)}"
        )
    return refs


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _json_functional_group(c: Dict[str, Any]) -> str:
    fg = _clean_text(c.get("functional_group") or (c.get("semantic") or {}).get("functional_group"))
    mid = _clean_text(c.get("module_id") or (c.get("module") or {}).get("module_id"))
    if fg.lower().startswith('module:'):
        return ''
    if mid and fg.lower() in {'', 'misc', 'other', 'free', 'none', 'null'}:
        return ''
    return fg or 'misc'


def _json_anchor_ref(c: Dict[str, Any]) -> Any:
    ref = _clean_text(c.get("ref"))
    aref = _clean_text(c.get("anchor_ref") or (c.get("semantic") or {}).get("anchor_ref"))
    mref = _clean_text(c.get("module_anchor_ref") or (c.get("module") or {}).get("anchor_ref"))
    if not aref or aref == ref or aref == mref:
        return None
    return aref


def _json_critical_neighbor_specs(c: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    vals = c.get("critical_neighbors") or (c.get("semantic") or {}).get("critical_neighbors") or []
    ref = _clean_text(c.get("ref"))
    aref = _json_anchor_ref(c) or ''
    mref = _clean_text(c.get("module_anchor_ref") or (c.get("module") or {}).get("anchor_ref"))
    excluded = {v for v in (ref, aref, mref) if v}
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, v in enumerate(vals):
        if isinstance(v, dict):
            vv = _clean_text(v.get("ref") or v.get("neighbor") or v.get("neighbor_ref"))
            if not vv or vv in excluded or vv in seen:
                continue
            item: Dict[str, Any] = {
                "ref": vv,
                "weight": float(v.get("weight", max(0.35, 1.0 - 0.08 * idx)) or max(0.35, 1.0 - 0.08 * idx)),
                "reason": _clean_text(v.get("reason") or v.get("type") or v.get("relation") or "explicit").lower() or "explicit",
            }
            if v.get("max_distance_mm", v.get("max_dist_mm", None)) not in (None, ""):
                item["max_distance_mm"] = float(v.get("max_distance_mm", v.get("max_dist_mm")))
            if v.get("preferred_subzone", v.get("subzone", None)) not in (None, ""):
                item["preferred_subzone"] = _clean_text(v.get("preferred_subzone", v.get("subzone")))
            out.append(item)
            seen.add(vv)
        else:
            vv = _clean_text(v)
            if vv and vv not in excluded and vv not in seen:
                out.append({
                    "ref": vv,
                    "weight": float(max(0.35, 1.0 - 0.08 * idx)),
                    "reason": "explicit",
                })
                seen.add(vv)
    return tuple(out)


def _json_critical_neighbors(c: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(item["ref"] for item in _json_critical_neighbor_specs(c))


def _json_nested_value(c: Dict[str, Any], key: str, default: Any = None) -> Any:
    if c.get(key, None) not in (None, ""):
        return c.get(key)
    semantic = c.get("semantic") if isinstance(c.get("semantic"), dict) else {}
    if semantic.get(key, None) not in (None, ""):
        return semantic.get(key)
    layout = c.get("layout") if isinstance(c.get("layout"), dict) else {}
    if layout.get(key, None) not in (None, ""):
        return layout.get(key)
    return default


def _json_float(c: Dict[str, Any], key: str, default: float) -> float:
    value = _json_nested_value(c, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _json_optional_float(c: Dict[str, Any], key: str) -> Any:
    value = _json_nested_value(c, key, None)
    if value in (None, ""):
        return None
    return float(value)


def _json_optional_int(c: Dict[str, Any], key: str) -> Any:
    value = _json_nested_value(c, key, None)
    if value in (None, ""):
        return None
    return int(value)


def _bbox_from_prior_component(c: Dict[str, Any]) -> Any:
    prior = c.get("prior") if isinstance(c.get("prior"), dict) else {}
    module = c.get("module") if isinstance(c.get("module"), dict) else {}
    module_prior = module.get("prior_region") if isinstance(module.get("prior_region"), dict) else {}
    return (
        prior.get("region_bbox_mm")
        or module_prior.get("bbox_mm")
        or module.get("region_bbox_mm")
    )


def _prior_conf_from_component(c: Dict[str, Any]) -> float:
    prior = c.get("prior") if isinstance(c.get("prior"), dict) else {}
    module = c.get("module") if isinstance(c.get("module"), dict) else {}
    module_prior = module.get("prior_region") if isinstance(module.get("prior_region"), dict) else {}
    val = (
        prior.get("region_confidence")
        if prior.get("region_confidence", None) not in (None, "") else
        module_prior.get("confidence")
        if module_prior.get("confidence", None) not in (None, "") else
        module.get("region_confidence", 1.0)
    )
    return float(val if val not in (None, "") else 1.0)


def _expert_module_bbox_from_component(c: Dict[str, Any]) -> Any:
    expert = c.get("expert") if isinstance(c.get("expert"), dict) else {}
    return expert.get("module_region_bbox_mm") or c.get("expert_module_region_bbox")

def task_from_json(path: str, *, sequence_policy: str = "rebuild", load_expert: bool = False) -> Task:
    """Load one task and resolve its placement sequence explicitly.

    ``rebuild`` ignores ``graph.sequence`` and computes the canonical
    six-phase module/semantic sequence. ``stored`` uses a validated
    ``graph.sequence`` exactly and does not reorder it. ``validate``
    rebuilds canonically and requires the stored sequence to match.

    ``load_expert`` is deliberately False by default. Runtime callers
    (training rollouts, replay and inference) must never load expert.xy_mm,
    expert.rot or expert module labels into Task/PlacementEnv. Offline
    structure-generation code may set it to True only while producing
    train-only supervision artifacts; PlacementEnv rejects such tasks.
    """
    data = drop_mounting_holes_from_task_json(load_json(path))
    validate_task_json(data, source=path)
    load_expert = bool(load_expert)
    board = data["board"]
    bbox = tuple(board["bbox_mm"])
    grid = float(board.get("grid_mm", 1.0))
    comps: List[Component] = []
    for c in data["components"]:
        pads = [(p["net"], tuple(p["rel_mm"])) for p in c.get("pads", [])]
        review = c.get("semantic_review") or {}
        comps.append(Component(
            ref=c["ref"],
            type=c.get("type","misc"),
            size_mm=tuple(c.get("size_mm",[1.0,1.0])),
            pads=pads,
            allowed_sides=c.get("allowed_sides", []) or [],
            must_touch_boundary=(
                c.get("must_touch_boundary")
                if c.get("must_touch_boundary", None) is not None
                else (c.get("semantic") or {}).get("must_touch_boundary")
            ),
            semantic_class=(c.get("semantic_class") or (c.get("semantic") or {}).get("semantic_class") or "other"),
            region_type=(c.get("region_type") or (c.get("semantic") or {}).get("region_type") or "free"),
            functional_group=_json_functional_group(c),
            side_preference=(c.get("side_preference") or (c.get("semantic") or {}).get("side_preference") or "free"),
            align_group=(c.get("align_group") or (c.get("semantic") or {}).get("align_group")),
            semantic_strength=_json_optional_float(c, "semantic_strength"),
            constraint_source=str(_json_nested_value(c, "constraint_source", "auto") or "auto"),
            constraint_level=str(_json_nested_value(c, "constraint_level", "soft") or "soft"),
            align_axis=str(_json_nested_value(c, "align_axis", "auto") or "auto"),
            align_strength=_json_float(c, "align_strength", 1.0),
            anchor_ref=_json_anchor_ref(c),
            subzone=(c.get("subzone") or (c.get("semantic") or {}).get("subzone") or "free"),
            same_side_group=(c.get("same_side_group") or (c.get("semantic") or {}).get("same_side_group") or None),
            boundary_order=(
                int(c.get("boundary_order")) if c.get("boundary_order", None) not in (None, "")
                else (int((c.get("semantic") or {}).get("boundary_order")) if (c.get("semantic") or {}).get("boundary_order", None) not in (None, "") else None)
            ),
            boundary_order_source=str(_json_nested_value(c, "boundary_order_source", "auto") or "auto"),
            spacing_policy=str(_json_nested_value(c, "spacing_policy", "equal") or "equal"),
            pitch_group=(_json_nested_value(c, "pitch_group", None) or None),
            pitch_strength=_json_float(c, "pitch_strength", 1.0),
            row_group=(_json_nested_value(c, "row_group", None) or None),
            row_axis=str(_json_nested_value(c, "row_axis", "auto") or "auto"),
            row_order=_json_optional_int(c, "row_order"),
            critical_nets=tuple(c.get("critical_nets") or (c.get("semantic") or {}).get("critical_nets") or []),
            critical_neighbors=_json_critical_neighbors(c),
            critical_neighbor_specs=_json_critical_neighbor_specs(c),
            review_status=str(review.get("review_status") or "untracked"),
            auto_confidence=float(review.get("auto_confidence", 1.0) if review.get("auto_confidence", None) not in (None, "") else 1.0),
            needs_review=bool(review.get("needs_review", False)),
            placement_role=str(c.get("placement_role") or (c.get("semantic") or {}).get("placement_role") or "member"),
            module_id=str(c.get("module_id") or (c.get("module") or {}).get("module_id") or ""),
            module_role=str(c.get("module_role") or (c.get("module") or {}).get("module_role") or "member"),
            module_anchor_ref=(c.get("module_anchor_ref") or (c.get("module") or {}).get("anchor_ref") or None),
            module_order=int(c.get("module_order", 0) if c.get("module_order", None) not in (None, "") else 0),
            module_local_order=int(c.get("module_local_order", 0) if c.get("module_local_order", None) not in (None, "") else 0),
            module_region_bbox=(
                tuple(float(v) for v in _bbox_from_prior_component(c))
                if _bbox_from_prior_component(c) else None
            ),
            module_region_confidence=_prior_conf_from_component(c),
            expert_module_region_bbox=(
                tuple(float(v) for v in _expert_module_bbox_from_component(c))
                if load_expert and _expert_module_bbox_from_component(c) else None
            ),
            module_region_source=str((c.get('prior') or {}).get('region_source') or (c.get('module') or {}).get('region_source') or 'prior_region'),
            module_shape_hint=(c.get("module_shape_hint") or (c.get("module") or {}).get("shape_hint") or None),
            module_subregion=str(c.get("module_subregion") or (c.get("module") or {}).get("subregion") or "free"),
            expert_xy=(
                (float(c["expert"]["xy_mm"][0]), float(c["expert"]["xy_mm"][1]))
                if load_expert and isinstance(c.get("expert"), dict) and c["expert"].get("xy_mm") is not None else None
            ),
            expert_rot=(
                int(c.get("expert", {}).get("rot", 0))
                if load_expert and isinstance(c.get("expert"), dict) else 0
            ),
        ))
    component_sequence = [
        str(c.get("ref"))
        for c in data.get("components", [])
        if c.get("ref")
    ]
    stored_sequence = _stored_graph_sequence(data)
    policy = normalize_sequence_policy(sequence_policy)
    nets = data.get("nets", {}) if isinstance(data.get("nets", {}), dict) else {}
    data_modules = (
        data.get("modules")
        or data.get("module_annotations")
        or data.get("graph", {}).get("modules")
        or data.get("graph", {}).get("module_sequence_detail")
        or []
    )

    if policy == "stored":
        if stored_sequence is None:
            raise ValueError(
                f"sequence_policy='stored' requires graph.sequence in {path}"
            )
        initial_sequence = _validate_complete_sequence(
            stored_sequence,
            component_sequence,
            source=f"{path}: graph.sequence",
        )
        task = Task(
            bbox_mm=bbox,
            grid_mm=grid,
            components=comps,
            nets=nets,
            sequence=initial_sequence,
            modules=None,
        )
        task = apply_modules_to_task(
            task,
            data_modules=data_modules,
            reorder_sequence=False,
        )
        task.sequence_policy = policy
        task.sequence_source = "graph.sequence"
        task.sequence_meta = {
            "policy": policy,
            "source": "graph.sequence",
            "stored_sequence_present": True,
            "rebuild_applied": False,
            "expert_fields_loaded": bool(load_expert),
        }
        if not load_expert:
            task = strip_expert_fields_from_task(task)
        return task

    # rebuild: graph.sequence is deliberately ignored. Component JSON order is
    # only a neutral tie-breaker before the canonical module/semantic reorder.
    task = Task(
        bbox_mm=bbox,
        grid_mm=grid,
        components=comps,
        nets=nets,
        sequence=component_sequence,
        modules=None,
    )
    task = apply_modules_to_task(
        task,
        data_modules=data_modules,
        reorder_sequence=True,
    )
    rebuilt_sequence = _validate_complete_sequence(
        list(task.sequence),
        component_sequence,
        source=f"{path}: rebuilt sequence",
    )

    if policy == "validate":
        if stored_sequence is None:
            raise ValueError(
                f"sequence_policy='validate' requires graph.sequence in {path}"
            )
        validated_stored = _validate_complete_sequence(
            stored_sequence,
            component_sequence,
            source=f"{path}: graph.sequence",
        )
        if validated_stored != rebuilt_sequence:
            mismatch_index = next(
                (
                    idx
                    for idx, (stored_ref, rebuilt_ref) in enumerate(
                        zip(validated_stored, rebuilt_sequence)
                    )
                    if stored_ref != rebuilt_ref
                ),
                None,
            )
            raise ValueError(
                f"Stored graph.sequence does not match canonical rebuilt sequence "
                f"for {path}; first_mismatch_index={mismatch_index}, "
                f"stored_ref={validated_stored[mismatch_index] if mismatch_index is not None else None}, "
                f"rebuilt_ref={rebuilt_sequence[mismatch_index] if mismatch_index is not None else None}"
            )

    previous_meta = dict(getattr(task, "sequence_meta", {}) or {})
    previous_meta.update(
        {
            "policy": policy,
            "source": "six_phase_anchor_frontier_v1",
            "stored_sequence_present": stored_sequence is not None,
            "stored_sequence_validated": policy == "validate",
            "rebuild_applied": True,
            "expert_fields_loaded": bool(load_expert),
        }
    )
    task.sequence = rebuilt_sequence
    task.sequence_policy = policy
    task.sequence_source = "six_phase_anchor_frontier_v1"
    task.sequence_meta = previous_meta
    if not load_expert:
        task = strip_expert_fields_from_task(task)
    return task


def strip_expert_fields_from_task(task: Task) -> Task:
    """Remove expert-derived fields from a Task before runtime inference/rollout.

    This is intentionally in-place: callers can sanitize the already resolved
    sequence/modules without re-parsing JSON.  Expert labels in the source JSON
    remain available to training code that reads the JSON directly.
    """
    for comp in getattr(task, "components", []) or []:
        comp.expert_xy = None
        comp.expert_rot = 0
        comp.expert_module_region_bbox = None
        hint = getattr(comp, "module_shape_hint", None)
        if isinstance(hint, dict) and str(hint.get("source", "")).startswith("derived_from_expert"):
            comp.module_shape_hint = None
    for module in getattr(task, "modules", None) or []:
        if not isinstance(module, dict):
            continue
        for key in ("expert_region_label", "expert_region_heatmap", "expert_bbox_mm"):
            module.pop(key, None)
        hint = module.get("shape_hint") or module.get("module_shape_hint")
        if isinstance(hint, dict) and str(hint.get("source", "")).startswith("derived_from_expert"):
            module.pop("shape_hint", None)
            module.pop("module_shape_hint", None)
    meta = dict(getattr(task, "sequence_meta", {}) or {})
    meta["expert_fields_loaded"] = False
    task.sequence_meta = meta
    return task
