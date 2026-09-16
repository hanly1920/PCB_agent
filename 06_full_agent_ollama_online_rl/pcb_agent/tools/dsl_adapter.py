from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from ..schemas import LayoutDSL


_EDGE_REGION = {
    "left": "edge_left",
    "right": "edge_right",
    "top": "edge_top",
    "bottom": "edge_bottom",
}


def _xy_from_value(value: Any) -> tuple[float, float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for keys in (("x", "y"), ("cx", "cy")):
            if all(k in value for k in keys):
                return (float(value[keys[0]]), float(value[keys[1]]))
        for key in ("xy_mm", "center_mm", "position_mm", "pos_mm", "at_mm", "fixed_xy_mm"):
            got = _xy_from_value(value.get(key))
            if got is not None:
                return got
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def _rot_from_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for key in ("rot", "rotation", "rot_deg", "angle", "angle_deg", "fixed_rot"):
            if value.get(key) not in (None, ""):
                return float(value.get(key)) % 360.0
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[2]) % 360.0
    try:
        return float(value) % 360.0
    except Exception:
        return None


def component_existing_xy(comp: dict[str, Any]) -> tuple[float, float] | None:
    for value in (
        comp.get("fixed_xy_mm"),
        comp.get("fixed_position_mm"),
        comp.get("locked_xy_mm"),
        comp.get("fixed"),
        comp.get("placement"),
        comp.get("layout"),
        comp.get("expert"),
    ):
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    return None


def component_existing_rot(comp: dict[str, Any]) -> float:
    for value in (
        comp.get("fixed_rot"),
        comp.get("locked_rot"),
        comp.get("fixed"),
        comp.get("placement"),
        comp.get("layout"),
        comp.get("expert"),
    ):
        rot = _rot_from_value(value)
        if rot is not None:
            return float(rot)
    return 0.0


class DSLTaskAdapter:
    @staticmethod
    def materialize_task(
        task_json_path: str | Path,
        dsl: LayoutDSL,
        output_path: str | Path | None = None,
    ) -> tuple[Path, list[str], dict[str, Any]]:
        path = Path(task_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        comps = data.get("components") or []
        by_ref = {str(c.get("ref")): c for c in comps if c.get("ref")}
        observations: list[str] = []
        applied: list[dict[str, Any]] = []

        def refs_for(patterns: list[str]) -> list[str]:
            matched: list[str] = []
            for pat in patterns:
                matched.extend(ref for ref in by_ref if fnmatch.fnmatchcase(ref, pat))
            return sorted(set(matched))

        for constraint in list(dsl.hard_constraints or []) + list(dsl.soft_constraints or []):
            ctype = str(constraint.type or "").lower()
            patterns = [str(r) for r in (constraint.refs or []) if str(r).strip()]
            if ctype in {"lock", "locked", "fix", "fixed", "freeze", "frozen"} and not patterns:
                patterns = [str(r) for r in (dsl.locked_refs or []) if str(r).strip()]
            if not patterns and ctype not in {"clearance"}:
                continue
            matched = refs_for(patterns)
            if patterns and not matched:
                observations.append(f"DSL constraint {ctype} matched no refs: {patterns}")
                continue
            record = constraint.model_dump(mode="json")
            record["matched_refs"] = matched
            applied.append(record)

            if ctype in {"lock", "locked", "fix", "fixed", "freeze", "frozen"}:
                missing: list[str] = []
                for ref in matched:
                    comp = by_ref[ref]
                    xy = component_existing_xy(comp)
                    if xy is None:
                        missing.append(ref)
                        continue
                    comp["fixed"] = True
                    comp["locked"] = True
                    comp["fixed_xy_mm"] = [float(xy[0]), float(xy[1])]
                    comp["fixed_rot"] = float(component_existing_rot(comp))
                    comp["fixed_source"] = "dsl_locked_ref"
                    comp.setdefault("agent_constraints", []).append(record)
                if missing:
                    raise ValueError(
                        "DSL lock constraints require existing coordinates. Missing fixed_xy_mm/placement/layout/expert.xy_mm for: "
                        + ", ".join(sorted(missing))
                    )
            elif ctype == "edge":
                side = str(constraint.side or "").lower()
                if side not in _EDGE_REGION:
                    observations.append(f"Unsupported edge side for DSL constraint: {constraint.side}")
                    continue
                for ref in matched:
                    comp = by_ref[ref]
                    comp["side_preference"] = side
                    comp["region_type"] = _EDGE_REGION[side]
                    if str(comp.get("semantic_class") or "").lower() in {"", "other"}:
                        comp["semantic_class"] = "interface"
                    comp.setdefault("agent_constraints", []).append(record)
            elif ctype == "prefer_region":
                region = str(constraint.region or "").lower()
                for ref in matched:
                    comp = by_ref[ref]
                    if region in {"center", "core"}:
                        comp["region_type"] = "core"
                        comp["side_preference"] = "free"
                    elif region in _EDGE_REGION:
                        comp["region_type"] = _EDGE_REGION[region]
                        comp["side_preference"] = region
                    comp.setdefault("agent_constraints", []).append(record)
            elif ctype == "near":
                for ref in matched:
                    comp = by_ref[ref]
                    comp["anchor_ref"] = constraint.target
                    comp["subzone"] = comp.get("subzone") or "around"
                    comp.setdefault("agent_constraints", []).append(record)
            elif ctype == "clearance":
                rules = data.setdefault("rules", {})
                if constraint.value_mm is not None:
                    rules["agent_clearance_mm"] = float(constraint.value_mm)

        if dsl.locked_refs:
            lock_patterns = [str(r) for r in dsl.locked_refs if str(r).strip()]
            locked_matched = refs_for(lock_patterns)
            if lock_patterns and not locked_matched:
                observations.append(f"DSL locked_refs matched no refs: {lock_patterns}")
            for ref in locked_matched:
                comp = by_ref[ref]
                xy = component_existing_xy(comp)
                if xy is None:
                    raise ValueError(
                        "DSL locked_refs require existing coordinates. Missing fixed_xy_mm/placement/layout/expert.xy_mm for: "
                        + ref
                    )
                comp["fixed"] = True
                comp["locked"] = True
                comp["fixed_xy_mm"] = [float(xy[0]), float(xy[1])]
                comp["fixed_rot"] = float(component_existing_rot(comp))
                comp["fixed_source"] = "dsl_locked_ref"

        meta = data.setdefault("meta", {})
        meta["agent_constraints"] = applied
        meta["agent_movable_refs"] = [str(r) for r in (dsl.movable_refs or [])]
        meta["agent_locked_refs"] = [str(r) for r in (dsl.locked_refs or [])]

        if not applied and not dsl.locked_refs and not dsl.movable_refs:
            return path, observations, {"applied_constraint_count": 0}

        out = Path(output_path) if output_path else path.with_name(path.stem + ".agent_dsl" + path.suffix)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out, observations, {"applied_constraint_count": len(applied)}
