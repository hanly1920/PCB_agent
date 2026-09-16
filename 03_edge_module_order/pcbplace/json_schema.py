from __future__ import annotations
from typing import Any

from .runtime_safety import (
    describe_runtime_prior_violation,
    describe_runtime_shape_hint_violation,
    is_runtime_prior_allowed,
    is_runtime_shape_hint_allowed,
)


VALID_ALLOWED_SIDES = {"left", "right", "top", "bottom"}


def _src_suffix(source: str | None) -> str:
    return f" in {source}" if source else ""


def _validate_component_semantic_fields(comp: dict, idx: int, source: str | None = None) -> None:
    suffix = _src_suffix(source)

    if "must_touch_boundary" in comp and comp["must_touch_boundary"] is not None and not isinstance(comp["must_touch_boundary"], bool):
        raise ValueError(f"components[{idx}].must_touch_boundary must be boolean or null{suffix}")

    semantic = comp.get("semantic")
    if semantic is not None and not isinstance(semantic, dict):
        raise ValueError(f"components[{idx}].semantic must be an object when present{suffix}")

    if isinstance(semantic, dict):
        if "must_touch_boundary" in semantic and semantic["must_touch_boundary"] is not None and not isinstance(semantic["must_touch_boundary"], bool):
            raise ValueError(f"components[{idx}].semantic.must_touch_boundary must be boolean or null{suffix}")


    # Optional engineering semantics.  These fields are intentionally permissive
    # so older datasets keep loading, but obvious malformed values fail fast.
    def _field(name: str) -> Any:
        if name in comp:
            return comp.get(name)
        if isinstance(semantic, dict) and name in semantic:
            return semantic.get(name)
        layout = comp.get("layout") if isinstance(comp.get("layout"), dict) else {}
        return layout.get(name)

    if _field("semantic_strength") not in (None, ""):
        try:
            val = float(_field("semantic_strength"))
        except Exception as exc:
            raise ValueError(f"components[{idx}].semantic_strength must be numeric{suffix}") from exc
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"components[{idx}].semantic_strength must be in [0, 1]{suffix}")

    if _field("constraint_level") not in (None, ""):
        if str(_field("constraint_level")).lower() not in {"hard", "soft", "hint"}:
            raise ValueError(f"components[{idx}].constraint_level must be hard|soft|hint{suffix}")

    if _field("align_axis") not in (None, ""):
        if str(_field("align_axis")).lower() not in {"x", "y", "auto"}:
            raise ValueError(f"components[{idx}].align_axis must be x|y|auto{suffix}")

    if _field("row_axis") not in (None, ""):
        if str(_field("row_axis")).lower() not in {"x", "y", "auto"}:
            raise ValueError(f"components[{idx}].row_axis must be x|y|auto{suffix}")

    if _field("spacing_policy") not in (None, ""):
        if str(_field("spacing_policy")).lower() not in {"equal", "preserve", "free"}:
            raise ValueError(f"components[{idx}].spacing_policy must be equal|preserve|free{suffix}")

    cn = _field("critical_neighbors")
    if cn is not None and not isinstance(cn, list):
        raise ValueError(f"components[{idx}].critical_neighbors must be a list{suffix}")
    if isinstance(cn, list):
        for j, item in enumerate(cn):
            if isinstance(item, dict):
                if not item.get("ref") and not item.get("neighbor") and not item.get("neighbor_ref"):
                    raise ValueError(f"components[{idx}].critical_neighbors[{j}] object must contain ref/neighbor/neighbor_ref{suffix}")
                if item.get("weight") not in (None, ""):
                    try:
                        float(item.get("weight"))
                    except Exception as exc:
                        raise ValueError(f"components[{idx}].critical_neighbors[{j}].weight must be numeric{suffix}") from exc
            elif not isinstance(item, str):
                raise ValueError(f"components[{idx}].critical_neighbors[{j}] must be a string or object{suffix}")

    # Optional fixed/locked component schema. `fixed` may be a bool, a dict,
    # or a compact [x, y, rot] list for DSL/post-layout edits.
    fixed = comp.get("fixed")
    if fixed is not None and not isinstance(fixed, (bool, dict, list, tuple, str, int, float)):
        raise ValueError(f"components[{idx}].fixed must be boolean/object/list when present{suffix}")
    for xy_key in ("fixed_xy_mm", "fixed_position_mm", "fixed_pos_mm", "locked_xy_mm"):
        xy = comp.get(xy_key)
        if xy is None:
            continue
        if not (isinstance(xy, list) and len(xy) >= 2):
            raise ValueError(f"components[{idx}].{xy_key} must be [x_mm, y_mm]{suffix}")
        try:
            float(xy[0]); float(xy[1])
        except Exception as exc:
            raise ValueError(f"components[{idx}].{xy_key} must contain numeric x/y values{suffix}") from exc

    sides = comp.get("allowed_sides")
    if sides is not None:
        if not isinstance(sides, list):
            raise ValueError(f"components[{idx}].allowed_sides must be a list{suffix}")
        bad = [s for s in sides if s not in VALID_ALLOWED_SIDES]
        if bad:
            raise ValueError(
                f"components[{idx}].allowed_sides contains invalid values: {bad}; "
                f"allowed values are {sorted(VALID_ALLOWED_SIDES)}{suffix}"
            )



def _validate_region_policy(data: dict[str, Any], source: str | None = None) -> None:
    suffix = _src_suffix(source)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    policy = meta.get("region_policy") if isinstance(meta.get("region_policy"), dict) else {}
    if policy.get("expert_region_as_input", False):
        raise ValueError(f"expert_region_as_input=true is not allowed for normal training/inference{suffix}")
    for midx, m in enumerate(data.get("modules") or []):
        if not isinstance(m, dict):
            continue
        prior = m.get("prior_region") if isinstance(m.get("prior_region"), dict) else None
        if prior is not None:
            bb = prior.get("bbox_mm")
            if not (isinstance(bb, list) and len(bb) == 4):
                raise ValueError(f"modules[{midx}].prior_region.bbox_mm must be a 4-element list{suffix}")
            if not is_runtime_prior_allowed(prior):
                reason = describe_runtime_prior_violation(prior, path=f"modules[{midx}].prior_region")
                raise ValueError(f"{reason}{suffix}")
        elif m.get("region_bbox_mm") is not None:
            raise ValueError(
                f"modules[{midx}].region_bbox_mm legacy runtime prior is not allowed without "
                f"a whitelisted prior_region{suffix}"
            )
        hint = m.get("shape_hint") or m.get("module_shape_hint")
        if isinstance(hint, dict) and not is_runtime_shape_hint_allowed(hint):
            reason = describe_runtime_shape_hint_violation(hint, path=f"modules[{midx}].shape_hint")
            raise ValueError(f"{reason}{suffix}")
        ex = m.get("expert_region_label") if isinstance(m.get("expert_region_label"), dict) else None
        if ex is not None and ex.get("use_as_input", False):
            raise ValueError(f"modules[{midx}].expert_region_label.use_as_input must be false{suffix}")

def validate_task_json(data: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    suffix = _src_suffix(source)

    if not isinstance(data, dict):
        raise ValueError(f"Task JSON root must be an object{suffix}")

    board = data.get("board")
    if not isinstance(board, dict):
        raise ValueError(f"Missing or invalid top-level 'board' object{suffix}")

    bbox = board.get("bbox_mm")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"board.bbox_mm must be a 4-element list{suffix}")

    comps = data.get("components")
    if not isinstance(comps, list):
        raise ValueError(f"Top-level 'components' must be a list{suffix}")

    seen_refs: set[str] = set()
    _validate_region_policy(data, source=source)

    for idx, comp in enumerate(comps):
        if not isinstance(comp, dict):
            raise ValueError(f"components[{idx}] must be an object{suffix}")

        ref = comp.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"components[{idx}].ref must be a non-empty string{suffix}")
        if ref in seen_refs:
            raise ValueError(f"Duplicate component ref '{ref}'{suffix}")
        seen_refs.add(ref)

        comp_module = comp.get("module") if isinstance(comp.get("module"), dict) else {}
        comp_prior = comp_module.get("prior_region") if isinstance(comp_module.get("prior_region"), dict) else None
        if comp_prior is not None and not is_runtime_prior_allowed(comp_prior):
            reason = describe_runtime_prior_violation(comp_prior, path=f"components[{idx}].module.prior_region")
            raise ValueError(f"{reason}{suffix}")
        comp_hint = comp.get("module_shape_hint") or comp_module.get("shape_hint")
        if isinstance(comp_hint, dict) and not is_runtime_shape_hint_allowed(comp_hint):
            reason = describe_runtime_shape_hint_violation(comp_hint, path=f"components[{idx}].module_shape_hint")
            raise ValueError(f"{reason}{suffix}")

        _validate_component_semantic_fields(comp, idx, source=source)

    return data
