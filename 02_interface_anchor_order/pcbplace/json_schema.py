from __future__ import annotations
from typing import Any


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
    for idx, comp in enumerate(comps):
        if not isinstance(comp, dict):
            raise ValueError(f"components[{idx}] must be an object{suffix}")

        ref = comp.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"components[{idx}].ref must be a non-empty string{suffix}")
        if ref in seen_refs:
            raise ValueError(f"Duplicate component ref '{ref}'{suffix}")
        seen_refs.add(ref)

        _validate_component_semantic_fields(comp, idx, source=source)

    return data
