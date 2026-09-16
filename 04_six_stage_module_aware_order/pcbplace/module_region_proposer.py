from __future__ import annotations

"""Leakage-safe module region proposer.

This module creates *prior_region* hints from information that is available at
inference time: board outline, refs/types/footprints, pads/nets, module type,
module membership and optional user/mechanical fixed hints.  It must not read
expert xy/rot or final-layout bboxes when generating priors.
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .region_prior import gaussian_heatmap_from_bbox
from .runtime_safety import RUNTIME_PRIOR_SCHEMA_VERSION, RUNTIME_SHAPE_HINT_SCHEMA_VERSION, is_runtime_prior_allowed

BBox = Tuple[float, float, float, float]

_POWER_WORDS = (
    "VIN", "VBUS", "VBAT", "BAT", "12V", "24V", "5V", "3V3", "3.3V", "VCC", "VDD",
    "PWR", "POWER", "FUSE", "PMIC", "DCDC", "DC-DC", "BUCK", "BOOST", "LDO", "REG",
    "INDUCTOR", "SWITCHER",
)
_MEMORY_WORDS = ("DDR", "SDRAM", "LPDDR", "RAM", "FLASH", "QSPI", "EMMC", "NAND", "NOR")
_CLOCK_WORDS = ("XTAL", "CRYSTAL", "OSC", "CLK", "CLOCK")
_INTERFACE_WORDS = ("USB", "HDMI", "MIPI", "FPC", "FFC", "RJ45", "PCIE", "PCI-E", "SATA", "ETH", "CONN", "HEADER", "JACK")
_MAIN_WORDS = ("MCU", "CPU", "SOC", "FPGA", "ZYNQ", "STM32", "ESP32", "RP2040", "ATSAM", "ATMEGA", "PIC")


def _bbox_valid(bb: Any) -> bool:
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in bb]
    except Exception:
        return False
    return x1 > x0 and y1 > y0


def _as_bbox(bb: Any) -> Optional[BBox]:
    if not _bbox_valid(bb):
        return None
    x0, y0, x1, y1 = [float(v) for v in bb]
    return (x0, y0, x1, y1)


def _clip_bbox(bb: BBox, board_bbox: BBox) -> BBox:
    bx0, by0, bx1, by1 = [float(v) for v in board_bbox]
    x0, y0, x1, y1 = [float(v) for v in bb]
    x0 = max(bx0, min(bx1, x0)); x1 = max(bx0, min(bx1, x1))
    y0 = max(by0, min(by1, y0)); y1 = max(by0, min(by1, y1))
    if x1 <= x0:
        cx = 0.5 * (bx0 + bx1); pad = max(1e-3, 0.05 * (bx1 - bx0))
        x0, x1 = max(bx0, cx - pad), min(bx1, cx + pad)
    if y1 <= y0:
        cy = 0.5 * (by0 + by1); pad = max(1e-3, 0.05 * (by1 - by0))
        y0, y1 = max(by0, cy - pad), min(by1, cy + pad)
    return (float(x0), float(y0), float(x1), float(y1))


def _norm_rect(board_bbox: BBox, x0: float, y0: float, x1: float, y1: float) -> BBox:
    bx0, by0, bx1, by1 = [float(v) for v in board_bbox]
    bw = max(1e-6, bx1 - bx0); bh = max(1e-6, by1 - by0)
    return _clip_bbox((bx0 + x0 * bw, by0 + y0 * bh, bx0 + x1 * bw, by0 + y1 * bh), board_bbox)


def _edge_bbox(board_bbox: BBox, side: str, band: float = 0.20) -> BBox:
    side = str(side or "").lower()
    b = max(0.08, min(0.35, float(band)))
    if side in {"left", "edge_left"}:
        return _norm_rect(board_bbox, 0.00, 0.12, b, 0.88)
    if side in {"right", "edge_right"}:
        return _norm_rect(board_bbox, 1.0 - b, 0.12, 1.00, 0.88)
    if side in {"top", "edge_top"}:
        return _norm_rect(board_bbox, 0.12, 1.0 - b, 0.88, 1.00)
    if side in {"bottom", "edge_bottom"}:
        return _norm_rect(board_bbox, 0.12, 0.00, 0.88, b)
    return _norm_rect(board_bbox, 0.12, 0.00, 0.88, b)


def _center_bbox(board_bbox: BBox, margin: float = 0.20) -> BBox:
    m = max(0.05, min(0.40, float(margin)))
    return _norm_rect(board_bbox, m, m, 1.0 - m, 1.0 - m)


def _heatmap_payload(board_bbox: BBox, bbox: BBox, confidence: float, *, grid_x: int = 6, grid_y: int = 6, sigma_cells: float = 0.85, source: str = "") -> Dict[str, Any]:
    values = gaussian_heatmap_from_bbox(tuple(float(v) for v in board_bbox), tuple(float(v) for v in bbox), int(grid_x), int(grid_y), float(sigma_cells), float(confidence)).reshape(int(grid_x), int(grid_y)).tolist()
    return {
        "grid_x": int(grid_x),
        "grid_y": int(grid_y),
        "sigma_cells": float(sigma_cells),
        "values": values,
        "source": str(source or "gaussian_bbox"),
    }


def _ref_prefix(ref: str) -> str:
    m = re.match(r"[A-Za-z]+", str(ref or ""))
    return m.group(0).upper() if m else ""


def _comp_text(comp: Any) -> str:
    vals = [
        getattr(comp, "ref", ""), getattr(comp, "type", ""), getattr(comp, "footprint", ""),
        getattr(comp, "semantic_class", ""), getattr(comp, "region_type", ""),
        getattr(comp, "functional_group", ""), getattr(comp, "side_preference", ""),
        getattr(comp, "placement_role", ""),
    ]
    pads = getattr(comp, "pads", []) or []
    for net, _xy in pads[:32]:
        vals.append(str(net or ""))
    return " ".join(str(v or "") for v in vals).upper()


def _contains_any(text: str, words: Iterable[str]) -> bool:
    t = str(text or "").upper()
    return any(w in t for w in words)


def _module_text(module: Dict[str, Any], comp_by_ref: Dict[str, Any]) -> str:
    vals = [module.get("module_id", ""), module.get("module_type", ""), module.get("anchor_ref", "")]
    for ref in module.get("members", []) or []:
        comp = comp_by_ref.get(str(ref))
        if comp is not None:
            vals.append(_comp_text(comp))
    return " ".join(vals).upper()


def _component_pin_count(comp: Any) -> int:
    return len(getattr(comp, "pads", []) or [])


def _is_physical_connector_for_prior(ref: str, comp: Any) -> bool:
    typ = str(getattr(comp, "type", "") or "").lower()
    sem = str(getattr(comp, "semantic_class", "") or "").lower()
    role = str(getattr(comp, "placement_role", "") or "").lower()
    pref = _ref_prefix(str(ref))
    connector_tokens = ("conn_", "connector", "header", "terminal", "usb", "rj", "jack", "fpc", "ffc", "socket", "sma", "coax")
    if typ.startswith("conn_") or any(tok in typ for tok in connector_tokens):
        return True
    if sem in {"interface", "mechanical_edge_interface", "rf"} and (pref in {"J", "P", "JP", "CN", "CON", "CONN", "RJ", "X", "XA"} or "edge" in role):
        return True
    return False


def _explicit_edge_side(comp: Any) -> str:
    for attr in ("side_preference", "region_type"):
        val = str(getattr(comp, attr, "") or "").lower().strip()
        if val in {"left", "right", "top", "bottom", "edge_left", "edge_right", "edge_top", "edge_bottom"}:
            return val
    allowed = getattr(comp, "allowed_sides", []) or []
    if allowed:
        val = str(allowed[0]).lower().strip()
        if val in {"left", "right", "top", "bottom", "edge_left", "edge_right", "edge_top", "edge_bottom"}:
            return val
    return ""


def _side_from_members(module: Dict[str, Any], comp_by_ref: Dict[str, Any]) -> str:
    """Use an edge side only when a physical edge-facing member establishes it."""
    for ref in module.get("members", []) or []:
        comp = comp_by_ref.get(str(ref))
        if comp is None:
            continue
        if _is_physical_connector_for_prior(str(ref), comp) or bool(getattr(comp, "must_touch_boundary", False)):
            side = _explicit_edge_side(comp)
            if side:
                return side
    return ""


def _module_family(module: Dict[str, Any], comp_by_ref: Dict[str, Any]) -> str:
    """Family selection is type/semantic-first, never a raw substring match."""
    mtype = str(module.get("module_type", "") or "").lower()
    members = [(str(r), comp_by_ref.get(str(r))) for r in module.get("members", []) or []]
    members = [(r, c) for r, c in members if c is not None]
    if mtype == "power_or_driver":
        return "power"
    if mtype == "core_ic":
        return "main_ic"
    if mtype == "clock_local":
        return "clock"
    if mtype == "interface":
        return "interface"
    if mtype == "passive_cluster":
        return "passive_cluster"
    if any(_is_physical_connector_for_prior(r, c) for r, c in members):
        return "interface"
    text = _module_text(module, comp_by_ref)
    if _contains_any(text, _MEMORY_WORDS):
        return "memory"
    if _contains_any(text, _MAIN_WORDS):
        return "main_ic"
    if _contains_any(text, _POWER_WORDS):
        return "power"
    if _contains_any(text, _CLOCK_WORDS):
        return "clock"
    return "generic_cluster"


def _module_has_confirmed_edge_anchor(module: Dict[str, Any], comp_by_ref: Dict[str, Any]) -> bool:
    for ref in module.get("members", []) or []:
        comp = comp_by_ref.get(str(ref))
        if comp is None:
            continue
        if (_is_physical_connector_for_prior(str(ref), comp) or bool(getattr(comp, "must_touch_boundary", False))) and _explicit_edge_side(comp):
            return True
    return False


def _best_main_anchor(modules: List[Dict[str, Any]], comp_by_ref: Dict[str, Any]) -> str:
    best_ref = ""
    best_score = -1.0
    for m in modules:
        for ref in m.get("members", []) or []:
            comp = comp_by_ref.get(str(ref))
            if comp is None:
                continue
            text = _comp_text(comp)
            prefix = _ref_prefix(str(ref))
            score = float(_component_pin_count(comp))
            if prefix in {"U", "IC", "MCU"}:
                score += 20.0
            if _contains_any(text, _MAIN_WORDS):
                score += 40.0
            if score > best_score:
                best_score = score
                best_ref = str(ref)
    return best_ref


def _broad_free_bbox(board_bbox: BBox, margin: float = 0.06) -> BBox:
    return _norm_rect(board_bbox, margin, margin, 1.0 - margin, 1.0 - margin)


def _make_region_candidates(family: str, board_bbox: BBox, side: str = "") -> List[Dict[str, Any]]:
    if family == "interface":
        sides = [side] if side else ["left", "right", "top", "bottom"]
        out = []
        score = 1.0 / max(1, len(sides))
        for s in sides:
            out.append({"bbox_mm": list(_edge_bbox(board_bbox, s, 0.18)), "score": float(score), "reason": f"{family}_{s or 'edge'}"})
        return out
    if family == "power":
        return [
            {"bbox_mm": list(_edge_bbox(board_bbox, side or "left", 0.28)), "score": 0.55, "reason": "power_near_entry_edge"},
            {"bbox_mm": list(_center_bbox(board_bbox, 0.18)), "score": 0.25, "reason": "power_between_entry_and_load_fallback"},
        ]
    if family == "main_ic":
        return [
            {"bbox_mm": list(_center_bbox(board_bbox, 0.22)), "score": 0.65, "reason": "main_ic_interior"},
            {"bbox_mm": list(_center_bbox(board_bbox, 0.14)), "score": 0.35, "reason": "main_ic_large_interior"},
        ]
    if family in {"memory", "clock"}:
        return [
            {"bbox_mm": list(_center_bbox(board_bbox, 0.18)), "score": 0.50, "reason": f"{family}_parent_near_fallback"},
            {"bbox_mm": list(_center_bbox(board_bbox, 0.28)), "score": 0.30, "reason": f"{family}_compact_interior"},
        ]
    return [
        {"bbox_mm": list(_center_bbox(board_bbox, 0.24)), "score": 0.45, "reason": "graph_cluster_center_fallback"},
        {"bbox_mm": list(_center_bbox(board_bbox, 0.14)), "score": 0.25, "reason": "generic_large_center_fallback"},
    ]


def _confidence_for_family(family: str, has_side: bool = False) -> float:
    if family == "main_ic":
        return 0.60
    if family == "interface":
        return 0.72 if has_side else 0.52
    if family == "power":
        return 0.62 if has_side else 0.55
    if family == "memory":
        return 0.68
    if family == "clock":
        return 0.66
    if family == "passive_cluster":
        return 0.38
    return 0.35


def propose_prior_region_for_module(
    module: Dict[str, Any],
    comp_by_ref: Dict[str, Any],
    board_bbox: BBox,
    *,
    all_modules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Return (prior_region, region_candidates, prior_shape_hint)."""
    # Honor an explicitly leakage-safe prior if one already exists.
    pr = module.get("prior_region") if isinstance(module.get("prior_region"), dict) else None
    if pr and is_runtime_prior_allowed(pr) and _bbox_valid(pr.get("bbox_mm")):
        bbox = _as_bbox(pr.get("bbox_mm"))
        conf = float(pr.get("confidence", 0.5))
        center = [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])] if bbox else None
        hint = {
            "version": 2,
            "schema_version": RUNTIME_SHAPE_HINT_SCHEMA_VERSION,
            "source": "prior_region_existing",
            "leakage_safe": True,
            "module_center_mm": center,
            "module_axis": "compact",
            "anchor_relative_zone": "around",
            "edge_corridor": "",
            "support_ring_radius_mm": max(1.0, 0.35 * max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))) if bbox else 1.0,
        }
        return dict(pr), list(module.get("region_candidates") or []), hint

    family = _module_family(module, comp_by_ref)
    side = _side_from_members(module, comp_by_ref)
    confirmed_edge = _module_has_confirmed_edge_anchor(module, comp_by_ref)

    # Only a connector with an explicit edge constraint earns a narrow edge
    # prior.  Unknown-side interfaces and power/relay islands instead receive
    # a broad, low-confidence hint so a bad semantic guess cannot pull a whole
    # module into one board edge.
    if family == "interface" and not (side and confirmed_edge):
        candidates = [
            {"bbox_mm": list(_broad_free_bbox(board_bbox, 0.08)), "score": 0.55, "reason": "interface_unconstrained_broad"},
            *[
                {"bbox_mm": list(_edge_bbox(board_bbox, edge, 0.18)), "score": 0.1125, "reason": f"interface_candidate_{edge}"}
                for edge in ("left", "right", "top", "bottom")
            ],
        ]
        conf = 0.28
    elif family == "power" and not (side and confirmed_edge):
        candidates = [
            {"bbox_mm": list(_broad_free_bbox(board_bbox, 0.06)), "score": 0.62, "reason": "power_broad_free"},
            {"bbox_mm": list(_center_bbox(board_bbox, 0.12)), "score": 0.38, "reason": "power_interior_fallback"},
        ]
        conf = 0.30
    else:
        candidates = _make_region_candidates(family, board_bbox, side)
        conf = _confidence_for_family(family, bool(side and confirmed_edge))
    best = max(candidates, key=lambda c: float(c.get("score", 0.0))) if candidates else {"bbox_mm": list(_center_bbox(board_bbox, 0.24)), "score": 0.1, "reason": "fallback"}
    bbox = _as_bbox(best.get("bbox_mm")) or _center_bbox(board_bbox, 0.24)
    center = [float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)]
    bw = max(1e-6, bbox[2] - bbox[0]); bh = max(1e-6, bbox[3] - bbox[1])
    axis = "horizontal" if bw > 1.35 * bh else ("vertical" if bh > 1.35 * bw else "compact")
    edge_corridor = ""
    if family == "interface" or str(best.get("reason", "")).startswith("power_near"):
        reason = str(best.get("reason", "")).lower()
        for s in ("left", "right", "top", "bottom"):
            if s in reason or side.endswith(s) or side == s:
                edge_corridor = f"edge_{s}"
                break
    main_anchor = _best_main_anchor(all_modules or [module], comp_by_ref)
    relative_prior: Dict[str, Any] = {}
    if family in {"memory", "clock"} and main_anchor:
        relative_prior = {
            "type": "very_near_parent" if family == "clock" else "near_parent",
            "parent_ref": main_anchor,
            "max_distance_norm": 0.08 if family == "clock" else 0.18,
            "same_side_preferred": True,
        }
    prior = {
        "bbox_mm": [float(v) for v in bbox],
        "center_mm": center,
        "confidence": float(conf),
        "source": f"rule_v2_{family}",
        "schema_version": RUNTIME_PRIOR_SCHEMA_VERSION,
        "leakage_safe": True,
        "family": family,
    }
    if relative_prior:
        prior["relative_prior"] = relative_prior
    hint = {
        "version": 2,
        "schema_version": RUNTIME_SHAPE_HINT_SCHEMA_VERSION,
        "source": f"prior_rule_v2_{family}",
        "leakage_safe": True,
        "module_center_mm": center,
        "module_axis": axis,
        "anchor_relative_zone": "edge_aligned" if edge_corridor else ("around" if family in {"main_ic", "memory", "clock"} else "fallback"),
        "edge_corridor": edge_corridor,
        "support_ring_radius_mm": float(max(1.0, 0.35 * max(bw, bh))),
    }
    return prior, candidates, hint


def attach_prior_and_expert_regions(
    modules: List[Dict[str, Any]],
    comp_by_ref: Dict[str, Any],
    board_bbox: BBox,
    *,
    legacy_region_as_expert: bool = True,
) -> List[Dict[str, Any]]:
    """Mutate modules so active region fields are prior-only.

    - modules[i].prior_region is the only intended model input.
    - modules[i].expert_region_label stores labels derived from final/expert layout.
    - modules[i].region_bbox_mm mirrors prior_region.bbox_mm for legacy callers.
    """
    board_bbox = tuple(float(v) for v in board_bbox)  # type: ignore[assignment]
    out: List[Dict[str, Any]] = []
    for m in modules:
        mm = dict(m)
        # Preserve old region bbox as expert label when no explicit expert label exists.
        expert_label = mm.get("expert_region_label") if isinstance(mm.get("expert_region_label"), dict) else {}
        expert_bbox = _as_bbox(expert_label.get("bbox_mm")) if expert_label else None
        if expert_bbox is None:
            legacy_bbox = None
            if legacy_region_as_expert:
                legacy_bbox = _as_bbox(mm.get("expert_bbox_mm")) or _as_bbox(mm.get("bbox_mm")) or _as_bbox(mm.get("region_bbox_mm"))
            if legacy_bbox is not None:
                expert_bbox = legacy_bbox
        if expert_bbox is not None:
            mm["expert_region_label"] = {
                "bbox_mm": [float(v) for v in expert_bbox],
                "center_mm": [float(0.5 * (expert_bbox[0] + expert_bbox[2])), float(0.5 * (expert_bbox[1] + expert_bbox[3]))],
                "source": str((expert_label or {}).get("source") or "final_layout_or_legacy_region_bbox"),
                "use_as_input": False,
            }
            mm["expert_region_heatmap"] = _heatmap_payload(board_bbox, expert_bbox, 1.0, source="expert_region_bbox_gaussian")
        # Generate leakage-safe prior.
        prior, candidates, hint = propose_prior_region_for_module(mm, comp_by_ref, board_bbox, all_modules=modules)
        bbox = _as_bbox(prior.get("bbox_mm")) or _center_bbox(board_bbox, 0.24)
        mm["prior_region"] = prior
        mm["prior_region_heatmap"] = _heatmap_payload(board_bbox, bbox, float(prior.get("confidence", 0.35)), source=str(prior.get("source") or "rule_v2"))
        mm["region_candidates"] = candidates
        mm["region_bbox_mm"] = [float(v) for v in bbox]  # legacy active field = prior only
        mm["region_confidence"] = float(prior.get("confidence", 0.35))
        mm["shape_hint"] = hint
        mm["module_subregion"] = str(hint.get("anchor_relative_zone") or "around")
        mm.pop("expert_bbox_mm", None)
        out.append(mm)
    return out
