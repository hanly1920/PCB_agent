from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict

from .module_region_proposer import attach_prior_and_expert_regions
from .runtime_safety import RUNTIME_PRIOR_SCHEMA_VERSION, RUNTIME_SHAPE_HINT_SCHEMA_VERSION

POWER_TOKENS = (
    "GND", "GROUND", "AGND", "DGND", "PGND",
    "VCC", "VDD", "VDDA", "VSSA", "VBUS", "VIN", "VOUT", "VBAT",
    "3V", "5V", "12V", "24V", "1V", "2V", "PWR", "POWER", "VREF",
)

PREFIX_ORDER = {
    "J": 0, "P": 0, "CN": 0, "CON": 0, "SW": 1,
    "U": 2, "IC": 2, "MCU": 2,
    "L": 3, "F": 3, "Q": 3, "D": 4,
    "Y": 5, "X": 5,
    "R": 6, "C": 7,
}

MODULE_TYPE_ORDER = {
    "interface": 0,
    "core_ic": 1,
    "power_or_driver": 2,
    "clock_local": 3,
    "mixed": 4,
    "passive_cluster": 5,
    "misc": 6,
}

def ref_prefix(ref: str) -> str:
    m = re.match(r"[A-Za-z]+", str(ref or ""))
    return m.group(0).upper() if m else ""

def is_power_net(net: str) -> bool:
    n = str(net or "").strip().upper().strip("/")
    if not n:
        return True
    if n in {"NC", "N/C"}:
        return True
    return any(tok in n for tok in POWER_TOKENS)


def _refs_from_top_level_net_value(value: Any, comp_refs: set[str]) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        candidates = value.get('refs') or value.get('components') or value.get('nodes') or value.get('pads') or value.get('members') or []
    else:
        candidates = value or []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    if not isinstance(candidates, (list, tuple, set)):
        candidates = [candidates]
    for item in candidates:
        ref = None
        if isinstance(item, str):
            ref = item
        elif isinstance(item, dict):
            ref = item.get('ref') or item.get('component') or item.get('component_ref') or item.get('refdes')
        elif isinstance(item, (list, tuple)) and item:
            ref = item[0]
        if ref is None:
            continue
        ref = str(ref).strip()
        # Accept either an exact ref or a KiCad-like pad token such as U1:3 / U1.3.
        if ref not in comp_refs:
            m = re.match(r'([A-Za-z]+[A-Za-z0-9_-]*?)(?:[:.].*)?$', ref)
            if m and m.group(1) in comp_refs:
                ref = m.group(1)
        if ref in comp_refs:
            refs.add(ref)
    return refs


def _top_level_netrefs(nets: Optional[Dict[str, List[str]]], comp_refs: set[str]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = defaultdict(set)
    if not isinstance(nets, dict):
        return out
    for net, value in nets.items():
        if is_power_net(str(net)):
            continue
        refs = _refs_from_top_level_net_value(value, comp_refs)
        if len(refs) >= 2:
            out[str(net)].update(refs)
    return out

def _component_xy(comp: Any) -> Tuple[float, float]:
    if getattr(comp, "expert_xy", None) is not None:
        return float(comp.expert_xy[0]), float(comp.expert_xy[1])
    # Fallback: no expert yet; use board center later by caller if needed.
    return (0.0, 0.0)

def _component_half_size(comp: Any) -> Tuple[float, float]:
    w, h = getattr(comp, "size_mm", (1.0, 1.0))
    return max(0.5, float(w) / 2.0), max(0.5, float(h) / 2.0)

def _infer_module_type(members: List[str], comp_by_ref: Dict[str, Any]) -> str:
    """Infer a compatible module type from *real* component semantics.

    A reference prefix alone is deliberately not enough to classify a whole
    connected component as an interface module.  In particular, net labels
    such as ``UNCONNECTED`` must never make an unrelated mixed group look like
    a connector cluster.
    """
    prefixes = Counter(ref_prefix(r) for r in members)
    comps = [(r, comp_by_ref.get(r)) for r in members if comp_by_ref.get(r) is not None]
    if any(_is_true_interface_connector(r, c) for r, c in comps):
        return "interface"
    if any(_is_power_anchor_component(r, c) for r, c in comps):
        return "power_or_driver"
    if any(_is_clock_source_component(r, c) for r, c in comps):
        return "clock_local"
    if any(_is_core_anchor_component(r, c) for r, c in comps):
        return "core_ic"
    if prefixes.get("R", 0) + prefixes.get("C", 0) >= max(2, len(members) * 0.7):
        return "passive_cluster"
    return "mixed"

def choose_anchor(members: List[str], comp_by_ref: Dict[str, Any]) -> str:
    return sorted(
        members,
        key=lambda r: (
            PREFIX_ORDER.get(ref_prefix(r), 5),
            -len(getattr(comp_by_ref.get(r), "pads", []) or []),
            r,
        )
    )[0]

def _bbox_for_members(
    members: List[str],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
) -> List[float]:
    xmin, ymin, xmax, ymax = board_bbox
    pts = []
    for ref in members:
        comp = comp_by_ref.get(ref)
        if comp is None:
            continue
        if getattr(comp, "expert_xy", None) is None:
            continue
        x, y = _component_xy(comp)
        hw, hh = _component_half_size(comp)
        pts.append((x - hw, y - hh))
        pts.append((x + hw, y + hh))
    if not pts:
        # Fallback: central soft region when no expert positions exist.
        bw, bh = xmax - xmin, ymax - ymin
        return [xmin + 0.15 * bw, ymin + 0.15 * bh, xmax - 0.15 * bw, ymax - 0.15 * bh]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = max(2.0 * float(grid_mm), 1.5)
    return [
        max(xmin, min(xs) - pad),
        max(ymin, min(ys) - pad),
        min(xmax, max(xs) + pad),
        min(ymax, max(ys) + pad),
    ]


def _bbox_area(bb: Any) -> float:
    if not bb or len(bb) != 4:
        return 0.0
    x0, y0, x1, y1 = [float(v) for v in bb]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_intersection_area(a: Any, b: Any) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def _area_confidence(area_frac: float) -> float:
    """Confidence for a module bbox as a useful region hint.

    Small module windows are informative; board-sized windows are mostly noise.
    """
    a = max(0.0, float(area_frac))
    if a <= 0.15:
        return 1.0
    if a <= 0.35:
        # 1.0 -> 0.55
        return 1.0 - 0.45 * ((a - 0.15) / 0.20)
    if a <= 0.75:
        # 0.55 -> 0.18
        return 0.55 - 0.37 * ((a - 0.35) / 0.40)
    if a <= 0.98:
        # 0.18 -> 0.04
        return 0.18 - 0.14 * ((a - 0.75) / 0.23)
    return 0.02


def _compute_region_confidence(
    module: Dict[str, Any],
    modules: List[Dict[str, Any]],
    board_bbox: Tuple[float, float, float, float],
) -> float:
    bb = (module.get("prior_region") or {}).get("bbox_mm") or module.get("region_bbox_mm")
    board_area = max(1e-6, _bbox_area(board_bbox))
    area = max(1e-6, _bbox_area(bb))
    area_frac = area / board_area
    overlap_area = 0.0
    for other in modules:
        if other is module:
            continue
        overlap_area += _bbox_intersection_area(bb, (other.get("prior_region") or {}).get("bbox_mm") or other.get("region_bbox_mm"))
    overlap_frac = overlap_area / area
    conf = _area_confidence(area_frac)
    # Overlap does not make a bbox invalid, but it makes it less informative.
    if overlap_frac > 0.0:
        conf *= max(0.20, 1.0 / (1.0 + overlap_frac))
    if not str(module.get("anchor_ref") or "").strip():
        conf *= 0.85
    member_count = len(module.get("members") or [])
    if member_count <= 1:
        conf *= 0.75
    return float(max(0.02, min(1.0, conf)))


def _nearest_board_edge(x: float, y: float, board_bbox: Tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = [float(v) for v in board_bbox]
    distances = {
        "edge_left": abs(x - xmin),
        "edge_right": abs(xmax - x),
        "edge_bottom": abs(y - ymin),
        "edge_top": abs(ymax - y),
    }
    return min(distances, key=distances.get)


def _edge_hint_from_component(comp: Any, board_bbox: Tuple[float, float, float, float]) -> str:
    for attr in ("side_preference", "region_type"):
        val = str(getattr(comp, attr, "") or "").strip().lower()
        if val in {"edge_left", "edge_right", "edge_top", "edge_bottom"}:
            return val
    typ = str(getattr(comp, "type", "") or "").lower()
    sem = str(getattr(comp, "semantic_class", "") or "").lower()
    if any(tok in typ for tok in ("conn", "header", "usb", "jack", "switch", "button", "antenna")) or sem in {"interface", "mechanical_edge_interface", "rf"}:
        xy = getattr(comp, "expert_xy", None)
        if xy is not None:
            return _nearest_board_edge(float(xy[0]), float(xy[1]), board_bbox)
    return ""


def _infer_module_shape_hint(
    module: Dict[str, Any],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
) -> Dict[str, Any]:
    members = [r for r in (module.get("members") or []) if r in comp_by_ref]
    pts: List[Tuple[float, float]] = []
    for ref in members:
        xy = getattr(comp_by_ref[ref], "expert_xy", None)
        if xy is not None:
            pts.append((float(xy[0]), float(xy[1])))
    bb = (module.get("prior_region") or {}).get("bbox_mm") or module.get("region_bbox_mm") or board_bbox
    if pts:
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        mid = len(pts) // 2
        cx = xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])
        cy = ys[mid] if len(ys) % 2 else 0.5 * (ys[mid - 1] + ys[mid])
        span_x = max(xs) - min(xs) if len(xs) > 1 else 0.0
        span_y = max(ys) - min(ys) if len(ys) > 1 else 0.0
    else:
        x0, y0, x1, y1 = [float(v) for v in bb]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        span_x, span_y = max(0.0, x1 - x0), max(0.0, y1 - y0)
    if span_x > 1.35 * max(span_y, 1e-6):
        axis = "horizontal"
    elif span_y > 1.35 * max(span_x, 1e-6):
        axis = "vertical"
    else:
        axis = "compact"

    anchor_ref = str(module.get("anchor_ref") or "").strip()
    anchor_xy = None
    edge_corridor = ""
    if anchor_ref in comp_by_ref:
        ac = comp_by_ref[anchor_ref]
        axy = getattr(ac, "expert_xy", None)
        if axy is not None:
            anchor_xy = (float(axy[0]), float(axy[1]))
        edge_corridor = _edge_hint_from_component(ac, board_bbox)
    if not edge_corridor:
        for ref in members:
            edge_corridor = _edge_hint_from_component(comp_by_ref[ref], board_bbox)
            if edge_corridor:
                break

    zone = "around"
    median_dx = 0.0
    median_dy = 0.0
    radius = 0.0
    if anchor_xy is not None:
        rel = [(x - anchor_xy[0], y - anchor_xy[1]) for x, y in pts if abs(x - anchor_xy[0]) > 1e-6 or abs(y - anchor_xy[1]) > 1e-6]
        if rel:
            dxs = sorted(v[0] for v in rel)
            dys = sorted(v[1] for v in rel)
            mid = len(rel) // 2
            median_dx = dxs[mid] if len(dxs) % 2 else 0.5 * (dxs[mid - 1] + dxs[mid])
            median_dy = dys[mid] if len(dys) % 2 else 0.5 * (dys[mid - 1] + dys[mid])
            dists = sorted(math.hypot(dx, dy) for dx, dy in rel)
            q = min(len(dists) - 1, max(0, int(0.75 * (len(dists) - 1))))
            radius = max(2.5 * float(grid_mm), float(dists[q]))
            if edge_corridor:
                zone = "edge_aligned"
            elif max(abs(median_dx), abs(median_dy)) <= max(2.5 * float(grid_mm), 0.15 * max(span_x, span_y, float(grid_mm))):
                zone = "around"
            elif abs(median_dx) >= 1.25 * abs(median_dy):
                zone = "right" if median_dx > 0 else "left"
            elif abs(median_dy) >= 1.25 * abs(median_dx):
                zone = "top" if median_dy > 0 else "bottom"
            else:
                zone = "around"
    elif edge_corridor:
        zone = "edge_aligned"
    if radius <= 0.0:
        radius = max(2.5 * float(grid_mm), 0.35 * max(span_x, span_y, float(grid_mm)))

    hint = {
        "version": 1,
        "source": "derived_from_expert_module_geometry",
        "module_center_mm": [float(cx), float(cy)],
        "module_axis": axis,
        "anchor_relative_zone": zone,
        "edge_corridor": edge_corridor,
        "support_ring_radius_mm": float(radius),
        "median_offset_from_anchor_mm": [float(median_dx), float(median_dy)],
    }
    if anchor_xy is not None:
        hint["anchor_xy_mm"] = [float(anchor_xy[0]), float(anchor_xy[1])]
    return hint


def _annotate_module_region_hints(
    modules: List[Dict[str, Any]],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
) -> None:
    for m in modules:
        m["region_confidence"] = float(m.get("region_confidence", _compute_region_confidence(m, modules, board_bbox)))
    for m in modules:
        if not isinstance(m.get("shape_hint"), dict):
            m["shape_hint"] = _infer_module_shape_hint(m, comp_by_ref, board_bbox, grid_mm)
        m["module_subregion"] = str((m.get("shape_hint") or {}).get("anchor_relative_zone") or "around")

def _normalize_external_modules(
    data_modules: List[Dict[str, Any]],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, m in enumerate(data_modules or []):
        members = [str(x) for x in m.get("members", []) if str(x) in comp_by_ref]
        if len(members) < 1:
            continue
        mid = str(m.get("module_id") or f"M{idx+1:02d}")
        mtype = str(m.get("module_type") or _infer_module_type(members, comp_by_ref))
        anchor = str(m.get("anchor_ref") or choose_anchor(members, comp_by_ref))
        # Existing prior regions are derived runtime hints, not source truth.
        # They are intentionally discarded here so reprocessing a dataset can
        # upgrade old rule versions instead of silently preserving stale priors.
        legacy_bbox = (
            m.get("expert_bbox_mm")
            or m.get("bbox_mm")
            or m.get("region_bbox_mm")
            or _bbox_for_members(members, comp_by_ref, board_bbox, grid_mm)
        )
        if not legacy_bbox or len(legacy_bbox) != 4:
            legacy_bbox = _bbox_for_members(members, comp_by_ref, board_bbox, grid_mm)
        row = {
            "module_id": mid,
            "module_type": mtype,
            "anchor_ref": anchor,
            "members": members,
            "region_bbox_mm": [float(v) for v in legacy_bbox],
            "source": "json",
        }
        if isinstance(m.get("expert_region_label"), dict):
            row["expert_region_label"] = dict(m.get("expert_region_label"))
        out.append(row)
    return _renumber_modules(out)

def _renumber_modules(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modules = sorted(
        modules,
        key=lambda m: (
            MODULE_TYPE_ORDER.get(str(m.get("module_type", "misc")), 9),
            -len(m.get("members", [])),
            str(m.get("anchor_ref", "")),
        )
    )
    for i, m in enumerate(modules, 1):
        old_type = str(m.get("module_type", "misc"))
        m["module_id"] = f"M{i:02d}_{old_type}"
        m["module_order"] = i - 1
    return modules



# ---------------------------------------------------------------------------
# Fine-grained, inference-safe module partitioning (v2)
# ---------------------------------------------------------------------------

_INTERFACE_TYPE_TOKENS = (
    "conn_", "connector", "header", "terminal", "usb", "rj", "jack",
    "fpc", "ffc", "sma", "coax", "socket", "card_edge",
)
_INTERFACE_PREFIXES = {"J", "P", "JP", "CN", "CON", "CONN", "RJ", "X", "XA"}
_INTERFACE_NET_TOKENS = (
    "USB", "UART", "I2C", "SCL", "SDA", "SPI", "MOSI", "MISO", "SCK",
    "CAN", "RS485", "RS232", "SWD", "JTAG", "HDMI", "DP", "DM", "D+", "D-",
    "ETH", "MDI", "TX", "RX", "PCIE", "SATA", "MIPI", "GPIO",
)


def _semantic_value(comp: Any, key: str = "semantic_class") -> str:
    return str(getattr(comp, key, "") or "").strip().lower()


def _is_true_interface_connector(ref: str, comp: Any) -> bool:
    """Return True only for a physical connector/interface object.

    This deliberately excludes generic strings such as ``UNCONNECTED`` and
    excludes ordinary relays/ICs merely because they share an interface net.
    """
    typ = _component_type_text(comp)
    sem = _semantic_value(comp)
    role = _semantic_value(comp, "placement_role")
    pref = ref_prefix(ref)
    if typ.startswith("conn_") or any(tok in typ for tok in _INTERFACE_TYPE_TOKENS):
        return True
    if sem in {"interface", "mechanical_edge_interface", "rf"} and (
        pref in _INTERFACE_PREFIXES or "edge" in role or _has_boundary_constraint(comp)
    ):
        return True
    if pref in _INTERFACE_PREFIXES and _has_boundary_constraint(comp):
        return True
    return False


def _interface_signal_count(comp: Any) -> int:
    count = 0
    for net in _component_signal_nets(comp):
        n = str(net or "").upper()
        if any(tok in n for tok in _INTERFACE_NET_TOKENS):
            count += 1
    return count


def _is_core_anchor_component(ref: str, comp: Any) -> bool:
    pref = ref_prefix(ref)
    sem = _semantic_value(comp)
    typ = _component_type_text(comp)
    if _is_true_interface_connector(ref, comp) or _is_power_anchor_component(ref, comp):
        return False
    if sem == "core":
        return True
    if pref in {"U", "IC", "MCU"} and not any(t in typ for t in ("opto", "isolator", "driver", "regulator", "ldo", "buck", "boost")):
        return True
    return False


def _is_power_anchor_component(ref: str, comp: Any) -> bool:
    """Identify genuine power/relay anchors without promoting large passives.

    ``placement_role=anchor_large`` is an ordering heuristic supplied by the
    semantic annotator.  It is intentionally *not* treated as electrical power
    evidence here: axial resistors can be physically large while belonging to
    neither a power island nor an interface module.
    """
    pref = ref_prefix(ref)
    sem = _semantic_value(comp)
    typ = _component_type_text(comp)
    group = _semantic_value(comp, "functional_group")
    area = _component_area_mm2(comp)
    text = f"{ref} {typ} {sem} {group}".upper()
    if pref == "K" or "RELAY" in text:
        return True
    power_semantic = sem in {"power", "power_active"} or group == "power_supply"
    if power_semantic:
        if pref in {"L", "F"}:
            return True
        if pref in {"U", "IC"} and any(tok in text for tok in ("REG", "LDO", "BUCK", "BOOST", "PMIC", "DCDC", "DC-DC")):
            return True
        if pref == "Q" and area >= 10.0:
            return True
        if pref == "D" and area >= 35.0:
            return True
        if area >= 80.0:
            return True
    if _is_power_loop_component(ref, comp):
        if pref in {"L", "F"}:
            return True
        if pref == "Q" and area >= 10.0:
            return True
        if pref == "D" and area >= 35.0:
            return True
    return False


def _is_interface_support_component(ref: str, comp: Any) -> bool:
    if _is_true_interface_connector(ref, comp) or _is_power_anchor_component(ref, comp) or _is_core_anchor_component(ref, comp):
        return False
    sem = _semantic_value(comp)
    typ = _component_type_text(comp)
    pref = ref_prefix(ref)
    return bool(
        sem == "interface_support"
        or (_interface_signal_count(comp) > 0 and pref in {"R", "C", "D", "L", "FB"})
        or (_interface_signal_count(comp) > 0 and _is_interface_protection_like(ref, comp))
        or ("esd" in typ or "tvs" in typ)
    )


def _module_type_for_anchor(ref: str, comp: Any) -> str:
    if _is_true_interface_connector(ref, comp):
        return "interface"
    if _is_power_anchor_component(ref, comp):
        return "power_or_driver"
    if _is_clock_source_component(ref, comp):
        return "clock_local"
    if _is_core_anchor_component(ref, comp):
        return "core_ic"
    return "mixed"


def _shared_signal_weight(a: Any, b: Any) -> float:
    na = _component_signal_nets(a)
    nb = _component_signal_nets(b)
    common = na.intersection(nb)
    if not common:
        return 0.0
    # Small nets are more informative than broad shared buses.  A direct net
    # match remains the strongest feature, but it is not a hard requirement.
    return float(len(common))


def _support_score_for_anchor(ref: str, comp: Any, anchor_ref: str, anchor: Any) -> float:
    """Score an attachment only when it has a defensible local relationship."""
    shared = _shared_signal_weight(comp, anchor)
    score = 2.0 * shared
    mtype = _module_type_for_anchor(anchor_ref, anchor)
    pref = ref_prefix(ref)
    sem = _semantic_value(comp)
    if mtype == "interface":
        # A semantic ``interface_support`` label by itself is not enough: many
        # boards have multiple unrelated interfaces.  Require a direct shared
        # signal net and restrict the member to a passive/protection role.
        if shared <= 0.0:
            return 0.0
        is_passive_support = sem in {"interface_support", "passive"} and pref in {"R", "C", "FB", "L"}
        is_protection = _is_interface_protection_like(ref, comp) or any(tok in _component_type_text(comp) for tok in ("esd", "tvs"))
        if not (is_passive_support or is_protection):
            return 0.0
        score += 3.0 if (sem == "interface_support" or is_protection) else 1.0
        return float(score)
    if mtype == "power_or_driver":
        if _is_power_loop_component(ref, comp) or _is_decap_like(ref, comp):
            score += 2.5
        if sem in {"power_support", "power"}:
            score += 1.5
    elif mtype == "clock_local":
        if _is_clock_like_component(ref, comp):
            score += 3.0
    elif mtype == "core_ic":
        if _is_decap_like(ref, comp):
            score += 2.0
        if pref in {"R", "C"} and shared > 0:
            score += 1.0
    return float(score)


def _leftover_module_type(ref: str, comp: Any) -> str:
    if _is_power_anchor_component(ref, comp) or _is_power_loop_component(ref, comp):
        return "power_or_driver"
    if _is_clock_source_component(ref, comp) or _is_clock_like_component(ref, comp):
        return "clock_local"
    if _is_core_anchor_component(ref, comp):
        return "core_ic"
    if _is_true_interface_connector(ref, comp):
        return "interface"
    if ref_prefix(ref) in {"R", "C", "D", "L", "FB"}:
        return "passive_cluster"
    return "misc"


def _split_members_into_fine_modules(
    members: List[str],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
    *,
    source: str,
) -> List[Dict[str, Any]]:
    """Split a coarse electrical component into anchor-local modules.

    The procedure uses only refs, semantic labels, package/type, constraints
    and pad connectivity.  Expert coordinates are used solely when building
    the train-only legacy expert bbox, never when deciding membership or prior.
    """
    refs = [str(r) for r in members if str(r) in comp_by_ref]
    if not refs:
        return []
    anchors = [
        r for r in refs
        if _module_type_for_anchor(r, comp_by_ref[r]) != "mixed"
    ]
    # If no strong anchor is available, retain a compact semantic group rather
    # than creating a misleading interface module from a reference prefix.
    if not anchors:
        mtype = _infer_module_type(refs, comp_by_ref)
        anchor = choose_anchor(refs, comp_by_ref)
        return [{
            "module_id": "pending",
            "module_type": mtype,
            "anchor_ref": anchor,
            "members": sorted(refs, key=lambda r: (ref_prefix(r), r)),
            "region_bbox_mm": _bbox_for_members(refs, comp_by_ref, board_bbox, grid_mm),
            "source": source,
        }]

    # A connector, relay, IC or clock source forms an independent local module.
    # This prevents one edge connector from donating its edge prior to a whole
    # power/control island.
    buckets: Dict[str, List[str]] = {r: [r] for r in anchors}
    unassigned = [r for r in refs if r not in buckets]
    leftovers: List[str] = []
    for r in unassigned:
        comp = comp_by_ref[r]
        scores = [(_support_score_for_anchor(r, comp, a, comp_by_ref[a]), a) for a in anchors]
        best_score, best_anchor = max(scores, key=lambda item: (item[0], item[1]))
        # Require an actual relationship.  This avoids pulling arbitrary
        # passives toward a connector merely because that connector was first.
        if best_score >= 2.0:
            buckets[best_anchor].append(r)
        else:
            leftovers.append(r)

    out: List[Dict[str, Any]] = []
    for anchor in sorted(buckets):
        mem = sorted(set(buckets[anchor]), key=lambda r: (r != anchor, ref_prefix(r), r))
        mtype = _module_type_for_anchor(anchor, comp_by_ref[anchor])
        out.append({
            "module_id": "pending",
            "module_type": mtype,
            "anchor_ref": anchor,
            "members": mem,
            "region_bbox_mm": _bbox_for_members(mem, comp_by_ref, board_bbox, grid_mm),
            "source": source,
        })

    # Remaining small parts stay in compact type-local clusters.  They are
    # intentionally capped to keep a large unconnected passive cloud from
    # becoming a broad, high-confidence module prior.
    by_type: Dict[str, List[str]] = defaultdict(list)
    for r in leftovers:
        by_type[_leftover_module_type(r, comp_by_ref[r])].append(r)
    for mtype, group in sorted(by_type.items()):
        group = sorted(group, key=lambda r: (ref_prefix(r), r))
        for start in range(0, len(group), 8):
            mem = group[start:start + 8]
            anchor = choose_anchor(mem, comp_by_ref)
            out.append({
                "module_id": "pending",
                "module_type": mtype,
                "anchor_ref": anchor,
                "members": mem,
                "region_bbox_mm": _bbox_for_members(mem, comp_by_ref, board_bbox, grid_mm),
                "source": source,
            })
    return out


def _refine_modules(
    modules: List[Dict[str, Any]],
    comp_by_ref: Dict[str, Any],
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float,
) -> List[Dict[str, Any]]:
    """Return a duplicate-free, fine-grained module partition for every ref."""
    used: set[str] = set()
    coarse_groups: List[Tuple[List[str], str]] = []
    for m in modules:
        members = [str(r) for r in (m.get("members") or []) if str(r) in comp_by_ref and str(r) not in used]
        if members:
            used.update(members)
            coarse_groups.append((members, str(m.get("source") or "auto")))
    for ref in sorted(set(comp_by_ref) - used):
        coarse_groups.append(([ref], "auto_unassigned"))

    refined: List[Dict[str, Any]] = []
    assigned: set[str] = set()
    for members, source in coarse_groups:
        clean = [r for r in members if r not in assigned]
        split = _split_members_into_fine_modules(clean, comp_by_ref, board_bbox, grid_mm, source="refined_v2" if source else "refined_v2")
        for row in split:
            row["members"] = [r for r in row.get("members", []) if r not in assigned]
            if not row["members"]:
                continue
            assigned.update(row["members"])
            # Keep only compatible module types in the runtime feature schema.
            if row.get("module_type") not in MODULE_TYPE_ORDER:
                row["module_type"] = "misc"
            refined.append(row)

    # A final safety pass guarantees every remaining component has exactly one module.
    for ref in sorted(set(comp_by_ref) - assigned):
        comp = comp_by_ref[ref]
        refined.append({
            "module_id": "pending",
            "module_type": _leftover_module_type(ref, comp),
            "anchor_ref": ref,
            "members": [ref],
            "region_bbox_mm": _bbox_for_members([ref], comp_by_ref, board_bbox, grid_mm),
            "source": "refined_v2_fallback",
        })
    return _renumber_modules(refined)


def infer_modules(
    components: List[Any],
    nets: Optional[Dict[str, List[str]]] = None,
    *,
    board_bbox: Tuple[float, float, float, float],
    grid_mm: float = 1.0,
    data_modules: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    comp_by_ref = {c.ref: c for c in components}
    if data_modules:
        modules = _normalize_external_modules(data_modules, comp_by_ref, board_bbox, grid_mm)
        if modules:
            return _refine_modules(modules, comp_by_ref, board_bbox, grid_mm)

    # Build non-power net -> refs from component pads first.  If pad-level
    # connectivity is missing or too sparse, fall back to top-level nets so
    # module inference still works for schematic-only JSON exports.
    netrefs: Dict[str, set[str]] = defaultdict(set)
    for comp in components:
        for net, _xy in getattr(comp, "pads", []) or []:
            if is_power_net(net):
                continue
            netrefs[str(net)].add(comp.ref)

    comp_refs = set(comp_by_ref)
    top_netrefs = _top_level_netrefs(nets, comp_refs)
    pad_multi_net_count = sum(1 for refs in netrefs.values() if len(refs) >= 2)
    fallback_threshold = max(1, min(4, len(components) // 12))
    if top_netrefs and pad_multi_net_count < fallback_threshold:
        for net, refs in top_netrefs.items():
            netrefs[net].update(refs)

    n = max(1, len(components))
    max_fanout = max(6, min(22, int(max(6, n * 0.18))))
    adj: Dict[str, Counter] = {c.ref: Counter() for c in components}
    for net, refs0 in netrefs.items():
        refs = sorted(refs0)
        if len(refs) < 2 or len(refs) > max_fanout:
            continue
        weight = 1.0 / max(1, len(refs) - 1)
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                adj[refs[i]][refs[j]] += weight
                adj[refs[j]][refs[i]] += weight

    # Connected components over the filtered graph.
    seen = set()
    raw_modules: List[List[str]] = []
    for comp in components:
        r = comp.ref
        if r in seen:
            continue
        stack = [r]
        seen.add(r)
        group = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v, w in adj.get(u, {}).items():
                if w > 0 and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(group) >= 2:
            raw_modules.append(sorted(group))

    # If too few modules were found, split by strong anchors and attach passives to the
    # connected anchor with the highest edge weight.
    if len(raw_modules) <= 1:
        anchors = []
        for c in components:
            p = ref_prefix(c.ref)
            if p in {"U", "IC", "MCU", "J", "P", "CN", "CON", "L", "Q", "D", "SW"}:
                anchors.append(c.ref)
        anchors = sorted(set(anchors), key=lambda r: (PREFIX_ORDER.get(ref_prefix(r), 5), r))
        if len(anchors) >= 2:
            buckets = {a: [a] for a in anchors}
            for c in components:
                if c.ref in buckets:
                    continue
                scores = [(sum(adj[c.ref][x] for x in [a]), a) for a in anchors]
                scores = [(s, a) for s, a in scores if s > 0]
                if scores:
                    _s, a = max(scores)
                    buckets[a].append(c.ref)
            raw_modules = [sorted(v) for v in buckets.values() if len(v) >= 2]

    modules: List[Dict[str, Any]] = []
    for members in raw_modules:
        members = sorted(set(members), key=lambda r: (ref_prefix(r), r))
        if len(members) < 2:
            continue
        mtype = _infer_module_type(members, comp_by_ref)
        anchor = choose_anchor(members, comp_by_ref)
        bbox = _bbox_for_members(members, comp_by_ref, board_bbox, grid_mm)
        modules.append({
            "module_id": f"M{len(modules)+1:02d}_{mtype}",
            "module_type": mtype,
            "anchor_ref": anchor,
            "members": members,
            "region_bbox_mm": bbox,
            "source": "auto",
        })
    return _refine_modules(modules, comp_by_ref, board_bbox, grid_mm)


def _component_type_text(comp: Any) -> str:
    return str(getattr(comp, "type", "") or "").strip().lower()


def _component_semantic_text(comp: Any) -> str:
    vals = [
        getattr(comp, "semantic_class", ""),
        getattr(comp, "placement_role", ""),
        getattr(comp, "functional_group", ""),
        getattr(comp, "subzone", ""),
        getattr(comp, "side_preference", ""),
        getattr(comp, "region_type", ""),
    ]
    return " ".join(str(v or "").lower() for v in vals)


def _component_area_mm2(comp: Any) -> float:
    try:
        w, h = getattr(comp, "size_mm", (1.0, 1.0))
        return max(1e-3, float(w) * float(h))
    except Exception:
        return 1.0


def _component_all_nets(comp: Any) -> set:
    nets = set()
    for pad in (getattr(comp, "pads", None) or []):
        try:
            net = str(pad[0])
        except Exception:
            continue
        if net:
            nets.add(net)
    return nets


def _component_signal_nets(comp: Any) -> set:
    return {n for n in _component_all_nets(comp) if not is_power_net(str(n))}


def _is_ground_like_net(net: str) -> bool:
    n = str(net or "").strip().upper().strip("/")
    return n in {"", "GND", "GROUND", "AGND", "DGND", "PGND", "VSS", "VSSA", "0", "0V"} or n.endswith("GND")


def _is_supply_like_net(net: str) -> bool:
    return is_power_net(net) and not _is_ground_like_net(net)


def _is_decap_like(ref: str, comp: Any) -> bool:
    if ref_prefix(ref) != "C" and "cap" not in _component_type_text(comp):
        return False
    nets = _component_all_nets(comp)
    return any(_is_ground_like_net(n) for n in nets) and any(_is_supply_like_net(n) for n in nets)


def _component_attr_text(comp: Any, *attrs: str) -> str:
    return " ".join(str(getattr(comp, a, "") or "").strip().lower() for a in attrs)


def _is_testpoint_component(ref: str, comp: Any) -> bool:
    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    return ref_prefix(ref) == "TP" or "TESTPOINT" in text or "TEST POINT" in text


def _has_boundary_constraint(comp: Any) -> bool:
    if bool(getattr(comp, "must_touch_boundary", False)):
        return True
    if bool(getattr(comp, "fixed", False)) or bool(getattr(comp, "is_fixed", False)):
        return True
    if bool(getattr(comp, "must_be_on_edge", False)) or bool(getattr(comp, "edge_locked", False)) or bool(getattr(comp, "boundary_locked", False)):
        return True
    if getattr(comp, "allowed_sides", None):
        return True

    text = _component_attr_text(
        comp,
        "placement_role",
        "semantic_class",
        "side_preference",
        "region_type",
        "functional_group",
        "subzone",
    )
    boundary_tokens = (
        "edge_anchor",
        "boundary_anchor",
        "fixed_edge",
        "must_edge",
        "must-touch-boundary",
        "must_touch_boundary",
        "mechanical_edge_interface",
        "edge_",
        "board_edge",
        "boundary",
    )
    return any(tok in text for tok in boundary_tokens)


def _is_clock_source_component(ref: str, comp: Any) -> bool:
    pref = ref_prefix(ref)

    # Passive clock support parts (load caps, damping/series resistors, filters)
    # are not clock sources. They stay in phase 2 as critical local neighbors.
    if pref in {"C", "R", "L", "D", "FB", "F"}:
        return False

    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    if pref in {"Y", "X", "XO", "OSC"}:
        return True
    return any(k in text for k in ("CLOCK SOURCE", "CLK SOURCE", "CRYSTAL", "XTAL", "OSCILLATOR", " TCXO", " VCXO"))


def _is_clock_like_component(ref: str, comp: Any) -> bool:
    # Clock-related local support: crystal load caps, damping resistors,
    # terminations/matching parts. The source itself is handled as a phase-1 anchor.
    if _is_clock_source_component(ref, comp):
        return False
    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    return any(k in text for k in ("XTAL", "CRYSTAL", "OSC", "CLOCK", "CLK", "LOAD CAP", "LOAD_CAP"))


def _is_power_loop_component(ref: str, comp: Any) -> bool:
    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    if any(k in text for k in ("BUCK", "BOOST", "LDO", "REG", "DCDC", "PMIC", "POWER", "FEEDBACK", "FB", "COMP", "SWITCHNODE", "SW_NODE")):
        return True
    return ref_prefix(ref) in {"L", "D", "Q", "F"} and any(_is_supply_like_net(n) for n in _component_all_nets(comp))


def _is_interface_protection_like(ref: str, comp: Any) -> bool:
    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    return any(k in text for k in ("ESD", "TVS", "PROTECT", "FILTER", "FERRITE", "CHOKE", "CMCHOKE", "COMMONMODE", "COMMON_MODE", "TERM", "TERMINATION", "MATCH", "MATCHING"))


def _is_bridge_like_component(ref: str, comp: Any) -> bool:
    text = f"{ref} {_component_type_text(comp)} {_component_semantic_text(comp)}".upper()
    return any(k in text for k in ("LEVEL", "SHIFT", "TRANSLATOR", "TRANSCEIVER", "BUFFER", "DRIVER", "DRV", "MUX", "DEMUX", "ISOLATOR", "OPTO", "REPEATER", "BRIDGE", "SERIES", "FILTER"))


def _is_edge_hard_component(ref: str, comp: Any) -> bool:
    typ = _component_type_text(comp)
    pref = ref_prefix(ref)

    # Test points are weak-tail inspection objects, never hard-boundary anchors.
    if _is_testpoint_component(ref, comp):
        return False

    # Mounting holes / explicit mechanical anchors remain hard-boundary objects.
    if pref in {"H", "MH"}:
        return True
    role = str(getattr(comp, "placement_role", "") or "").lower()
    if role == "mechanical_anchor" or any(k in typ for k in ("mount", "hole")):
        return True

    # Connectors and UI/RF edge-facing parts only become phase-0 objects when
    # they carry an actual edge/boundary/fixed-side constraint.
    if not _has_boundary_constraint(comp):
        return False

    if pref in {"J", "P", "JP", "CN", "CON", "SW"}:
        return True
    sem = str(getattr(comp, "semantic_class", "") or "").lower()
    if sem in {"mechanical_edge_interface", "rf"}:
        return True
    return any(k in typ for k in ("conn", "header", "usb", "jack", "button", "switch", "led", "antenna", "rf"))


def _is_soft_interface_anchor_component(ref: str, comp: Any) -> bool:
    """Early-order soft interface anchors without imposing hard legality.

    Many real board connectors lack an explicit side_preference/allowed_sides in
    generated JSON.  Keeping them out of phase 0 means the model may not learn
    empty-board connector placement under suffix-only curriculum.  This function
    only affects ordering; actual boundary legality still comes from
    _is_edge_hard_component()/env masks.
    """
    if _is_testpoint_component(ref, comp):
        return False

    pref = ref_prefix(ref)
    typ = _component_type_text(comp)
    sem = str(getattr(comp, "semantic_class", "") or "").lower()
    role = str(getattr(comp, "placement_role", "") or "").lower()

    if pref not in {"J", "P", "JP", "CN", "CON", "CONN", "RJ", "X", "XA"}:
        return False

    if role in {"edge_anchor", "boundary_anchor", "interface_anchor"}:
        return True
    if sem in {"interface", "mechanical_edge_interface", "rf"}:
        return True
    return any(k in typ for k in ("conn", "header", "usb", "jack", "socket", "fpc", "ffc", "terminal", "rj"))


def _is_large_or_electrical_anchor(ref: str, comp: Any) -> bool:
    typ = _component_type_text(comp)
    sem = str(getattr(comp, "semantic_class", "") or "").lower()
    role = str(getattr(comp, "placement_role", "") or "").lower()
    pref = ref_prefix(ref)
    area = _component_area_mm2(comp)
    if _is_clock_source_component(ref, comp):
        return True
    # Do not let stale semantic/module metadata promote ordinary passives to
    # phase-1 anchors.  Older generated JSONs marked axial resistors and small
    # capacitors as anchor_large or singleton module anchors, which lets them
    # occupy scarce slots before the real IC/display/connector anchors.
    if role in {"anchor_large", "main_anchor", "core_anchor", "power_anchor"}:
        if (pref in {"R", "C", "FB"} and area < 45.0) or (pref == "D" and area < 35.0):
            return False
        if pref == "D" and area < 35.0 and not _is_power_loop_component(ref, comp):
            return False
        return True
    if str(getattr(comp, "module_role", "") or "").lower() == "anchor":
        if (pref in {"R", "C", "FB"} and area < 45.0) or (pref == "D" and area < 35.0):
            return False
        if pref == "D" and area < 35.0 and not _is_power_loop_component(ref, comp):
            return False
        return True
    if pref in {"U", "IC", "MCU"} or typ in {"chip", "ic", "mcu", "processor", "driver"}:
        return True
    if sem in {"core", "power", "power_active"}:
        return True
    if sem == "clock" and _is_clock_source_component(ref, comp):
        return True
    if _is_power_loop_component(ref, comp) and area >= 8.0:
        return True
    return area >= 45.0


def _is_high_priority_local(ref: str, comp: Any) -> bool:
    if getattr(comp, "anchor_ref", None):
        return True
    if getattr(comp, "critical_neighbors", None):
        return True
    return bool(
        _is_decap_like(ref, comp)
        or _is_clock_like_component(ref, comp)
        or _is_power_loop_component(ref, comp)
        or _is_interface_protection_like(ref, comp)
    )


def _module_connection_stats(comp_by_ref: Dict[str, Any], modules: List[Dict[str, Any]]) -> Tuple[Dict[str, set], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return signal nets, per-ref peer weights, and per-ref per-module weights."""
    signal_nets = {r: _component_signal_nets(c) for r, c in comp_by_ref.items()}
    net_to_refs: Dict[str, List[str]] = defaultdict(list)
    for ref, nets in signal_nets.items():
        for net in nets:
            net_to_refs[net].append(ref)
    peer_w: Dict[str, Dict[str, float]] = {r: defaultdict(float) for r in comp_by_ref}
    for net, refs in net_to_refs.items():
        if len(refs) < 2:
            continue
        if len(refs) > max(8, int(0.25 * max(1, len(comp_by_ref)))):
            continue
        w = 1.0 / math.sqrt(max(1, len(refs) - 1))
        for i, a in enumerate(refs):
            for b in refs[i + 1:]:
                peer_w[a][b] += w
                peer_w[b][a] += w
    ref_module: Dict[str, str] = {}
    for m in modules:
        mid = str(m.get("module_id") or "")
        for r in m.get("members", []) or []:
            ref_module[str(r)] = mid
    module_w: Dict[str, Dict[str, float]] = {r: defaultdict(float) for r in comp_by_ref}
    for r, peers in peer_w.items():
        for p, w in peers.items():
            mid = ref_module.get(p, "")
            if mid:
                module_w[r][mid] += float(w)
    return signal_nets, peer_w, module_w


def _module_members_by_ref(modules: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in modules:
        mid = str(m.get("module_id") or "")
        for r in m.get("members", []) or []:
            out[str(r)] = mid
    return out


def component_order_key(ref: str, module: Dict[str, Any], comp_by_ref: Dict[str, Any], original_index: Dict[str, int]) -> Tuple[int, float, float, int, str]:
    """Module-local order based on relationship readiness, not just prefix/type.

    Local order is only a tie-breaker for the global frontier sequence.  It puts
    the module anchor first, then critical/anchor-local parts, then power/clock/
    interface support, then connected ordinary members, then weak passives.
    """
    comp = comp_by_ref[ref]
    anchor = str(module.get("anchor_ref") or "").strip()
    pref = ref_prefix(ref)
    anchor_comp = comp_by_ref.get(anchor)
    anchor_nets = _component_signal_nets(anchor_comp) if anchor_comp is not None else set()
    nets = _component_signal_nets(comp)
    conn_to_anchor = len(nets.intersection(anchor_nets))
    crit_count = len(getattr(comp, "critical_neighbors", None) or ())
    area = _component_area_mm2(comp)
    if ref == anchor or str(getattr(comp, "module_role", "") or "").lower() == "anchor":
        role = 0
    elif getattr(comp, "anchor_ref", None) or crit_count > 0:
        role = 1
    elif _is_decap_like(ref, comp) or _is_clock_like_component(ref, comp) or _is_power_loop_component(ref, comp) or _is_interface_protection_like(ref, comp):
        role = 2
    elif conn_to_anchor > 0 or nets:
        role = 3
    elif pref in {"C", "R", "D", "L", "Q", "F"}:
        role = 4
    elif _is_bridge_like_component(ref, comp):
        role = 5
    else:
        role = 6
    return (role, -float(crit_count + conn_to_anchor), -float(area), original_index.get(ref, 10**9), ref)


def _component_phase(ref: str, comp: Any, module: Optional[Dict[str, Any]]) -> int:
    """Return the intended global placement phase 0..5."""
    anchor = str((module or {}).get("anchor_ref") or getattr(comp, "module_anchor_ref", "") or "").strip()
    if _is_testpoint_component(ref, comp):
        return 5
    if _is_edge_hard_component(ref, comp):
        return 0
    if _is_soft_interface_anchor_component(ref, comp):
        return 0
    if ref == anchor:
        pref = ref_prefix(ref)
        area = _component_area_mm2(comp)
        mtype = str((module or {}).get("module_type") or "").lower()
        if not (mtype in {"passive_cluster", "mixed", "misc"} and (pref in {"R", "C", "FB"} and area < 45.0) or (pref == "D" and area < 35.0)):
            return 1
    if _is_large_or_electrical_anchor(ref, comp):
        return 1
    if _is_high_priority_local(ref, comp):
        return 2
    if _is_bridge_like_component(ref, comp):
        return 4
    pref = ref_prefix(ref)
    if pref in {"R", "C", "L", "D", "Q", "F", "U", "IC", "MCU"} or str(getattr(comp, "module_id", "") or ""):
        return 3
    return 5


def _phase_name(phase: int) -> str:
    return {
        0: "hard_boundary",
        1: "module_anchor",
        2: "critical_local",
        3: "module_member_frontier",
        4: "bridge",
        5: "weak_tail",
    }.get(int(phase), "weak_tail")


def _module_frontier_sequence(
    modules: List[Dict[str, Any]],
    comp_by_ref: Dict[str, Any],
    original_sequence: List[str],
    *,
    return_meta: bool = False,
) -> Any:
    """Six-stage anchor-first / frontier sequence.

    Phase 0: fixed/hard boundary objects.
    Phase 1: module anchors and large electrical anchors.
    Phase 2: local critical parts around already established anchors.
    Phase 3: ordinary module members chosen by frontier readiness across modules.
    Phase 4: bridge components between modules/interfaces.
    Phase 5: weak tail / remaining passives.
    """
    original_index = {r: i for i, r in enumerate(original_sequence)}
    module_by_ref: Dict[str, Dict[str, Any]] = {}
    local_idx: Dict[str, int] = {}
    module_order: Dict[str, int] = {}
    anchors: set = set()
    for m in modules:
        order = int(m.get("module_order", len(module_order)))
        anchor = str(m.get("anchor_ref") or "").strip()
        if anchor and anchor in comp_by_ref:
            anchors.add(anchor)
        for j, r in enumerate(m.get("members", []) or []):
            r = str(r)
            if r not in comp_by_ref:
                continue
            module_by_ref[r] = m
            local_idx[r] = int(j)
            module_order[r] = order

    all_refs = list(comp_by_ref.keys())
    assigned_module_refs = {r for m in modules for r in (m.get("members", []) or []) if r in comp_by_ref}
    signal_nets, peer_w, module_w = _module_connection_stats(comp_by_ref, modules)
    ref_module_id = _module_members_by_ref(modules)

    phase_of: Dict[str, int] = {}
    for r in all_refs:
        phase = _component_phase(r, comp_by_ref[r], module_by_ref.get(r))
        if r in anchors and phase > 1:
            mtype = str((module_by_ref.get(r) or {}).get("module_type") or "").lower()
            # Singleton passive/misc modules are bookkeeping clusters, not true
            # establishment anchors. Keep them in local/member phases so large
            # ICs and UI/mechanical anchors reserve space first.
            pref = ref_prefix(r)
            area = _component_area_mm2(comp_by_ref[r])
            small_passive_singleton = (
                mtype in {"passive_cluster", "mixed", "misc"}
                and (
                    (pref in {"R", "C", "FB"} and area < 45.0)
                    or (pref == "D" and area < 35.0)
                )
            )
            if not small_passive_singleton:
                phase = 1
        # Components with meaningful signal ties to more than one other module are
        # better treated as bridge/frontier objects instead of being buried inside
        # one module's passive tail.  Keep hard anchors and critical local parts
        # in their earlier phases.
        my_mid = ref_module_id.get(r, "")
        cross_mods = [mid for mid, wt in module_w.get(r, {}).items() if mid and mid != my_mid and wt > 0.0]
        cross_strength = sum(float(module_w.get(r, {}).get(mid, 0.0)) for mid in cross_mods)
        if phase >= 3 and (len(cross_mods) >= 2 or cross_strength >= 0.75):
            phase = 4
        phase_of[r] = phase

    seq: List[str] = []
    seen: set = set()
    phase_assigned: Dict[str, List[str]] = {name: [] for name in (_phase_name(i) for i in range(6))}

    def add(r: str) -> None:
        if r in comp_by_ref and r not in seen:
            seq.append(r)
            seen.add(r)
            phase_assigned[_phase_name(phase_of.get(r, 5))].append(r)

    def size_score(r: str) -> float:
        return math.sqrt(max(1e-6, _component_area_mm2(comp_by_ref[r])))

    def placed_conn_strength(r: str, placed: set) -> float:
        return sum(float(peer_w.get(r, {}).get(p, 0.0)) for p in placed)

    def placed_module_conn_strength(r: str, placed: set) -> float:
        my_mid = ref_module_id.get(r, "")
        if not my_mid:
            return 0.0
        return sum(float(peer_w.get(r, {}).get(p, 0.0)) for p in placed if ref_module_id.get(p, "") == my_mid)

    def cross_module_conn_strength(r: str, placed: set) -> float:
        my_mid = ref_module_id.get(r, "")
        return sum(float(peer_w.get(r, {}).get(p, 0.0)) for p in placed if ref_module_id.get(p, "") and ref_module_id.get(p, "") != my_mid)

    def readiness_score(r: str, placed: set, target_phase: int) -> float:
        c = comp_by_ref[r]
        m = module_by_ref.get(r, {})
        module_anchor = str(getattr(c, "module_anchor_ref", "") or m.get("anchor_ref", "") or "").strip()
        explicit_anchor = str(getattr(c, "anchor_ref", "") or "").strip()
        crit = {str(x) for x in (getattr(c, "critical_neighbors", None) or ()) if str(x)}
        my_mid = ref_module_id.get(r, "")
        same_module_placed = sum(1 for p in placed if my_mid and ref_module_id.get(p, "") == my_mid)
        anchor_ready = 1.0 if ((module_anchor and module_anchor in placed) or (explicit_anchor and explicit_anchor in placed)) else 0.0
        critical_ready = float(sum(1 for p in placed if p in crit or (bool(explicit_anchor) and p == explicit_anchor)))
        conn = placed_conn_strength(r, placed)
        conn_mod = placed_module_conn_strength(r, placed)
        conn_cross = cross_module_conn_strength(r, placed)
        local_progress = min(1.0, same_module_placed / max(1.0, float(len((m or {}).get("members", []) or []))))
        edge_bonus = 1.0 if _is_edge_hard_component(r, c) else 0.0
        weak_tail_bonus = 1.0 if _is_testpoint_component(r, c) else 0.0
        large_bonus = min(1.0, size_score(r) / 8.0)
        phase = phase_of.get(r, 5)
        val = 0.0
        val += 5.0 * anchor_ready
        val += 3.0 * critical_ready
        val += 2.0 * conn_mod
        val += 1.2 * conn
        val += 0.7 * local_progress
        val += 0.5 * large_bonus
        if target_phase == 0:
            val += 4.0 * edge_bonus
        elif target_phase == 1:
            val += 3.0 * float(r in anchors or _is_large_or_electrical_anchor(r, c))
        elif target_phase == 2:
            val += 2.2 * float(_is_high_priority_local(r, c))
            val += 0.8 * anchor_ready
        elif target_phase == 3:
            val += 1.2 * local_progress + 1.0 * conn_mod
        elif target_phase == 4:
            val += 2.0 * float(_is_bridge_like_component(r, c)) + 1.5 * conn_cross
        elif target_phase == 5:
            val += 3.0 * weak_tail_bonus + 1.2 * conn + 0.2 * size_score(r)
        # Do not let a late-stage bridge/passive leap ahead of critical local parts
        # unless it has clear placed context.
        val -= 0.35 * max(0, phase - target_phase)
        return float(val)

    # Large boards can contain several hundred independently labelled
    # connectors/passives.  The exact dynamic frontier below is O(N^3) because
    # each candidate rescans the placed set; use a deterministic O(N log N)
    # phase-preserving approximation for those boards.  It preserves the same
    # hard-boundary -> anchors -> critical-local -> members ordering while
    # avoiding a preprocessing timeout that would leave part of a dataset stale.
    if len(all_refs) > 256:
        peer_total = {r: float(sum(peer_w.get(r, {}).values())) for r in all_refs}
        anchor_of = {
            r: str((module_by_ref.get(r) or {}).get("anchor_ref") or "").strip()
            for r in all_refs
        }
        for phase in range(6):
            pool = [r for r in all_refs if r not in seen and phase_of.get(r, 5) == phase]
            def fast_key(r: str) -> Tuple[int, int, float, float, int, str]:
                c = comp_by_ref[r]
                m = module_by_ref.get(r, {})
                anchor = anchor_of.get(r, "")
                is_anchor = int(r == anchor or r in anchors)
                is_critical = int(_is_high_priority_local(r, c))
                is_edge = int(_is_edge_hard_component(r, c))
                # Negative values put stronger establishment/local context first.
                if phase == 0:
                    stage = -is_edge
                elif phase == 1:
                    stage = -is_anchor
                elif phase == 2:
                    stage = -is_critical
                else:
                    stage = 0
                return (
                    stage,
                    module_order.get(r, 9999),
                    -peer_total.get(r, 0.0),
                    -size_score(r),
                    local_idx.get(r, 9999),
                    r,
                )
            for r in sorted(pool, key=fast_key):
                add(r)
        for r in original_sequence:
            add(r)
        for r in all_refs:
            add(r)
        meta = {
            "sequence_policy": "six_phase_anchor_frontier_v1",
            "sequence_mode": "scalable_static_frontier",
            "phase_counts": {k: len(v) for k, v in phase_assigned.items()},
            "phases": phase_assigned,
        }
        return (seq, meta) if return_meta else seq

    # Phase 0 and 1 are explicit establishment phases.
    for phase in (0, 1):
        pool = [r for r in all_refs if r not in seen and phase_of.get(r, 5) == phase]
        pool.sort(key=lambda r: (
            -readiness_score(r, set(seen), phase),
            module_order.get(r, 9999),
            local_idx.get(r, 9999),
            original_index.get(r, 999999),
            r,
        ))
        for r in pool:
            add(r)

    # Remaining stages are frontier expansions across all modules, not module-by-module.
    for phase in (2, 3, 4, 5):
        pool = {r for r in all_refs if r not in seen and phase_of.get(r, 5) == phase}
        if phase == 5:
            # The tail also absorbs any unclassified leftovers.
            pool |= {r for r in all_refs if r not in seen}
        while pool:
            placed = set(seen)
            def key(r: str) -> Tuple[float, float, float, int, int, int, str]:
                c = comp_by_ref[r]
                m = module_by_ref.get(r, {})
                anchor = str(getattr(c, "module_anchor_ref", "") or m.get("anchor_ref", "") or "").strip()
                anchor_rank = 0 if anchor and anchor in placed else 1
                crit_rank = 0 if any(str(x) in placed for x in (getattr(c, "critical_neighbors", None) or ())) else 1
                return (
                    -readiness_score(r, placed, phase),
                    -placed_conn_strength(r, placed),
                    -size_score(r),
                    anchor_rank,
                    crit_rank,
                    local_idx.get(r, 9999),
                    r,
                )
            pick = sorted(pool, key=key)[0]
            add(pick)
            pool.remove(pick)
        if phase == 5:
            break

    # Safety: preserve any omitted refs in original order.
    for r in original_sequence:
        add(r)
    for r in all_refs:
        add(r)

    meta = {
        "sequence_policy": "six_phase_anchor_frontier_v1",
        "phase_counts": {k: len(v) for k, v in phase_assigned.items()},
        "phases": phase_assigned,
    }
    return (seq, meta) if return_meta else seq


def export_task_modules_to_json(data: Dict[str, Any], task: Any) -> Dict[str, Any]:
    """Write task.modules and per-component module/prior/expert fields back to JSON.

    The exported active legacy fields mirror prior_region only.  Expert module
    bboxes are stored under expert_region_label and component.expert.module_region_bbox_mm.
    """
    modules = [dict(m) for m in (getattr(task, "modules", None) or [])]
    data["modules"] = modules
    graph = data.setdefault("graph", {})
    graph["modules"] = modules
    graph["module_region_policy"] = {
        "active_region_source": "prior_region",
        "expert_region_as_input": False,
        "leakage_checked": True,
        "prior_generator": "rule_v1",
    }
    meta = data.setdefault("meta", {})
    meta["region_policy"] = dict(graph["module_region_policy"])
    comp_json = {str(c.get("ref")): c for c in data.get("components", []) if isinstance(c, dict) and c.get("ref")}
    for c in getattr(task, "components", []) or []:
        row = comp_json.get(str(c.ref))
        if row is None:
            continue
        row["module_id"] = str(getattr(c, "module_id", "") or "")
        row["module_role"] = str(getattr(c, "module_role", "member") or "member")
        row["module_anchor_ref"] = getattr(c, "module_anchor_ref", None)
        row["module_order"] = int(getattr(c, "module_order", 0) or 0)
        row["module_local_order"] = int(getattr(c, "module_local_order", 0) or 0)
        if getattr(c, "module_region_bbox", None):
            row["module_region_bbox"] = [float(v) for v in c.module_region_bbox]
        row["module_region_confidence"] = float(getattr(c, "module_region_confidence", 1.0) or 1.0)
        row["module_region_source"] = str(getattr(c, "module_region_source", "prior_region") or "prior_region")
        shape_hint = dict(getattr(c, "module_shape_hint", {}) or {})
        if shape_hint:
            row["module_shape_hint"] = shape_hint
        else:
            row.pop("module_shape_hint", None)
        row["module_subregion"] = str(getattr(c, "module_subregion", "free") or "free")
        prior = row.setdefault("prior", {})
        if getattr(c, "module_region_bbox", None):
            prior["region_bbox_mm"] = [float(v) for v in c.module_region_bbox]
        prior["region_confidence"] = float(getattr(c, "module_region_confidence", 1.0) or 1.0)
        prior["region_source"] = str(getattr(c, "module_region_source", "prior_region") or "prior_region")
        prior["schema_version"] = RUNTIME_PRIOR_SCHEMA_VERSION
        prior["leakage_safe"] = True
        ex = row.setdefault("expert", {}) if isinstance(row.get("expert", {}), dict) else {}
        if getattr(c, "expert_module_region_bbox", None):
            ex["module_region_bbox_mm"] = [float(v) for v in c.expert_module_region_bbox]
            row["expert"] = ex
        # Mirror module-level region heatmaps when available.
        module_heatmap = None
        expert_heatmap = None
        for mm in modules:
            if str(mm.get("module_id", "")) == str(row.get("module_id", "")):
                module_heatmap = mm.get("prior_region_heatmap")
                expert_heatmap = mm.get("expert_region_heatmap")
                break
        row["module"] = {
            "module_id": row.get("module_id", ""),
            "anchor_ref": row.get("module_anchor_ref"),
            "region_bbox_mm": row.get("module_region_bbox"),
            "region_confidence": row.get("module_region_confidence"),
            "prior_region": {
                "bbox_mm": row.get("module_region_bbox"),
                "confidence": row.get("module_region_confidence"),
                "source": row.get("module_region_source"),
                "schema_version": RUNTIME_PRIOR_SCHEMA_VERSION,
                "leakage_safe": True,
            },
        }
        if shape_hint:
            row["module"]["shape_hint"] = shape_hint
        if module_heatmap is not None:
            row["module"]["prior_region_heatmap"] = dict(module_heatmap)
            row.setdefault("prior", {})["region_heatmap"] = dict(module_heatmap)
        if expert_heatmap is not None:
            row.setdefault("expert", {})["region_heatmap"] = dict(expert_heatmap)
    return data

def apply_modules_to_task(
    task: Any,
    data_modules: Optional[List[Dict[str, Any]]] = None,
    *,
    reorder_sequence: bool = True,
) -> Any:
    """Infer/attach modules, module regions and module-ordered component sequence.

    This mutates and returns ``task`` so existing call sites do not need to change.
    """
    original_sequence = list(getattr(task, "sequence", []) or [c.ref for c in task.components])
    original_index = {r: i for i, r in enumerate(original_sequence)}
    comp_by_ref = {c.ref: c for c in task.components}

    modules = infer_modules(
        task.components,
        getattr(task, "nets", {}) or {},
        board_bbox=tuple(task.bbox_mm),
        grid_mm=float(task.grid_mm),
        data_modules=data_modules,
    )

    # Normalize module member order and preserve any legacy region bbox as an expert label.
    # Then replace active region fields with leakage-safe prior_region.
    for m in modules:
        members = [r for r in m.get("members", []) if r in comp_by_ref]
        ordered = sorted(members, key=lambda r: component_order_key(r, m, comp_by_ref, original_index))
        m["members"] = ordered
        if "region_bbox_mm" not in m or not m.get("region_bbox_mm"):
            m["region_bbox_mm"] = _bbox_for_members(ordered, comp_by_ref, task.bbox_mm, task.grid_mm)

    modules = attach_prior_and_expert_regions(
        modules,
        comp_by_ref,
        tuple(task.bbox_mm),
        legacy_region_as_expert=True,
    )

    # Attach module metadata and derived hints to components.
    for m in modules:
        ordered = [r for r in m.get("members", []) if r in comp_by_ref]
        shape_hint = dict(m.get("shape_hint") or {})
        region_confidence = float(m.get("region_confidence", 1.0))
        module_subregion = str(m.get("module_subregion") or shape_hint.get("anchor_relative_zone") or "around")
        for local_idx, ref in enumerate(ordered):
            c = comp_by_ref[ref]
            c.module_id = str(m["module_id"])
            c.module_role = "anchor" if ref == m.get("anchor_ref") else "member"
            c.module_anchor_ref = str(m.get("anchor_ref") or "")
            c.module_order = int(m.get("module_order", 0))
            c.module_local_order = int(local_idx)
            c.module_region_bbox = tuple(float(v) for v in m["region_bbox_mm"])
            c.module_region_confidence = region_confidence
            c.module_region_source = str((m.get('prior_region') or {}).get('source') or 'prior_region')
            expert_label = m.get('expert_region_label') if isinstance(m.get('expert_region_label'), dict) else {}
            eb = expert_label.get('bbox_mm') if isinstance(expert_label, dict) else None
            c.expert_module_region_bbox = tuple(float(v) for v in eb) if eb and len(eb) == 4 else None
            c.module_shape_hint = dict(shape_hint)
            c.module_subregion = module_subregion

    assigned = set(r for m in modules for r in m.get("members", []))
    leftovers = [r for r in original_sequence if r in comp_by_ref and r not in assigned]

    if reorder_sequence:
        seq, seq_meta = _module_frontier_sequence(modules, comp_by_ref, original_sequence, return_meta=True)
        task.sequence = seq
        # Dataclass has no slots; attach diagnostics for debugging/training logs.
        task.sequence_meta = seq_meta
    task.modules = modules
    return task
