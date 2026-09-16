from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

POWER_NET_RE = re.compile(
    r"(^|[_\-])(VIN|VCC|VDD|VSS|VBAT|BATT|BAT|5V|3V3|3\.3V|12V|24V|AVDD|DVDD|VOUT|VREG|BOOST|BUCK|PWR|POWER|GND|PGND|AGND|DGND|VBUS)([_\-]|$)",
    re.IGNORECASE,
)
INTERFACE_NET_RE = re.compile(
    r"(^|[_\-])(USB|UART|TX|RX|SCL|SDA|I2C|SPI|MOSI|MISO|SCK|CLK|CS|CAN|SWD|JTAG|GPIO|ADC|DAC|PWM|D\+|D\-|DP|DM|IRQ|INT|SCL1|SDA1|SCL2|SDA2)([_\-]|$)",
    re.IGNORECASE,
)
CLOCK_NET_RE = re.compile(r"(^|[_\-])(CLK|CLOCK|XTAL|OSC|XIN|XOUT|MCLK)([_\-]|$)", re.IGNORECASE)
RESET_NET_RE = re.compile(r"(^|[_\-])(RST|RESET|EN|BOOT|PROG|PGM)([_\-]|$)", re.IGNORECASE)
RF_NET_RE = re.compile(r"(^|[_\-])(RF|ANT|ANTENNA|LNA|PA)([_\-]|$)", re.IGNORECASE)
GROUND_NET_RE = re.compile(r"(^|[_\-])(GND|PGND|AGND|DGND|EARTH)([_\-]|$)", re.IGNORECASE)
POWER_REF_RE = re.compile(r"^(L|TR|RL|K|BT|F|POW)", re.IGNORECASE)
INTERFACE_REF_RE = re.compile(r"^(J|P|JP|CON|CONN|RJ|X|XA)", re.IGNORECASE)
UI_REF_RE = re.compile(r"^(SW|LED|RST|PROG|S)$", re.IGNORECASE)
MECH_REF_RE = re.compile(r"^(H|MH|LOGO|G)$", re.IGNORECASE)
CORE_REF_RE = re.compile(r"^(U|IC|ESP|BMP|ARD|TMP|A)$", re.IGNORECASE)

SEMANTIC_REVIEW_VERSION = "semi_manual_semantic_v2"


# -----------------------------
# Low-level helpers
# -----------------------------

def _ref_prefix(ref: str) -> str:
    m = re.match(r"[A-Za-z_]+", ref or "")
    return (m.group(0) if m else (ref or "")).upper()


def _component_nets(comp: dict) -> List[str]:
    nets: List[str] = []
    for pad in comp.get("pads", []) or []:
        if isinstance(pad, dict):
            net = str(pad.get("net", "") or "").strip()
        elif isinstance(pad, (list, tuple)) and pad:
            net = str(pad[0] or "").strip()
        else:
            net = ""
        if net:
            nets.append(net)
    return nets


def _is_power_net(net: str) -> bool:
    return bool(POWER_NET_RE.search(net or ""))


def _is_interface_net(net: str) -> bool:
    return bool(INTERFACE_NET_RE.search(net or ""))


def _is_clock_net(net: str) -> bool:
    return bool(CLOCK_NET_RE.search(net or ""))


def _is_reset_net(net: str) -> bool:
    return bool(RESET_NET_RE.search(net or ""))


def _is_rf_net(net: str) -> bool:
    return bool(RF_NET_RE.search(net or ""))


def _is_ground_net(net: str) -> bool:
    return bool(GROUND_NET_RE.search(net or ""))


def _safe_size(comp: dict) -> Tuple[float, float]:
    size = comp.get("size_mm", [0.0, 0.0])
    try:
        return float(size[0]), float(size[1])
    except Exception:
        return 0.0, 0.0


def _xy(comp: dict) -> Tuple[float, float]:
    expert = comp.get("expert", {}) or {}
    xy = expert.get("xy_mm", [0.0, 0.0])
    try:
        return float(xy[0]), float(xy[1])
    except Exception:
        return 0.0, 0.0


def _boundary_axis_value(side: str, x: float, y: float) -> float:
    return float(y) if side in {"edge_left", "edge_right"} else float(x)


def _component_bbox_diag(comp: dict) -> float:
    w, h = _safe_size(comp)
    return math.hypot(w, h)


def _is_anchor_semantic(semantic_class: str, functional_group: str, comp: dict) -> bool:
    ctype = (comp.get("type") or "").lower()
    if semantic_class in {"core", "power", "interface", "rf", "clock"}:
        return True
    if functional_group in {"control_processing", "power_supply", "interface_io", "timing", "rf_frontend"}:
        return True
    return ctype == 'chip'


def _infer_subzone_from_xy(comp_xy: Tuple[float, float], anchor_xy: Tuple[float, float]) -> str:
    dx = float(comp_xy[0] - anchor_xy[0])
    dy = float(comp_xy[1] - anchor_xy[1])
    adx, ady = abs(dx), abs(dy)
    if max(adx, ady) < 1e-6:
        return 'around'
    if max(adx, ady) <= 1.25 * min(adx, ady):
        return 'around'
    if adx > ady:
        return 'right' if dx > 0 else 'left'
    return 'top' if dy > 0 else 'bottom'


def populate_layout_semantic_fields(data: dict, overwrite: bool = False) -> dict:
    comps = data.get('components', []) or []
    if not comps:
        return refresh_semantic_meta(data)

    comp_by_ref = {str(c.get('ref', '')): c for c in comps}
    sem_class = {ref: (c.get('semantic_class') or (c.get('semantic') or {}).get('semantic_class') or infer_semantic_class(c)) for ref, c in comp_by_ref.items()}
    fgroup = {ref: (c.get('functional_group') or (c.get('semantic') or {}).get('functional_group') or infer_functional_group(c, sem_class[ref])) for ref, c in comp_by_ref.items()}
    region = {ref: (c.get('region_type') or (c.get('semantic') or {}).get('region_type') or 'free') for ref, c in comp_by_ref.items()}
    side = {ref: (c.get('side_preference') or (c.get('semantic') or {}).get('side_preference') or region[ref] or 'free') for ref, c in comp_by_ref.items()}
    neighbors = {ref: list(c.get('critical_neighbors') or (c.get('semantic') or {}).get('critical_neighbors') or []) for ref, c in comp_by_ref.items()}
    align_group = {ref: (c.get('align_group') or (c.get('semantic') or {}).get('align_group')) for ref, c in comp_by_ref.items()}
    coords = {ref: _xy(c) for ref, c in comp_by_ref.items()}

    anchor_candidates = {ref for ref, c in comp_by_ref.items() if _is_anchor_semantic(sem_class[ref], fgroup[ref], c)}

    anchors: Dict[str, str | None] = {}
    for ref, comp in comp_by_ref.items():
        cur = comp.get('anchor_ref') or (comp.get('semantic') or {}).get('anchor_ref')
        if cur and not overwrite:
            anchors[ref] = str(cur)
            continue
        if ref in anchor_candidates and sem_class[ref] in {'core', 'power', 'interface', 'rf', 'clock'}:
            anchors[ref] = ref
            continue
        chosen = None
        for nb in neighbors.get(ref, []):
            if nb not in comp_by_ref:
                continue
            if nb in anchor_candidates and (fgroup.get(nb) == fgroup.get(ref) or sem_class[nb] in {'core', 'power', 'interface', 'rf', 'clock'}):
                chosen = nb
                break
        if chosen is None:
            same_group = [r for r in comp_by_ref if r != ref and fgroup.get(r) == fgroup.get(ref) and r in anchor_candidates]
            if same_group:
                chosen = min(same_group, key=lambda r: ((coords[r][0] - coords[ref][0]) ** 2 + (coords[r][1] - coords[ref][1]) ** 2, r))
        anchors[ref] = chosen

    subzones: Dict[str, str] = {}
    for ref, comp in comp_by_ref.items():
        cur = comp.get('subzone') or (comp.get('semantic') or {}).get('subzone')
        if cur and not overwrite:
            subzones[ref] = str(cur)
            continue
        anchor_ref = anchors.get(ref)
        if not anchor_ref or anchor_ref == ref or anchor_ref not in coords:
            subzones[ref] = 'free' if sem_class[ref] in {'core', 'interface'} else 'around'
        else:
            subzones[ref] = _infer_subzone_from_xy(coords[ref], coords[anchor_ref])

    side_groups: Dict[str, str | None] = {}
    boundary_order: Dict[str, int | None] = {}
    raw_groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for ref, comp in comp_by_ref.items():
        pref = side.get(ref, 'free')
        if pref not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            continue
        prefix = _ref_prefix(ref)
        group_key = (pref, fgroup.get(ref, 'misc'), align_group.get(ref) or prefix)
        raw_groups[group_key].append(ref)
    for (pref, fg, gid), refs in raw_groups.items():
        refs_sorted = sorted(refs, key=lambda r: (_boundary_axis_value(pref, *coords[r]), r))
        group_name = f"{fg}:{gid}:{pref}"
        for idx, ref in enumerate(refs_sorted):
            comp = comp_by_ref[ref]
            cur_group = comp.get('same_side_group') or (comp.get('semantic') or {}).get('same_side_group')
            cur_order = comp.get('boundary_order') if comp.get('boundary_order', None) not in (None, '') else (comp.get('semantic') or {}).get('boundary_order')
            side_groups[ref] = group_name if (overwrite or not cur_group) else str(cur_group)
            boundary_order[ref] = idx if (overwrite or cur_order in (None, '')) else int(cur_order)

    for ref, comp in comp_by_ref.items():
        sem = comp.setdefault('semantic', {})
        review = comp.setdefault('semantic_review', {})
        auto = review.setdefault('auto_seed', {})
        if overwrite or (comp.get('anchor_ref', None) in (None, '')):
            comp['anchor_ref'] = anchors.get(ref)
        if overwrite or (sem.get('anchor_ref', None) in (None, '')):
            sem['anchor_ref'] = anchors.get(ref)
        auto['anchor_ref'] = anchors.get(ref)
        if overwrite or (comp.get('subzone', None) in (None, '')):
            comp['subzone'] = subzones.get(ref, 'free')
        if overwrite or (sem.get('subzone', None) in (None, '')):
            sem['subzone'] = subzones.get(ref, 'free')
        auto['subzone'] = subzones.get(ref, 'free')
        if side_groups.get(ref) is not None and (overwrite or (comp.get('same_side_group', None) in (None, ''))):
            comp['same_side_group'] = side_groups[ref]
        if side_groups.get(ref) is not None and (overwrite or (sem.get('same_side_group', None) in (None, ''))):
            sem['same_side_group'] = side_groups[ref]
        if side_groups.get(ref) is not None:
            auto['same_side_group'] = side_groups[ref]
        if boundary_order.get(ref) is not None and (overwrite or comp.get('boundary_order', None) in (None, '')):
            comp['boundary_order'] = int(boundary_order[ref])
        if boundary_order.get(ref) is not None and (overwrite or sem.get('boundary_order', None) in (None, '')):
            sem['boundary_order'] = int(boundary_order[ref])
        if boundary_order.get(ref) is not None:
            auto['boundary_order'] = int(boundary_order[ref])
        cands = review.setdefault('candidate_labels', {})
        cands['subzone'] = ['left', 'right', 'top', 'bottom', 'around', 'free']
        if side_groups.get(ref) is not None:
            cands['same_side_group'] = [side_groups[ref]]
        if boundary_order.get(ref) is not None:
            cands['boundary_order'] = [int(boundary_order[ref])]
    refresh_semantic_meta(data)
    meta = data.setdefault('meta', {})
    meta.setdefault('semantic_annotation', {})['layout_semantic_fields'] = ['anchor_ref', 'subzone', 'same_side_group', 'boundary_order']
    return data


def _nearest_edge(x: float, y: float, bbox: List[float]) -> Tuple[str, float]:
    x0, y0, x1, y1 = bbox
    dists = {
        "edge_left": abs(x - x0),
        "edge_right": abs(x1 - x),
        "edge_bottom": abs(y - y0),
        "edge_top": abs(y1 - y),
    }
    side = min(dists, key=dists.get)
    return side, float(dists[side])


def _net_profile(nets: Iterable[str]) -> Dict[str, int]:
    nets = list(nets)
    return {
        "power": sum(1 for n in nets if _is_power_net(n)),
        "power_non_ground": sum(1 for n in nets if _is_power_net(n) and not _is_ground_net(n)),
        "interface": sum(1 for n in nets if _is_interface_net(n)),
        "clock": sum(1 for n in nets if _is_clock_net(n)),
        "reset": sum(1 for n in nets if _is_reset_net(n)),
        "rf": sum(1 for n in nets if _is_rf_net(n)),
        "ground": sum(1 for n in nets if _is_ground_net(n)),
        "signal": sum(1 for n in nets if not _is_ground_net(n)),
        "total": len(nets),
    }


def _board_split_from_path(path_like: str) -> str:
    s = str(path_like).replace('\\', '/')
    if '/seq_k15_20_train/' in s:
        return 'seq_k15_20_train'
    if '/seq_k15_20_test/' in s:
        return 'seq_k15_20_test'
    if '/seq_test/' in s:
        return 'seq_test'
    if '/seq/' in s:
        return 'seq'
    return 'unknown'


# -----------------------------
# Rule-based seed label inference
# -----------------------------

def infer_semantic_class(comp: dict) -> str:
    ctype = (comp.get("type") or "").lower()
    footprint = (comp.get("footprint") or "").lower()
    ref = _ref_prefix(comp.get("ref", ""))
    nets = _component_nets(comp)
    prof = _net_profile(nets)

    if ctype.startswith("mech_") or MECH_REF_RE.match(ref):
        return "mechanical"
    if ctype in {"ui_button", "ui_led"} or UI_REF_RE.match(ref):
        return "ui"
    if ctype == "conn_rf" or "antenna" in footprint or prof["rf"] > 0:
        return "rf"
    if ctype.startswith("conn_") or INTERFACE_REF_RE.match(ref):
        if ctype in {"conn_barrel_jack", "conn_terminal_block"} or prof["power_non_ground"] >= max(1, prof["interface"]):
            return "power"
        return "interface"
    if ctype == "inductor" or POWER_REF_RE.match(ref) or "transform" in footprint or "inductor" in footprint:
        return "power"
    if ctype in {"diode", "transistor"} and (prof["power"] >= 1 or prof["rf"] >= 1):
        return "power" if prof["power"] >= prof["rf"] else "rf"
    if ctype == "chip":
        if prof["rf"] >= 1:
            return "rf"
        if prof["power_non_ground"] >= 2 and prof["interface"] <= 1 and prof["clock"] == 0:
            return "power"
        return "core"
    if ctype == "misc":
        if "crystal" in footprint or prof["clock"] >= 1:
            return "clock"
        if "transform" in footprint or prof["power_non_ground"] >= 2:
            return "power"
        return "support"
    if ctype in {"capacitor", "resistor", "diode", "transistor"}:
        if prof["power_non_ground"] >= 1 and prof["interface"] == 0:
            return "power_support"
        if prof["interface"] >= 1:
            return "interface_support"
        if prof["clock"] >= 1:
            return "clock"
        return "passive"
    if CORE_REF_RE.match(ref):
        return "core"
    return "support"


def infer_functional_group(comp: dict, semantic_class: str) -> str:
    nets = _component_nets(comp)
    prof = _net_profile(nets)
    ctype = (comp.get("type") or "").lower()
    footprint = (comp.get("footprint") or "").lower()

    if semantic_class == "mechanical":
        return "mechanics"
    if semantic_class == "ui":
        return "user_interface"
    if semantic_class == "rf":
        return "rf_frontend"
    if semantic_class == "interface":
        return "interface_io"
    if semantic_class in {"power", "power_support"}:
        return "power_supply"
    if semantic_class == "clock":
        return "timing"
    if semantic_class == "core":
        if "memory" in footprint or "flash" in footprint or "eeprom" in footprint:
            return "memory"
        return "control_processing"
    if semantic_class == "interface_support":
        return "interface_io"
    if ctype in {"diode", "transistor"} and (prof["power_non_ground"] >= 1 or prof["reset"] >= 1):
        return "protection"
    if prof["clock"] >= 1:
        return "timing"
    if prof["interface"] >= 1:
        return "interface_io"
    if prof["power_non_ground"] >= 1:
        return "power_supply"
    return "passive_support"


def infer_region_type(comp: dict, semantic_class: str, functional_group: str, bbox: List[float]) -> Tuple[str, str]:
    x, y = _xy(comp)
    side, dist = _nearest_edge(x, y, bbox)
    x0, y0, x1, y1 = bbox
    w = max(1e-6, x1 - x0)
    h = max(1e-6, y1 - y0)
    edge_band = 0.14 * min(w, h)
    near_edge = dist <= edge_band

    if semantic_class in {"mechanical", "ui", "interface", "rf"}:
        return side, side
    if functional_group == "power_supply":
        if near_edge:
            return side, side
        return "free", side
    if semantic_class == "core" or functional_group in {"control_processing", "memory", "timing"}:
        return "core", "core"
    if semantic_class in {"passive", "power_support", "interface_support", "support", "clock"}:
        return (side if near_edge else "free"), side
    return (side if near_edge else "free"), side


def _net_weight(net: str, comps_on_net: List[str], comp_classes: Dict[str, str]) -> float:
    degree = len(set(comps_on_net))
    if degree <= 1:
        return 0.0
    score = 1.0 + 0.25 * max(0, degree - 2)
    if _is_power_net(net):
        score += 1.0
    if _is_interface_net(net):
        score += 0.8
    if _is_clock_net(net) or _is_reset_net(net):
        score += 0.8
    if _is_rf_net(net):
        score += 1.0
    if _is_ground_net(net):
        score -= 0.4
    classes = {comp_classes.get(r, "support") for r in comps_on_net}
    if "core" in classes and "interface" in classes:
        score += 0.6
    if "core" in classes and ("power" in classes or "power_support" in classes):
        score += 0.4
    return max(0.25, score)


# -----------------------------
# Human/semi-human review metadata
# -----------------------------

def _semantic_rule_hits(comp: dict, semantic_class: str, prof: Dict[str, int]) -> List[str]:
    ref = _ref_prefix(comp.get('ref', ''))
    ctype = (comp.get('type') or '').lower()
    footprint = (comp.get('footprint') or '').lower()
    hits: List[str] = []

    if ctype.startswith('mech_') and semantic_class == 'mechanical':
        hits.append('type:mechanical')
    if ctype.startswith('conn_') and semantic_class in {'interface', 'power', 'rf'}:
        hits.append('type:connector')
    if ctype == 'chip' and semantic_class in {'core', 'power', 'rf'}:
        hits.append('type:chip')
    if ctype in {'capacitor', 'resistor', 'diode', 'transistor'} and semantic_class in {'passive', 'power_support', 'interface_support', 'clock'}:
        hits.append('type:passive_family')
    if MECH_REF_RE.match(ref) and semantic_class == 'mechanical':
        hits.append('ref_prefix:mechanical')
    if UI_REF_RE.match(ref) and semantic_class == 'ui':
        hits.append('ref_prefix:ui')
    if INTERFACE_REF_RE.match(ref) and semantic_class in {'interface', 'power'}:
        hits.append('ref_prefix:connector')
    if POWER_REF_RE.match(ref) and semantic_class == 'power':
        hits.append('ref_prefix:power')
    if CORE_REF_RE.match(ref) and semantic_class == 'core':
        hits.append('ref_prefix:core')
    if prof['rf'] > 0 and semantic_class == 'rf':
        hits.append('nets:rf')
    if prof['interface'] > 0 and semantic_class in {'interface', 'interface_support'}:
        hits.append('nets:interface')
    if prof['power_non_ground'] > 0 and semantic_class in {'power', 'power_support'}:
        hits.append('nets:power')
    if prof['clock'] > 0 and semantic_class == 'clock':
        hits.append('nets:clock')
    if 'antenna' in footprint and semantic_class == 'rf':
        hits.append('footprint:antenna')
    if 'transform' in footprint and semantic_class == 'power':
        hits.append('footprint:transformer')
    if 'crystal' in footprint and semantic_class == 'clock':
        hits.append('footprint:crystal')
    return hits


def _group_rule_hits(functional_group: str, semantic_class: str, prof: Dict[str, int], footprint: str) -> List[str]:
    hits: List[str] = []
    if functional_group == 'power_supply' and semantic_class in {'power', 'power_support'}:
        hits.append('group_from_semantic:power')
    if functional_group == 'interface_io' and semantic_class in {'interface', 'interface_support'}:
        hits.append('group_from_semantic:interface')
    if functional_group in {'control_processing', 'memory'} and semantic_class == 'core':
        hits.append('group_from_semantic:core')
    if functional_group == 'timing' and (semantic_class == 'clock' or prof['clock'] > 0):
        hits.append('group_from_clock')
    if functional_group == 'memory' and any(k in footprint for k in ['memory', 'flash', 'eeprom']):
        hits.append('footprint:memory')
    if functional_group == 'rf_frontend' and semantic_class == 'rf':
        hits.append('group_from_rf')
    return hits


def _region_rule_hits(region_type: str, semantic_class: str, functional_group: str, side: str, near_edge: bool) -> List[str]:
    hits: List[str] = []
    if semantic_class in {'interface', 'ui', 'mechanical', 'rf'} and region_type == side:
        hits.append('semantic_requires_edge_side')
    if semantic_class == 'core' and region_type == 'core':
        hits.append('semantic_requires_core')
    if functional_group in {'control_processing', 'memory', 'timing'} and region_type == 'core':
        hits.append('functional_group_requires_core')
    if functional_group == 'power_supply' and region_type in {'free', side}:
        hits.append('power_prefers_edge_or_free')
    if near_edge and region_type == side:
        hits.append('expert_position_near_edge')
    return hits


def _candidate_semantic_labels(comp: dict, prof: Dict[str, int], current: str) -> List[str]:
    cand: List[str] = [current]
    ctype = (comp.get('type') or '').lower()
    ref = _ref_prefix(comp.get('ref', ''))
    if ctype.startswith('conn_') or INTERFACE_REF_RE.match(ref):
        cand.extend(['interface', 'power'])
    if ctype == 'chip' or CORE_REF_RE.match(ref):
        cand.extend(['core', 'power', 'rf'])
    if ctype in {'capacitor', 'resistor', 'diode', 'transistor'}:
        cand.extend(['passive', 'power_support', 'interface_support'])
    if prof['clock'] > 0:
        cand.append('clock')
    if prof['rf'] > 0:
        cand.append('rf')
    seen = set()
    out: List[str] = []
    for x in cand:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out[:5]


def _candidate_functional_groups(semantic_class: str, current: str) -> List[str]:
    cand = [current]
    mapping = {
        'core': ['control_processing', 'memory'],
        'power': ['power_supply'],
        'power_support': ['power_supply', 'protection'],
        'interface': ['interface_io'],
        'interface_support': ['interface_io'],
        'rf': ['rf_frontend'],
        'clock': ['timing'],
        'ui': ['user_interface'],
        'mechanical': ['mechanics'],
        'passive': ['passive_support'],
        'support': ['passive_support'],
    }
    cand.extend(mapping.get(semantic_class, ['passive_support']))
    out: List[str] = []
    seen = set()
    for x in cand:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out[:4]


def _candidate_region_types(current: str, side: str, semantic_class: str) -> List[str]:
    cand = [current]
    if side:
        cand.append(side)
    if semantic_class == 'core':
        cand.append('core')
        cand.append('free')
    elif semantic_class in {'interface', 'ui', 'mechanical', 'rf'}:
        cand.extend([side, 'free'])
    else:
        cand.extend(['free', side])
    out: List[str] = []
    seen = set()
    for x in cand:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:4]


def _infer_confidence_and_review(comp: dict, semantic_class: str, functional_group: str, region_type: str, side: str, bbox: List[float]) -> Dict[str, Any]:
    nets = _component_nets(comp)
    prof = _net_profile(nets)
    x, y = _xy(comp)
    _, dist = _nearest_edge(x, y, bbox)
    x0, y0, x1, y1 = bbox
    edge_band = 0.14 * max(1e-6, min(x1 - x0, y1 - y0))
    near_edge = dist <= edge_band
    footprint = (comp.get('footprint') or '').lower()
    ctype = (comp.get('type') or '').lower()

    semantic_hits = _semantic_rule_hits(comp, semantic_class, prof)
    group_hits = _group_rule_hits(functional_group, semantic_class, prof, footprint)
    region_hits = _region_rule_hits(region_type, semantic_class, functional_group, side, near_edge)

    # Ambiguity / conflict checks
    ambiguity: List[str] = []
    if semantic_class in {'interface', 'power'} and prof['interface'] > 0 and prof['power_non_ground'] > 0:
        ambiguity.append('mixed_interface_and_power_nets')
    if semantic_class == 'core' and ctype != 'chip' and not CORE_REF_RE.match(_ref_prefix(comp.get('ref', ''))):
        ambiguity.append('core_without_chip_or_core_ref')
    if semantic_class in {'interface', 'ui', 'mechanical', 'rf'} and region_type == 'core':
        ambiguity.append('edge_oriented_part_labeled_core')
    if semantic_class == 'core' and region_type.startswith('edge_'):
        ambiguity.append('core_part_labeled_edge')
    if semantic_class in {'support', 'other'}:
        ambiguity.append('generic_semantic_class')
    if functional_group == 'passive_support' and semantic_class in {'power_support', 'interface_support'}:
        ambiguity.append('generic_group_for_specific_support_part')
    if region_type == 'free' and semantic_class in {'interface', 'ui', 'mechanical', 'rf'}:
        ambiguity.append('edge_oriented_part_labeled_free')
    if not semantic_hits:
        ambiguity.append('few_semantic_rule_hits')

    score = 0.55
    score += min(0.28, 0.08 * len(semantic_hits))
    score += min(0.14, 0.05 * len(group_hits))
    score += min(0.12, 0.04 * len(region_hits))
    if ctype.startswith('conn_') or ctype == 'chip':
        score += 0.06
    if len(ambiguity) >= 1:
        score -= 0.10
    if len(ambiguity) >= 2:
        score -= 0.08
    if len(ambiguity) >= 3:
        score -= 0.06
    if semantic_class in {'interface', 'ui', 'mechanical', 'rf'} and near_edge:
        score += 0.04
    if semantic_class == 'core' and region_type == 'core':
        score += 0.04
    score = max(0.05, min(0.98, score))

    relaxed_classes = {'passive', 'mechanical', 'ui'}
    needs_review = bool(score < 0.62 or len(ambiguity) >= 2 or semantic_class in {'support', 'other'} or (region_type == 'free' and semantic_class in {'interface', 'rf'}))
    if semantic_class in relaxed_classes and len(ambiguity) == 0 and score >= 0.55:
        needs_review = False
    if not needs_review:
        priority = 'low'
    elif semantic_class in {'core', 'power', 'interface', 'rf'} or functional_group in {'control_processing', 'power_supply', 'interface_io', 'rf_frontend'}:
        priority = 'high'
    elif semantic_class in {'power_support', 'interface_support', 'clock'} or len(ambiguity) >= 2:
        priority = 'medium'
    else:
        priority = 'low'

    evidence = {
        'ref_prefix': _ref_prefix(comp.get('ref', '')),
        'component_type': ctype,
        'footprint': comp.get('footprint', ''),
        'net_profile': prof,
        'expert_side': side,
        'expert_near_edge': near_edge,
        'semantic_rule_hits': semantic_hits,
        'functional_group_rule_hits': group_hits,
        'region_rule_hits': region_hits,
        'ambiguity_flags': ambiguity,
    }

    return {
        'auto_confidence': round(score, 3),
        'needs_review': needs_review,
        'review_priority': priority,
        'evidence': evidence,
        'candidate_labels': {
            'semantic_class': _candidate_semantic_labels(comp, prof, semantic_class),
            'functional_group': _candidate_functional_groups(semantic_class, functional_group),
            'region_type': _candidate_region_types(region_type, side, semantic_class),
        },
    }


def refresh_semantic_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    comps = data.get('components', []) or []
    meta = data.setdefault('meta', {})
    sem = meta.setdefault('semantic_annotation', {})
    sem.update({
        'version': SEMANTIC_REVIEW_VERSION,
        'description': 'Seed semantic labels with review metadata for human / semi-human curation. Top-level semantic fields are editable and can be reviewed through exported CSV manifests.',
        'fields': [
            'semantic_class',
            'region_type',
            'functional_group',
            'side_preference',
            'align_group',
            'anchor_ref',
            'subzone',
            'same_side_group',
            'boundary_order',
            'critical_nets',
            'critical_neighbors',
            'semantic_review',
        ],
        'class_counts': dict(Counter((c.get('semantic_class') or (c.get('semantic') or {}).get('semantic_class') or 'other') for c in comps)),
        'group_counts': dict(Counter((c.get('functional_group') or (c.get('semantic') or {}).get('functional_group') or 'misc') for c in comps)),
        'region_counts': dict(Counter((c.get('region_type') or (c.get('semantic') or {}).get('region_type') or 'free') for c in comps)),
        'review_status_counts': dict(Counter(((c.get('semantic_review') or {}).get('review_status') or 'untracked') for c in comps)),
        'needs_review_count': int(sum(1 for c in comps if bool((c.get('semantic_review') or {}).get('needs_review', False)))),
        'manual_override_count': int(sum(1 for c in comps if bool((c.get('semantic_review') or {}).get('manual_overrides')))),
    })
    return data


def annotate_board(data: dict) -> dict:
    comps = data.get("components", []) or []
    bbox = (data.get("board", {}) or {}).get("bbox_mm", [0.0, 0.0, 100.0, 100.0])

    comp_class: Dict[str, str] = {}
    comp_group: Dict[str, str] = {}
    comp_region: Dict[str, str] = {}
    side_pref: Dict[str, str] = {}
    comp_nets: Dict[str, List[str]] = {}

    for comp in comps:
        ref = comp.get("ref", "")
        sclass = infer_semantic_class(comp)
        fgroup = infer_functional_group(comp, sclass)
        region_type, side = infer_region_type(comp, sclass, fgroup, bbox)
        comp_class[ref] = sclass
        comp_group[ref] = fgroup
        comp_region[ref] = region_type
        side_pref[ref] = side
        comp_nets[ref] = _component_nets(comp)

    # Net -> components
    net_to_comps: Dict[str, List[str]] = defaultdict(list)
    for ref, nets in comp_nets.items():
        for net in nets:
            net_to_comps[net].append(ref)

    net_weights = {
        net: _net_weight(net, refs, comp_class) for net, refs in net_to_comps.items()
    }

    # Per-component critical nets / neighbors
    neighbor_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for net, refs in net_to_comps.items():
        w = net_weights.get(net, 0.0)
        if w <= 0:
            continue
        uniq_refs = list(dict.fromkeys(refs))
        for i, a in enumerate(uniq_refs):
            for b in uniq_refs[i + 1 :]:
                neighbor_scores[a][b] += w
                neighbor_scores[b][a] += w

    # Initial per-component annotations
    for comp in comps:
        ref = comp.get("ref", "")
        nets = comp_nets[ref]
        weighted_nets = sorted(
            [(net, net_weights.get(net, 0.0)) for net in set(nets) if not _is_ground_net(net)],
            key=lambda x: (-x[1], x[0]),
        )
        critical_nets = [net for net, w in weighted_nets[:3] if w >= 1.0]
        neighbors = sorted(neighbor_scores[ref].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        semantic_class = comp_class[ref]
        region_type = comp_region[ref]
        functional_group = comp_group[ref]
        side = side_pref[ref]
        review_meta = _infer_confidence_and_review(comp, semantic_class, functional_group, region_type, side, bbox)

        comp["semantic_class"] = semantic_class
        comp["region_type"] = region_type
        comp["functional_group"] = functional_group
        comp["side_preference"] = side
        comp["critical_nets"] = critical_nets
        comp["critical_neighbors"] = [n for n, _ in neighbors]
        comp["semantic"] = {
            "semantic_class": semantic_class,
            "region_type": region_type,
            "functional_group": functional_group,
            "side_preference": side,
            "anchor_ref": None,
            "subzone": "free",
            "same_side_group": None,
            "boundary_order": None,
            "critical_nets": critical_nets,
            "critical_neighbors": [n for n, _ in neighbors],
            "label_source": "auto_seed_semantic_v2",
        }
        comp["semantic_review"] = {
            "schema_version": SEMANTIC_REVIEW_VERSION,
            "label_source": "auto_seed_semantic_v2",
            "review_status": "seeded",
            "review_notes": "",
            "manual_overrides": {},
            "auto_seed": {
                "semantic_class": semantic_class,
                "region_type": region_type,
                "functional_group": functional_group,
                "side_preference": side,
                "anchor_ref": None,
                "subzone": "free",
                "same_side_group": None,
                "boundary_order": None,
                "critical_nets": critical_nets,
                "critical_neighbors": [n for n, _ in neighbors],
            },
            **review_meta,
        }

    # Align groups: same prefix + function + region with count>=2, limited to useful classes.
    group_counter: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for comp in comps:
        ref = comp.get("ref", "")
        prefix = _ref_prefix(ref)
        key = (prefix, comp_group[ref], comp_region[ref])
        group_counter[key].append(ref)

    group_names: Dict[str, str] = {}
    for (prefix, fgroup, region), refs in group_counter.items():
        if len(refs) < 2:
            continue
        if prefix in {"C", "R", "D", "Q", "J", "P", "JP", "SW", "LED", "H", "U"} or fgroup in {"interface_io", "user_interface", "power_supply"}:
            gid = f"{fgroup}:{prefix}:{region}"
            for ref in refs:
                group_names[ref] = gid

    for comp in comps:
        ref = comp.get("ref", "")
        align_group = group_names.get(ref)
        comp["align_group"] = align_group
        comp["semantic"]["align_group"] = align_group
        comp["semantic_review"]["auto_seed"]["align_group"] = align_group
        comp["semantic_review"]["candidate_labels"]["align_group"] = [align_group] if align_group else []

    populate_layout_semantic_fields(data, overwrite=True)

    # Meta summary
    meta = data.setdefault("meta", {})
    meta["semantic_annotation"] = {
        "version": SEMANTIC_REVIEW_VERSION,
        "description": "Seed semantic labels inferred from component type, reference designator, connected nets, and expert placement side; enriched with review metadata for human / semi-human correction.",
        "fields": [
            "semantic_class",
            "region_type",
            "functional_group",
            "side_preference",
            "align_group",
            "anchor_ref",
            "subzone",
            "same_side_group",
            "boundary_order",
            "critical_nets",
            "critical_neighbors",
            "semantic_review",
        ],
        "class_counts": dict(Counter(comp_class.values())),
        "group_counts": dict(Counter(comp_group.values())),
        "region_counts": dict(Counter(comp_region.values())),
        "critical_net_weights": {
            net: round(weight, 3)
            for net, weight in sorted(net_weights.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        },
        "needs_review_count": int(sum(1 for comp in comps if bool((comp.get('semantic_review') or {}).get('needs_review', False)))),
        "review_priority_counts": dict(Counter(((comp.get('semantic_review') or {}).get('review_priority') or 'low') for comp in comps)),
    }
    return data


def iter_dataset_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob('*.json'):
        s = str(path).replace('\\', '/')
        if '/data/' not in s:
            continue
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'components' in data and 'board' in data:
                yield path
        except Exception:
            continue


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
