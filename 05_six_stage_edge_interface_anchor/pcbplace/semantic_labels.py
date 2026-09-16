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

SEMANTIC_REVIEW_VERSION = "semi_manual_semantic_v3"


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



def _module_id_of(comp: dict) -> str:
    return str(comp.get('module_id') or (comp.get('module') or {}).get('module_id') or '').strip()


def _module_anchor_ref_of(comp: dict, module_by_id: Dict[str, dict] | None = None) -> str:
    mid = _module_id_of(comp)
    module_by_id = module_by_id or {}
    module = module_by_id.get(mid) or {}
    return str(
        comp.get('module_anchor_ref')
        or (comp.get('module') or {}).get('anchor_ref')
        or module.get('anchor_ref')
        or ''
    ).strip()


GENERIC_FUNCTIONAL_GROUPS = {
    '', 'misc', 'other', 'free', 'none', 'null',
    # Broad rule-inferred classes are not fine-grained layout relations.
    # When a component already has module_id, these would duplicate or blur
    # module-level grouping, so keep them out of explicit functional_group.
    'passive_support', 'power_supply', 'interface_io', 'control_processing',
    'memory', 'timing', 'rf_frontend', 'user_interface', 'mechanics',
    'protection',
}


def _is_generic_functional_group(label: str) -> bool:
    text = str(label or '').strip().lower()
    return (not text) or text in GENERIC_FUNCTIONAL_GROUPS or text.startswith('module:')


def _module_functional_group(comp: dict, base_group: str = '') -> str:
    """Return only explicit, fine-grained functional groups.

    Module identity is represented by module_id/module_anchor_ref, not by a
    semantic functional_group label.  In particular, labels like
    ``module:M03_core_ic`` are treated as missing schema defaults.  PlacementEnv
    can still fall back to module:<module_id> internally when it needs a weak
    same-module group, but the JSON should not carry that duplicate semantic
    label.
    """
    mid = _module_id_of(comp)
    raw = str(base_group or comp.get('functional_group') or (comp.get('semantic') or {}).get('functional_group') or '').strip()
    if not raw:
        return '' if mid else 'misc'
    if raw.lower().startswith('module:'):
        return '' if mid else 'misc'
    if mid and raw.lower() in GENERIC_FUNCTIONAL_GROUPS:
        return ''
    return raw


def _component_area(comp: dict) -> float:
    w, h = _safe_size(comp)
    return max(0.0, float(w) * float(h))


def _is_connector_like(comp: dict, semantic_class: str | None = None) -> bool:
    ctype = str(comp.get('type') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    return (
        ctype.startswith('conn_')
        or prefix in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'RJ', 'X', 'XA'}
        or sem in {'interface', 'rf'}
    )


def _is_ui_like(comp: dict, semantic_class: str | None = None) -> bool:
    ctype = str(comp.get('type') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    return ctype in {'ui_button', 'ui_led'} or prefix in {'SW', 'LED', 'S'} or sem == 'ui'




EDGE_IO_ROLES = {'none', 'edge_hard', 'edge_soft', 'internal_connector'}
EDGE_SIDE_LABELS = {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}

_HARD_EXTERNAL_IO_RE = re.compile(
    r"(usb|type\s*c|typec|hdmi|display\s*port|displayport|rj\s*45|rj45|rj\s*11|rj11|"
    r"barrel|dc\s*jack|power\s*jack|audio\s*jack|jack|terminal|screw\s*terminal|"
    r"bornier|phoenix|xt60|xt30|jst|sd\s*card|sim|"
    r"edge\s*connector|external|switch|button|tact|toggle|slide|dip\s*switch|pot|potentiometer|encoder)",
    re.IGNORECASE,
)
_INTERNAL_CONNECTOR_RE = re.compile(
    r"(test\s*point|testpoint|\btp\b|probe|jumper|shunt|solder\s*bridge|debug|programming|prog|swd|jtag|isp|icsp)",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r"(pin\s*header|header|hdr|conn_?pin|pinheader|socket|female|male)", re.IGNORECASE)


def _component_text_blob(comp: dict) -> str:
    fields = [
        comp.get('ref', ''), comp.get('type', ''), comp.get('footprint', ''), comp.get('value', ''),
        comp.get('description', ''), comp.get('lib_id', ''), comp.get('part', ''), comp.get('name', ''),
    ]
    return ' '.join(str(v or '') for v in fields).lower()


def _has_expert_xy(comp: dict) -> bool:
    expert = comp.get('expert') if isinstance(comp.get('expert'), dict) else {}
    xy = expert.get('xy_mm') if isinstance(expert, dict) else None
    return isinstance(xy, (list, tuple)) and len(xy) >= 2


def _edge_band_mm_for_component(comp: dict) -> float:
    w, h = _safe_size(comp)
    return float(max(2.5, 0.5 * max(float(w), float(h)) + 1.0))


def _manual_edge_info(comp: dict, board_bbox: Iterable[float]) -> Tuple[str, float, bool, float]:
    """Return (edge_side_label, min_dist_mm, near_edge, band_mm) from expert/manual geometry.

    A part is considered truly edge-near when either its center or footprint bbox
    is inside ``max(2.5, 0.5 * max(w, h) + 1.0)`` from a board edge.
    """
    x0, y0, x1, y1 = [float(v) for v in board_bbox]
    band = _edge_band_mm_for_component(comp)
    if not _has_expert_xy(comp):
        return 'free', float('inf'), False, band
    x, y = _xy(comp)
    w, h = _safe_size(comp)
    a, b, c, d = x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0
    dists = {
        'edge_left': min(abs(x - x0), abs(a - x0)),
        'edge_right': min(abs(x1 - x), abs(x1 - c)),
        'edge_bottom': min(abs(y - y0), abs(b - y0)),
        'edge_top': min(abs(y1 - y), abs(y1 - d)),
    }
    side, dist = min(dists.items(), key=lambda kv: (kv[1], kv[0]))
    return side, float(dist), bool(dist <= band + 1e-6), band


def _is_testpoint_like(comp: dict) -> bool:
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    text = _component_text_blob(comp)
    ctype = str(comp.get('type') or '').lower()
    return bool(prefix == 'TP' or ctype == 'testpoint' or 'testpoint' in text or 'test point' in text)


def _is_internal_connector_hint(comp: dict) -> bool:
    text = _component_text_blob(comp)
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    if _is_testpoint_like(comp):
        return True
    if prefix in {'JP'} and _INTERNAL_CONNECTOR_RE.search(text):
        return True
    return bool(_INTERNAL_CONNECTOR_RE.search(text))


def _is_hard_external_io_hint(comp: dict, semantic_class: str | None = None) -> bool:
    text = _component_text_blob(comp)
    ctype = str(comp.get('type') or '').lower()
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    if _is_testpoint_like(comp):
        return False
    if ctype in {'conn_usb', 'conn_typec', 'conn_hdmi', 'conn_displayport', 'conn_rj45', 'conn_rj11', 'conn_rf', 'conn_audio_jack', 'conn_barrel_jack', 'conn_terminal_block', 'conn_sd_card', 'conn_sim'}:
        return True
    if ctype in {'ui_button', 'ui_switch', 'potentiometer'} or prefix in {'SW', 'S'}:
        return True
    if sem == 'rf' and ('antenna' in text or 'sma' in text or 'u.fl' in text or 'ufl' in text):
        return True
    return bool(_HARD_EXTERNAL_IO_RE.search(text))


def _is_header_like(comp: dict) -> bool:
    text = _component_text_blob(comp)
    ctype = str(comp.get('type') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    return bool(ctype in {'conn_pin_header', 'conn_jst', 'conn_fpc', 'conn_other'} or prefix in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'X', 'XA'} or _HEADER_RE.search(text))


def _is_io_like(comp: dict, semantic_class: str | None = None) -> bool:
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    return bool(
        _is_connector_like(comp, sem)
        or _is_ui_like(comp, sem)
        or _is_hard_external_io_hint(comp, sem)
        or _is_header_like(comp)
        or sem in {'interface', 'rf'}
    )


def _normalize_external_io_role(role: Any) -> str:
    text = str(role or '').strip().lower()
    return text if text in EDGE_IO_ROLES else 'none'


def _infer_external_io_role(
    comp: dict,
    semantic_class: str,
    board_bbox: Iterable[float],
    *,
    mode: str,
) -> Tuple[str, str, bool, float, List[str]]:
    """Classify physical interfaces into edge_hard / edge_soft / internal_connector.

    TRAIN mode uses expert/manual position and only promotes connector-like parts
    to edge roles when the component center or bbox is genuinely near the board
    edge.  INFER mode is conservative: hard external-looking parts with an
    explicit side become edge_hard; ambiguous headers become edge_soft; generic
    connector-like parts without a side remain internal_connector/free.
    """
    sem = str(semantic_class or '').lower()
    existing = _normalize_external_io_role(comp.get('external_io_role') or (comp.get('semantic') or {}).get('external_io_role'))
    side_label, dist, near, band = _manual_edge_info(comp, board_bbox)
    explicit_side = _normalize_side_preference_for_component(
        comp,
        semantic_class,
        str(comp.get('side_preference') or (comp.get('semantic') or {}).get('side_preference') or 'free'),
    )
    if explicit_side in EDGE_SIDE_LABELS and not near:
        side_label = explicit_side
    allowed = comp.get('allowed_sides') or []
    if allowed and side_label == 'free':
        s = str(allowed[0]).strip().lower()
        side_label = _SIDE_TO_REGION.get(s, _SIDE_TO_REGION.get(f'edge_{s}', 'free')) if '_SIDE_TO_REGION' in globals() else f'edge_{s}'
    side_is_edge = side_label in EDGE_SIDE_LABELS
    reasons: List[str] = [f'edge_band_mm:{band:.3f}']
    if near:
        reasons.append('manual_bbox_or_center_near_edge')
    if _is_testpoint_like(comp):
        return 'internal_connector', 'free', False, band, reasons + ['testpoint_is_internal']
    if existing in {'edge_hard', 'edge_soft', 'internal_connector'}:
        # Preserve explicit/manual labels but still normalize edge side by manual geometry when available.
        if existing.startswith('edge_') and side_is_edge:
            return existing, side_label, near or mode == 'infer', band, reasons + ['explicit_external_io_role']
        if existing == 'internal_connector':
            return existing, 'free', False, band, reasons + ['explicit_internal_connector']
    if not _is_io_like(comp, sem):
        return 'none', 'free', False, band, reasons + ['not_io_like']
    hard_hint = _is_hard_external_io_hint(comp, sem)
    internal_hint = _is_internal_connector_hint(comp)
    header_like = _is_header_like(comp)
    connector_like = _is_connector_like(comp, sem) or header_like

    if str(mode or '').lower() == 'train':
        if near or comp.get('must_touch_boundary') or comp.get('allowed_sides'):
            if hard_hint or connector_like:
                return 'edge_hard', side_label if side_is_edge else 'free', True, band, reasons + ['train_true_edge_external_io']
            return 'edge_soft', side_label if side_is_edge else 'free', True, band, reasons + ['train_true_edge_soft_io']
        if connector_like or internal_hint:
            return 'internal_connector', 'free', False, band, reasons + ['train_connector_not_near_edge']
        return 'none', 'free', False, band, reasons + ['train_io_not_near_edge']

    # Infer-safe: no expert geometry, so only explicit/mechanical/hard footprint
    # evidence may become hard.  Ambiguous headers are soft or internal.
    if hard_hint:
        return 'edge_hard', side_label if side_is_edge else 'free', bool(side_is_edge), band, reasons + ['infer_hard_external' + ('_with_side' if side_is_edge else '_without_side')]
    if side_is_edge and (connector_like or _is_ui_like(comp, sem)):
        return 'edge_soft', side_label, True, band, reasons + ['infer_soft_external_with_side']
    if connector_like or internal_hint:
        return 'internal_connector', 'free', False, band, reasons + ['infer_connector_without_safe_side']
    return 'none', 'free', False, band, reasons + ['infer_no_safe_external_side']


def _footprint_bucket(comp: dict) -> str:
    raw = str(comp.get('footprint') or comp.get('type') or _ref_prefix(str(comp.get('ref') or '')) or 'unknown').strip().lower()
    raw = raw.rsplit(':', 1)[-1]
    raw = re.sub(r'[^a-z0-9]+', '_', raw).strip('_')
    return raw[:48] or 'unknown'


def _io_group_key(comp: dict, role: str, side: str) -> str:
    mid = _module_id_of(comp) or 'global'
    return f"{mid}::{_footprint_bucket(comp)}::{role}::{side or 'free'}"


def _apply_external_io_role_labels(data: dict, *, mode: str) -> dict:
    comps = data.get('components', []) or []
    if not comps:
        return data
    board_bbox = (data.get('board', {}) or {}).get('bbox_mm', [0.0, 0.0, 100.0, 100.0])
    comp_by_ref = {str(c.get('ref') or ''): c for c in comps if str(c.get('ref') or '')}
    info: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[str]] = defaultdict(list)

    for ref, comp in comp_by_ref.items():
        sem_class = str(comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or infer_semantic_class(comp)).strip()
        role, side, is_edge, band, reasons = _infer_external_io_role(comp, sem_class, board_bbox, mode=mode)
        role = _normalize_external_io_role(role)
        side = side if side in EDGE_SIDE_LABELS else 'free'
        if role == 'internal_connector':
            side = 'free'
            is_edge = False
        info[ref] = {'role': role, 'side': side, 'is_edge': is_edge, 'band': band, 'reasons': reasons}
        if role in {'edge_hard', 'edge_soft'} and side in EDGE_SIDE_LABELS:
            groups[_io_group_key(comp, role, side)].append(ref)
        elif role == 'internal_connector' and not _is_testpoint_like(comp):
            groups[_io_group_key(comp, role, 'free')].append(ref)

    # Boundary ordering / pitch is only meaningful within the same module,
    # footprint bucket, role and side.  Never align unrelated modules globally.
    group_order: Dict[str, Tuple[str, int]] = {}
    for g, refs in groups.items():
        if len(refs) < 2:
            continue
        side = info[refs[0]]['side']
        axis = 'y' if side in {'edge_left', 'edge_right'} else 'x'
        if side == 'free':
            axis = 'auto'
        def key_fn(r: str):
            x, y = _xy(comp_by_ref[r]) if _has_expert_xy(comp_by_ref[r]) else (0.0, 0.0)
            return (y if axis == 'y' else x, r)
        for idx, ref in enumerate(sorted(refs, key=key_fn)):
            group_order[ref] = (g, idx)

    for ref, comp in comp_by_ref.items():
        sem = comp.setdefault('semantic', {})
        review = comp.setdefault('semantic_review', {})
        auto = review.setdefault('auto_seed', {})
        cands = review.setdefault('candidate_labels', {})
        item = info[ref]
        role = item['role']
        side = item['side']

        comp['external_io_role'] = role
        sem['external_io_role'] = role
        auto['external_io_role'] = role
        cands['external_io_role'] = ['edge_hard', 'edge_soft', 'internal_connector', 'none']
        comp['io_edge_band_mm'] = round(float(item['band']), 3)
        sem['io_edge_band_mm'] = comp['io_edge_band_mm']
        sem['external_io_evidence'] = item['reasons']

        if role == 'edge_hard':
            comp['semantic_strength'] = max(float(comp.get('semantic_strength') or 0.0), 0.95)
            comp['constraint_source'] = 'expert_manual_geometry' if mode == 'train' else 'infer_hard_external_io'
            comp['constraint_level'] = 'soft' if mode == 'train' else 'hard'
            comp['placement_role'] = 'edge_anchor'
            if side in EDGE_SIDE_LABELS:
                comp['side_preference'] = side
                comp['region_type'] = side
        elif role == 'edge_soft':
            comp['semantic_strength'] = max(float(comp.get('semantic_strength') or 0.0), 0.70)
            comp['constraint_source'] = 'expert_manual_geometry' if mode == 'train' else 'infer_soft_external_io'
            comp['constraint_level'] = 'hint'
            comp['placement_role'] = 'edge_anchor' if side in EDGE_SIDE_LABELS else 'member'
            if side in EDGE_SIDE_LABELS:
                comp['side_preference'] = side
                comp['region_type'] = side
        elif role == 'internal_connector':
            comp['semantic_strength'] = min(float(comp.get('semantic_strength') or 0.55), 0.55)
            comp['constraint_source'] = 'expert_manual_geometry' if mode == 'train' else 'infer_internal_connector'
            comp['constraint_level'] = 'soft'
            comp['placement_role'] = 'internal_connector'
            comp['side_preference'] = 'free'
            if str(comp.get('region_type') or '').startswith('edge_'):
                comp['region_type'] = 'free'
        else:
            comp.setdefault('constraint_level', 'soft')
            if comp.get('placement_role') == 'internal_connector':
                comp['placement_role'] = 'member'

        # Keep nested mirror synchronized.
        sem.update({
            'side_preference': comp.get('side_preference', 'free'),
            'region_type': comp.get('region_type', 'free'),
            'placement_role': comp.get('placement_role', 'member'),
            'semantic_strength': comp.get('semantic_strength'),
            'constraint_source': comp.get('constraint_source'),
            'constraint_level': comp.get('constraint_level'),
        })
        auto.update({
            'side_preference': comp.get('side_preference', 'free'),
            'region_type': comp.get('region_type', 'free'),
            'placement_role': comp.get('placement_role', 'member'),
            'semantic_strength': comp.get('semantic_strength'),
            'constraint_level': comp.get('constraint_level'),
        })

        # Grouped alignment / pitch: only same module + footprint + role + side.
        group_tuple = group_order.get(ref)
        if group_tuple:
            g, idx = group_tuple
            align_axis = 'x' if side in {'edge_left', 'edge_right'} else ('y' if side in {'edge_top', 'edge_bottom'} else 'auto')
            row_axis = 'y' if align_axis == 'x' else ('x' if align_axis == 'y' else 'auto')
            comp['align_group'] = g
            comp['same_side_group'] = g if side in EDGE_SIDE_LABELS else None
            comp['pitch_group'] = g
            comp['row_group'] = g
            comp['align_axis'] = align_axis
            comp['row_axis'] = row_axis
            comp['boundary_order'] = int(idx) if side in EDGE_SIDE_LABELS else None
            comp['row_order'] = int(idx)
            comp['align_strength'] = 1.0 if role == 'edge_hard' else 0.75
            comp['pitch_strength'] = 1.0 if role == 'edge_hard' else 0.75
        else:
            # Remove old broad/global alignment for interface-like parts to avoid
            # pulling internal headers or unrelated modules to the same x/y lines.
            if role in {'edge_hard', 'edge_soft', 'internal_connector'}:
                comp['align_group'] = None
                comp['same_side_group'] = None
                comp['pitch_group'] = None
                comp['row_group'] = None
                comp['boundary_order'] = None
                comp['row_order'] = None
                comp['align_axis'] = 'x' if side in {'edge_left', 'edge_right'} else ('y' if side in {'edge_top', 'edge_bottom'} else 'auto')
                comp['row_axis'] = 'y' if comp['align_axis'] == 'x' else ('x' if comp['align_axis'] == 'y' else 'auto')

        sem.update({
            'align_group': comp.get('align_group'),
            'same_side_group': comp.get('same_side_group'),
            'pitch_group': comp.get('pitch_group'),
            'row_group': comp.get('row_group'),
            'align_axis': comp.get('align_axis'),
            'row_axis': comp.get('row_axis'),
            'boundary_order': comp.get('boundary_order'),
            'row_order': comp.get('row_order'),
            'align_strength': comp.get('align_strength'),
            'pitch_strength': comp.get('pitch_strength'),
        })
        auto.update({
            'align_group': comp.get('align_group'),
            'same_side_group': comp.get('same_side_group'),
            'pitch_group': comp.get('pitch_group'),
            'row_group': comp.get('row_group'),
            'align_axis': comp.get('align_axis'),
            'row_axis': comp.get('row_axis'),
            'boundary_order': comp.get('boundary_order'),
            'row_order': comp.get('row_order'),
        })
        cands['placement_role'] = ['member', 'anchor_large', 'edge_anchor', 'internal_connector']

    meta = data.setdefault('meta', {}).setdefault('semantic_annotation', {})
    meta['external_io_role_mode'] = str(mode)
    meta['external_io_role_counts'] = dict(Counter(info[r]['role'] for r in info))
    fields = list(meta.get('fields') or [])
    for field in ['external_io_role', 'io_edge_band_mm', 'constraint_level', 'semantic_strength', 'pitch_group', 'row_axis', 'row_order']:
        if field not in fields:
            fields.append(field)
    meta['fields'] = fields
    return data


def _is_edge_preference_component(comp: dict, semantic_class: str | None = None) -> bool:
    """Only true edge/physical-interface parts should keep edge side labels."""
    ctype = str(comp.get('type') or '').lower()
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    footprint = str(comp.get('footprint') or '').lower()
    if comp.get('must_touch_boundary') or comp.get('allowed_sides'):
        return True
    if _is_connector_like(comp, sem) or _is_ui_like(comp, sem):
        return True
    if sem in {'interface', 'rf', 'mechanical', 'mechanical_edge_interface'}:
        return True
    if ctype.startswith('conn_') or ctype.startswith('mech_') or ctype in {'ui_button', 'ui_led', 'antenna'}:
        return True
    if prefix in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'RJ', 'X', 'XA', 'SW', 'LED', 'S', 'H', 'MH', 'TP'}:
        return True
    if 'antenna' in footprint:
        return True
    return False


def _normalize_side_preference_for_component(comp: dict, semantic_class: str, side: str) -> str:
    text = str(side or 'free').strip()
    if text in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'} and not _is_edge_preference_component(comp, semantic_class):
        return 'free'
    return text or 'free'


def _is_true_align_component(comp: dict, semantic_class: str | None = None) -> bool:
    """Keep align_group only for real arrays / edge-aligned physical groups."""
    ctype = str(comp.get('type') or '').lower()
    sem = str(semantic_class or comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    if _is_connector_like(comp, sem) or _is_ui_like(comp, sem):
        return True
    if sem in {'interface', 'rf', 'mechanical', 'mechanical_edge_interface'}:
        return True
    if ctype.startswith('conn_') or ctype.startswith('mech_') or ctype in {'ui_button', 'ui_led', 'testpoint'}:
        return True
    return prefix in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'RJ', 'X', 'XA', 'SW', 'LED', 'S', 'H', 'MH', 'TP'}

def _is_anchor_like_component(comp: dict, semantic_class: str, functional_group: str, board_area: float) -> bool:
    ctype = str(comp.get('type') or '').lower()
    sem = str(semantic_class or '').lower()
    group = str(functional_group or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    w, h = _safe_size(comp)
    diag = math.hypot(w, h)
    area = _component_area(comp)
    is_passive = ctype in {'resistor', 'capacitor'}
    if comp.get('module_role') == 'anchor':
        return True
    if _is_connector_like(comp, sem):
        return True
    if ctype == 'chip' or prefix in {'U', 'IC', 'MCU'}:
        return True
    if prefix in {'J', 'P', 'JP', 'CN', 'CON', 'SW', 'LED', 'Y', 'X', 'L', 'F'}:
        return True
    if ctype in {'inductor', 'transistor'} or prefix in {'Q', 'BT', 'K'}:
        return True
    if ctype == 'diode' and sem in {'power', 'rf', 'clock'}:
        return True
    if (not is_passive) and sem in {'core', 'power', 'interface', 'rf', 'clock', 'ui', 'mechanical'}:
        return True
    if (not is_passive) and group in {'control_processing', 'power_supply', 'interface_io', 'timing', 'rf_frontend'}:
        return True
    # Very large passives can still act as coarse placement anchors; ordinary
    # R/C parts must not become anchors because that reintroduces local collapse.
    large = area >= max(25.0, 0.035 * max(1e-6, board_area)) or diag >= 12.0
    return bool(large)


def _edge_category(comp: dict, semantic_class: str, functional_group: str) -> str:
    sem = str(semantic_class or '').lower()
    group = str(functional_group or '').lower()
    prefix = _ref_prefix(str(comp.get('ref') or ''))
    ctype = str(comp.get('type') or '').lower()
    if _is_connector_like(comp, sem):
        return 'connectors'
    if _is_ui_like(comp, sem):
        return 'ui'
    if sem == 'rf':
        return 'rf'
    if sem in {'mechanical', 'mechanical_edge_interface'} or ctype.startswith('mech_') or prefix in {'H', 'MH'}:
        return 'mechanical'
    if sem in {'power', 'power_support'} or group == 'power_supply':
        return 'power'
    if prefix in {'LED'}:
        return 'ui'
    return 'edge_components'


def _net_to_refs_from_components(comps: List[dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for comp in comps:
        ref = str(comp.get('ref') or '')
        for net in _component_nets(comp):
            if ref and ref not in out[net]:
                out[net].append(ref)
    return out


def _net_anchor_priority(net: str, refs: List[str], comp_classes: Dict[str, str]) -> float:
    """Higher is more useful for anchor / critical-neighbor labels.

    Ground and high-fanout rails should not pull whole boards together.  Clock,
    reset, RF, and low-fanout signal nets are the best module-local evidence.
    """
    n = str(net or '').strip()
    deg = len(set(refs))
    if deg <= 1:
        return 0.0
    if _is_ground_net(n):
        return 0.0
    max_good_fanout = 8
    if deg > max_good_fanout and not (_is_clock_net(n) or _is_reset_net(n) or _is_rf_net(n)):
        return 0.0

    score = 1.0
    if _is_clock_net(n) or _is_reset_net(n):
        score += 4.0
    if _is_rf_net(n):
        score += 4.0
    if deg <= 3 and not _is_power_net(n):
        score += 2.5
    elif deg <= 6 and not _is_power_net(n):
        score += 1.5
    if _is_interface_net(n):
        score += 1.5
    if _is_power_net(n):
        # Non-ground power can identify decoupling / power support, but should
        # be weaker than low-fanout functional nets.
        score += 0.45
        if deg > 6:
            score *= 0.35

    classes = {comp_classes.get(r, 'support') for r in refs}
    if 'core' in classes and ('interface' in classes or 'interface_support' in classes):
        score += 0.8
    if 'core' in classes and ('power' in classes or 'power_support' in classes):
        score += 0.5
    return float(max(0.0, score / math.log1p(float(deg))))


def _pair_scores_from_nets(
    comps: List[dict],
    comp_classes: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    net_to_refs = _net_to_refs_from_components(comps)
    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for net, refs0 in net_to_refs.items():
        refs = list(dict.fromkeys(refs0))
        w = _net_anchor_priority(net, refs, comp_classes)
        if w <= 0.0:
            continue
        for i, a in enumerate(refs):
            for b in refs[i + 1:]:
                scores[a][b] += w
                scores[b][a] += w
    return scores


def _same_module_bonus(ref: str, other: str, module_id: Dict[str, str]) -> float:
    return 1.25 if module_id.get(ref) and module_id.get(ref) == module_id.get(other) else 0.0


def _select_anchor_for_component(
    ref: str,
    comp_by_ref: Dict[str, dict],
    anchors_like: set[str],
    pair_scores: Dict[str, Dict[str, float]],
    module_anchor: Dict[str, str],
    module_id: Dict[str, str],
    comp_classes: Dict[str, str],
) -> str | None:
    comp = comp_by_ref[ref]
    m_anchor = str(module_anchor.get(ref, '') or '').strip()
    explicit = comp.get('anchor_ref') or (comp.get('semantic') or {}).get('anchor_ref')
    if explicit and str(explicit).strip() and str(explicit).strip() != ref:
        explicit_ref = str(explicit).strip()
        # anchor_ref is only a local override.  If it duplicates module_anchor_ref,
        # keep JSON anchor_ref empty and let PlacementEnv fall back to module anchor.
        return None if explicit_ref == m_anchor else explicit_ref

    if m_anchor == ref:
        return None

    candidates: Dict[str, float] = defaultdict(float)

    # module_anchor_ref is not an explicit semantic anchor.  It is kept only as
    # a weak baseline when deciding whether a net-derived anchor is stronger; if
    # it wins, return None so JSON does not duplicate module_anchor_ref.
    if m_anchor and m_anchor in comp_by_ref:
        candidates[m_anchor] += 1.0

    for other, score in pair_scores.get(ref, {}).items():
        if other == ref or other not in comp_by_ref:
            continue
        if other not in anchors_like:
            continue
        sem = comp_classes.get(other, '')
        role_bonus = 0.0
        if sem in {'core', 'interface', 'power', 'rf', 'clock'}:
            role_bonus += 1.1
        if _is_connector_like(comp_by_ref[other], sem):
            role_bonus += 0.9
        if str(other) == m_anchor:
            role_bonus += 0.25
        # Same-module is only a weak tie-breaker here, not a reason to write
        # module_anchor_ref into anchor_ref.
        candidates[other] += float(score) + role_bonus + 0.25 * _same_module_bonus(ref, other, module_id)

    if candidates:
        best, best_score = max(candidates.items(), key=lambda kv: (kv[1], kv[0]))
        if best_score >= 1.0 and str(best) != m_anchor:
            return str(best)
    return None


def _select_critical_neighbors(
    ref: str,
    comp_by_ref: Dict[str, dict],
    pair_scores: Dict[str, Dict[str, float]],
    anchor_ref: str | None,
    module_anchor_ref: str | None,
    comp_classes: Dict[str, str],
    max_neighbors: int = 4,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    excluded = {str(ref)}
    if anchor_ref:
        excluded.add(str(anchor_ref))
    if module_anchor_ref:
        excluded.add(str(module_anchor_ref))

    for other, score in pair_scores.get(ref, {}).items():
        if other == ref or other not in comp_by_ref or str(other) in excluded:
            continue
        sem = comp_classes.get(other, '')
        bonus = 0.0
        if _is_anchor_like_component(comp_by_ref[other], sem, '', 1.0):
            bonus += 0.8
        # Do not add same-module-only bonus here. critical_neighbors should mean
        # real electrical-critical evidence from nets, not duplicate module group.
        scores[other] += float(score) + bonus

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [r for r, s in ranked[:max_neighbors] if s > 0.0]


def _auto_confidence_for_component(
    comp: dict,
    anchor_ref: str | None,
    critical_neighbors: List[str],
    same_side_group: str | None,
) -> float:
    conf = 0.62
    if _module_id_of(comp):
        conf += 0.08
    if anchor_ref:
        conf += 0.12
    if critical_neighbors:
        conf += 0.08
    if same_side_group:
        conf += 0.06
    if comp.get('semantic_review') and (comp.get('semantic_review') or {}).get('review_status') in {'approved', 'edited'}:
        conf = 1.0
    return float(max(0.50, min(0.92, conf)))


def populate_layout_semantic_fields(data: dict, overwrite: bool = False) -> dict:
    """Populate layout-aware semantic labels used by train/infer.

    This pass intentionally writes top-level fields and the nested ``semantic``
    mirror because the dataset parser, PlacementEnv, region prior and CUDA
    context all consume those fields.  Module identity stays in module_id /
    module_anchor_ref.  functional_group is only for explicit, fine-grained
    local relations; PlacementEnv may use module_id as an internal fallback,
    but this function does not write module:<module_id> into JSON.
    """
    comps = data.get('components', []) or []
    if not comps:
        return refresh_semantic_meta(data)

    board_bbox = (data.get('board', {}) or {}).get('bbox_mm', [0.0, 0.0, 100.0, 100.0])
    x0, y0, x1, y1 = [float(v) for v in board_bbox]
    bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    board_area = max(1e-6, bw * bh)
    edge_band = 0.14 * min(bw, bh)

    comp_by_ref = {str(c.get('ref', '')): c for c in comps if str(c.get('ref', ''))}
    module_by_id = {
        str(m.get('module_id')): m
        for m in (data.get('modules') or data.get('module_annotations') or data.get('graph', {}).get('modules') or [])
        if isinstance(m, dict) and m.get('module_id')
    }

    # Base labels come from annotate_board() or fallback rules.
    sem_class: Dict[str, str] = {}
    base_group: Dict[str, str] = {}
    functional_group: Dict[str, str] = {}
    region: Dict[str, str] = {}
    side: Dict[str, str] = {}
    module_id: Dict[str, str] = {}
    module_anchor: Dict[str, str] = {}
    coords: Dict[str, Tuple[float, float]] = {}
    for ref, comp in comp_by_ref.items():
        sclass = str(comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or infer_semantic_class(comp)).strip()
        bgroup = str(comp.get('functional_group') or (comp.get('semantic') or {}).get('functional_group') or infer_functional_group(comp, sclass)).strip()
        fgroup = _module_functional_group(comp, bgroup)
        inferred_region, inferred_side = infer_region_type(comp, sclass, fgroup or bgroup, board_bbox)
        rtype = str(comp.get('region_type') or (comp.get('semantic') or {}).get('region_type') or inferred_region).strip()
        spref = str(comp.get('side_preference') or (comp.get('semantic') or {}).get('side_preference') or inferred_side).strip()
        spref = _normalize_side_preference_for_component(comp, sclass, spref)
        if rtype in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'} and spref == 'free' and not _is_edge_preference_component(comp, sclass):
            rtype = 'free'
        sem_class[ref] = sclass
        base_group[ref] = bgroup
        functional_group[ref] = fgroup
        region[ref] = rtype
        side[ref] = spref
        module_id[ref] = _module_id_of(comp)
        module_anchor[ref] = _module_anchor_ref_of(comp, module_by_id)
        coords[ref] = _xy(comp)

    pair_scores = _pair_scores_from_nets(comps, sem_class)
    anchor_like = {
        ref for ref, comp in comp_by_ref.items()
        if _is_anchor_like_component(comp, sem_class.get(ref, 'support'), base_group.get(ref, ''), board_area)
    }

    # Respect module anchors as anchor-like even if their footprint is small.
    for ref, anch in module_anchor.items():
        if anch in comp_by_ref:
            anchor_like.add(anch)

    anchors: Dict[str, str | None] = {}
    for ref, comp in comp_by_ref.items():
        cur = comp.get('anchor_ref') or (comp.get('semantic') or {}).get('anchor_ref')
        if cur and not overwrite:
            anchors[ref] = str(cur)
            continue
        anchors[ref] = _select_anchor_for_component(
            ref,
            comp_by_ref,
            anchor_like,
            pair_scores,
            module_anchor,
            module_id,
            sem_class,
        )

    critical_neighbors: Dict[str, List[str]] = {}
    for ref in comp_by_ref:
        cur = comp_by_ref[ref].get('critical_neighbors') or (comp_by_ref[ref].get('semantic') or {}).get('critical_neighbors')
        if cur and not overwrite:
            excluded = {str(ref)}
            if anchors.get(ref):
                excluded.add(str(anchors.get(ref)))
            if module_anchor.get(ref):
                excluded.add(str(module_anchor.get(ref)))
            critical_neighbors[ref] = [str(v) for v in cur if str(v) in comp_by_ref and str(v) not in excluded][:4]
        else:
            critical_neighbors[ref] = _select_critical_neighbors(
                ref,
                comp_by_ref,
                pair_scores,
                anchors.get(ref),
                module_anchor.get(ref),
                sem_class,
                max_neighbors=4,
            )

    subzones: Dict[str, str] = {}
    for ref, comp in comp_by_ref.items():
        cur = comp.get('subzone') or (comp.get('semantic') or {}).get('subzone')
        if cur and not overwrite:
            subzones[ref] = str(cur)
            continue
        anchor_ref = anchors.get(ref)
        if not anchor_ref or anchor_ref == ref or anchor_ref not in coords:
            subzones[ref] = 'free' if sem_class[ref] in {'core', 'interface', 'mechanical'} else 'around'
        else:
            subzones[ref] = _infer_subzone_from_xy(coords[ref], coords[anchor_ref])

    # Edge sequencing labels for connectors, buttons, LEDs, headers and long
    # edge parts.  This is intentionally side/category based rather than
    # module-only so same-edge connectors keep their global order.
    edge_candidates: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for ref, comp in comp_by_ref.items():
        pref = side.get(ref, region.get(ref, 'free'))
        if pref not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            continue
        x, y = coords[ref]
        nearest_side, dist = _nearest_edge(x, y, board_bbox)
        w, h = _safe_size(comp)
        elongated = max(w, h) >= max(6.0, 2.2 * max(1e-6, min(w, h)))
        edge_like = (
            _is_connector_like(comp, sem_class[ref])
            or _is_ui_like(comp, sem_class[ref])
            or sem_class[ref] in {'mechanical', 'rf'}
            or elongated
            or dist <= edge_band
        )
        if not edge_like:
            continue
        # Prefer the actual nearest edge when close; otherwise keep semantic side.
        side_key = nearest_side if dist <= 1.5 * edge_band else pref
        cat = _edge_category(comp, sem_class[ref], base_group[ref])
        edge_candidates[(side_key, cat)].append(ref)

    same_side_group: Dict[str, str | None] = {ref: None for ref in comp_by_ref}
    boundary_order: Dict[str, int | None] = {ref: None for ref in comp_by_ref}
    for (side_key, cat), refs in edge_candidates.items():
        if len(refs) < 2:
            continue
        refs_sorted = sorted(refs, key=lambda r: (_boundary_axis_value(side_key, *coords[r]), r))
        group_name = f"{side_key}_{cat}"
        for idx, ref in enumerate(refs_sorted):
            same_side_group[ref] = group_name
            boundary_order[ref] = int(idx)

    # Placement role: consumed by the env as an additional large/anchor cue.
    placement_role: Dict[str, str] = {}
    for ref, comp in comp_by_ref.items():
        if ref in anchor_like or str(comp.get('module_role') or '') == 'anchor':
            if side.get(ref, '').startswith('edge_') or _is_connector_like(comp, sem_class[ref]):
                placement_role[ref] = 'edge_anchor'
            else:
                placement_role[ref] = 'anchor_large'
        else:
            placement_role[ref] = 'member'

    for ref, comp in comp_by_ref.items():
        sem = comp.setdefault('semantic', {})
        review = comp.setdefault('semantic_review', {})
        auto = review.setdefault('auto_seed', {})

        # Canonical top-level labels.
        comp['semantic_class'] = sem_class[ref]
        comp['region_type'] = region[ref]
        comp['functional_group'] = functional_group[ref]
        comp['side_preference'] = side[ref]
        comp['placement_role'] = placement_role[ref]

        # Anchor self is stored as null by design; env has the same self-filter.
        aref = anchors.get(ref)
        if aref and module_anchor.get(ref) and str(aref) == str(module_anchor.get(ref)):
            aref = None
        if overwrite or comp.get('anchor_ref', None) in (None, ''):
            comp['anchor_ref'] = aref if aref and aref != ref else None
        if overwrite or comp.get('subzone', None) in (None, ''):
            comp['subzone'] = subzones.get(ref, 'free')
        if overwrite or comp.get('critical_neighbors', None) in (None, ''):
            comp['critical_neighbors'] = critical_neighbors.get(ref, [])
        if same_side_group.get(ref) is not None and (overwrite or comp.get('same_side_group', None) in (None, '')):
            comp['same_side_group'] = same_side_group[ref]
        elif overwrite:
            comp['same_side_group'] = None
        if boundary_order.get(ref) is not None and (overwrite or comp.get('boundary_order', None) in (None, '')):
            comp['boundary_order'] = int(boundary_order[ref])
        elif overwrite:
            comp['boundary_order'] = None

        # Keep module role explicit for anchors.
        if module_anchor.get(ref) and module_anchor.get(ref) == ref:
            comp['module_role'] = 'anchor'

        # Nested mirror for loaders that read semantic.*.
        sem.update({
            'semantic_class': sem_class[ref],
            'region_type': region[ref],
            'functional_group': functional_group[ref],
            'side_preference': side[ref],
            'anchor_ref': comp.get('anchor_ref'),
            'subzone': comp.get('subzone'),
            'same_side_group': comp.get('same_side_group'),
            'boundary_order': comp.get('boundary_order'),
            'critical_neighbors': comp.get('critical_neighbors') or [],
            'placement_role': placement_role[ref],
            'label_source': 'auto_seed_semantic_v3',
        })

        conf = _auto_confidence_for_component(comp, comp.get('anchor_ref'), comp.get('critical_neighbors') or [], comp.get('same_side_group'))
        review.setdefault('schema_version', SEMANTIC_REVIEW_VERSION)
        review.setdefault('label_source', 'auto_seed_semantic_v3')
        if overwrite or not review.get('review_status'):
            review['review_status'] = 'seeded'
        review['auto_confidence'] = conf
        review['needs_review'] = bool(review.get('needs_review', False))
        auto.update({
            'semantic_class': sem_class[ref],
            'region_type': region[ref],
            'functional_group': functional_group[ref],
            'side_preference': side[ref],
            'anchor_ref': comp.get('anchor_ref'),
            'subzone': comp.get('subzone'),
            'same_side_group': comp.get('same_side_group'),
            'boundary_order': comp.get('boundary_order'),
            'critical_neighbors': comp.get('critical_neighbors') or [],
            'placement_role': placement_role[ref],
        })
        cands = review.setdefault('candidate_labels', {})
        cands['subzone'] = ['left', 'right', 'top', 'bottom', 'around', 'free']
        cands['placement_role'] = ['member', 'anchor_large', 'edge_anchor']
        if comp.get('same_side_group') is not None:
            cands['same_side_group'] = [comp.get('same_side_group')]
        if comp.get('boundary_order') is not None:
            cands['boundary_order'] = [int(comp.get('boundary_order'))]

    refresh_semantic_meta(data)
    meta = data.setdefault('meta', {})
    meta.setdefault('semantic_annotation', {})['layout_semantic_fields'] = [
        'anchor_ref', 'subzone', 'critical_neighbors',
        'same_side_group', 'boundary_order', 'placement_role',
        'external_io_role', 'constraint_level', 'semantic_strength',
    ]
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
    side, _dist, near_edge, _band = _manual_edge_info(comp, bbox)

    if _is_edge_preference_component(comp, semantic_class):
        # Interface-looking refs/footprints are not enough: in train mode the
        # manual/expert center or bbox must be genuinely near an edge.  Explicit
        # mechanical constraints are still respected.
        if near_edge or comp.get('must_touch_boundary') or comp.get('allowed_sides'):
            return side, side
    if semantic_class == "core" or functional_group in {"control_processing", "memory", "timing"}:
        return "core", "core"
    # Ordinary module members/passives and internal headers remain free.
    return "free", "free"


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

    # Align groups are only for true arrays / physical edge groups (connectors,
    # buttons/LEDs, mounting holes, test points, RF/interface rows).  Do not
    # generate broad groups from passive_support, power_supply, module_id, etc.
    group_counter: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for comp in comps:
        ref = comp.get("ref", "")
        if not _is_true_align_component(comp, comp_class.get(ref, '')):
            continue
        prefix = _ref_prefix(ref)
        side = side_pref.get(ref, 'free')
        if side not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'} and comp_class.get(ref, '') not in {'interface', 'ui', 'mechanical', 'rf'}:
            continue
        key = (prefix, side if side.startswith('edge_') else comp_class.get(ref, ''))
        group_counter[key].append(ref)

    group_names: Dict[str, str] = {}
    for (prefix, bucket), refs in group_counter.items():
        if len(refs) < 2:
            continue
        gid = f"align:{bucket}:{prefix}"
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


# -----------------------------------------------------------------------------
# Explicit train / inference-safe semantic entry points
# -----------------------------------------------------------------------------

_SAFE_REGION_TYPES = {'edge_top', 'edge_bottom', 'edge_left', 'edge_right', 'core', 'free'}
_SAFE_SIDES = {'left', 'right', 'top', 'bottom', 'edge_left', 'edge_right', 'edge_top', 'edge_bottom', 'free'}
_SIDE_TO_REGION = {
    'left': 'edge_left', 'edge_left': 'edge_left',
    'right': 'edge_right', 'edge_right': 'edge_right',
    'top': 'edge_top', 'edge_top': 'edge_top',
    'bottom': 'edge_bottom', 'edge_bottom': 'edge_bottom',
}
_REGION_TO_SIDE = {
    'edge_left': 'edge_left',
    'edge_right': 'edge_right',
    'edge_top': 'edge_top',
    'edge_bottom': 'edge_bottom',
    'core': 'free',
    'free': 'free',
}


def _first_valid_side_from_component(comp: dict) -> str:
    allowed = comp.get('allowed_sides') or []
    if isinstance(allowed, list):
        for val in allowed:
            text = str(val or '').strip().lower()
            if text in _SIDE_TO_REGION:
                return text
    sem = comp.get('semantic') if isinstance(comp.get('semantic'), dict) else {}
    for key in ('side_preference', 'region_type'):
        val = comp.get(key)
        if val in (None, ''):
            val = sem.get(key)
        text = str(val or '').strip().lower()
        if text in _SIDE_TO_REGION:
            return text
    return ''


def _infer_region_type_infer_safe(comp: dict, semantic_class: str, functional_group: str) -> Tuple[str, str, List[str], float]:
    """Infer region/side without expert xy or final-layout bboxes.

    Inputs are limited to ref/type/footprint, pads/nets, explicit constraints,
    and user-provided side/region hints that remain after infer sanitization.
    Missing spatial information is intentionally represented as lower confidence
    instead of a hard edge/core guess.
    """
    sem = comp.get('semantic') if isinstance(comp.get('semantic'), dict) else {}
    evidence: List[str] = []
    confidence_delta = 0.0

    # Explicit user/mechanical side constraints are strong and inference-safe.
    side_hint = _first_valid_side_from_component(comp)
    if side_hint in _SIDE_TO_REGION:
        evidence.append('explicit_side_or_allowed_sides')
        confidence_delta += 0.14
        return _SIDE_TO_REGION[side_hint], _SIDE_TO_REGION[side_hint], evidence, confidence_delta

    # Explicit user region hints are acceptable only when they are still present
    # after infer sanitization.  The unified generator removes old auto-derived
    # fields before calling this function in infer mode.
    region_hint = str(comp.get('region_type') or sem.get('region_type') or '').strip().lower()
    if region_hint in _SAFE_REGION_TYPES and region_hint not in {'', 'free'}:
        evidence.append('explicit_region_hint')
        confidence_delta += 0.08
        return region_hint, _REGION_TO_SIDE.get(region_hint, 'free'), evidence, confidence_delta

    # Core / main active components are a safe center-ish prior; edge-capable
    # parts without a side remain free+review instead of being forced to an edge.
    sclass = str(semantic_class or '').lower()
    fgroup = str(functional_group or '').lower()
    if sclass == 'core' or fgroup in {'control_processing', 'memory'}:
        evidence.append('safe_core_prior')
        confidence_delta += 0.05
        return 'core', 'free', evidence, confidence_delta
    if _is_edge_preference_component(comp, semantic_class) or sclass in {'interface', 'ui', 'rf', 'mechanical'}:
        evidence.append('edge_capable_but_no_safe_side')
        confidence_delta -= 0.12
        return 'free', 'free', evidence, confidence_delta
    evidence.append('no_safe_region_hint')
    confidence_delta -= 0.04
    return 'free', 'free', evidence, confidence_delta


def _build_net_neighbor_scores(comps: List[dict], comp_class: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, float], Dict[str, Dict[str, float]]]:
    comp_nets: Dict[str, List[str]] = {}
    for comp in comps:
        ref = str(comp.get('ref') or '')
        comp_nets[ref] = _component_nets(comp)
    net_to_comps: Dict[str, List[str]] = defaultdict(list)
    for ref, nets in comp_nets.items():
        for net in nets:
            net_to_comps[net].append(ref)
    net_weights = {net: _net_weight(net, refs, comp_class) for net, refs in net_to_comps.items()}
    neighbor_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for net, refs0 in net_to_comps.items():
        w = float(net_weights.get(net, 0.0))
        if w <= 0:
            continue
        refs = list(dict.fromkeys(refs0))
        for i, a in enumerate(refs):
            for b in refs[i + 1:]:
                neighbor_scores[a][b] += w
                neighbor_scores[b][a] += w
    return comp_nets, net_weights, neighbor_scores


def annotate_board_train(data: dict, *, overwrite: bool = True) -> dict:
    """Training semantic annotator.

    This mode may use expert/final layout geometry to derive local spatial
    labels such as anchor-relative subzone and same-edge ordering.  It is meant
    for building supervised labels, not for preparing inference inputs.
    """
    out = _apply_external_io_role_labels(annotate_board(data), mode='train')
    meta = out.setdefault('meta', {}).setdefault('semantic_annotation', {})
    meta['mode'] = 'train'
    meta['uses_expert_layout'] = True
    meta['strong_supervision_fields'] = [
        'semantic_class', 'region_type', 'side_preference',
        'anchor_ref', 'subzone', 'same_side_group', 'boundary_order',
        'critical_neighbors',
    ]
    meta['weak_supervision_fields'] = ['functional_group', 'placement_role', 'align_group', 'external_io_role']
    meta['review_hint_fields'] = ['semantic_review.auto_confidence', 'semantic_review.needs_review']
    return out


def annotate_board_infer_safe(data: dict, *, overwrite: bool = True) -> dict:
    """Inference-safe semantic annotator.

    This mode never reads expert xy/rot, expert module bboxes, or
    expert-derived shape hints.  It uses only ref/type/footprint, pads/nets,
    schematic relations, explicit constraints and user-provided hints.  Missing
    spatial labels are assigned lower confidence and marked for review rather
    than hard-filled from nonexistent geometry.
    """
    comps = data.get('components', []) or []
    if not comps:
        return refresh_semantic_meta(data)

    comp_by_ref = {str(c.get('ref', '')): c for c in comps if str(c.get('ref', ''))}
    module_by_id = {
        str(m.get('module_id')): m
        for m in (data.get('modules') or data.get('module_annotations') or data.get('graph', {}).get('modules') or [])
        if isinstance(m, dict) and m.get('module_id')
    }
    board_bbox = (data.get('board', {}) or {}).get('bbox_mm', [0.0, 0.0, 100.0, 100.0])
    x0, y0, x1, y1 = [float(v) for v in board_bbox]
    board_area = max(1e-6, (x1 - x0) * (y1 - y0))

    comp_class: Dict[str, str] = {}
    base_group: Dict[str, str] = {}
    functional_group: Dict[str, str] = {}
    comp_region: Dict[str, str] = {}
    side_pref: Dict[str, str] = {}
    region_evidence: Dict[str, List[str]] = {}
    region_conf_delta: Dict[str, float] = {}
    module_id: Dict[str, str] = {}
    module_anchor: Dict[str, str] = {}

    for ref, comp in comp_by_ref.items():
        sclass = str(comp.get('semantic_class') or (comp.get('semantic') or {}).get('semantic_class') or infer_semantic_class(comp)).strip()
        bgroup = str(comp.get('functional_group') or (comp.get('semantic') or {}).get('functional_group') or infer_functional_group(comp, sclass)).strip()
        fgroup = _module_functional_group(comp, bgroup)
        rtype, side, ev, delta = _infer_region_type_infer_safe(comp, sclass, fgroup or bgroup)
        side = _normalize_side_preference_for_component(comp, sclass, side)
        comp_class[ref] = sclass
        base_group[ref] = bgroup
        functional_group[ref] = fgroup
        comp_region[ref] = rtype
        side_pref[ref] = side
        region_evidence[ref] = ev
        region_conf_delta[ref] = delta
        module_id[ref] = _module_id_of(comp)
        module_anchor[ref] = _module_anchor_ref_of(comp, module_by_id)

    comp_nets, net_weights, neighbor_scores = _build_net_neighbor_scores(comps, comp_class)
    pair_scores = _pair_scores_from_nets(comps, comp_class)
    anchor_like = {
        ref for ref, comp in comp_by_ref.items()
        if _is_anchor_like_component(comp, comp_class.get(ref, 'support'), base_group.get(ref, ''), board_area)
    }
    for ref, anch in module_anchor.items():
        if anch in comp_by_ref:
            anchor_like.add(anch)

    anchors: Dict[str, str | None] = {}
    for ref, comp in comp_by_ref.items():
        anchors[ref] = _select_anchor_for_component(
            ref,
            comp_by_ref,
            anchor_like,
            pair_scores,
            module_anchor,
            module_id,
            comp_class,
        )

    critical_neighbors: Dict[str, List[str]] = {}
    for ref in comp_by_ref:
        critical_neighbors[ref] = _select_critical_neighbors(
            ref,
            comp_by_ref,
            pair_scores,
            anchors.get(ref),
            module_anchor.get(ref),
            comp_class,
            max_neighbors=4,
        )

    for comp in comps:
        ref = str(comp.get('ref') or '')
        if not ref:
            continue
        nets = comp_nets.get(ref, [])
        weighted_nets = sorted(
            [(net, net_weights.get(net, 0.0)) for net in set(nets) if not _is_ground_net(net)],
            key=lambda x: (-x[1], x[0]),
        )
        critical_nets = [net for net, w in weighted_nets[:3] if w >= 1.0]
        semantic_class = comp_class[ref]
        rtype = comp_region[ref]
        side = side_pref[ref]
        fgroup = functional_group[ref]
        aref = anchors.get(ref)
        if aref and module_anchor.get(ref) and str(aref) == str(module_anchor.get(ref)):
            aref = None
        subzone = 'around' if aref or critical_neighbors.get(ref) else 'free'

        placement_role = 'member'
        if ref in anchor_like or str(comp.get('module_role') or '') == 'anchor':
            placement_role = 'edge_anchor' if side.startswith('edge_') or _is_connector_like(comp, semantic_class) else 'anchor_large'

        comp['semantic_class'] = semantic_class
        comp['region_type'] = rtype
        comp['functional_group'] = fgroup
        comp['side_preference'] = side
        comp['critical_nets'] = critical_nets
        comp['critical_neighbors'] = critical_neighbors.get(ref, [])
        comp['anchor_ref'] = aref if aref and aref != ref else None
        comp['subzone'] = subzone
        comp['same_side_group'] = None
        comp['boundary_order'] = None
        comp['placement_role'] = placement_role

        sem = comp.setdefault('semantic', {})
        sem.update({
            'semantic_class': semantic_class,
            'region_type': rtype,
            'functional_group': fgroup,
            'side_preference': side,
            'anchor_ref': comp.get('anchor_ref'),
            'subzone': subzone,
            'same_side_group': None,
            'boundary_order': None,
            'critical_nets': critical_nets,
            'critical_neighbors': critical_neighbors.get(ref, []),
            'placement_role': placement_role,
            'label_source': 'infer_safe_semantic_v1',
        })

        review_meta = _infer_confidence_and_review(comp, semantic_class, fgroup or base_group[ref], rtype, side, board_bbox)
        conf = float(review_meta.get('auto_confidence', 0.55)) + float(region_conf_delta.get(ref, 0.0))
        # Lack of spatially safe evidence should down-weight supervision and
        # region prior influence rather than fabricating a precise label.
        if 'edge_capable_but_no_safe_side' in region_evidence.get(ref, []):
            review_meta['needs_review'] = True
            review_meta['review_priority'] = 'medium' if semantic_class in {'interface', 'rf', 'power'} else review_meta.get('review_priority', 'low')
        if not comp.get('anchor_ref') and semantic_class in {'core', 'power', 'interface', 'rf', 'clock'}:
            conf -= 0.04
        conf = max(0.20, min(0.86, conf))
        review_meta['auto_confidence'] = round(conf, 3)
        ev = review_meta.setdefault('evidence', {})
        ev.pop('expert_side', None)
        ev.pop('expert_near_edge', None)
        ev['inference_safe_region_evidence'] = list(region_evidence.get(ref, []))
        ev['uses_expert_layout'] = False

        review = comp.setdefault('semantic_review', {})
        review.clear()
        review.update({
            'schema_version': SEMANTIC_REVIEW_VERSION,
            'label_source': 'infer_safe_semantic_v1',
            'review_status': 'seeded',
            'review_notes': '',
            'manual_overrides': {},
            'auto_seed': {
                'semantic_class': semantic_class,
                'region_type': rtype,
                'functional_group': fgroup,
                'side_preference': side,
                'anchor_ref': comp.get('anchor_ref'),
                'subzone': subzone,
                'same_side_group': None,
                'boundary_order': None,
                'critical_nets': critical_nets,
                'critical_neighbors': critical_neighbors.get(ref, []),
                'placement_role': placement_role,
            },
            **review_meta,
        })
        comp['align_group'] = None
        sem['align_group'] = None
        review['auto_seed']['align_group'] = None
        review.setdefault('candidate_labels', {})['align_group'] = []
        review['candidate_labels']['subzone'] = ['around', 'free']
        review['candidate_labels']['same_side_group'] = []
        review['candidate_labels']['boundary_order'] = []

    meta = data.setdefault('meta', {})
    meta['semantic_annotation'] = {
        'version': SEMANTIC_REVIEW_VERSION,
        'mode': 'infer',
        'description': 'Inference-safe semantic labels generated from ref/type/footprint, pads/nets, schematic relations, explicit constraints and user hints only. No expert xy, expert bbox or expert-derived shape hint is used.',
        'uses_expert_layout': False,
        'fields': [
            'semantic_class', 'region_type', 'functional_group', 'side_preference',
            'align_group', 'anchor_ref', 'subzone', 'same_side_group',
            'boundary_order', 'critical_nets', 'critical_neighbors', 'semantic_review',
        ],
        'strong_supervision_fields': ['semantic_class', 'critical_neighbors'],
        'weak_supervision_fields': ['region_type', 'side_preference', 'anchor_ref', 'subzone', 'functional_group', 'placement_role'],
        'review_hint_fields': ['semantic_review.auto_confidence', 'semantic_review.needs_review'],
        'class_counts': dict(Counter(comp_class.values())),
        'group_counts': dict(Counter(functional_group.values())),
        'region_counts': dict(Counter(comp_region.values())),
        'critical_net_weights': {
            net: round(weight, 3)
            for net, weight in sorted(net_weights.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        },
        'needs_review_count': int(sum(1 for comp in comps if bool((comp.get('semantic_review') or {}).get('needs_review', False)))),
        'review_priority_counts': dict(Counter(((comp.get('semantic_review') or {}).get('review_priority') or 'low') for comp in comps)),
    }
    _apply_external_io_role_labels(data, mode='infer')
    return data
