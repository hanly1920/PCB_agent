from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from .utils import (
    hpwl_from_pins,
    w_hpwl_from_pins,
    is_edge_required_type,
    is_connector_type,
    coarse_type_from_fine,
    is_edge_preferred_type,
)
from .runtime_safety import (
    RUNTIME_SHAPE_HINT_SCHEMA_VERSION,
    describe_runtime_prior_violation,
    describe_runtime_shape_hint_violation,
    is_runtime_prior_allowed,
    is_runtime_shape_hint_allowed,
)


@dataclass
class Component:
    ref: str
    type: str
    size_mm: Tuple[float, float]
    pads: List[Tuple[str, Tuple[float, float]]]  # (net, rel_xy_mm) where rel_xy is w.r.t. component center
    allowed_sides: List[str]
    must_touch_boundary: Optional[bool] = None
    semantic_class: str = 'other'
    region_type: str = 'free'
    functional_group: str = 'misc'
    side_preference: str = 'free'
    align_group: Optional[str] = None
    # Optional engineering-semantics controls.  These are backward-compatible:
    # old JSONs may omit them and the environment will infer conservative defaults.
    semantic_strength: Optional[float] = None
    constraint_source: str = 'auto'
    constraint_level: str = 'soft'
    align_axis: str = 'auto'  # 'x'=shared x/vertical column, 'y'=shared y/horizontal row, 'auto'=infer
    align_strength: float = 1.0
    anchor_ref: Optional[str] = None
    subzone: str = 'free'
    same_side_group: Optional[str] = None
    boundary_order: Optional[int] = None
    boundary_order_source: str = 'auto'
    spacing_policy: str = 'equal'  # same-side / pitch policy: equal|preserve|free
    pitch_group: Optional[str] = None
    pitch_strength: float = 1.0
    row_group: Optional[str] = None
    row_axis: str = 'auto'
    row_order: Optional[int] = None
    critical_nets: Tuple[str, ...] = ()
    critical_neighbors: Tuple[str, ...] = ()
    critical_neighbor_specs: Tuple[Dict[str, Any], ...] = ()
    review_status: str = 'untracked'
    auto_confidence: float = 1.0
    needs_review: bool = False
    placement_role: str = 'member'
    module_id: str = ''
    module_role: str = 'member'
    module_anchor_ref: Optional[str] = None
    module_order: int = 0
    module_local_order: int = 0
    module_region_bbox: Optional[Tuple[float, float, float, float]] = None
    module_region_confidence: float = 1.0
    expert_module_region_bbox: Optional[Tuple[float, float, float, float]] = None
    module_region_source: str = 'prior_region'
    module_shape_hint: Optional[Dict[str, Any]] = None
    module_subregion: str = 'free'
    expert_xy: Optional[Tuple[float, float]] = None
    expert_rot: int = 0


@dataclass
class Task:
    bbox_mm: Tuple[float, float, float, float]
    grid_mm: float
    components: List[Component]
    nets: Dict[str, List[str]]  # optional
    sequence: List[str]  # resolved single source of truth used by PlacementEnv
    modules: Optional[List[Dict[str, Any]]] = None
    sequence_policy: str = "rebuild"
    sequence_source: str = "unspecified"
    sequence_meta: Optional[Dict[str, Any]] = None


@dataclass
class PlacementObjectiveConfig:
    hpwl_weight: float = 1.0
    w_hpwl_weight: float = 0.20
    nslw_weight: float = 0.05
    region_weight: float = 0.55
    module_region_weight: float = 0.35
    conn_weight: float = 0.50
    align_weight: float = 0.28
    group_weight: float = 0.12
    anchor_weight: float = 0.18
    boundary_group_weight: float = 0.22
    pitch_weight: float = 0.22
    orientation_weight: float = 0.14
    edge_clearance_weight: float = 0.40
    interior_weight: float = 0.30
    density_weight: float = 0.45
    soft_spacing_weight: float = 0.32
    neatness_weight: float = 0.12
    boundary_reward: float = 0.0


def _normalize_net_name(net: str) -> str:
    return str(net or '').strip().upper()


def _is_power_net(net: str) -> bool:
    n = _normalize_net_name(net)
    if not n:
        return False
    power_tokens = (
        'VCC', 'VDD', 'VAA', 'VEE', 'VBUS', 'VIN', 'VOUT', 'BAT', 'PWR', 'POWER',
        '3V', '5V', '12V', '1V', '1P', '2V', 'AVDD', 'DVDD', 'VREF',
    )
    return any(tok in n for tok in power_tokens)


def _is_ground_net(net: str) -> bool:
    n = _normalize_net_name(net)
    return n in ('', 'GND', 'GROUND', 'AGND', 'DGND', 'PGND')


_VALID_REGION_TYPES = {'edge_top', 'edge_bottom', 'edge_left', 'edge_right', 'core', 'free'}


def _normalize_semantic_text(value: Any, default: str = '') -> str:
    return str(value or default).strip().lower()


def _clamp01(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return float(max(0.0, min(1.0, v)))


def _component_constraint_level(comp: Component) -> str:
    raw = str(getattr(comp, 'constraint_level', '') or '').strip().lower()
    if raw in {'hard', 'soft', 'hint'}:
        return raw
    # Manual/mechanical constraints should be treated stronger than pure auto labels.
    src = str(getattr(comp, 'constraint_source', '') or '').strip().lower()
    if src in {'manual', 'expert', 'mechanical', 'schematic'}:
        return 'soft'
    return 'soft'


def _component_constraint_source(comp: Component) -> str:
    raw = str(getattr(comp, 'constraint_source', '') or '').strip().lower()
    return raw or 'auto'


def _source_strength_multiplier(source: str) -> float:
    source = str(source or '').strip().lower()
    if source in {'manual', 'expert', 'mechanical', 'schematic'}:
        return 1.0
    if source in {'derived', 'heuristic', 'auto'}:
        return 0.82
    if source in {'weak', 'guess', 'llm'}:
        return 0.65
    return 0.78


def _level_strength_multiplier(level: str) -> float:
    level = str(level or '').strip().lower()
    if level == 'hard':
        return 1.0
    if level == 'soft':
        return 0.78
    if level == 'hint':
        return 0.45
    return 0.75


def _allowed_side_to_region(side: str) -> str:
    s = _normalize_semantic_text(side)
    mapping = {
        'top': 'edge_top',
        'bottom': 'edge_bottom',
        'left': 'edge_left',
        'right': 'edge_right',
        'edge_top': 'edge_top',
        'edge_bottom': 'edge_bottom',
        'edge_left': 'edge_left',
        'edge_right': 'edge_right',
    }
    return mapping.get(s, 'free')




def _component_must_touch_boundary(comp: Component) -> bool:
    explicit = getattr(comp, 'must_touch_boundary', None)
    if explicit is not None:
        return bool(explicit)
    semantic = _normalize_semantic_text(getattr(comp, 'semantic_class', ''), '')
    if semantic in {'mechanical_edge_interface'}:
        return True
    return bool(is_edge_required_type(comp.type))

def _default_semantic_class(comp: Component) -> str:
    if is_connector_type(comp.type) or _component_must_touch_boundary(comp):
        return 'interface'
    tt = coarse_type_from_fine(comp.type)
    if tt == 'chip':
        return 'core'
    if tt in ('inductor', 'diode', 'transistor'):
        return 'power'
    if tt == 'mechanical':
        return 'mechanical'
    if tt in ('capacitor', 'resistor'):
        power_hits = 0
        signal_hits = 0
        for net, _ in comp.pads:
            if _is_ground_net(net):
                continue
            if _is_power_net(net):
                power_hits += 1
            else:
                signal_hits += 1
        if power_hits > signal_hits:
            return 'power_support'
        if signal_hits > 0:
            return 'passive'
    return 'support'


def _component_semantic_class(comp: Component) -> str:
    label = _normalize_semantic_text(getattr(comp, 'semantic_class', ''), '')
    return label or _default_semantic_class(comp)




def _component_edge_preference_capable(comp: Component) -> bool:
    ctype = str(getattr(comp, 'type', '') or '').lower()
    sem = _normalize_semantic_text(getattr(comp, 'semantic_class', ''), '')
    ref = str(getattr(comp, 'ref', '') or '').upper()
    prefix = re.match(r'[A-Z_]+', ref)
    prefix_text = prefix.group(0) if prefix else ref
    if getattr(comp, 'must_touch_boundary', None) or getattr(comp, 'allowed_sides', None):
        return True
    if is_connector_type(ctype) or _component_must_touch_boundary(comp):
        return True
    if sem in {'interface', 'ui', 'rf', 'mechanical', 'mechanical_edge_interface'}:
        return True
    if ctype.startswith('conn_') or ctype.startswith('mech_') or ctype in {'ui_button', 'ui_led', 'antenna', 'testpoint'}:
        return True
    if prefix_text in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'RJ', 'X', 'XA', 'SW', 'LED', 'S', 'H', 'MH', 'TP'}:
        return True
    return False


def _component_align_capable(comp: Component) -> bool:
    ctype = str(getattr(comp, 'type', '') or '').lower()
    sem = _normalize_semantic_text(getattr(comp, 'semantic_class', ''), '')
    ref = str(getattr(comp, 'ref', '') or '').upper()
    prefix = re.match(r'[A-Z_]+', ref)
    prefix_text = prefix.group(0) if prefix else ref
    if _component_edge_preference_capable(comp):
        return True
    if sem in {'interface', 'ui', 'rf', 'mechanical', 'mechanical_edge_interface'}:
        return True
    if ctype.startswith('conn_') or ctype.startswith('mech_') or ctype in {'ui_button', 'ui_led', 'testpoint'}:
        return True
    return prefix_text in {'J', 'P', 'JP', 'CN', 'CON', 'CONN', 'RJ', 'X', 'XA', 'SW', 'LED', 'S', 'H', 'MH', 'TP'}

def _component_explicit_functional_group(comp: Component) -> str:
    """Return only a real, fine-grained semantic functional group.

    module:<module_id> is a structural fallback, not an explicit semantic group.
    Keeping this distinction prevents the objective from treating every member
    of a module as a strong same-functional-group pair.
    """
    label = _normalize_semantic_text(getattr(comp, 'functional_group', ''), '')
    if not label or label in {'misc', 'other', 'free', 'none', 'null'}:
        return ''
    if label.startswith('module:'):
        return ''
    return label


def _component_functional_group(comp: Component) -> str:
    explicit = _component_explicit_functional_group(comp)
    if explicit:
        return explicit
    mid = str(getattr(comp, 'module_id', '') or '').strip()
    if mid:
        return f'module:{mid}'
    return _component_group_key(comp)


def _component_region_type(comp: Component) -> str:
    label = _normalize_semantic_text(getattr(comp, 'region_type', ''), '')
    if label in _VALID_REGION_TYPES:
        return label
    sclass = _component_semantic_class(comp)
    if sclass in {'core', 'clock'}:
        return 'core'
    side = _component_side_preference(comp)
    if side != 'free' and sclass in {'interface', 'ui', 'rf', 'mechanical'}:
        return side
    return 'free'


def _component_side_preference(comp: Component) -> str:
    label = _normalize_semantic_text(getattr(comp, 'side_preference', ''), '')
    edge_labels = {'edge_top', 'edge_bottom', 'edge_left', 'edge_right'}
    if label in edge_labels:
        return label if _component_edge_preference_capable(comp) else 'free'
    if label in {'core', 'free'}:
        return label
    if getattr(comp, 'allowed_sides', None):
        return _allowed_side_to_region(comp.allowed_sides[0])
    region = _normalize_semantic_text(getattr(comp, 'region_type', ''), '')
    if region in edge_labels and _component_edge_preference_capable(comp):
        return region
    return 'free'


def _component_align_group(comp: Component) -> Optional[str]:
    value = getattr(comp, 'align_group', None)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    if low.startswith('module:') or any(tok in low for tok in ('passive_support', 'power_supply', 'control_processing', 'module:')):
        return None
    if not _component_align_capable(comp):
        return None
    return text


def _component_anchor_ref(comp: Component) -> Optional[str]:
    # Prefer explicit component anchor, but fall back to module_anchor_ref so
    # module-local members are tied to their true anchor during train/infer.
    for attr in ('anchor_ref', 'module_anchor_ref'):
        value = getattr(comp, attr, None)
        if value is None:
            continue
        text = str(value).strip()
        if text and text != str(getattr(comp, 'ref', '')).strip():
            return text
    return None


def _component_subzone(comp: Component) -> str:
    value = _normalize_semantic_text(getattr(comp, 'subzone', ''), 'free')
    return value if value in {'left', 'right', 'top', 'bottom', 'around', 'free'} else 'free'


def _component_same_side_group(comp: Component) -> Optional[str]:
    value = getattr(comp, 'same_side_group', None)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _component_boundary_order(comp: Component) -> Optional[int]:
    value = getattr(comp, 'boundary_order', None)
    if value in (None, ''):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _component_align_axis(comp: Component) -> str:
    value = str(getattr(comp, 'align_axis', '') or '').strip().lower()
    if value in {'x', 'y', 'auto'}:
        return value
    # For edge groups, align on the edge coordinate and sort/pitch along the opposite axis.
    side = _component_side_preference(comp)
    if side in {'edge_left', 'edge_right'}:
        return 'x'
    if side in {'edge_top', 'edge_bottom'}:
        return 'y'
    row_axis = str(getattr(comp, 'row_axis', '') or '').strip().lower()
    if row_axis in {'x', 'y'}:
        # row_axis is the pitch/sort axis, so alignment is the perpendicular axis.
        return 'y' if row_axis == 'x' else 'x'
    return 'auto'


def _component_align_strength(comp: Component) -> float:
    return _clamp01(getattr(comp, 'align_strength', 1.0), 1.0)


def _component_boundary_order_source(comp: Component) -> str:
    raw = str(getattr(comp, 'boundary_order_source', '') or '').strip().lower()
    return raw or _component_constraint_source(comp)


def _component_spacing_policy(comp: Component) -> str:
    raw = str(getattr(comp, 'spacing_policy', '') or '').strip().lower()
    if raw in {'equal', 'preserve', 'free'}:
        return raw
    return 'equal'


def _component_pitch_group(comp: Component) -> Optional[str]:
    for attr in ('pitch_group', 'row_group'):
        value = getattr(comp, attr, None)
        if value not in (None, '', 'none', 'null'):
            text = str(value).strip()
            if text:
                return text
    return None


def _component_pitch_strength(comp: Component) -> float:
    return _clamp01(getattr(comp, 'pitch_strength', 1.0), 1.0)


def _component_row_axis(comp: Component) -> str:
    value = str(getattr(comp, 'row_axis', '') or '').strip().lower()
    return value if value in {'x', 'y', 'auto'} else 'auto'


def _component_row_order(comp: Component) -> Optional[int]:
    value = getattr(comp, 'row_order', None)
    if value in (None, ''):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _component_placement_role(comp: Component) -> str:
    value = getattr(comp, 'placement_role', None)
    text = str(value or '').strip().lower()
    if text in {'anchor_large', 'edge_anchor', 'main_anchor', 'member'}:
        return text
    if str(getattr(comp, 'module_role', '') or '').strip().lower() == 'anchor':
        return 'anchor_large'
    return 'member'


def _semantic_strength_for_component(comp: Component) -> float:
    """Reliability of semantic/layout annotations for this component.

    Backward compatibility:
    - Existing review_status/auto_confidence/needs_review still work.
    - New fields semantic_strength, constraint_source, and constraint_level can
      explicitly turn an auto label into a weak hint or a strong engineering rule.
    """
    explicit = getattr(comp, "semantic_strength", None)
    if explicit not in (None, ""):
        base = _clamp01(explicit, 1.0)
    else:
        status = str(getattr(comp, "review_status", "seeded") or "seeded").strip().lower()
        try:
            conf = float(getattr(comp, "auto_confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        needs_review = bool(getattr(comp, "needs_review", False))
        if status in {"approved", "edited", "manual_reviewed", "accepted", "confirmed"}:
            base = 1.0
        elif status in {"rejected", "ignored", "ignore", "discarded"}:
            base = 0.0
        elif status in {"seeded", "auto_seeded", "pending", "untracked", ""}:
            base = 0.6 + 0.2 * conf
        else:
            base = 0.55 + 0.25 * conf
        if needs_review and base < 0.999:
            base *= 0.65

    source_mul = _source_strength_multiplier(_component_constraint_source(comp))
    level_mul = _level_strength_multiplier(_component_constraint_level(comp))
    # Hard manual/mechanical constraints should not be weakened by the generic soft-level default.
    if _component_constraint_level(comp) == 'hard':
        level_mul = 1.0
    return float(max(0.0, min(1.0, float(base) * source_mul * level_mul)))




def _component_critical_nets(comp: Component) -> Tuple[str, ...]:
    vals = getattr(comp, 'critical_nets', ()) or ()
    out: List[str] = []
    for net in vals:
        nn = _normalize_net_name(net)
        if nn and (not _is_ground_net(nn)):
            out.append(nn)
    return tuple(dict.fromkeys(out))


def _component_critical_neighbor_specs(comp: Component) -> Tuple[Dict[str, Any], ...]:
    vals = list(getattr(comp, 'critical_neighbor_specs', ()) or ())
    # Backward-compatible fallback: old critical_neighbors can be ["U1"] or
    # [{"ref":"U1","weight":1.0,"reason":"decoupling"}].
    if not vals:
        vals = list(getattr(comp, 'critical_neighbors', ()) or ())
    self_ref = str(getattr(comp, 'ref', '') or '').strip()
    anchor_ref = str(getattr(comp, 'anchor_ref', '') or '').strip()
    module_anchor_ref = str(getattr(comp, 'module_anchor_ref', '') or '').strip()
    excluded = {v for v in (self_ref, anchor_ref, module_anchor_ref) if v}
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(vals):
        if isinstance(item, dict):
            ref = str(item.get('ref') or item.get('neighbor') or item.get('neighbor_ref') or '').strip()
            reason = str(item.get('reason') or item.get('type') or item.get('relation') or 'explicit').strip().lower()
            weight = item.get('weight', None)
            max_dist = item.get('max_distance_mm', item.get('max_dist_mm', None))
            subzone = item.get('preferred_subzone', item.get('subzone', None))
        else:
            ref = str(item or '').strip()
            reason = 'explicit'
            weight = None
            max_dist = None
            subzone = None
        if not ref or ref in excluded or ref in seen:
            continue
        try:
            w = float(weight) if weight not in (None, '') else max(0.35, 1.0 - 0.08 * float(idx))
        except Exception:
            w = max(0.35, 1.0 - 0.08 * float(idx))
        spec: Dict[str, Any] = {
            'ref': ref,
            'weight': float(max(0.0, min(3.0, w))),
            'reason': reason or 'explicit',
        }
        if max_dist not in (None, ''):
            try:
                spec['max_distance_mm'] = float(max_dist)
            except Exception:
                pass
        if subzone not in (None, ''):
            spec['preferred_subzone'] = str(subzone)
        out.append(spec)
        seen.add(ref)
    return tuple(out)


def _component_critical_neighbors(comp: Component) -> Tuple[str, ...]:
    return tuple(spec['ref'] for spec in _component_critical_neighbor_specs(comp) if spec.get('ref'))


def _component_group_key(comp: Component) -> str:
    tt = coarse_type_from_fine(comp.type)
    if is_connector_type(comp.type) or _component_must_touch_boundary(comp):
        return 'edge_io'
    if tt in ('chip',):
        return 'core_active'
    if tt in ('inductor', 'transistor', 'diode'):
        return 'power_active'
    if tt in ('capacitor', 'resistor'):
        power_hits = 0
        signal_hits = 0
        for net, _ in comp.pads:
            if _is_ground_net(net):
                continue
            if _is_power_net(net):
                power_hits += 1
            else:
                signal_hits += 1
        if power_hits > signal_hits:
            return 'power_passive'
        if signal_hits > 0:
            return 'signal_passive'
    if tt == 'mechanical':
        return 'mechanical'
    return 'misc'



def _assert_runtime_task_is_leakage_safe(task: Task) -> None:
    """PlacementEnv is runtime-only: expert labels must stay outside it.

    Expert xy/rot and expert module regions are supervision labels. Loading
    them into the environment makes it too easy for masks, context features or
    objective terms to depend on the answer.  Offline generators may still use
    task_from_json(..., load_expert=True), but such Tasks are intentionally
    rejected here.
    """
    leaked = []
    for comp in getattr(task, 'components', []) or []:
        ref = str(getattr(comp, 'ref', '') or '<unknown>')
        if getattr(comp, 'expert_xy', None) is not None:
            leaked.append(f'{ref}.expert_xy')
        if int(getattr(comp, 'expert_rot', 0) or 0) != 0:
            leaked.append(f'{ref}.expert_rot')
        if getattr(comp, 'expert_module_region_bbox', None) is not None:
            leaked.append(f'{ref}.expert_module_region_bbox')
        hint = getattr(comp, 'module_shape_hint', None)
        if isinstance(hint, dict) and not is_runtime_shape_hint_allowed(hint):
            leaked.append(f"{ref}.module_shape_hint: {describe_runtime_shape_hint_violation(hint)}")
    for module in getattr(task, 'modules', None) or []:
        if not isinstance(module, dict):
            continue
        mid = str(module.get('module_id', '') or '<module>')
        for key in ('expert_region_label', 'expert_region_heatmap', 'expert_bbox_mm'):
            if key in module:
                leaked.append(f'{mid}.{key}')
        prior = module.get('prior_region') if isinstance(module.get('prior_region'), dict) else None
        if prior is not None and not is_runtime_prior_allowed(prior):
            leaked.append(f"{mid}.prior_region: {describe_runtime_prior_violation(prior)}")
        elif prior is None and module.get('region_bbox_mm') is not None:
            leaked.append(f'{mid}.region_bbox_mm legacy prior without whitelisted prior_region')
        hint = module.get('shape_hint') or module.get('module_shape_hint')
        if isinstance(hint, dict) and not is_runtime_shape_hint_allowed(hint):
            leaked.append(f"{mid}.shape_hint: {describe_runtime_shape_hint_violation(hint)}")
    if leaked:
        preview = ', '.join(leaked[:10])
        more = '' if len(leaked) <= 10 else f', ... (+{len(leaked) - 10} more)'
        raise ValueError(
            'PlacementEnv received expert/train-only labels. '
            'Load runtime tasks with task_from_json(..., load_expert=False) '
            f'or sanitize the task first. Leaked fields: {preview}{more}'
        )

class PlacementEnv:
    """Deterministic grid placement environment with hard constraints and bias.
    Fixes vs baseline:
      - Illegal action now terminates env (prevents infinite loops).
      - Interface/boundary constraints use a tolerance (not strict equality).
      - allowed_sides / region_type / side_preference can define soft edge bands; only hard-boundary parts must touch the board edge.
      - Action mask is vectorized (much faster on larger boards).
      - Tracks placement order for building sequence tokens.
      - Reward objective and teacher objective delta use the same unified terms
        (HPWL + w_HPWL - NSLW + region + conn + align + group + edge_clearance + interior + density + soft_spacing + neatness).
    """

    def __init__(
        self,
        task: Task,
        min_spacing_mm: float = 0.2,
        rotations: Tuple[int, ...] = (0, 90, 180, 270),
        enforce_interface_on_boundary: bool = True,
        alignment_bonus: float = 0.05,
        edge_bonus: float = 0.15,
        hpwl_weight: float = 1.0,
        w_hpwl_weight: float = 0.20,
        nslw_weight: float = 0.05,
        edge_eps_mm: float | None = None,
        non_interface_edge_penalty: float = 10.0,
        non_interface_edge_margin_mm: float | None = None,
        density_penalty: float = 3.0,
        density_radius_mm: float | None = None,
        same_group_density_scale: float = 0.40,
        critical_neighbor_density_scale: float = 0.25,
        anchor_group_density_scale: float = 0.50,
        large_pair_density_scale: float = 1.20,
        same_group_soft_spacing_scale: float = 1.00,
        critical_neighbor_soft_spacing_scale: float = 0.90,
        anchor_group_soft_spacing_scale: float = 1.10,
        cross_group_soft_spacing_scale: float = 1.15,
        large_pair_soft_spacing_scale: float = 1.30,
        interior_penalty: float = 1.0,
        interior_margin_ratio: float = 0.18,
        region_weight: float = 0.55,
        module_region_weight: float = 0.35,
        module_region_bias: float = 0.20,
        module_region_margin_mm: float | None = None,
        conn_weight: float = 0.50,
        objective_align_weight: float = 0.28,
        group_weight: float = 0.12,
        anchor_weight: float = 0.18,
        boundary_group_weight: float = 0.22,
        pitch_weight: float = 0.22,
        orientation_weight: float = 0.14,
        edge_clearance_weight: float = 0.40,
        interior_weight: float = 0.30,
        density_weight: float = 0.45,
        soft_spacing_weight: float = 0.32,
        neatness_weight: float = 0.12,
        objective_edge_clearance_weight: float | None = None,
        objective_interior_weight: float | None = None,
        objective_density_weight: float | None = None,
        objective_soft_spacing_weight: float | None = None,
        objective_neatness_weight: float | None = None,
        edge_band_ratio: float = 0.12,
        edge_band_center_ratio: float = 0.55,
        soft_spacing_same_group_extra_mm: float = 0.6,
        soft_spacing_cross_group_extra_mm: float = 1.4,
        soft_spacing_large_extra_mm: float = 0.7,
    ):
        _assert_runtime_task_is_leakage_safe(task)
        self.task = task
        self.min_spacing = float(min_spacing_mm)
        self.rotations = rotations
        self.enforce_interface_on_boundary = enforce_interface_on_boundary
        self.alignment_bonus = float(alignment_bonus)
        self.edge_bonus = float(edge_bonus)
        self.edge_eps_mm = float(edge_eps_mm) if edge_eps_mm is not None else 1.5

        if objective_edge_clearance_weight is not None:
            edge_clearance_weight = float(objective_edge_clearance_weight)
        if objective_interior_weight is not None:
            interior_weight = float(objective_interior_weight)
        if objective_density_weight is not None:
            density_weight = float(objective_density_weight)
        if objective_soft_spacing_weight is not None:
            soft_spacing_weight = float(objective_soft_spacing_weight)
        if objective_neatness_weight is not None:
            neatness_weight = float(objective_neatness_weight)

        self.refs = [c.ref for c in task.components]
        self.comp_by_ref = {c.ref: c for c in task.components}
        self.sequence = list(task.sequence)

        self.module_region_bias = float(module_region_bias)
        self.module_region_margin_mm = (
            float(module_region_margin_mm)
            if module_region_margin_mm is not None
            else max(2.0 * float(task.grid_mm), 1.5)
        )
        self._component_module_id: Dict[str, str] = {
            c.ref: str(getattr(c, 'module_id', '') or '')
            for c in task.components
        }
        self._module_regions: Dict[str, Tuple[float, float, float, float]] = {}
        self._module_region_sources: Dict[str, str] = {}
        for m in (getattr(task, 'modules', None) or []):
            mid = str(m.get('module_id', '') or '')
            prior = m.get('prior_region') if isinstance(m.get('prior_region'), dict) else None
            if prior is None:
                continue
            # _assert_runtime_task_is_leakage_safe has already rejected unsafe priors;
            # keep this guard here so direct future callers cannot accidentally re-enable
            # legacy region_bbox_mm fallback as a runtime input.
            if not is_runtime_prior_allowed(prior):
                continue
            bbox = prior.get('bbox_mm')
            src = str(prior.get('source') or 'prior_region')
            if mid and bbox and len(bbox) == 4:
                self._module_regions[mid] = tuple(float(v) for v in bbox)
                self._module_region_sources[mid] = src
        self._module_anchor_refs: Dict[str, str] = {}
        self._module_region_confidence: Dict[str, float] = {}
        self._module_shape_hints: Dict[str, Dict[str, Any]] = {}
        for m in (getattr(task, 'modules', None) or []):
            mid = str(m.get('module_id', '') or '')
            if not mid:
                continue
            anchor = str(m.get('anchor_ref', '') or '')
            if anchor:
                self._module_anchor_refs[mid] = anchor
            prior = m.get('prior_region') if isinstance(m.get('prior_region'), dict) else None
            conf_val = prior.get('confidence') if isinstance(prior, dict) and is_runtime_prior_allowed(prior) else None
            if conf_val not in (None, ''):
                self._module_region_confidence[mid] = float(conf_val)
            hint = m.get('shape_hint') or m.get('module_shape_hint')
            if isinstance(hint, dict) and is_runtime_shape_hint_allowed(hint):
                self._module_shape_hints[mid] = dict(hint)
        for c in task.components:
            mid = str(getattr(c, 'module_id', '') or '')
            if mid and getattr(c, 'module_anchor_ref', None) and mid not in self._module_anchor_refs:
                self._module_anchor_refs[mid] = str(getattr(c, 'module_anchor_ref') or '')
            if mid and getattr(c, 'module_region_confidence', None) not in (None, '') and mid in self._module_regions and mid not in self._module_region_confidence:
                self._module_region_confidence[mid] = float(getattr(c, 'module_region_confidence'))
            hint = getattr(c, 'module_shape_hint', None)
            if mid and isinstance(hint, dict) and is_runtime_shape_hint_allowed(hint) and mid not in self._module_shape_hints:
                self._module_shape_hints[mid] = dict(hint)
        for mid in list(self._module_regions.keys()):
            if mid not in self._module_region_confidence:
                self._module_region_confidence[mid] = self._estimate_module_region_confidence(mid)
            if mid not in self._module_shape_hints:
                self._module_shape_hints[mid] = self._derive_module_shape_hint(mid)

        self.objective_cfg = PlacementObjectiveConfig(
            hpwl_weight=float(hpwl_weight),
            w_hpwl_weight=float(w_hpwl_weight),
            nslw_weight=float(nslw_weight),
            region_weight=float(region_weight),
            module_region_weight=float(module_region_weight),
            conn_weight=float(conn_weight),
            align_weight=float(objective_align_weight),
            group_weight=float(group_weight),
            anchor_weight=float(anchor_weight),
            boundary_group_weight=float(boundary_group_weight),
            pitch_weight=float(pitch_weight),
            orientation_weight=float(orientation_weight),
            edge_clearance_weight=float(edge_clearance_weight),
            interior_weight=float(interior_weight),
            density_weight=float(density_weight),
            soft_spacing_weight=float(soft_spacing_weight),
            neatness_weight=float(neatness_weight),
            boundary_reward=float(edge_bonus),
        )

        # Legacy-named shaping knobs remain public CLI/env parameters and are
        # intentionally multiplied into the corresponding objective terms below.
        # The objective_cfg.*_weight values are the outer term weights; these
        # scalars control the internal strength/shape of each legacy term.
        self.non_interface_edge_penalty = float(non_interface_edge_penalty)
        self.non_interface_edge_margin_mm = (
            float(non_interface_edge_margin_mm) if non_interface_edge_margin_mm is not None else 2.5
        )
        self.density_penalty = float(density_penalty)
        self.density_radius_mm = (
            float(density_radius_mm) if density_radius_mm is not None else max(4.0, 2.0 * float(task.grid_mm))
        )
        self.interior_penalty = float(interior_penalty)
        self.interior_margin_ratio = float(interior_margin_ratio)
        self.edge_band_ratio = float(edge_band_ratio)
        self.edge_band_center_ratio = float(edge_band_center_ratio)
        self.soft_spacing_same_group_extra_mm = float(soft_spacing_same_group_extra_mm)
        self.soft_spacing_anchor_or_critical_extra_mm = 0.5
        self.soft_spacing_cross_group_extra_mm = float(soft_spacing_cross_group_extra_mm)
        self.soft_spacing_large_extra_mm = float(soft_spacing_large_extra_mm)
        self.same_group_density_scale = float(same_group_density_scale)
        self.critical_neighbor_density_scale = float(critical_neighbor_density_scale)
        self.anchor_group_density_scale = float(anchor_group_density_scale)
        self.large_pair_density_scale = float(large_pair_density_scale)
        self.same_group_soft_spacing_scale = float(same_group_soft_spacing_scale)
        self.critical_neighbor_soft_spacing_scale = float(critical_neighbor_soft_spacing_scale)
        self.anchor_group_soft_spacing_scale = float(anchor_group_soft_spacing_scale)
        self.cross_group_soft_spacing_scale = float(cross_group_soft_spacing_scale)
        self.large_pair_soft_spacing_scale = float(large_pair_soft_spacing_scale)

        bx0, by0, bx1, by1 = self.task.bbox_mm
        self.board_diag = max(1e-6, math.hypot(bx1 - bx0, by1 - by0))
        self.board_area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        self._comp_nets_cache = {ref: self._component_nets(ref) for ref in self.refs}
        self._semantic_class = {ref: _component_semantic_class(self.comp_by_ref[ref]) for ref in self.refs}
        self._explicit_functional_group = {ref: _component_explicit_functional_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._functional_group = {ref: _component_functional_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._module_ids = {ref: str(getattr(self.comp_by_ref[ref], 'module_id', '') or '').strip() for ref in self.refs}
        self._region_targets = {ref: _component_region_type(self.comp_by_ref[ref]) for ref in self.refs}
        self._side_preferences = {ref: _component_side_preference(self.comp_by_ref[ref]) for ref in self.refs}
        self._align_groups = {ref: _component_align_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._align_axes = {ref: _component_align_axis(self.comp_by_ref[ref]) for ref in self.refs}
        self._align_strengths = {ref: _component_align_strength(self.comp_by_ref[ref]) for ref in self.refs}
        self._anchor_refs = {ref: _component_anchor_ref(self.comp_by_ref[ref]) for ref in self.refs}
        self._subzones = {ref: _component_subzone(self.comp_by_ref[ref]) for ref in self.refs}
        self._same_side_groups = {ref: _component_same_side_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._boundary_orders = {ref: _component_boundary_order(self.comp_by_ref[ref]) for ref in self.refs}
        self._boundary_order_sources = {ref: _component_boundary_order_source(self.comp_by_ref[ref]) for ref in self.refs}
        self._spacing_policies = {ref: _component_spacing_policy(self.comp_by_ref[ref]) for ref in self.refs}
        self._pitch_groups = {ref: _component_pitch_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._pitch_strengths = {ref: _component_pitch_strength(self.comp_by_ref[ref]) for ref in self.refs}
        self._row_axes = {ref: _component_row_axis(self.comp_by_ref[ref]) for ref in self.refs}
        self._row_orders = {ref: _component_row_order(self.comp_by_ref[ref]) for ref in self.refs}
        self._semantic_strengths = {ref: _semantic_strength_for_component(self.comp_by_ref[ref]) for ref in self.refs}
        self._placement_roles = {ref: _component_placement_role(self.comp_by_ref[ref]) for ref in self.refs}
        self._critical_nets = {ref: _component_critical_nets(self.comp_by_ref[ref]) for ref in self.refs}
        self._critical_neighbor_specs = {ref: _component_critical_neighbor_specs(self.comp_by_ref[ref]) for ref in self.refs}
        self._critical_neighbors = {ref: _component_critical_neighbors(self.comp_by_ref[ref]) for ref in self.refs}
        self._group_keys = dict(self._functional_group)
        self._suppress_overlapping_module_region_confidences()
        self._conn_weights = self._build_conn_weight_map()
        self.reset()

    @staticmethod
    def _bbox_iou_static(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(float(ax0), float(bx0))
        iy0 = max(float(ay0), float(by0))
        ix1 = min(float(ax1), float(bx1))
        iy1 = min(float(ay1), float(by1))
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        aa = max(0.0, float(ax1) - float(ax0)) * max(0.0, float(ay1) - float(ay0))
        bb = max(0.0, float(bx1) - float(bx0)) * max(0.0, float(by1) - float(by0))
        den = aa + bb - inter
        return float(inter / den) if den > 1e-9 else 0.0

    def _suppress_overlapping_module_region_confidences(self) -> None:
        """Downgrade conflicting auto module regions so priors do not fight each other.

        This does not delete module regions.  It only lowers confidence for lower-trust
        overlapping regions, which weakens module_region_penalty and prior heatmap pull.
        """
        mids = sorted(self._module_regions.keys())
        if len(mids) < 2:
            return
        strong_sources = {'manual', 'expert', 'mechanical', 'schematic'}
        for i, a in enumerate(mids):
            for b in mids[i + 1:]:
                iou = self._bbox_iou_static(self._module_regions[a], self._module_regions[b])
                if iou < 0.35:
                    continue
                ca = float(self._module_region_confidence.get(a, 1.0))
                cb = float(self._module_region_confidence.get(b, 1.0))
                sa = str(self._module_region_sources.get(a, 'auto') or 'auto').lower()
                sb = str(self._module_region_sources.get(b, 'auto') or 'auto').lower()
                a_strong = any(tok in sa for tok in strong_sources)
                b_strong = any(tok in sb for tok in strong_sources)
                if a_strong and not b_strong:
                    loser = b
                elif b_strong and not a_strong:
                    loser = a
                else:
                    loser = b if ca >= cb else a
                old_conf = float(self._module_region_confidence.get(loser, 1.0))
                # Higher overlap -> stronger downgrade, but keep a weak hint.
                new_conf = max(0.15, old_conf * max(0.30, 1.0 - 0.75 * iou))
                self._module_region_confidence[loser] = float(min(old_conf, new_conf))

    def _component_nets(self, ref: str) -> set[str]:
        nets = set()
        for net, _ in self.comp_by_ref[ref].pads:
            n = _normalize_net_name(net)
            if _is_ground_net(n):
                continue
            nets.add(n)
        return nets

    def _build_conn_weight_map(self) -> Dict[Tuple[str, str], float]:
        net_to_refs: Dict[str, List[str]] = {}
        for ref in self.refs:
            for net in self._component_nets(ref):
                net_to_refs.setdefault(net, []).append(ref)

        weights: Dict[Tuple[str, str], float] = {}

        def _add(a: str, b: str, w: float) -> None:
            if a == b or w <= 0.0:
                return
            key = (a, b) if a < b else (b, a)
            weights[key] = weights.get(key, 0.0) + float(w)

        # Primary semantic source: explicit critical-neighbor annotations.
        # New format supports per-neighbor weight/reason/max_distance_mm while
        # old critical_neighbors=["U1", ...] still maps to sensible defaults.
        reason_scale = {
            'decoupling': 1.35,
            'bypass': 1.30,
            'crystal_load': 1.45,
            'clock': 1.35,
            'esd': 1.30,
            'hot_loop': 1.50,
            'power_loop': 1.45,
            'switching_loop': 1.45,
            'connector_protection': 1.25,
            'explicit': 1.00,
        }
        for ref in self.refs:
            specs = list(self._critical_neighbor_specs.get(ref, ()) or ())
            if not specs:
                specs = [{'ref': nb, 'weight': max(0.35, 1.0 - 0.08 * rank), 'reason': 'explicit'} for rank, nb in enumerate(self._critical_neighbors.get(ref, ()))]
            for rank, spec in enumerate(specs):
                nb = str(spec.get('ref', '') or '').strip()
                if nb not in self.comp_by_ref:
                    continue
                try:
                    ann_weight = float(spec.get('weight', 1.0))
                except Exception:
                    ann_weight = 1.0
                reason = str(spec.get('reason', 'explicit') or 'explicit').strip().lower()
                base = max(0.45, 2.05 - 0.22 * float(rank)) * max(0.0, min(3.0, ann_weight))
                base *= float(reason_scale.get(reason, 1.0))
                if self._same_explicit_functional_group(ref, nb):
                    base += 0.15
                _add(ref, nb, base)

        # Secondary semantic source: critical nets are stronger than generic shared nets.
        for net, refs in net_to_refs.items():
            uniq_refs = list(dict.fromkeys(refs))
            deg = len(uniq_refs)
            if deg <= 1:
                continue
            net_scale = 1.0 / max(1.0, math.log1p(float(deg)))
            for i in range(len(uniq_refs)):
                for j in range(i + 1, len(uniq_refs)):
                    a, b = uniq_refs[i], uniq_refs[j]
                    crit = 0.0
                    if net in self._critical_nets.get(a, ()):
                        crit += 0.8
                    if net in self._critical_nets.get(b, ()):
                        crit += 0.8
                    base = 0.0
                    if crit > 0.0:
                        base = crit * net_scale
                    elif _is_power_net(net):
                        base = 0.28 * net_scale
                    else:
                        base = 0.08 * net_scale
                    _add(a, b, base)
        return weights

    def _pair_conn_weight(self, ref_a: str, ref_b: str) -> float:
        key = (ref_a, ref_b) if ref_a < ref_b else (ref_b, ref_a)
        return float(self._conn_weights.get(key, 0.0))

    def _ref_prefix(self, ref: str) -> str:
        return ''.join([ch for ch in str(ref) if not ch.isdigit()])

    def _align_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        align_group = self._align_groups.get(ref)
        if not align_group:
            return 0.0
        target_refs = [pref for pref in placed_order if pref != ref and self._align_groups.get(pref) == align_group]
        if not target_refs:
            return 0.0
        bx0, by0, bx1, by1 = self.task.bbox_mm
        bw = max(1e-6, bx1 - bx0)
        bh = max(1e-6, by1 - by0)
        axis = str(self._align_axes.get(ref, 'auto') or 'auto')
        vals: List[float] = []
        for pref in target_refs:
            px, py = placed_xy[pref]
            peer_axis = str(self._align_axes.get(pref, axis) or 'auto')
            use_axis = axis if axis in {'x', 'y'} else peer_axis
            if use_axis == 'x':
                vals.append(abs(x - px) / bw)
            elif use_axis == 'y':
                vals.append(abs(y - py) / bh)
            else:
                vals.append(min(abs(x - px) / bw, abs(y - py) / bh))
        vals.sort()
        top = vals[: min(3, len(vals))]
        strength = 0.5 * (float(self._align_strengths.get(ref, 1.0)) + float(self._semantic_strength(ref)))
        return float(max(0.0, strength) * sum(top) / max(1, len(top)))

    def _group_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        my_group = self._explicit_functional_group.get(ref, '')
        if not my_group:
            return 0.0
        group_refs = [r for r in placed_order if self._explicit_functional_group.get(r, '') == my_group]
        if not group_refs:
            return 0.0
        bbs: Dict[str, Tuple[float, float, float, float]] = {
            r: self._group_proxy_bbox(r, *placed_xy[r]) for r in group_refs
        }
        bbs[ref] = self._group_proxy_bbox(ref, x, y)
        full_group_refs = list(dict.fromkeys(group_refs + [ref]))
        span_pen = self._group_span_penalty_from_bbs(full_group_refs, bbs)
        sep_pen = self._group_separation_penalty(ref, float(x), float(y), placed_order, placed_xy)
        return float(self._semantic_strength(ref) * span_pen + sep_pen)

    def _align_delta_for_candidate(self, ref: str, x: float, y: float) -> float:
        current_order = list(self.placed_order)
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in current_order}
        new_order = current_order + [ref]
        new_xy = dict(current_xy)
        new_xy[ref] = (float(x), float(y))
        align_group = self._align_groups.get(ref)
        if not align_group:
            return 0.0
        affected = {ref}
        for pref in current_order:
            if self._align_groups.get(pref) == align_group:
                affected.add(pref)
        delta = 0.0
        for pref in affected:
            if pref == ref:
                old_pen = 0.0
                new_pen = self._align_penalty_for_layout(pref, float(x), float(y), new_order, new_xy)
            else:
                px, py, _ = self.placed[pref]
                old_pen = self._align_penalty_for_layout(pref, float(px), float(py), current_order, current_xy)
                new_pen = self._align_penalty_for_layout(pref, float(px), float(py), new_order, new_xy)
            delta += (new_pen - old_pen)
        return float(delta)

    def _group_delta_for_candidate(self, ref: str, x: float, y: float) -> float:
        current_order = list(self.placed_order)
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in current_order}
        new_order = current_order + [ref]
        new_xy = dict(current_xy)
        new_xy[ref] = (float(x), float(y))
        my_group = self._explicit_functional_group.get(ref, '')
        if not my_group:
            return 0.0
        affected = [pref for pref in current_order if self._explicit_functional_group.get(pref, '') == my_group]
        delta = 0.0
        for pref in affected:
            px, py, _ = self.placed[pref]
            old_pen = self._group_penalty_for_layout(pref, float(px), float(py), current_order, current_xy)
            new_pen = self._group_penalty_for_layout(pref, float(px), float(py), new_order, new_xy)
            delta += (new_pen - old_pen)
        delta += self._group_penalty_for_layout(ref, float(x), float(y), new_order, new_xy)
        return float(delta)


    def _anchor_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        anchor_ref = self._anchor_refs.get(ref)
        if (not anchor_ref) or anchor_ref == ref or anchor_ref not in placed_xy:
            return 0.0
        ax, ay = placed_xy[anchor_ref]
        dx = float(x - ax)
        dy = float(y - ay)
        dist = math.hypot(dx, dy)
        c = self.comp_by_ref[ref]
        a = self.comp_by_ref[anchor_ref]
        comp_span = 0.5 * math.hypot(float(c.size_mm[0]), float(c.size_mm[1]))
        anchor_span = 0.5 * math.hypot(float(a.size_mm[0]), float(a.size_mm[1]))
        min_r = max(self.task.grid_mm, 0.35 * (comp_span + anchor_span))
        max_r = max(4.0 * self.task.grid_mm, 1.35 * (comp_span + anchor_span), 0.18 * self.board_diag)
        pen = 0.0
        if dist < min_r:
            pen += ((min_r - dist) / self.board_diag) ** 2
        if dist > max_r:
            pen += ((dist - max_r) / self.board_diag) ** 2
        sub = self._subzones.get(ref, 'free')
        strength = max(0.0, self._semantic_strength(ref))
        if sub == 'left':
            pen += 1.60 * ((max(0.0, dx) / self.board_diag) ** 2) + 0.45 * ((max(0.0, abs(dy) - abs(dx)) / self.board_diag) ** 2)
        elif sub == 'right':
            pen += 1.60 * ((max(0.0, -dx) / self.board_diag) ** 2) + 0.45 * ((max(0.0, abs(dy) - abs(dx)) / self.board_diag) ** 2)
        elif sub == 'top':
            pen += 1.60 * ((max(0.0, -dy) / self.board_diag) ** 2) + 0.45 * ((max(0.0, abs(dx) - abs(dy)) / self.board_diag) ** 2)
        elif sub == 'bottom':
            pen += 1.60 * ((max(0.0, dy) / self.board_diag) ** 2) + 0.45 * ((max(0.0, abs(dx) - abs(dy)) / self.board_diag) ** 2)
        elif sub == 'around':
            target = 0.5 * (min_r + max_r)
            pen += 0.10 * (((dist - target) / self.board_diag) ** 2)
        return float(strength * pen)

    def _anchor_delta_for_candidate(self, ref: str, x: float, y: float) -> float:
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in self.placed_order}
        new_xy = dict(current_xy)
        new_xy[ref] = (float(x), float(y))
        affected = {ref}
        anchor_ref = self._anchor_refs.get(ref)
        if anchor_ref and anchor_ref in current_xy:
            affected.add(anchor_ref)
        for pref in self.placed_order:
            if self._anchor_refs.get(pref) == ref:
                affected.add(pref)
        delta = 0.0
        for pref in affected:
            if pref == ref:
                old_pen = 0.0
                new_pen = self._anchor_penalty_for_layout(pref, float(x), float(y), new_xy)
            elif pref in current_xy:
                px, py = current_xy[pref]
                old_pen = self._anchor_penalty_for_layout(pref, float(px), float(py), current_xy)
                new_pen = self._anchor_penalty_for_layout(pref, float(px), float(py), new_xy)
            else:
                continue
            delta += (new_pen - old_pen)
        return float(delta)

    def _boundary_axis_value(self, side: str, x: float, y: float) -> float:
        if side in {'edge_left', 'edge_right'}:
            return float(y)
        return float(x)

    def _boundary_group_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        group = self._same_side_groups.get(ref)
        if not group:
            return 0.0
        my_side = self._side_preferences.get(ref, self._region_targets.get(ref, 'free'))
        if my_side not in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            return 0.0
        my_order = self._boundary_orders.get(ref, None)
        me_axis = self._boundary_axis_value(my_side, x, y)
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        span = max(1e-6, (ymax - ymin) if my_side in {'edge_left', 'edge_right'} else (xmax - xmin))
        pen = 0.0
        strength = max(0.0, self._semantic_strength(ref))
        for pref in placed_order:
            if pref == ref or self._same_side_groups.get(pref) != group:
                continue
            peer_side = self._side_preferences.get(pref, self._region_targets.get(pref, 'free'))
            peer_strength = max(0.0, self._semantic_strength(pref))
            pair_strength = 0.5 * (strength + peer_strength)
            if peer_side != my_side:
                pen += pair_strength * 0.25
                continue
            peer_order = self._boundary_orders.get(pref, None)
            if my_order is None or peer_order is None or my_order == peer_order:
                continue
            if self._spacing_policies.get(ref, 'equal') == 'free' or self._spacing_policies.get(pref, 'equal') == 'free':
                continue
            src_a = str(self._boundary_order_sources.get(ref, 'auto') or 'auto')
            src_b = str(self._boundary_order_sources.get(pref, 'auto') or 'auto')
            order_strength = 1.0
            if src_a not in {'manual', 'expert', 'mechanical', 'schematic'}:
                order_strength *= 0.70
            if src_b not in {'manual', 'expert', 'mechanical', 'schematic'}:
                order_strength *= 0.70
            px, py = placed_xy[pref]
            peer_axis = self._boundary_axis_value(my_side, float(px), float(py))
            if my_order < peer_order:
                pen += pair_strength * order_strength * (max(0.0, me_axis - peer_axis) / span) ** 2
            else:
                pen += pair_strength * order_strength * (max(0.0, peer_axis - me_axis) / span) ** 2
        return float(pen)

    def _boundary_group_delta_for_candidate(self, ref: str, x: float, y: float) -> float:
        current_order = list(self.placed_order)
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in current_order}
        new_order = current_order + [ref]
        new_xy = dict(current_xy)
        new_xy[ref] = (float(x), float(y))
        group = self._same_side_groups.get(ref)
        if not group:
            return 0.0
        affected = {ref}
        for pref in current_order:
            if self._same_side_groups.get(pref) == group:
                affected.add(pref)
        delta = 0.0
        for pref in affected:
            if pref == ref:
                old_pen = 0.0
                new_pen = self._boundary_group_penalty_for_layout(pref, float(x), float(y), new_order, new_xy)
            else:
                px, py, _ = self.placed[pref]
                old_pen = self._boundary_group_penalty_for_layout(pref, float(px), float(py), current_order, current_xy)
                new_pen = self._boundary_group_penalty_for_layout(pref, float(px), float(py), new_order, new_xy)
            delta += (new_pen - old_pen)
        return float(delta)

    def _pitch_group_name(self, ref: str) -> Optional[str]:
        g = self._pitch_groups.get(ref)
        if g:
            return f'pitch::{g}'
        g = self._same_side_groups.get(ref)
        if g:
            if self._spacing_policies.get(ref, 'equal') == 'free':
                return None
            return f'side::{g}'
        g = self._align_groups.get(ref)
        if g:
            return f'align::{g}'
        return None

    def _pitch_group_refs(self, ref: str, placed_order: List[str]) -> List[str]:
        g = self._pitch_group_name(ref)
        if not g:
            return []
        return [pref for pref in placed_order if self._pitch_group_name(pref) == g]

    def _pitch_axis_for_group(self, refs: List[str], placed_xy: Dict[str, Tuple[float, float]]) -> str:
        # Explicit row_axis/pitch axis wins when present.
        row_votes = [self._row_axes.get(r, 'auto') for r in refs if self._row_axes.get(r, 'auto') in {'x', 'y'}]
        if row_votes:
            return max(sorted(set(row_votes)), key=row_votes.count)
        side_votes = [self._side_preferences.get(r, self._region_targets.get(r, 'free')) for r in refs]
        side = max(sorted(set(side_votes)), key=side_votes.count) if side_votes else 'free'
        if side in {'edge_left', 'edge_right'}:
            return 'y'
        if side in {'edge_top', 'edge_bottom'}:
            return 'x'
        align_votes = [self._align_axes.get(r, 'auto') for r in refs if self._align_axes.get(r, 'auto') in {'x', 'y'}]
        if align_votes:
            align_axis = max(sorted(set(align_votes)), key=align_votes.count)
            return 'y' if align_axis == 'x' else 'x'
        xs = [placed_xy[r][0] for r in refs if r in placed_xy]
        ys = [placed_xy[r][1] for r in refs if r in placed_xy]
        if len(xs) < 2:
            return 'x'
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        return 'y' if spread_x <= spread_y else 'x'

    def _pitch_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        refs = self._pitch_group_refs(ref, placed_order)
        if len(refs) < 3:
            return 0.0
        local_xy = dict(placed_xy)
        local_xy[ref] = (float(x), float(y))
        axis = self._pitch_axis_for_group(refs, local_xy)
        vals = sorted(float(local_xy[r][0] if axis == 'x' else local_xy[r][1]) for r in refs if r in local_xy)
        if len(vals) < 3:
            return 0.0
        gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if not gaps:
            return 0.0
        mean_gap = sum(gaps) / max(1, len(gaps))
        if mean_gap <= 1e-6:
            return 0.0
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        span = max(1e-6, (xmax - xmin) if axis == 'x' else (ymax - ymin))
        pen = sum((g - mean_gap) ** 2 for g in gaps) / max(1, len(gaps))
        strength = float(self._semantic_strength(ref)) * float(self._pitch_strengths.get(ref, 1.0))
        return float(max(0.0, strength) * pen / (span * span))

    def _pitch_delta_for_candidate(self, ref: str, x: float, y: float) -> float:
        current_order = list(self.placed_order)
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in current_order}
        new_order = current_order + [ref]
        new_xy = dict(current_xy)
        new_xy[ref] = (float(x), float(y))
        refs = self._pitch_group_refs(ref, new_order)
        if len(refs) < 3:
            return 0.0
        delta = 0.0
        affected = set(refs)
        for pref in affected:
            if pref == ref:
                old_pen = 0.0
                new_pen = self._pitch_penalty_for_layout(pref, float(x), float(y), new_order, new_xy)
            else:
                px, py, _ = self.placed[pref]
                old_pen = self._pitch_penalty_for_layout(pref, float(px), float(py), current_order, current_xy)
                new_pen = self._pitch_penalty_for_layout(pref, float(px), float(py), new_order, new_xy)
            delta += (new_pen - old_pen)
        return float(delta)

    def _orientation_group_name(self, ref: str) -> Optional[str]:
        g = self._same_side_groups.get(ref)
        if g:
            return f'side::{g}'
        g = self._align_groups.get(ref)
        if g:
            return f'align::{g}'
        return None

    def _orientation_group_refs(self, ref: str, placed_order: List[str]) -> List[str]:
        g = self._orientation_group_name(ref)
        if not g:
            return []
        return [pref for pref in placed_order if self._orientation_group_name(pref) == g]

    def _orientation_sensitive(self, ref: str) -> bool:
        c = self.comp_by_ref[ref]
        w, h = c.size_mm
        aspect = max(w, h) / max(1e-6, min(w, h))
        return bool(aspect >= 1.15 or is_connector_type(c.type) or c.allowed_sides)

    def _canonical_rot(self, rot: int) -> int:
        return int(rot) % 360

    def _default_orientation_for_side(self, side: str) -> int:
        side = str(side or 'free')
        if side == 'edge_left':
            return 90
        if side == 'edge_right':
            return 270
        if side == 'edge_top':
            return 0
        if side == 'edge_bottom':
            return 180
        return 0

    def _target_orientation_for_group(
        self,
        refs: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
        placed_rot: Dict[str, int],
    ) -> int:
        side_votes = [self._side_preferences.get(r, self._region_targets.get(r, 'free')) for r in refs]
        side_counts: Dict[str, int] = {}
        for side in side_votes:
            side_counts[str(side)] = side_counts.get(str(side), 0) + 1
        side = 'free'
        if side_counts:
            side = max(sorted(side_counts), key=lambda k: side_counts[k])

        abs_counts = {int(rot) % 360: 0 for rot in self.rotations}
        for r in refs:
            if r in placed_rot and self._orientation_sensitive(r):
                canon = self._canonical_rot(placed_rot[r])
                abs_counts[canon] = abs_counts.get(canon, 0) + 1

        if max(abs_counts.values(), default=0) > 0:
            return int(max(sorted(abs_counts), key=lambda rot: abs_counts[rot]))

        return self._default_orientation_for_side(side)

    def _orientation_penalty_for_layout(
        self,
        ref: str,
        rot: int,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
        placed_rot: Dict[str, int],
    ) -> float:
        if not self._orientation_sensitive(ref):
            return 0.0
        refs = self._orientation_group_refs(ref, placed_order)
        if len(refs) < 2:
            return 0.0
        local_rot = dict(placed_rot)
        local_rot[ref] = int(rot)
        target = self._target_orientation_for_group(refs, placed_xy, local_rot)
        pen = 0.0 if self._canonical_rot(int(rot)) == self._canonical_rot(int(target)) else 1.0
        return float(self._semantic_strength(ref) * pen)

    def _orientation_delta_for_candidate(self, ref: str, rot: int) -> float:
        current_order = list(self.placed_order)
        current_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in current_order}
        current_rot = {pref: int(self.placed[pref][2]) for pref in current_order}
        new_order = current_order + [ref]
        refs = self._orientation_group_refs(ref, new_order)
        if len(refs) < 2:
            return 0.0
        delta = 0.0
        affected = set(refs)
        for pref in affected:
            if pref == ref:
                old_pen = 0.0
                new_pen = self._orientation_penalty_for_layout(pref, int(rot), new_order, current_xy, current_rot)
            else:
                px, py, prot = self.placed[pref]
                old_pen = self._orientation_penalty_for_layout(pref, int(prot), current_order, current_xy, current_rot)
                new_rot = dict(current_rot)
                new_rot[ref] = int(rot)
                new_pen = self._orientation_penalty_for_layout(pref, int(prot), new_order, current_xy, new_rot)
            delta += (new_pen - old_pen)
        return float(delta)

    def _region_role(self, ref: str) -> str:
        return self._region_targets.get(ref, 'free')

    def _distance_to_region(self, bb: Tuple[float, float, float, float], region: str, side_pref: str) -> float:
        bx0, by0, bx1, by1 = self.task.bbox_mm
        bw = max(1e-6, bx1 - bx0)
        bh = max(1e-6, by1 - by0)
        cx, cy = self._bbox_center(bb)

        if region == 'edge_left':
            return self._distance_to_edge_band_target(bb, 'left')
        if region == 'edge_right':
            return self._distance_to_edge_band_target(bb, 'right')
        if region == 'edge_bottom':
            return self._distance_to_edge_band_target(bb, 'bottom')
        if region == 'edge_top':
            return self._distance_to_edge_band_target(bb, 'top')
        if region == 'core':
            mx = min(0.40 * bw, max(self.task.grid_mm, 0.28 * bw))
            my = min(0.40 * bh, max(self.task.grid_mm, 0.28 * bh))
            core_x0 = bx0 + mx
            core_x1 = bx1 - mx
            core_y0 = by0 + my
            core_y1 = by1 - my
            dx = 0.0
            if cx < core_x0:
                dx = core_x0 - cx
            elif cx > core_x1:
                dx = cx - core_x1
            dy = 0.0
            if cy < core_y0:
                dy = core_y0 - cy
            elif cy > core_y1:
                dy = cy - core_y1
            return math.hypot(dx, dy)
        if side_pref in {'edge_top', 'edge_bottom', 'edge_left', 'edge_right'}:
            return 0.35 * self._distance_to_region(bb, side_pref, 'free')
        return 0.0

    def _region_penalty_from_bbox(self, ref: str, bb: Tuple[float, float, float, float]) -> float:
        role = self._region_role(ref)
        side_pref = self._side_preferences.get(ref, 'free')
        dist = self._distance_to_region(bb, role, side_pref)
        return float(self._semantic_strength(ref) * ((dist / self.board_diag) ** 2))

    def _module_region_bbox_for_ref(self, ref: str) -> Optional[Tuple[float, float, float, float]]:
        mid = self._component_module_id.get(ref, '')
        if not mid:
            return None
        return self._module_regions.get(mid)

    def _module_id_for_ref(self, ref: str) -> str:
        return str(self._component_module_id.get(ref, '') or '')

    def _module_region_confidence_for_ref(self, ref: str) -> float:
        mid = self._module_id_for_ref(ref)
        if not mid:
            return 0.0
        return float(max(0.0, min(1.0, self._module_region_confidence.get(mid, 1.0))))

    def _module_shape_hint_for_ref(self, ref: str) -> Dict[str, Any]:
        mid = self._module_id_for_ref(ref)
        if not mid:
            return {}
        return self._module_shape_hints.get(mid, {}) or {}

    def _module_bbox_area(self, bb: Tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = [float(v) for v in bb]
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def _bbox_intersection_area(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax0, ay0, ax1, ay1 = [float(v) for v in a]
        bx0, by0, bx1, by1 = [float(v) for v in b]
        return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))

    def _estimate_module_region_confidence(self, mid: str) -> float:
        rb = self._module_regions.get(mid)
        if rb is None:
            return 0.0
        board_area = max(1e-6, self._module_bbox_area(tuple(self.task.bbox_mm)))
        area = max(1e-6, self._module_bbox_area(rb))
        area_frac = area / board_area
        if area_frac <= 0.15:
            conf = 1.0
        elif area_frac <= 0.35:
            conf = 1.0 - 0.45 * ((area_frac - 0.15) / 0.20)
        elif area_frac <= 0.75:
            conf = 0.55 - 0.37 * ((area_frac - 0.35) / 0.40)
        elif area_frac <= 0.98:
            conf = 0.18 - 0.14 * ((area_frac - 0.75) / 0.23)
        else:
            conf = 0.02
        overlap_area = 0.0
        for other_mid, other in self._module_regions.items():
            if other_mid == mid:
                continue
            overlap_area += self._bbox_intersection_area(rb, other)
        overlap_frac = overlap_area / area
        if overlap_frac > 0.0:
            conf *= max(0.20, 1.0 / (1.0 + overlap_frac))
        if not self._module_anchor_refs.get(mid):
            conf *= 0.85
        member_count = sum(1 for c in self.task.components if str(getattr(c, 'module_id', '') or '') == mid)
        if member_count <= 1:
            conf *= 0.75
        return float(max(0.02, min(1.0, conf)))

    def _ref_xy_hint(self, ref: str) -> Optional[Tuple[float, float]]:
        if ref in self.placed:
            x, y, _ = self.placed[ref]
            return float(x), float(y)
        # Runtime hints may only use already placed components.  Expert
        # coordinates are labels, not observable state.
        return None

    def _nearest_board_edge_name(self, x: float, y: float) -> str:
        xmin, ymin, xmax, ymax = [float(v) for v in self.task.bbox_mm]
        vals = {
            'edge_left': abs(x - xmin),
            'edge_right': abs(xmax - x),
            'edge_bottom': abs(y - ymin),
            'edge_top': abs(ymax - y),
        }
        return min(vals, key=vals.get)

    def _derive_module_shape_hint(self, mid: str) -> Dict[str, Any]:
        # Derive runtime shape hints only from leakage-safe module regions.
        # Expert component coordinates must not influence objective terms.
        rb = self._module_regions.get(mid) or tuple(self.task.bbox_mm)
        x0, y0, x1, y1 = [float(v) for v in rb]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        span_x, span_y = x1 - x0, y1 - y0
        axis = 'compact'
        if span_x > 1.35 * max(1e-6, span_y):
            axis = 'horizontal'
        elif span_y > 1.35 * max(1e-6, span_x):
            axis = 'vertical'
        anchor_ref = self._module_anchor_refs.get(mid, '')
        anchor_xy = self._ref_xy_hint(anchor_ref) if anchor_ref else None
        edge_corridor = ''
        if anchor_xy is not None:
            ac = self.comp_by_ref.get(anchor_ref)
            side = _normalize_semantic_text(getattr(ac, 'side_preference', '') if ac is not None else '', '')
            role = _normalize_semantic_text(getattr(ac, 'region_type', '') if ac is not None else '', '')
            if side in {'edge_top', 'edge_bottom', 'edge_left', 'edge_right'}:
                edge_corridor = side
            elif role in {'edge_top', 'edge_bottom', 'edge_left', 'edge_right'}:
                edge_corridor = role
            elif ac is not None and (is_connector_type(ac.type) or _component_must_touch_boundary(ac)):
                edge_corridor = self._nearest_board_edge_name(anchor_xy[0], anchor_xy[1])
        zone = 'around'
        median_dx = 0.0
        median_dy = 0.0
        radius = max(2.5 * float(self.task.grid_mm), 0.35 * max(span_x, span_y, float(self.task.grid_mm)))
        if edge_corridor:
            zone = 'edge_aligned'
        hint = {
            'version': 1,
            'schema_version': RUNTIME_SHAPE_HINT_SCHEMA_VERSION,
            'source': 'runtime_derived_module_geometry',
            'leakage_safe': True,
            'module_center_mm': [float(cx), float(cy)],
            'module_axis': axis,
            'anchor_relative_zone': zone,
            'edge_corridor': edge_corridor,
            'support_ring_radius_mm': float(radius),
            'median_offset_from_anchor_mm': [float(median_dx), float(median_dy)],
        }
        if anchor_xy is not None:
            hint['anchor_xy_mm'] = [float(anchor_xy[0]), float(anchor_xy[1])]
        return hint

    def _module_shape_hint_penalty_for_center(self, ref: str, x: float, y: float) -> float:
        hint = self._module_shape_hint_for_ref(ref)
        if not hint:
            return 0.0
        diag = max(1e-6, float(self.board_diag))
        p = 0.0
        center = hint.get('module_center_mm')
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
            p += 0.20 * ((math.hypot(float(x) - cx, float(y) - cy) / diag) ** 2)
        anchor_xy = hint.get('anchor_xy_mm')
        zone = _normalize_semantic_text(hint.get('anchor_relative_zone', 'around'), 'around')
        if isinstance(anchor_xy, (list, tuple)) and len(anchor_xy) == 2:
            ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
            tol = max(2.0 * float(self.task.grid_mm), 1.0)
            if zone == 'left':
                p += 0.55 * ((max(0.0, float(x) - (ax - tol)) / diag) ** 2)
            elif zone == 'right':
                p += 0.55 * ((max(0.0, (ax + tol) - float(x)) / diag) ** 2)
            elif zone == 'bottom':
                p += 0.55 * ((max(0.0, float(y) - (ay - tol)) / diag) ** 2)
            elif zone == 'top':
                p += 0.55 * ((max(0.0, (ay + tol) - float(y)) / diag) ** 2)
            elif zone in {'around', 'edge_aligned'}:
                radius = max(float(hint.get('support_ring_radius_mm') or 0.0), 2.5 * float(self.task.grid_mm))
                dist = math.hypot(float(x) - ax, float(y) - ay)
                p += 0.35 * ((max(0.0, dist - radius) / diag) ** 2)
        edge = _normalize_semantic_text(hint.get('edge_corridor', ''), '')
        if edge in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            xmin, ymin, xmax, ymax = [float(v) for v in self.task.bbox_mm]
            band = max(2.5 * float(self.task.grid_mm), 0.08 * min(max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)))
            if edge == 'edge_left':
                dist = float(x) - xmin
            elif edge == 'edge_right':
                dist = xmax - float(x)
            elif edge == 'edge_bottom':
                dist = float(y) - ymin
            else:
                dist = ymax - float(y)
            p += 0.40 * ((max(0.0, dist - band) / diag) ** 2)
        return float(min(1.0, max(0.0, p)))

    def _module_shape_hint_penalty_grid(self, ref: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        hint = self._module_shape_hint_for_ref(ref)
        if not hint:
            return np.zeros_like(x, dtype=np.float32)
        diag = max(1e-6, float(self.board_diag))
        p = np.zeros_like(x + y, dtype=np.float32)
        center = hint.get('module_center_mm')
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
            p += 0.20 * (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / diag) ** 2
        anchor_xy = hint.get('anchor_xy_mm')
        zone = _normalize_semantic_text(hint.get('anchor_relative_zone', 'around'), 'around')
        if isinstance(anchor_xy, (list, tuple)) and len(anchor_xy) == 2:
            ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
            tol = max(2.0 * float(self.task.grid_mm), 1.0)
            if zone == 'left':
                p += 0.55 * (np.maximum(0.0, x - (ax - tol)) / diag) ** 2
            elif zone == 'right':
                p += 0.55 * (np.maximum(0.0, (ax + tol) - x) / diag) ** 2
            elif zone == 'bottom':
                p += 0.55 * (np.maximum(0.0, y - (ay - tol)) / diag) ** 2
            elif zone == 'top':
                p += 0.55 * (np.maximum(0.0, (ay + tol) - y) / diag) ** 2
            elif zone in {'around', 'edge_aligned'}:
                radius = max(float(hint.get('support_ring_radius_mm') or 0.0), 2.5 * float(self.task.grid_mm))
                dist = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
                p += 0.35 * (np.maximum(0.0, dist - radius) / diag) ** 2
        edge = _normalize_semantic_text(hint.get('edge_corridor', ''), '')
        if edge in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            xmin, ymin, xmax, ymax = [float(v) for v in self.task.bbox_mm]
            band = max(2.5 * float(self.task.grid_mm), 0.08 * min(max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)))
            if edge == 'edge_left':
                dist = x - xmin
            elif edge == 'edge_right':
                dist = xmax - x
            elif edge == 'edge_bottom':
                dist = y - ymin
            else:
                dist = ymax - y
            p += 0.40 * (np.maximum(0.0, dist - band) / diag) ** 2
        return np.clip(p, 0.0, 1.0).astype(np.float32)

    def _module_region_penalty_from_bbox(self, ref: str, bb: Tuple[float, float, float, float]) -> float:
        """Confidence-scaled module-region hint.

        The bbox is treated as a coarse, low-confidence allowed zone. More
        specific shape hints (anchor-relative zone, edge corridor, support ring)
        provide the medium-confidence signal that tells the model where inside
        a broad/overlapping module region it should prefer to place a member.
        """
        rb = self._module_region_bbox_for_ref(ref)
        conf = self._module_region_confidence_for_ref(ref)
        bbox_penalty = 0.0
        if rb is not None and conf > 0.0:
            margin = float(self.module_region_margin_mm)
            x0, y0, x1, y1 = rb
            x0 -= margin
            y0 -= margin
            x1 += margin
            y1 += margin
            a, b, c, d = bb
            over_x = max(0.0, x0 - a) + max(0.0, c - x1)
            over_y = max(0.0, y0 - b) + max(0.0, d - y1)
            bbox_penalty = ((over_x + over_y) / max(1e-6, self.board_diag)) ** 2
        x = 0.5 * (float(bb[0]) + float(bb[2]))
        y = 0.5 * (float(bb[1]) + float(bb[3]))
        shape_penalty = self._module_shape_hint_penalty_for_center(ref, x, y)
        shape_conf = max(conf, 0.35 if self._module_shape_hint_for_ref(ref) else 0.0)
        return float(conf * bbox_penalty + shape_conf * shape_penalty)

    def _module_region_penalty_grid(
        self,
        ref: str,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        rb = self._module_region_bbox_for_ref(ref)
        conf = self._module_region_confidence_for_ref(ref)
        bbox_penalty = np.zeros_like(a + b, dtype=np.float32)
        if rb is not None and conf > 0.0:
            margin = float(self.module_region_margin_mm)
            x0, y0, x1, y1 = rb
            x0 -= margin
            y0 -= margin
            x1 += margin
            y1 += margin
            over_x = np.maximum(0.0, x0 - a) + np.maximum(0.0, c - x1)
            over_y = np.maximum(0.0, y0 - b) + np.maximum(0.0, d - y1)
            bbox_penalty = (((over_x + over_y) / max(1e-6, self.board_diag)) ** 2).astype(np.float32)
        x = 0.5 * (a + c)
        y = 0.5 * (b + d)
        shape_penalty = self._module_shape_hint_penalty_grid(ref, x, y)
        shape_conf = max(conf, 0.35 if self._module_shape_hint_for_ref(ref) else 0.0)
        return (conf * bbox_penalty + shape_conf * shape_penalty).astype(np.float32)

    def _conn_penalty_for_candidate(self, ref: str, x: float, y: float) -> float:
        total = 0.0
        for pref in self.placed_order:
            w = self._pair_conn_weight(ref, pref)
            if w <= 0.0:
                continue
            px, py, _ = self.placed[pref]
            total += w * ((abs(x - px) + abs(y - py)) / self.board_diag)
        return float(total)

    def _align_penalty_for_candidate(self, ref: str, x: float, y: float) -> float:
        placed_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in self.placed_order}
        return self._align_penalty_for_layout(ref, float(x), float(y), list(self.placed_order), placed_xy)

    def _group_penalty_for_candidate(self, ref: str, x: float, y: float) -> float:
        placed_xy = {pref: (float(self.placed[pref][0]), float(self.placed[pref][1])) for pref in self.placed_order}
        return self._group_penalty_for_layout(ref, float(x), float(y), list(self.placed_order), placed_xy)

    def objective_delta_mask(self, ref: str) -> Dict[str, np.ndarray]:
        w_cells, h_cells = self.grid_shape()
        xmin, ymin, _, _ = self.task.bbox_mm
        ix = np.arange(w_cells, dtype=np.float32)[:, None]
        iy = np.arange(h_cells, dtype=np.float32)[None, :]
        Xc = xmin + (ix + 0.5) * self.task.grid_mm
        Yc = ymin + (iy + 0.5) * self.task.grid_mm
        Xc = np.broadcast_to(Xc, (w_cells, h_cells)).astype(np.float32)
        Yc = np.broadcast_to(Yc, (w_cells, h_cells)).astype(np.float32)
        R = len(self.rotations)
        wire_terms = self.wire_delta_mask(ref)
        hpwl = wire_terms['hpwl'].astype(np.float32)
        w_hpwl = wire_terms['w_hpwl'].astype(np.float32)
        nslw = wire_terms['nslw'].astype(np.float32)
        region = np.zeros((R, w_cells, h_cells), dtype=np.float32)
        module_region = np.zeros_like(region)
        conn = np.zeros_like(region)
        align = np.zeros_like(region)
        group = np.zeros_like(region)
        anchor = np.zeros_like(region)
        boundary_group = np.zeros_like(region)
        pitch = np.zeros_like(region)
        orientation = np.zeros_like(region)
        edge_clearance = np.zeros_like(region)
        interior = np.zeros_like(region)
        density = np.zeros_like(region)
        soft_spacing = np.zeros_like(region)
        neatness = np.zeros_like(region)
        for ri, rot in enumerate(self.rotations):
            c = self.comp_by_ref[ref]
            w_mm, h_mm = c.size_mm
            if rot % 180 != 0:
                w_mm, h_mm = h_mm, w_mm
            a = Xc - w_mm / 2.0
            b = Yc - h_mm / 2.0
            cc = Xc + w_mm / 2.0
            d = Yc + h_mm / 2.0
            for xi in range(w_cells):
                for yi in range(h_cells):
                    bb = (float(a[xi, yi]), float(b[xi, yi]), float(cc[xi, yi]), float(d[xi, yi]))
                    x = float(Xc[xi, yi])
                    y = float(Yc[xi, yi])
                    region[ri, xi, yi] = self._region_penalty_from_bbox(ref, bb)
                    module_region[ri, xi, yi] = self._module_region_penalty_from_bbox(ref, bb)
                    conn[ri, xi, yi] = self._conn_penalty_for_candidate(ref, x, y)
                    align[ri, xi, yi] = self._align_delta_for_candidate(ref, x, y)
                    group[ri, xi, yi] = self._group_delta_for_candidate(ref, x, y)
                    anchor[ri, xi, yi] = self._anchor_delta_for_candidate(ref, x, y)
                    boundary_group[ri, xi, yi] = self._boundary_group_delta_for_candidate(ref, x, y)
                    pitch[ri, xi, yi] = self._pitch_delta_for_candidate(ref, x, y)
                    orientation[ri, xi, yi] = self._orientation_delta_for_candidate(ref, int(rot))
                    edge_clearance[ri, xi, yi] = self._edge_penalty_from_bbox(ref, bb)
                    interior[ri, xi, yi] = self._interior_penalty_from_center(ref, x, y)
                    density_total = 0.0
                    for pref in self.placed_order:
                        density_total += self._density_pair_penalty(ref, bb, pref, self._ref_bbox(pref, *self.placed[pref]))
                    density[ri, xi, yi] = float(density_total)
                    soft_spacing[ri, xi, yi] = self._soft_spacing_penalty_for_candidate(ref, bb)
                    neatness[ri, xi, yi] = self._neatness_penalty_for_candidate(ref, x, y, bb)
        total = (
            self.objective_cfg.hpwl_weight * hpwl
            + self.objective_cfg.w_hpwl_weight * w_hpwl
            - self.objective_cfg.nslw_weight * nslw
            + self.objective_cfg.region_weight * region
            + self.objective_cfg.module_region_weight * module_region
            + self.objective_cfg.conn_weight * conn
            + self.objective_cfg.align_weight * align
            + self.objective_cfg.group_weight * group
            + self.objective_cfg.anchor_weight * anchor
            + self.objective_cfg.boundary_group_weight * boundary_group
            + self.objective_cfg.pitch_weight * pitch
            + self.objective_cfg.orientation_weight * orientation
            + self.objective_cfg.edge_clearance_weight * edge_clearance
            + self.objective_cfg.interior_weight * interior
            + self.objective_cfg.density_weight * density
            + self.objective_cfg.soft_spacing_weight * soft_spacing
            + self.objective_cfg.neatness_weight * neatness
        ).astype(np.float32)
        return {
            'hpwl': hpwl,
            'w_hpwl': w_hpwl,
            'nslw': nslw,
            'region': region,
            'module_region': module_region,
            'conn': conn,
            'align': align,
            'group': group,
            'anchor': anchor,
            'boundary_group': boundary_group,
            'pitch': pitch,
            'orientation': orientation,
            'edge_clearance': edge_clearance,
            'interior': interior,
            'density': density,
            'soft_spacing': soft_spacing,
            'neatness': neatness,
            'total': total,
        }

    def reset(self):
        self.t = 0
        self.terminated = False
        self.placed: Dict[str, Tuple[float, float, int]] = {}  # ref -> (x,y,rot_deg)
        self.placed_order: List[str] = []
        self.occupied: List[Tuple[float, float, float, float]] = []  # list of bboxes (xmin,ymin,xmax,ymax)
        self.prev_obj = self._objective()
        return self.observe()

    def done(self) -> bool:
        return self.terminated or (self.t >= len(self.sequence))

    def current_ref(self) -> str:
        if self.done():
            raise IndexError("env is done")
        return self.sequence[self.t]

    def observe(self) -> Dict[str, Any]:
        ref = None if self.done() else self.sequence[self.t]
        mask, bias = self.action_mask_and_bias(ref) if ref else (None, None)
        return {
            "t": self.t,
            "ref": ref,
            "placed": dict(self.placed),
            "action_mask": mask,   # shape [R, X, Y]
            "action_bias": bias,   # shape [R, X, Y]
            "terminated": self.terminated,
        }

    def grid_shape(self) -> Tuple[int, int]:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        w = max(1, int(math.floor((xmax - xmin) / self.task.grid_mm)))
        h = max(1, int(math.floor((ymax - ymin) / self.task.grid_mm)))
        return w, h

    def _ref_bbox(self, ref: str, x: float, y: float, rot_deg: int) -> Tuple[float, float, float, float]:
        c = self.comp_by_ref[ref]
        w, h = c.size_mm
        if rot_deg % 180 != 0:
            w, h = h, w
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def _inside(self, bb: Tuple[float, float, float, float]) -> bool:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        a, b, c, d = bb
        return (a >= xmin) and (b >= ymin) and (c <= xmax) and (d <= ymax)

    def _touch_sides(self, bb: Tuple[float, float, float, float], allowed_sides: List[str]) -> bool:
        """Touch one of allowed sides within tolerance.
        allowed_sides: ["left","right","top","bottom"] (case-insensitive).
        """
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        a, b, c, d = bb
        eps = self.edge_eps_mm
        left = abs(a - xmin) <= eps
        right = abs(c - xmax) <= eps
        bottom = abs(b - ymin) <= eps
        top = abs(d - ymax) <= eps
        if not allowed_sides:
            return left or right or bottom or top
        s = {str(x).lower() for x in allowed_sides}
        ok = False
        if "left" in s:
            ok = ok or left
        if "right" in s:
            ok = ok or right
        if "bottom" in s:
            ok = ok or bottom
        if "top" in s:
            ok = ok or top
        return ok

    def _violates_spacing(self, bb: Tuple[float, float, float, float]) -> bool:
        a, b, c, d = bb
        eps = 1e-6  # mm，避免浮点边界误差
        for oa, ob, oc, od in self.occupied:
            oa2 = oa - self.min_spacing
            ob2 = ob - self.min_spacing
            oc2 = oc + self.min_spacing
            od2 = od + self.min_spacing
            if not (c <= oa2 + eps or a >= oc2 - eps or d <= ob2 + eps or b >= od2 - eps):
                return True
        return False

    def _wire_net_ignored(self, net: str) -> bool:
        """Return True for global nets that should not dominate wire objectives."""
        return _is_ground_net(net) or _is_power_net(net)

    def _placed_pin_records(self) -> Dict[str, List[Tuple[str, float, float]]]:
        """Absolute pin records by net: net -> [(ref, x, y), ...]."""
        pins_by_net: Dict[str, List[Tuple[str, float, float]]] = {}
        for ref, (x, y, rot) in self.placed.items():
            c = self.comp_by_ref[ref]
            angle = math.radians(rot)
            ca, sa = math.cos(angle), math.sin(angle)
            for net, (rx, ry) in c.pads:
                if self._wire_net_ignored(net):
                    continue
                ax = ca * rx - sa * ry
                ay = sa * rx + ca * ry
                pins_by_net.setdefault(str(net), []).append((ref, x + ax, y + ay))
        return pins_by_net

    def _candidate_pin_records(self, ref: str, x: float, y: float, rot: int) -> List[Tuple[str, float, float]]:
        """Absolute candidate pins: [(net, x, y), ...]."""
        c = self.comp_by_ref[ref]
        angle = math.radians(rot)
        ca, sa = math.cos(angle), math.sin(angle)
        out: List[Tuple[str, float, float]] = []
        for net, (rx, ry) in c.pads:
            if self._wire_net_ignored(net):
                continue
            ax = ca * rx - sa * ry
            ay = sa * rx + ca * ry
            out.append((str(net), x + ax, y + ay))
        return out

    def _closest_pad_edges(
        self,
        pad_xy: Tuple[float, float],
        bb: Tuple[float, float, float, float],
    ) -> set[str]:
        """Two nearest component edges for a pad, approximating PCBAgent fan-out rule."""
        px, py = pad_xy
        a, b, c, d = bb
        ds = [
            (abs(px - a), "left"),
            (abs(px - c), "right"),
            (abs(py - b), "bottom"),
            (abs(py - d), "top"),
        ]
        ds.sort(key=lambda x: x[0])
        return {ds[0][1], ds[1][1]}

    def _axis_edge_between(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        axis: str,
        reverse: bool = False,
    ) -> str:
        """Fan-out edge for p0 when connecting to p1 along the chosen axis."""
        x0, y0 = p0
        x1, y1 = p1
        if axis == "x":
            edge = "right" if x1 >= x0 else "left"
        else:
            edge = "top" if y1 >= y0 else "bottom"
        if reverse:
            edge = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[edge]
        return edge

    def _axis_segment_hits_rect(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        bb: Tuple[float, float, float, float],
        axis: str,
        eps: float = 1e-6,
    ) -> bool:
        """Check whether an axis-aligned segment intersects a component rectangle."""
        x0, y0 = p0
        x1, y1 = p1
        a, b, c, d = bb
        if axis == "x":
            y = 0.5 * (y0 + y1)
            lo, hi = sorted((x0, x1))
            return (b + eps < y < d - eps) and (max(lo, a) + eps < min(hi, c) - eps)
        x = 0.5 * (x0 + x1)
        lo, hi = sorted((y0, y1))
        return (a + eps < x < c - eps) and (max(lo, b) + eps < min(hi, d) - eps)

    def _is_surface_layer_wire_between(
        self,
        ref_a: str,
        pad_a: Tuple[float, float],
        ref_b: str,
        pad_b: Tuple[float, float],
        placed_bbs: Dict[str, Tuple[float, float, float, float]],
        align_tol_mm: float | None = None,
    ) -> bool:
        """Approximate PCBAgent SLW for a candidate wire between two pads.

        A wire is counted when:
        1. pads are aligned in horizontal or vertical projection;
        2. the fan-out direction from each component uses one of the pad's two
           nearest component edges;
        3. the straight axis-aligned segment does not intersect other components.
        """
        if align_tol_mm is None:
            align_tol_mm = max(0.05, 0.25 * float(self.task.grid_mm))
        xa, ya = pad_a
        xb, yb = pad_b
        axes: List[str] = []
        if abs(ya - yb) <= align_tol_mm:
            axes.append("x")
        if abs(xa - xb) <= align_tol_mm:
            axes.append("y")
        if not axes:
            return False

        bb_a = placed_bbs.get(ref_a)
        bb_b = placed_bbs.get(ref_b)
        if bb_a is None or bb_b is None:
            return False
        edges_a = self._closest_pad_edges(pad_a, bb_a)
        edges_b = self._closest_pad_edges(pad_b, bb_b)

        for axis in axes:
            edge_a = self._axis_edge_between(pad_a, pad_b, axis)
            edge_b = self._axis_edge_between(pad_b, pad_a, axis)
            if edge_a not in edges_a or edge_b not in edges_b:
                continue
            blocked = False
            for other_ref, bb in placed_bbs.items():
                if other_ref in (ref_a, ref_b):
                    continue
                if self._axis_segment_hits_rect(pad_a, pad_b, bb, axis):
                    blocked = True
                    break
            if not blocked:
                return True
        return False

    def _nslw_count_for_current_layout(self) -> int:
        """Count surface-layer wires in the current placement.

        This is the PCBAgent-style NSLW term. It is a count to maximize, unlike
        HPWL and w_HPWL terms which are costs to minimize.
        """
        if len(self.placed) < 2:
            return 0
        pin_records = self._placed_pin_records()
        placed_bbs = {
            ref: self._ref_bbox(ref, *self.placed[ref])
            for ref in self.placed.keys()
        }
        count = 0
        for net, recs in pin_records.items():
            if len(recs) <= 1:
                continue
            for i in range(len(recs)):
                ref_i, xi, yi = recs[i]
                for j in range(i + 1, len(recs)):
                    ref_j, xj, yj = recs[j]
                    if ref_i == ref_j:
                        continue
                    if self._is_surface_layer_wire_between(ref_i, (xi, yi), ref_j, (xj, yj), placed_bbs):
                        count += 1
        return int(count)


    def _alignment_sets(self) -> Tuple[set[int], set[int]]:
        """Alignment in index space to avoid float eps trouble."""
        if not self.placed:
            return set(), set()
        xmin, ymin, _, _ = self.task.bbox_mm
        g = self.task.grid_mm
        xs: set[int] = set()
        ys: set[int] = set()
        for _, (x, y, _) in self.placed.items():
            ix = int(round((x - xmin) / g - 0.5))
            iy = int(round((y - ymin) / g - 0.5))
            xs.add(ix)
            ys.add(iy)
        return xs, ys

    def _hard_boundary_required(self, ref: str) -> bool:
        return _component_must_touch_boundary(self.comp_by_ref[ref])

    def _edge_band_sides(self, ref: str) -> List[str]:
        """Hard boundary sides only.

        Semantic region / side preferences are intentionally excluded here so they
        cannot turn into hard legality constraints. Only explicit side constraints
        (allowed_sides) and true boundary-required components may restrict action
        legality.
        """
        comp = self.comp_by_ref[ref]
        sides: List[str] = []
        for side in (getattr(comp, 'allowed_sides', None) or []):
            s = _normalize_semantic_text(side)
            if s in {'left', 'right', 'top', 'bottom'} and s not in sides:
                sides.append(s)
        if not sides and self._hard_boundary_required(ref):
            region = self._region_targets.get(ref, 'free')
            if region in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
                sides.append(region.split('_', 1)[1])
        if not sides and self._hard_boundary_required(ref):
            side_pref = self._side_preferences.get(ref, 'free')
            if side_pref in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
                sides.append(side_pref.split('_', 1)[1])
        if not sides and self._hard_boundary_required(ref) and is_edge_preferred_type(comp.type):
            sides = ['left', 'right', 'top', 'bottom']
        return sides

    def _soft_edge_preference_sides(self, ref: str) -> List[str]:
        """Semantic edge hints used for bias/reward only, never hard legality."""
        comp = self.comp_by_ref[ref]
        sides: List[str] = []
        region = self._region_targets.get(ref, 'free')
        if region in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            sides.append(region.split('_', 1)[1])
        side_pref = self._side_preferences.get(ref, 'free')
        if side_pref in {'edge_left', 'edge_right', 'edge_top', 'edge_bottom'}:
            s = side_pref.split('_', 1)[1]
            if s not in sides:
                sides.append(s)
        if not sides and is_edge_preferred_type(comp.type):
            sides = ['left', 'right', 'top', 'bottom']
        return sides

    def _edge_band_width_mm(self) -> float:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        return float(max(self.task.grid_mm, self.edge_band_ratio * min(xmax - xmin, ymax - ymin)))

    def _semantic_strength(self, ref: str) -> float:
        return float(self._semantic_strengths.get(ref, 1.0))

    def _edge_target_clearance_mm(self, ref: str, side: str) -> float:
        _ = side
        band = float(self._edge_band_width_mm())
        sem = self._semantic_class.get(ref, '')
        role = self._placement_roles.get(ref, 'member')
        if role == 'edge_anchor' or sem in {'interface', 'mechanical_edge_interface'}:
            ratio = 0.15
        elif sem in {'ui', 'mechanical', 'rf'}:
            ratio = 0.25
        elif sem in {'interface_support', 'power_support'}:
            ratio = 0.35
        else:
            ratio = float(getattr(self, 'edge_band_center_ratio', 0.55))
        ratio = min(1.0, max(0.0, float(ratio)))
        return float(max(0.0, ratio * band))

    def _bbox_side_clearances(self, bb: Tuple[float, float, float, float]) -> Dict[str, float]:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        a, b, c, d = bb
        return {
            'left': float(a - xmin),
            'right': float(xmax - c),
            'bottom': float(b - ymin),
            'top': float(ymax - d),
        }

    def _within_edge_band(self, bb: Tuple[float, float, float, float], sides: List[str]) -> bool:
        if not sides:
            return False
        band = self._edge_band_width_mm()
        clearances = self._bbox_side_clearances(bb)
        return any(clearances.get(str(side).lower(), float('inf')) <= band + 1e-6 for side in sides)

    def _distance_to_edge_band_target(self, bb: Tuple[float, float, float, float], side: str, ref: Optional[str] = None) -> float:
        s = _normalize_semantic_text(side)
        if s.startswith('edge_'):
            s = s.split('_', 1)[1]
        clearances = self._bbox_side_clearances(bb)
        clearance = clearances.get(s, float('inf'))
        target = self._edge_target_clearance_mm(ref or '', s)
        return float(abs(clearance - target))

    def _is_regularized_non_interface(self, c: Component) -> bool:
        tt = (c.type or "").lower()
        if is_connector_type(tt):
            return False
        if _component_must_touch_boundary(c):
            return False
        if tt.startswith("mech_") or tt == "mechanical":
            return False
        return True

    def _bbox_center(self, bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
        a, b, c, d = bb
        return 0.5 * (a + c), 0.5 * (b + d)

    def _bbox_edge_clearance(self, bb: Tuple[float, float, float, float]) -> float:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        a, b, c, d = bb
        return float(min(a - xmin, xmax - c, b - ymin, ymax - d))

    def _edge_penalty_from_bbox(self, ref: str, bb: Tuple[float, float, float, float]) -> float:
        c = self.comp_by_ref[ref]
        if not self._is_regularized_non_interface(c):
            return 0.0
        margin = max(1e-6, float(self.non_interface_edge_margin_mm))
        clearance = self._bbox_edge_clearance(bb)
        deficit = max(0.0, margin - clearance)
        if deficit <= 0.0:
            return 0.0
        u = deficit / margin
        return float(max(0.0, float(self.non_interface_edge_penalty)) * (u * u))

    def _interior_penalty_from_center(self, ref: str, x: float, y: float) -> float:
        c = self.comp_by_ref[ref]
        if not self._is_regularized_non_interface(c):
            return 0.0
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        bw = max(1e-6, xmax - xmin)
        bh = max(1e-6, ymax - ymin)
        mx = min(0.45 * bw, max(self.task.grid_mm, bw * max(0.0, self.interior_margin_ratio)))
        my = min(0.45 * bh, max(self.task.grid_mm, bh * max(0.0, self.interior_margin_ratio)))
        inner_xmin = xmin + mx
        inner_xmax = xmax - mx
        inner_ymin = ymin + my
        inner_ymax = ymax - my
        dx = 0.0
        if x < inner_xmin:
            dx = inner_xmin - x
        elif x > inner_xmax:
            dx = x - inner_xmax
        dy = 0.0
        if y < inner_ymin:
            dy = inner_ymin - y
        elif y > inner_ymax:
            dy = y - inner_ymax
        if dx <= 0.0 and dy <= 0.0:
            return 0.0
        base = (dx / max(1e-6, mx)) ** 2 + (dy / max(1e-6, my)) ** 2
        return float(max(0.0, float(self.interior_penalty)) * base)

    def _same_explicit_functional_group(self, ref_a: str, ref_b: str) -> bool:
        ga = self._explicit_functional_group.get(ref_a, '')
        gb = self._explicit_functional_group.get(ref_b, '')
        return bool(ga and gb and ga == gb)

    def _same_module_pair(self, ref_a: str, ref_b: str) -> bool:
        ma = self._module_ids.get(ref_a, '')
        mb = self._module_ids.get(ref_b, '')
        return bool(ma and mb and ma == mb)

    def _direct_anchor_pair(self, ref_a: str, ref_b: str) -> bool:
        a_anchor = self._anchor_refs.get(ref_a)
        b_anchor = self._anchor_refs.get(ref_b)
        return bool((a_anchor and a_anchor == ref_b) or (b_anchor and b_anchor == ref_a))

    def _shared_anchor_pair(self, ref_a: str, ref_b: str) -> bool:
        a_anchor = self._anchor_refs.get(ref_a)
        b_anchor = self._anchor_refs.get(ref_b)
        return bool(a_anchor and b_anchor and a_anchor == b_anchor and not self._direct_anchor_pair(ref_a, ref_b))

    def _critical_pair(self, ref_a: str, ref_b: str) -> bool:
        return bool(
            ref_b in self._critical_neighbors.get(ref_a, ())
            or ref_a in self._critical_neighbors.get(ref_b, ())
        )

    def _pair_primary_relation(self, ref_a: str, ref_b: str) -> str:
        """Return a single dominant pair relation.

        Relation priority prevents the same pair from simultaneously acting as
        same_group + anchor + critical + connected.  Explicit fine-grained
        functional groups are strong; module:<id> fallback is intentionally
        downgraded to the weaker same_module relation.
        """
        if self._critical_pair(ref_a, ref_b):
            return 'critical'
        if self._direct_anchor_pair(ref_a, ref_b):
            return 'anchor'
        if self._same_explicit_functional_group(ref_a, ref_b):
            return 'functional_group'
        if self._same_module_pair(ref_a, ref_b):
            return 'module'
        if self._pair_conn_weight(ref_a, ref_b) > 1e-9:
            return 'connected'
        if self._shared_anchor_pair(ref_a, ref_b):
            return 'weak_related'
        return 'none'

    def _pair_relation_flags(self, ref_a: str, ref_b: str) -> Dict[str, bool]:
        relation = self._pair_primary_relation(ref_a, ref_b)

        sem_a = self._semantic_class.get(ref_a, '')
        sem_b = self._semantic_class.get(ref_b, '')
        group_a = self._group_keys.get(ref_a, '')
        group_b = self._group_keys.get(ref_b, '')
        large_pair = self._is_large_component(ref_a) and self._is_large_component(ref_b)
        core_support_pair = (
            ((sem_a == 'core') or (group_a == 'core_active'))
            and ((sem_b in {'support', 'power_support', 'passive'}) or (group_b in {'signal_passive', 'power_passive'}))
        ) or (
            ((sem_b == 'core') or (group_b == 'core_active'))
            and ((sem_a in {'support', 'power_support', 'passive'}) or (group_a in {'signal_passive', 'power_passive'}))
        )
        return {
            'relation': relation,
            'critical_pair': relation == 'critical',
            'direct_anchor_pair': relation == 'anchor',
            'same_functional_group': relation == 'functional_group',
            'same_module': relation == 'module',
            'connected_pair': relation == 'connected',
            'weak_related': relation == 'weak_related',
            # Compatibility aliases for older diagnostics.  These now reflect
            # gated primary relations instead of independent additive labels.
            'same_group': relation == 'functional_group',
            'shared_anchor': relation in {'anchor', 'weak_related'},
            'large_pair': bool(large_pair),
            'core_support_pair': bool(core_support_pair),
        }

    def _density_scale(self, ref_a: str, ref_b: str) -> float:
        flags = self._pair_relation_flags(ref_a, ref_b)
        relation = flags['relation']

        if relation == 'critical':
            scale = float(self.critical_neighbor_density_scale)
        elif relation == 'anchor':
            scale = float(self.anchor_group_density_scale)
        elif relation == 'functional_group':
            scale = float(self.same_group_density_scale)
        elif relation == 'module':
            # Same module is a coarse placement hint, not a strong collapse cue.
            scale = min(0.90, 0.5 * (1.0 + float(self.same_group_density_scale)))
        elif relation == 'connected':
            scale = 0.95
        elif relation == 'weak_related':
            scale = 0.98
        else:
            scale = 1.0

        if flags['large_pair'] or flags['core_support_pair']:
            scale = max(scale, float(self.large_pair_density_scale))
        return float(scale)

    def _soft_spacing_scale(self, ref_a: str, ref_b: str) -> float:
        flags = self._pair_relation_flags(ref_a, ref_b)
        if flags['large_pair'] or flags['core_support_pair']:
            return float(self.large_pair_soft_spacing_scale)
        relation = flags['relation']
        if relation == 'critical':
            return float(self.critical_neighbor_soft_spacing_scale)
        if relation == 'anchor':
            return float(self.anchor_group_soft_spacing_scale)
        if relation == 'functional_group':
            return float(self.same_group_soft_spacing_scale)
        if relation == 'module':
            # Keep same-module members organized without forcing tight packing.
            return max(float(self.same_group_soft_spacing_scale), 1.05)
        if relation == 'connected':
            return 1.05
        if relation == 'weak_related':
            return 1.10
        return float(self.cross_group_soft_spacing_scale)

    def _pair_compactness_scale(self, ref_a: str, ref_b: str) -> float:
        # Backwards-compatible alias for older code paths. Density and soft spacing
        # now use separate scales to avoid rewarding same-group over-collapse.
        scale = self._density_scale(ref_a, ref_b)
        return float(scale)

    def _group_proxy_bbox(self, ref: str, x: float, y: float) -> Tuple[float, float, float, float]:
        c = self.comp_by_ref[ref]
        w = max(float(self.task.grid_mm), float(c.size_mm[0]))
        h = max(float(self.task.grid_mm), float(c.size_mm[1]))
        return float(x - 0.5 * w), float(y - 0.5 * h), float(x + 0.5 * w), float(y + 0.5 * h)

    def _group_span_targets(self, refs: List[str]) -> Tuple[float, float, float, float, float, float]:
        n = max(1, len(refs))
        widths = [max(float(self.task.grid_mm), float(self.comp_by_ref[r].size_mm[0])) for r in refs]
        heights = [max(float(self.task.grid_mm), float(self.comp_by_ref[r].size_mm[1])) for r in refs]
        area_equiv = math.sqrt(max(1e-6, sum(w * h for w, h in zip(widths, heights))))
        spread_pitch = max(float(self.task.grid_mm), 0.65 * float(self.task.grid_mm) * max(1, n - 1))
        base_x = 0.58 * sum(widths) / max(1.0, math.sqrt(float(n))) + spread_pitch + 0.12 * area_equiv
        base_y = 0.58 * sum(heights) / max(1.0, math.sqrt(float(n))) + spread_pitch + 0.12 * area_equiv

        if n <= 3:
            min_scale = 0.92 + 0.12 * max(0, n - 2)
            max_scale = 2.15 + 0.12 * max(0, n - 2)
        else:
            min_scale = 1.18 + 0.12 * min(4, n - 4)
            max_scale = 2.65 + 0.18 * min(4, n - 4)

        role_mult = 1.0
        role_semantics = {self._semantic_class.get(r, '') for r in refs}
        role_groups = {self._group_keys.get(r, '') for r in refs}
        if role_semantics & {'core', 'power', 'power_support', 'support', 'clock'}:
            role_mult = max(role_mult, 1.18)
        if role_groups & {'core_active', 'power_active', 'signal_passive', 'power_passive'}:
            role_mult = max(role_mult, 1.12)

        min_x = max(2.0 * float(self.task.grid_mm), base_x * min_scale * role_mult)
        min_y = max(2.0 * float(self.task.grid_mm), base_y * min_scale * role_mult)
        max_x = max(min_x + float(self.task.grid_mm), base_x * max_scale * role_mult)
        max_y = max(min_y + float(self.task.grid_mm), base_y * max_scale * role_mult)
        min_area = 0.72 * min_x * min_y
        max_area = 1.35 * max_x * max_y
        return float(min_x), float(max_x), float(min_y), float(max_y), float(min_area), float(max_area)

    def _group_span_penalty_from_bbs(
        self,
        refs: List[str],
        bbs: Dict[str, Tuple[float, float, float, float]],
    ) -> float:
        uniq_refs = [r for r in refs if r in bbs]
        if len(uniq_refs) <= 1:
            return 0.0
        min_x = min(bbs[r][0] for r in uniq_refs)
        max_x = max(bbs[r][2] for r in uniq_refs)
        min_y = min(bbs[r][1] for r in uniq_refs)
        max_y = max(bbs[r][3] for r in uniq_refs)
        span_x = max(0.0, float(max_x - min_x))
        span_y = max(0.0, float(max_y - min_y))
        area = max(1e-6, span_x * span_y)
        tgt_min_x, tgt_max_x, tgt_min_y, tgt_max_y, tgt_min_area, tgt_max_area = self._group_span_targets(uniq_refs)
        collapse_x = max(0.0, tgt_min_x - span_x) / max(1e-6, tgt_min_x)
        collapse_y = max(0.0, tgt_min_y - span_y) / max(1e-6, tgt_min_y)
        spread_x = max(0.0, span_x - tgt_max_x) / max(1e-6, tgt_max_x)
        spread_y = max(0.0, span_y - tgt_max_y) / max(1e-6, tgt_max_y)
        collapse_area = max(0.0, tgt_min_area - area) / max(1e-6, tgt_min_area)
        spread_area = max(0.0, area - tgt_max_area) / max(1e-6, tgt_max_area)
        return float(
            0.28 * (collapse_x * collapse_x + collapse_y * collapse_y)
            + 0.12 * (collapse_area * collapse_area)
            + 0.18 * (spread_x * spread_x + spread_y * spread_y)
            + 0.06 * (spread_area * spread_area)
        )

    def _group_separation_pair_weight(self, group_a: str, group_b: str) -> float:
        ga = str(group_a or 'misc')
        gb = str(group_b or 'misc')
        key = frozenset((ga, gb))
        strong = {
            frozenset(("core_active", "interface_support")): 1.55,
            frozenset(("power_active", "signal_passive")): 1.45,
            frozenset(("rf", "misc")): 1.35,
            frozenset(("ui", "misc")): 1.30,
            frozenset(("mechanical", "signal_passive")): 1.35,
            frozenset(("mechanical", "power_passive")): 1.35,
        }
        return float(strong.get(key, 1.0))

    def _group_separation_penalty(self, ref: str, x: float, y: float, placed_order: List[str], placed_xy: Dict[str, Tuple[float, float]]) -> float:
        my_group = self._explicit_functional_group.get(ref, '')
        if not my_group:
            return 0.0
        my_bb = self._group_proxy_bbox(ref, float(x), float(y))
        penalty = 0.0
        seen_groups = set()
        for pref in placed_order:
            if pref == ref:
                continue
            peer_group = self._explicit_functional_group.get(pref, '')
            if not peer_group or peer_group == my_group or peer_group in seen_groups:
                continue
            members = [r for r in placed_order if self._explicit_functional_group.get(r, '') == peer_group and r in placed_xy]
            if not members:
                continue
            seen_groups.add(peer_group)
            peer_bbs = {r: self._group_proxy_bbox(r, *placed_xy[r]) for r in members}
            xmin = min(bb[0] for bb in peer_bbs.values())
            ymin = min(bb[1] for bb in peer_bbs.values())
            xmax = max(bb[2] for bb in peer_bbs.values())
            ymax = max(bb[3] for bb in peer_bbs.values())
            dx = max(0.0, max(xmin - my_bb[2], my_bb[0] - xmax))
            dy = max(0.0, max(ymin - my_bb[3], my_bb[1] - ymax))
            bbox_gap = dy if dx <= 0.0 else dx if dy <= 0.0 else math.hypot(dx, dy)
            peer_cx = sum(placed_xy[r][0] for r in members) / len(members)
            peer_cy = sum(placed_xy[r][1] for r in members) / len(members)
            centroid_gap = math.hypot(float(x) - peer_cx, float(y) - peer_cy)
            min_bbox_gap = 1.2 * self.task.grid_mm
            min_centroid_gap = max(2.0 * self.task.grid_mm, 0.05 * self.board_diag)
            weight = self._group_separation_pair_weight(my_group, peer_group)
            if bbox_gap < min_bbox_gap:
                penalty += weight * ((min_bbox_gap - bbox_gap) / max(1e-6, min_bbox_gap)) ** 2
            if centroid_gap < min_centroid_gap:
                penalty += 0.6 * weight * ((min_centroid_gap - centroid_gap) / max(1e-6, min_centroid_gap)) ** 2
        return float(self._semantic_strength(ref) * penalty)

    def _density_pair_penalty(
        self,
        ref_a: str,
        bb_a: Tuple[float, float, float, float],
        ref_b: str,
        bb_b: Tuple[float, float, float, float],
    ) -> float:
        ca = self.comp_by_ref[ref_a]
        cb = self.comp_by_ref[ref_b]
        if (not self._is_regularized_non_interface(ca)) or (not self._is_regularized_non_interface(cb)):
            return 0.0
        xa, ya = self._bbox_center(bb_a)
        xb, yb = self._bbox_center(bb_b)
        dist = math.hypot(xa - xb, ya - yb)
        da = math.hypot(float(ca.size_mm[0]), float(ca.size_mm[1]))
        db = math.hypot(float(cb.size_mm[0]), float(cb.size_mm[1]))
        soft_radius = 0.5 * (da + db) + float(self.density_radius_mm)
        if soft_radius <= 1e-6:
            return 0.0
        overflow = max(0.0, soft_radius - dist)
        if overflow <= 0.0:
            return 0.0
        u = overflow / soft_radius
        return float(max(0.0, float(self.density_penalty)) * self._density_scale(ref_a, ref_b) * (u * u))

    def _bbox_gap(
        self,
        bb_a: Tuple[float, float, float, float],
        bb_b: Tuple[float, float, float, float],
    ) -> float:
        ax0, ay0, ax1, ay1 = bb_a
        bx0, by0, bx1, by1 = bb_b
        dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
        dy = max(0.0, max(by0 - ay1, ay0 - by1))
        if dx <= 0.0:
            return float(dy)
        if dy <= 0.0:
            return float(dx)
        return float(math.hypot(dx, dy))

    def _is_large_component(self, ref: str) -> bool:
        c = self.comp_by_ref[ref]
        diag = math.hypot(float(c.size_mm[0]), float(c.size_mm[1]))
        sem = self._semantic_class.get(ref, '')
        group = self._group_keys.get(ref, '')
        role = self._placement_roles.get(ref, 'member')
        return bool(
            role in {'anchor_large', 'edge_anchor', 'main_anchor'}
            or str(getattr(c, 'module_role', '') or '').strip().lower() == 'anchor'
            or diag >= max(6.0 * self.task.grid_mm, 0.12 * self.board_diag)
            or sem in {'core', 'power', 'power_support', 'mechanical'}
            or group in {'core_active', 'power_active'}
        )

    def _preferred_gap_mm(self, ref_a: str, ref_b: str) -> float:
        preferred = float(self.min_spacing)
        relation = self._pair_primary_relation(ref_a, ref_b)
        if relation == 'critical':
            preferred += 0.5 * self.soft_spacing_anchor_or_critical_extra_mm
        elif relation == 'anchor':
            preferred += self.soft_spacing_anchor_or_critical_extra_mm
        elif relation == 'functional_group':
            preferred += self.soft_spacing_same_group_extra_mm
        elif relation == 'module':
            # Module fallback should not behave like a strong semantic group;
            # leave enough air so the whole module does not collapse.
            preferred += max(self.soft_spacing_same_group_extra_mm, 0.65 * self.soft_spacing_cross_group_extra_mm)
        elif relation == 'connected':
            preferred += 0.75 * self.soft_spacing_cross_group_extra_mm
        elif relation == 'weak_related':
            preferred += 0.85 * self.soft_spacing_cross_group_extra_mm
        else:
            preferred += self.soft_spacing_cross_group_extra_mm
        if self._is_large_component(ref_a) or self._is_large_component(ref_b):
            preferred += self.soft_spacing_large_extra_mm
        return float(max(self.min_spacing, preferred))

    def _soft_spacing_pair_penalty(
        self,
        ref_a: str,
        bb_a: Tuple[float, float, float, float],
        ref_b: str,
        bb_b: Tuple[float, float, float, float],
    ) -> float:
        ca = self.comp_by_ref[ref_a]
        cb = self.comp_by_ref[ref_b]
        if (not self._is_regularized_non_interface(ca)) or (not self._is_regularized_non_interface(cb)):
            return 0.0
        gap = self._bbox_gap(bb_a, bb_b)
        if gap < float(self.min_spacing):
            return 0.0
        preferred_gap = self._preferred_gap_mm(ref_a, ref_b)
        if gap >= preferred_gap:
            return 0.0
        denom = max(1e-6, preferred_gap - float(self.min_spacing))
        u = (preferred_gap - gap) / denom
        return float(self._soft_spacing_scale(ref_a, ref_b) * (u * u))

    def _soft_spacing_penalty_for_candidate(
        self,
        ref: str,
        bb: Tuple[float, float, float, float],
    ) -> float:
        total = 0.0
        for pref in self.placed_order:
            if pref == ref:
                continue
            total += self._soft_spacing_pair_penalty(ref, bb, pref, self._ref_bbox(pref, *self.placed[pref]))
        return float(total)

    def _line_neatness_penalty_for_candidate(
        self,
        ref: str,
        x: float,
        y: float,
    ) -> float:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        board_w = max(1e-6, float(xmax - xmin))
        board_h = max(1e-6, float(ymax - ymin))
        my_group = self._explicit_functional_group.get(ref, '')
        my_align = self._align_groups.get(ref)
        nearest_x = None
        nearest_y = None
        for pref in self.placed_order:
            if pref == ref or pref not in self.placed:
                continue
            if self._hard_boundary_required(pref):
                continue
            px, py, _ = self.placed[pref]
            weight = 1.0
            if my_align and self._align_groups.get(pref) == my_align:
                weight = max(weight, 2.0)
            if my_group and self._explicit_functional_group.get(pref, '') == my_group:
                weight = max(weight, 1.6)
            dx = abs(float(x) - float(px)) / weight
            dy = abs(float(y) - float(py)) / weight
            nearest_x = dx if nearest_x is None else min(nearest_x, dx)
            nearest_y = dy if nearest_y is None else min(nearest_y, dy)
        if nearest_x is None or nearest_y is None:
            return 0.0
        return float(min(nearest_x / board_w, nearest_y / board_h) ** 2)

    def _local_directional_clearances(
        self,
        ref: str,
        bb: Tuple[float, float, float, float],
        placed_bbs: Dict[str, Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        a, b, c, d = bb
        left = max(0.0, float(a - xmin))
        right = max(0.0, float(xmax - c))
        bottom = max(0.0, float(b - ymin))
        top = max(0.0, float(ymax - d))
        for pref, other in placed_bbs.items():
            if pref == ref:
                continue
            oa, ob, oc, od = other
            overlap_y = min(d, od) - max(b, ob)
            overlap_x = min(c, oc) - max(a, oa)
            if overlap_y > 0.0:
                if oc <= a:
                    left = min(left, max(0.0, float(a - oc)))
                elif oa >= c:
                    right = min(right, max(0.0, float(oa - c)))
            if overlap_x > 0.0:
                if od <= b:
                    bottom = min(bottom, max(0.0, float(b - od)))
                elif ob >= d:
                    top = min(top, max(0.0, float(ob - d)))
        return float(left), float(right), float(top), float(bottom)

    def _whitespace_balance_penalty_for_layout(
        self,
        ref: str,
        bb: Tuple[float, float, float, float],
        placed_bbs: Dict[str, Tuple[float, float, float, float]],
    ) -> float:
        large_like = self._is_large_component(ref) or self._semantic_class.get(ref, '') in {
            'core', 'power', 'power_support', 'clock', 'support'
        }
        role_weight = 1.0 if large_like else 0.35
        if role_weight <= 0.0:
            return 0.0
        left, right, top, bottom = self._local_directional_clearances(ref, bb, placed_bbs)
        eps = 1e-6
        lr = ((left - right) / (left + right + eps)) ** 2
        tb = ((top - bottom) / (top + bottom + eps)) ** 2
        return float(role_weight * (lr + tb))

    def _whitespace_floor_profile(self, ref: str) -> Tuple[float, float]:
        sem = self._semantic_class.get(ref, '')
        group = self._group_keys.get(ref, '')
        if sem in {'interface', 'mechanical_edge_interface', 'ui'}:
            return 0.0, 0.0
        if self._is_large_component(ref) or sem in {'core', 'mechanical'} or group in {'core_active'}:
            return 2.0, 1.0
        if sem in {'power', 'power_support', 'support', 'clock'} or group in {'power_active'}:
            return 1.5, 0.8
        if sem == 'passive' or group in {'signal_passive', 'power_passive'}:
            return 1.0, 0.4
        return 0.9, 0.25

    def _whitespace_floor_penalty_for_layout(
        self,
        ref: str,
        bb: Tuple[float, float, float, float],
        placed_bbs: Dict[str, Tuple[float, float, float, float]],
    ) -> float:
        target_mm, role_weight = self._whitespace_floor_profile(ref)
        if role_weight <= 0.0 or target_mm <= 0.0:
            return 0.0
        vals = list(self._local_directional_clearances(ref, bb, placed_bbs))
        vals.sort()
        deficits = [max(0.0, target_mm - v) / max(1e-6, target_mm) for v in vals[:2]]
        return float(role_weight * sum(d * d for d in deficits) / max(1, len(deficits)))

    def _neatness_penalty_for_candidate(
        self,
        ref: str,
        x: float,
        y: float,
        bb: Tuple[float, float, float, float],
    ) -> float:
        line_neatness = self._line_neatness_penalty_for_candidate(ref, x, y)
        placed_bbs: Dict[str, Tuple[float, float, float, float]] = {}
        for pref in self.placed_order:
            if pref == ref:
                continue
            if pref in self.placed:
                placed_bbs[pref] = self._ref_bbox(pref, *self.placed[pref])
        whitespace_balance = self._whitespace_balance_penalty_for_layout(ref, bb, placed_bbs)
        whitespace_floor = self._whitespace_floor_penalty_for_layout(ref, bb, placed_bbs)
        return float(0.50 * line_neatness + 0.20 * whitespace_balance + 0.30 * whitespace_floor)

    def wire_delta_mask(self, ref: str) -> Dict[str, np.ndarray]:
        w_cells, h_cells = self.grid_shape()
        xmin, ymin, _, _ = self.task.bbox_mm

        ix = np.arange(w_cells, dtype=np.float32)[:, None]
        iy = np.arange(h_cells, dtype=np.float32)[None, :]
        Xc = xmin + (ix + 0.5) * self.task.grid_mm
        Yc = ymin + (iy + 0.5) * self.task.grid_mm
        Xc = np.broadcast_to(Xc, (w_cells, h_cells)).astype(np.float32)
        Yc = np.broadcast_to(Yc, (w_cells, h_cells)).astype(np.float32)

        pins = self._pin_positions()
        net_bbox = {}
        net_deg = {}
        for net, pts in pins.items():
            if self._wire_net_ignored(net):
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            net = str(net)
            net_bbox[net] = (min(xs), max(xs), min(ys), max(ys))
            net_deg[net] = int(len(pts))

        c = self.comp_by_ref[ref]
        pads_by_net = {}
        for net, (rx, ry) in c.pads:
            if self._wire_net_ignored(net):
                continue
            pads_by_net.setdefault(str(net), []).append((float(rx), float(ry)))

        R = len(self.rotations)
        hpwl_delta = np.zeros((R, w_cells, h_cells), dtype=np.float32)
        w_hpwl_delta = np.zeros((R, w_cells, h_cells), dtype=np.float32)
        nslw_delta = np.zeros((R, w_cells, h_cells), dtype=np.float32)

        for ri, rot in enumerate(self.rotations):
            angle = math.radians(rot)
            ca, sa = math.cos(angle), math.sin(angle)
            hpwl_total = np.zeros((w_cells, h_cells), dtype=np.float32)
            w_hpwl_total = np.zeros((w_cells, h_cells), dtype=np.float32)
            nslw_total = np.zeros((w_cells, h_cells), dtype=np.float32)

            if pads_by_net:
                for net, rels in pads_by_net.items():
                    d0 = int(net_deg.get(net, 0))
                    if d0 > 0 and net in net_bbox:
                        ex_minx, ex_maxx, ex_miny, ex_maxy = net_bbox[net]
                    else:
                        ex_minx = ex_maxx = ex_miny = ex_maxy = 0.0

                    dxs, dys = [], []
                    for rx, ry in rels:
                        dxs.append(ca * rx - sa * ry)
                        dys.append(sa * rx + ca * ry)

                    px_stack = np.stack([Xc + dx for dx in dxs], axis=0)
                    py_stack = np.stack([Yc + dy for dy in dys], axis=0)
                    comp_minx = px_stack.min(axis=0)
                    comp_maxx = px_stack.max(axis=0)
                    comp_miny = py_stack.min(axis=0)
                    comp_maxy = py_stack.max(axis=0)

                    d_comp = int(len(rels))
                    d_new = d0 + d_comp

                    if d0 <= 1:
                        hpwl_old = 0.0
                        w_hpwl_old = 0.0
                    else:
                        hpwl_old = (ex_maxx - ex_minx) + (ex_maxy - ex_miny)
                        w_hpwl_old = hpwl_old * math.log(1.0 + d0)

                    if d0 == 0:
                        new_minx, new_maxx = comp_minx, comp_maxx
                        new_miny, new_maxy = comp_miny, comp_maxy
                    else:
                        new_minx = np.minimum(comp_minx, ex_minx)
                        new_maxx = np.maximum(comp_maxx, ex_maxx)
                        new_miny = np.minimum(comp_miny, ex_miny)
                        new_maxy = np.maximum(comp_maxy, ex_maxy)

                    if d_new <= 1:
                        hpwl_new = 0.0
                        w_hpwl_new = 0.0
                    else:
                        hpwl_new = (new_maxx - new_minx) + (new_maxy - new_miny)
                        w_hpwl_new = hpwl_new * math.log(1.0 + d_new)

                    hpwl_total += np.asarray(hpwl_new - hpwl_old, dtype=np.float32)
                    w_hpwl_total += np.asarray(w_hpwl_new - w_hpwl_old, dtype=np.float32)

            hpwl_delta[ri] = hpwl_total
            w_hpwl_delta[ri] = w_hpwl_total

            # True NSLW delta: count new surface-layer wires to already placed pads.
            # Higher NSLW is better, so it enters the objective with a negative sign.
            if self.objective_cfg.nslw_weight != 0.0 and self.placed_order:
                placed_pin_records = self._placed_pin_records()
                placed_bbs_base = {
                    pref: self._ref_bbox(pref, *self.placed[pref])
                    for pref in self.placed_order
                }
                cand_nets = set(pads_by_net.keys())
                if cand_nets:
                    c_w, c_h = c.size_mm
                    if int(rot) % 180 != 0:
                        c_w, c_h = c_h, c_w
                    for xi in range(w_cells):
                        for yi in range(h_cells):
                            x0 = float(Xc[xi, yi])
                            y0 = float(Yc[xi, yi])
                            cand_bb = (x0 - c_w / 2.0, y0 - c_h / 2.0, x0 + c_w / 2.0, y0 + c_h / 2.0)
                            placed_bbs = dict(placed_bbs_base)
                            placed_bbs[ref] = cand_bb
                            cnt = 0
                            for net in cand_nets:
                                if net not in placed_pin_records:
                                    continue
                                for cand_net, px, py in self._candidate_pin_records(ref, x0, y0, int(rot)):
                                    if cand_net != net:
                                        continue
                                    for pref, qx, qy in placed_pin_records.get(net, []):
                                        if self._is_surface_layer_wire_between(ref, (px, py), pref, (qx, qy), placed_bbs):
                                            cnt += 1
                            nslw_delta[ri, xi, yi] = float(cnt)

        total = (
            self.objective_cfg.hpwl_weight * hpwl_delta
            + self.objective_cfg.w_hpwl_weight * w_hpwl_delta
            - self.objective_cfg.nslw_weight * nslw_delta
        ).astype(np.float32)
        return {
            'hpwl': hpwl_delta,
            'w_hpwl': w_hpwl_delta,
            'nslw': nslw_delta,
            'total': total,
        }

    def action_mask_and_bias(self, ref: str) -> Tuple[np.ndarray, np.ndarray]:
        xmin, ymin, xmax, ymax = self.task.bbox_mm
        w_cells, h_cells = self.grid_shape()
        R = len(self.rotations)

        # Precompute center grids
        ix = np.arange(w_cells, dtype=np.float32)[:, None]
        iy = np.arange(h_cells, dtype=np.float32)[None, :]
        X = xmin + (ix + 0.5) * self.task.grid_mm
        Y = ymin + (iy + 0.5) * self.task.grid_mm
        I = np.arange(w_cells, dtype=np.int32)[:, None]
        J = np.arange(h_cells, dtype=np.int32)[None, :]

        c = self.comp_by_ref[ref]
        mask = np.zeros((R, w_cells, h_cells), dtype=np.float32)
        bias = np.zeros((R, w_cells, h_cells), dtype=np.float32)

        current_align_group = self._align_groups.get(ref)
        align_refs = [pref for pref in self.placed_order if current_align_group and self._align_groups.get(pref) == current_align_group]
        align_ix = set()
        align_iy = set()
        for pref in align_refs:
            px, py, _ = self.placed[pref]
            pxi = int(math.floor((px - xmin) / self.task.grid_mm))
            pyi = int(math.floor((py - ymin) / self.task.grid_mm))
            align_ix.add(pxi)
            align_iy.add(pyi)
        if align_ix:
            ax = np.isin(I, np.array(sorted(align_ix), dtype=np.int32)).astype(np.float32)
        else:
            ax = np.zeros((w_cells, h_cells), dtype=np.float32)
        if align_iy:
            ay = np.isin(J, np.array(sorted(align_iy), dtype=np.int32)).astype(np.float32)
        else:
            ay = np.zeros((w_cells, h_cells), dtype=np.float32)
        align_score = ax + ay

        hard_boundary = self.enforce_interface_on_boundary and self._hard_boundary_required(ref)
        edge_band_sides = self._edge_band_sides(ref)
        soft_edge_sides = self._soft_edge_preference_sides(ref)
        semantic_strength = max(0.0, self._semantic_strength(ref))
        eps = self.edge_eps_mm
        band = self._edge_band_width_mm()
        semantic_region = self._region_targets.get(ref, 'free')
        semantic_side = self._side_preferences.get(ref, 'free')

        for ri, rot in enumerate(self.rotations):
            w_mm, h_mm = c.size_mm
            if rot % 180 != 0:
                w_mm, h_mm = h_mm, w_mm

            a = X - w_mm / 2
            b = Y - h_mm / 2
            cc = X + w_mm / 2
            d = Y + h_mm / 2

            inside = (a >= xmin) & (b >= ymin) & (cc <= xmax) & (d <= ymax)

            ok = inside.copy()
            if self.occupied:
                eps_sp = 1e-6
                for oa, ob, oc, od in self.occupied:
                    oa2 = oa - self.min_spacing
                    ob2 = ob - self.min_spacing
                    oc2 = oc + self.min_spacing
                    od2 = od + self.min_spacing
                    overlap = ~((cc <= oa2 + eps_sp) | (a >= oc2 - eps_sp) | (d <= ob2 + eps_sp) | (b >= od2 - eps_sp))
                    ok &= ~overlap

            left = np.abs(a - xmin) <= eps
            right = np.abs(cc - xmax) <= eps
            bottom = np.abs(b - ymin) <= eps
            top = np.abs(d - ymax) <= eps
            touch_any = left | right | bottom | top
            left_clear = a - xmin
            right_clear = xmax - cc
            bottom_clear = b - ymin
            top_clear = ymax - d
            if hard_boundary:
                if edge_band_sides:
                    s = {str(x).lower() for x in edge_band_sides}
                    touch = np.zeros_like(touch_any, dtype=bool)
                    if "left" in s:
                        touch |= left
                    if "right" in s:
                        touch |= right
                    if "bottom" in s:
                        touch |= bottom
                    if "top" in s:
                        touch |= top
                    ok &= touch
                else:
                    ok &= touch_any

            mask[ri] = ok.astype(np.float32)

            region_pen = np.zeros((w_cells, h_cells), dtype=np.float32)
            if semantic_region == 'edge_left':
                region_pen = np.abs(left_clear - self._edge_target_clearance_mm(ref, 'left'))
            elif semantic_region == 'edge_right':
                region_pen = np.abs(right_clear - self._edge_target_clearance_mm(ref, 'right'))
            elif semantic_region == 'edge_bottom':
                region_pen = np.abs(bottom_clear - self._edge_target_clearance_mm(ref, 'bottom'))
            elif semantic_region == 'edge_top':
                region_pen = np.abs(top_clear - self._edge_target_clearance_mm(ref, 'top'))
            elif semantic_region == 'core':
                mx = min(0.40 * (xmax - xmin), max(self.task.grid_mm, 0.28 * (xmax - xmin)))
                my = min(0.40 * (ymax - ymin), max(self.task.grid_mm, 0.28 * (ymax - ymin)))
                core_x0 = xmin + mx
                core_x1 = xmax - mx
                core_y0 = ymin + my
                core_y1 = ymax - my
                dx = np.maximum(np.maximum(core_x0 - X, 0.0), X - core_x1)
                dy = np.maximum(np.maximum(core_y0 - Y, 0.0), Y - core_y1)
                region_pen = np.sqrt(dx * dx + dy * dy)
            elif semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'}:
                if semantic_side == 'edge_left':
                    region_pen = 0.35 * np.abs(left_clear - self._edge_target_clearance_mm(ref, 'left'))
                elif semantic_side == 'edge_right':
                    region_pen = 0.35 * np.abs(right_clear - self._edge_target_clearance_mm(ref, 'right'))
                elif semantic_side == 'edge_bottom':
                    region_pen = 0.35 * np.abs(bottom_clear - self._edge_target_clearance_mm(ref, 'bottom'))
                else:
                    region_pen = 0.35 * np.abs(top_clear - self._edge_target_clearance_mm(ref, 'top'))
            region_bonus = -semantic_strength * self.objective_cfg.region_weight * (region_pen / self.board_diag) ** 2
            soft_edge_bonus = np.zeros((w_cells, h_cells), dtype=np.float32)
            if (not hard_boundary) and soft_edge_sides:
                pref_terms = []
                if 'left' in soft_edge_sides:
                    pref_terms.append(np.abs(left_clear - self._edge_target_clearance_mm(ref, 'left')))
                if 'right' in soft_edge_sides:
                    pref_terms.append(np.abs(right_clear - self._edge_target_clearance_mm(ref, 'right')))
                if 'bottom' in soft_edge_sides:
                    pref_terms.append(np.abs(bottom_clear - self._edge_target_clearance_mm(ref, 'bottom')))
                if 'top' in soft_edge_sides:
                    pref_terms.append(np.abs(top_clear - self._edge_target_clearance_mm(ref, 'top')))
                if pref_terms:
                    pref_pen = pref_terms[0]
                    for term in pref_terms[1:]:
                        pref_pen = np.minimum(pref_pen, term)
                    soft_edge_bonus = -0.25 * semantic_strength * self.objective_cfg.region_weight * (pref_pen / self.board_diag) ** 2
            edge_bonus_map = np.zeros((w_cells, h_cells), dtype=np.float32)
            edge_bonus_sides = list(edge_band_sides or soft_edge_sides or [])
            if edge_bonus_sides:
                edge_terms = []
                if 'left' in edge_bonus_sides:
                    edge_terms.append(np.abs(left_clear - self._edge_target_clearance_mm(ref, 'left')))
                if 'right' in edge_bonus_sides:
                    edge_terms.append(np.abs(right_clear - self._edge_target_clearance_mm(ref, 'right')))
                if 'bottom' in edge_bonus_sides:
                    edge_terms.append(np.abs(bottom_clear - self._edge_target_clearance_mm(ref, 'bottom')))
                if 'top' in edge_bonus_sides:
                    edge_terms.append(np.abs(top_clear - self._edge_target_clearance_mm(ref, 'top')))
                if edge_terms:
                    edge_pen = edge_terms[0]
                    for term in edge_terms[1:]:
                        edge_pen = np.minimum(edge_pen, term)
                    denom = max(self.task.grid_mm, self._edge_band_width_mm(), 1e-6)
                    edge_bonus_map = float(self.edge_bonus) * semantic_strength * np.exp(-((edge_pen / denom) ** 2))

            anchor_bonus = np.zeros((w_cells, h_cells), dtype=np.float32)
            anchor_ref = self._anchor_refs.get(ref)
            subzone = self._subzones.get(ref, 'free')
            if anchor_ref and anchor_ref in self.placed:
                ax0, ay0, _ = self.placed[anchor_ref]
                dx = X - float(ax0)
                dy = Y - float(ay0)
                dist2 = dx * dx + dy * dy
                anchor_bonus += -semantic_strength * self.objective_cfg.anchor_weight * 0.20 * (dist2 / max(1e-6, self.board_diag * self.board_diag))
                if subzone == 'left':
                    anchor_bonus += semantic_strength * 0.18 * np.tanh(np.maximum(0.0, -dx) / max(self.task.grid_mm, 1e-6))
                elif subzone == 'right':
                    anchor_bonus += semantic_strength * 0.18 * np.tanh(np.maximum(0.0, dx) / max(self.task.grid_mm, 1e-6))
                elif subzone == 'top':
                    anchor_bonus += semantic_strength * 0.18 * np.tanh(np.maximum(0.0, dy) / max(self.task.grid_mm, 1e-6))
                elif subzone == 'bottom':
                    anchor_bonus += semantic_strength * 0.18 * np.tanh(np.maximum(0.0, -dy) / max(self.task.grid_mm, 1e-6))
            boundary_order_bonus = np.zeros((w_cells, h_cells), dtype=np.float32)
            same_side_group = self._same_side_groups.get(ref)
            my_order = self._boundary_orders.get(ref, None)
            if same_side_group and (semantic_side in {'edge_left', 'edge_right', 'edge_bottom', 'edge_top'}) and my_order is not None:
                peers = [pref for pref in self.placed_order if self._same_side_groups.get(pref) == same_side_group and self._boundary_orders.get(pref, None) is not None]
                if peers:
                    if semantic_side in {'edge_left', 'edge_right'}:
                        axis_grid = Y
                    else:
                        axis_grid = X
                    lower_vals = []
                    upper_vals = []
                    for pref in peers:
                        peer_order = self._boundary_orders.get(pref, None)
                        if peer_order is None:
                            continue
                        px, py, _ = self.placed[pref]
                        peer_axis = float(py) if semantic_side in {'edge_left', 'edge_right'} else float(px)
                        if peer_order < my_order:
                            lower_vals.append(peer_axis)
                        elif peer_order > my_order:
                            upper_vals.append(peer_axis)
                    if lower_vals:
                        boundary_order_bonus += semantic_strength * 0.14 * np.tanh((axis_grid - max(lower_vals)) / max(self.task.grid_mm, 1e-6))
                    if upper_vals:
                        boundary_order_bonus += semantic_strength * 0.14 * np.tanh((min(upper_vals) - axis_grid) / max(self.task.grid_mm, 1e-6))
            module_region_penalty = self._module_region_penalty_grid(ref, a, b, cc, d)
            module_region_bonus = -float(self.module_region_bias) * module_region_penalty
            bmap = (
                region_bonus
                + soft_edge_bonus
                + edge_bonus_map
                + self.alignment_bonus * align_score
                + anchor_bonus
                + boundary_order_bonus
                + module_region_bonus
            )
            bias[ri] = bmap.astype(np.float32)

        return mask, bias

    def step(self, action: Tuple[int, int, int], assume_legal: bool = False, return_observation: bool = True, compute_objective: bool = True) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """action: (rot_index, x_index, y_index).

        When assume_legal=True, skip the full action_mask_and_bias() rebuild.
        Training already chooses actions from a freshly computed legal mask, so
        this avoids doing the W×H CPU legality pass a second time. The cheap
        single-candidate geometry postcheck below is still kept as a safety net.

        When return_observation=False, skip observe() in the return path.  This
        avoids accidentally rebuilding the next-step CPU action_mask when the
        caller will compute the next mask on GPU anyway.

        When compute_objective=False, skip the full CPU objective/reward pass.
        This is useful for training code that already builds the GPU objective
        map for supervision and only needs step() to mutate the placement state.
        """
        def _obs() -> Dict[str, Any]:
            return self.observe() if bool(return_observation) else {}

        if self.done():
            return _obs(), 0.0, True, {}

        rix, ix, iy = int(action[0]), int(action[1]), int(action[2])
        ref = self.sequence[self.t]
        w, h = self.grid_shape()
        if rix < 0 or rix >= len(self.rotations) or ix < 0 or ix >= w or iy < 0 or iy >= h:
            self.terminated = True
            self.t = len(self.sequence)
            return _obs(), -100.0, True, {"illegal": True, "reason": "index_oob"}

        if not assume_legal:
            mask, _ = self.action_mask_and_bias(ref)
            if mask[rix, ix, iy] < 0.5:
                self.terminated = True
                self.t = len(self.sequence)
                return _obs(), -100.0, True, {"illegal": True, "reason": "masked"}

        xmin, ymin, _, _ = self.task.bbox_mm
        x = xmin + (ix + 0.5) * self.task.grid_mm
        y = ymin + (iy + 0.5) * self.task.grid_mm
        rot = int(self.rotations[rix])
        bb = self._ref_bbox(ref, x, y, rot)

        post_hard_boundary = self.enforce_interface_on_boundary and self._hard_boundary_required(ref)
        post_band_sides = self._edge_band_sides(ref)
        post_boundary_fail = False
        if post_hard_boundary:
            if post_band_sides:
                post_boundary_fail = not self._touch_sides(bb, post_band_sides)
            else:
                post_boundary_fail = not self._touch_sides(bb, ['left', 'right', 'top', 'bottom'])

        if (
            (not self._inside(bb))
            or self._violates_spacing(bb)
            or post_boundary_fail
        ):
            self.terminated = True
            self.t = len(self.sequence)
            return _obs(), -100.0, True, {"illegal": True, "reason": "postcheck"}

        self.placed[ref] = (x, y, rot)
        self.placed_order.append(ref)
        self.occupied.append(bb)
        self.t += 1

        done = self.done()
        if bool(compute_objective):
            obj = self._objective()
            reward = self.prev_obj - obj  # improvement
            self.prev_obj = obj
            return _obs(), reward, done, {"obj": obj}
        return _obs(), 0.0, done, {"obj": self.prev_obj, "objective_skipped": True}

    def _pin_positions(self) -> Dict[str, List[Tuple[float, float]]]:
        """Compute absolute pin positions for nets from placed components."""
        pins_by_net: Dict[str, List[Tuple[float, float]]] = {}
        for ref, (x, y, rot) in self.placed.items():
            c = self.comp_by_ref[ref]
            angle = math.radians(rot)
            ca, sa = math.cos(angle), math.sin(angle)
            for net, (rx, ry) in c.pads:
                ax = ca * rx - sa * ry
                ay = sa * rx + ca * ry
                pins_by_net.setdefault(net, []).append((x + ax, y + ay))
        return pins_by_net

    def _objective(self) -> float:
        pins = self._pin_positions()
        hpwl = 0.0
        w_hpwl = 0.0
        for net, pts in pins.items():
            if len(pts) <= 1 or self._wire_net_ignored(net):
                continue
            hpwl += hpwl_from_pins(pts)
            w_hpwl += w_hpwl_from_pins(pts)
        nslw = float(self._nslw_count_for_current_layout())

        region_pen = 0.0
        module_region_pen = 0.0
        conn_pen = 0.0
        align_pen = 0.0
        group_pen = 0.0
        anchor_pen = 0.0
        boundary_group_pen = 0.0
        pitch_pen = 0.0
        orientation_pen = 0.0
        edge_clearance_pen = 0.0
        interior_pen = 0.0
        density_pen = 0.0
        soft_spacing_pen = 0.0
        neatness_pen = 0.0
        refs = list(self.placed.keys())
        placed_bbs: Dict[str, Tuple[float, float, float, float]] = {}
        placed_xy_full = {r: (float(self.placed[r][0]), float(self.placed[r][1])) for r in refs}
        placed_rot_full = {r: int(self.placed[r][2]) for r in refs}
        for ref in refs:
            x, y, rot = self.placed[ref]
            bb = self._ref_bbox(ref, x, y, rot)
            placed_bbs[ref] = bb
            region_pen += self._region_penalty_from_bbox(ref, bb)
            module_region_pen += self._module_region_penalty_from_bbox(ref, bb)
            edge_clearance_pen += self._edge_penalty_from_bbox(ref, bb)
            interior_pen += self._interior_penalty_from_center(ref, x, y)
            neatness_pen += self._neatness_penalty_for_candidate(ref, x, y, bb)

        for i in range(len(refs)):
            ref_i = refs[i]
            xi, yi, _ = self.placed[ref_i]
            group_pen += self._group_penalty_for_candidate(ref_i, xi, yi)
            align_pen += self._align_penalty_for_candidate(ref_i, xi, yi)
            anchor_pen += self._anchor_penalty_for_layout(ref_i, xi, yi, placed_xy_full)
            boundary_group_pen += self._boundary_group_penalty_for_layout(ref_i, xi, yi, refs, placed_xy_full)
            pitch_pen += self._pitch_penalty_for_layout(ref_i, xi, yi, refs, placed_xy_full)
            orientation_pen += self._orientation_penalty_for_layout(ref_i, int(self.placed[ref_i][2]), refs, placed_xy_full, placed_rot_full)
            for j in range(i + 1, len(refs)):
                ref_j = refs[j]
                xj, yj, _ = self.placed[ref_j]
                w = self._pair_conn_weight(ref_i, ref_j)
                if w > 0.0:
                    conn_pen += w * ((abs(xi - xj) + abs(yi - yj)) / self.board_diag)
                density_pen += self._density_pair_penalty(ref_i, placed_bbs[ref_i], ref_j, placed_bbs[ref_j])
                soft_spacing_pen += self._soft_spacing_pair_penalty(ref_i, placed_bbs[ref_i], ref_j, placed_bbs[ref_j])

        return float(
            self.objective_cfg.hpwl_weight * hpwl
            + self.objective_cfg.w_hpwl_weight * w_hpwl
            - self.objective_cfg.nslw_weight * nslw
            + self.objective_cfg.region_weight * region_pen
            + self.objective_cfg.module_region_weight * module_region_pen
            + self.objective_cfg.conn_weight * conn_pen
            + self.objective_cfg.align_weight * align_pen
            + self.objective_cfg.group_weight * group_pen
            + self.objective_cfg.anchor_weight * anchor_pen
            + self.objective_cfg.boundary_group_weight * boundary_group_pen
            + self.objective_cfg.pitch_weight * pitch_pen
            + self.objective_cfg.orientation_weight * orientation_pen
            + self.objective_cfg.edge_clearance_weight * edge_clearance_pen
            + self.objective_cfg.interior_weight * interior_pen
            + self.objective_cfg.density_weight * density_pen
            + self.objective_cfg.soft_spacing_weight * soft_spacing_pen
            + self.objective_cfg.neatness_weight * neatness_pen
        )
