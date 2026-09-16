from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from .utils import (
    hpwl_from_pins,
    nslw_from_pins,
    is_edge_required_type,
    is_connector_type,
    coarse_type_from_fine,
    is_edge_preferred_type,
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
    anchor_ref: Optional[str] = None
    subzone: str = 'free'
    same_side_group: Optional[str] = None
    boundary_order: Optional[int] = None
    critical_nets: Tuple[str, ...] = ()
    critical_neighbors: Tuple[str, ...] = ()
    review_status: str = 'untracked'
    auto_confidence: float = 1.0
    needs_review: bool = False


@dataclass
class Task:
    bbox_mm: Tuple[float, float, float, float]
    grid_mm: float
    components: List[Component]
    nets: Dict[str, List[str]]  # optional
    sequence: List[str]  # single source of truth for ordering


@dataclass
class PlacementObjectiveConfig:
    hpwl_weight: float = 1.0
    nslw_weight: float = 0.20
    region_weight: float = 0.55
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


def _component_functional_group(comp: Component) -> str:
    label = _normalize_semantic_text(getattr(comp, 'functional_group', ''), '')
    return label or _component_group_key(comp)


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
    if label in _VALID_REGION_TYPES:
        return label
    if getattr(comp, 'allowed_sides', None):
        return _allowed_side_to_region(comp.allowed_sides[0])
    region = _normalize_semantic_text(getattr(comp, 'region_type', ''), '')
    if region in {'edge_top', 'edge_bottom', 'edge_left', 'edge_right'}:
        return region
    return 'free'


def _component_align_group(comp: Component) -> Optional[str]:
    value = getattr(comp, 'align_group', None)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _component_anchor_ref(comp: Component) -> Optional[str]:
    value = getattr(comp, 'anchor_ref', None)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


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


def _semantic_strength_for_component(comp: Component) -> float:
    status = str(getattr(comp, "review_status", "seeded") or "seeded").strip().lower()
    try:
        conf = float(getattr(comp, "auto_confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    needs_review = bool(getattr(comp, "needs_review", False))
    if status in {"approved", "edited", "manual_reviewed", "accepted", "confirmed"}:
        strength = 1.0
    elif status in {"rejected", "ignored", "ignore", "discarded"}:
        strength = 0.0
    elif status in {"seeded", "auto_seeded", "pending", "untracked", ""}:
        strength = 0.6 + 0.2 * conf
    else:
        strength = 0.55 + 0.25 * conf
    if needs_review and strength < 0.999:
        strength *= 0.65
    return float(max(0.0, min(1.0, strength)))


def _component_critical_nets(comp: Component) -> Tuple[str, ...]:
    vals = getattr(comp, 'critical_nets', ()) or ()
    out: List[str] = []
    for net in vals:
        nn = _normalize_net_name(net)
        if nn and (not _is_ground_net(nn)):
            out.append(nn)
    return tuple(dict.fromkeys(out))


def _component_critical_neighbors(comp: Component) -> Tuple[str, ...]:
    vals = getattr(comp, 'critical_neighbors', ()) or ()
    out: List[str] = []
    for ref in vals:
        rr = str(ref or '').strip()
        if rr:
            out.append(rr)
    return tuple(dict.fromkeys(out))


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


class PlacementEnv:
    """Deterministic grid placement environment with hard constraints and bias.
    Fixes vs baseline:
      - Illegal action now terminates env (prevents infinite loops).
      - Interface/boundary constraints use a tolerance (not strict equality).
      - allowed_sides / region_type / side_preference can define soft edge bands; only hard-boundary parts must touch the board edge.
      - Action mask is vectorized (much faster on larger boards).
      - Tracks placement order for building sequence tokens.
      - Reward objective and teacher objective delta use the same unified terms
        (HPWL + NSLW + region + conn + align + group + edge_clearance + interior + density + soft_spacing + neatness).
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
        nslw_weight: float = 0.20,
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

        self.objective_cfg = PlacementObjectiveConfig(
            hpwl_weight=float(hpwl_weight),
            nslw_weight=float(nslw_weight),
            region_weight=float(region_weight),
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

        # legacy shaping knobs are kept only for backwards-compatible constructor arguments.
        # Objective magnitudes now flow through objective_cfg.*_weight instead of these legacy scalars.
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
        self._functional_group = {ref: _component_functional_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._region_targets = {ref: _component_region_type(self.comp_by_ref[ref]) for ref in self.refs}
        self._side_preferences = {ref: _component_side_preference(self.comp_by_ref[ref]) for ref in self.refs}
        self._align_groups = {ref: _component_align_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._anchor_refs = {ref: _component_anchor_ref(self.comp_by_ref[ref]) for ref in self.refs}
        self._subzones = {ref: _component_subzone(self.comp_by_ref[ref]) for ref in self.refs}
        self._same_side_groups = {ref: _component_same_side_group(self.comp_by_ref[ref]) for ref in self.refs}
        self._boundary_orders = {ref: _component_boundary_order(self.comp_by_ref[ref]) for ref in self.refs}
        self._semantic_strengths = {ref: _semantic_strength_for_component(self.comp_by_ref[ref]) for ref in self.refs}
        self._critical_nets = {ref: _component_critical_nets(self.comp_by_ref[ref]) for ref in self.refs}
        self._critical_neighbors = {ref: _component_critical_neighbors(self.comp_by_ref[ref]) for ref in self.refs}
        self._group_keys = dict(self._functional_group)
        self._conn_weights = self._build_conn_weight_map()
        self.reset()

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
        for ref in self.refs:
            for rank, nb in enumerate(self._critical_neighbors.get(ref, ())):
                if nb not in self.comp_by_ref:
                    continue
                base = max(0.6, 2.2 - 0.28 * float(rank))
                if self._functional_group.get(nb) == self._functional_group.get(ref):
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
        vals: List[float] = []
        for pref in target_refs:
            px, py = placed_xy[pref]
            vals.append(min(abs(x - px) / bw, abs(y - py) / bh))
        vals.sort()
        top = vals[: min(3, len(vals))]
        return float(sum(top) / max(1, len(top)))

    def _group_penalty_for_layout(
        self,
        ref: str,
        x: float,
        y: float,
        placed_order: List[str],
        placed_xy: Dict[str, Tuple[float, float]],
    ) -> float:
        my_group = self._functional_group.get(ref, 'misc')
        group_refs = [r for r in placed_order if self._functional_group.get(r, 'misc') == my_group]
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
        my_group = self._functional_group.get(ref, 'misc')
        affected = [pref for pref in current_order if self._functional_group.get(pref, 'misc') == my_group]
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
            px, py = placed_xy[pref]
            peer_axis = self._boundary_axis_value(my_side, float(px), float(py))
            if my_order < peer_order:
                pen += pair_strength * (max(0.0, me_axis - peer_axis) / span) ** 2
            else:
                pen += pair_strength * (max(0.0, peer_axis - me_axis) / span) ** 2
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
        g = self._same_side_groups.get(ref)
        if g:
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
        side_votes = [self._side_preferences.get(r, self._region_targets.get(r, 'free')) for r in refs]
        side = max(set(side_votes), key=side_votes.count) if side_votes else 'free'
        if side in {'edge_left', 'edge_right'}:
            return 'y'
        if side in {'edge_top', 'edge_bottom'}:
            return 'x'
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
        return float(self._semantic_strength(ref) * pen / (span * span))

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
        nslw = wire_terms['nslw'].astype(np.float32)
        region = np.zeros((R, w_cells, h_cells), dtype=np.float32)
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
            + self.objective_cfg.nslw_weight * nslw
            + self.objective_cfg.region_weight * region
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
            'nslw': nslw,
            'region': region,
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
        if sem in {'interface', 'mechanical_edge_interface'}:
            ratio = 0.15
        elif sem in {'ui', 'mechanical', 'rf'}:
            ratio = 0.25
        elif sem in {'interface_support', 'power_support'}:
            ratio = 0.35
        else:
            ratio = 0.55
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
        return float(u * u)

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
        return float((dx / max(1e-6, mx)) ** 2 + (dy / max(1e-6, my)) ** 2)

    def _pair_relation_flags(self, ref_a: str, ref_b: str) -> Dict[str, bool]:
        same_group = self._functional_group.get(ref_a, 'misc') == self._functional_group.get(ref_b, 'misc')
        a_anchor = self._anchor_refs.get(ref_a)
        b_anchor = self._anchor_refs.get(ref_b)
        shared_anchor = bool(
            (a_anchor and a_anchor == b_anchor)
            or (a_anchor and a_anchor == ref_b)
            or (b_anchor and b_anchor == ref_a)
        )
        critical_pair = bool(
            ref_b in self._critical_neighbors.get(ref_a, ())
            or ref_a in self._critical_neighbors.get(ref_b, ())
        )

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
            'same_group': bool(same_group),
            'shared_anchor': bool(shared_anchor),
            'critical_pair': bool(critical_pair),
            'large_pair': bool(large_pair),
            'core_support_pair': bool(core_support_pair),
        }

    def _density_scale(self, ref_a: str, ref_b: str) -> float:
        flags = self._pair_relation_flags(ref_a, ref_b)

        scale = 1.0
        if flags['same_group']:
            scale = min(scale, float(self.same_group_density_scale))
        if flags['critical_pair']:
            scale = min(scale, float(self.critical_neighbor_density_scale))
        elif flags['shared_anchor']:
            scale = min(scale, float(self.anchor_group_density_scale))
        if flags['large_pair'] or flags['core_support_pair']:
            scale = max(scale, float(self.large_pair_density_scale))
        return float(scale)

    def _soft_spacing_scale(self, ref_a: str, ref_b: str) -> float:
        flags = self._pair_relation_flags(ref_a, ref_b)
        if flags['large_pair'] or flags['core_support_pair']:
            return float(self.large_pair_soft_spacing_scale)
        if flags['same_group']:
            return float(self.same_group_soft_spacing_scale)
        if flags['critical_pair']:
            return float(self.critical_neighbor_soft_spacing_scale)
        if flags['shared_anchor']:
            return float(self.anchor_group_soft_spacing_scale)
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
        my_group = self._functional_group.get(ref, 'misc')
        if my_group in {'', 'misc'}:
            return 0.0
        my_bb = self._group_proxy_bbox(ref, float(x), float(y))
        penalty = 0.0
        seen_groups = set()
        for pref in placed_order:
            if pref == ref:
                continue
            peer_group = self._functional_group.get(pref, 'misc')
            if peer_group in {'', 'misc'} or peer_group == my_group or peer_group in seen_groups:
                continue
            members = [r for r in placed_order if self._functional_group.get(r, 'misc') == peer_group and r in placed_xy]
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
        return float(self._density_scale(ref_a, ref_b) * (u * u))

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
        return bool(
            diag >= max(6.0 * self.task.grid_mm, 0.12 * self.board_diag)
            or sem in {'core', 'power', 'power_support', 'mechanical'}
            or group in {'core_active', 'power_active'}
        )

    def _preferred_gap_mm(self, ref_a: str, ref_b: str) -> float:
        preferred = float(self.min_spacing)
        same_group = self._functional_group.get(ref_a, 'misc') == self._functional_group.get(ref_b, 'misc')
        a_anchor = self._anchor_refs.get(ref_a)
        b_anchor = self._anchor_refs.get(ref_b)
        shared_anchor = bool(
            (a_anchor and a_anchor == b_anchor)
            or (a_anchor and a_anchor == ref_b)
            or (b_anchor and b_anchor == ref_a)
        )
        critical_pair = bool(
            ref_b in self._critical_neighbors.get(ref_a, ())
            or ref_a in self._critical_neighbors.get(ref_b, ())
        )
        if same_group:
            preferred += self.soft_spacing_same_group_extra_mm
        elif shared_anchor or critical_pair:
            preferred += self.soft_spacing_anchor_or_critical_extra_mm
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
        my_group = self._functional_group.get(ref, 'misc')
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
            if self._functional_group.get(pref, 'misc') == my_group:
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
            if not net or str(net).upper() in ("", "GND", "GROUND"):
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            net = str(net)
            net_bbox[net] = (min(xs), max(xs), min(ys), max(ys))
            net_deg[net] = int(len(pts))

        c = self.comp_by_ref[ref]
        pads_by_net = {}
        for net, (rx, ry) in c.pads:
            if not net or str(net).upper() in ("", "GND", "GROUND"):
                continue
            pads_by_net.setdefault(str(net), []).append((float(rx), float(ry)))

        R = len(self.rotations)
        hpwl_delta = np.zeros((R, w_cells, h_cells), dtype=np.float32)
        nslw_delta = np.zeros((R, w_cells, h_cells), dtype=np.float32)

        for ri, rot in enumerate(self.rotations):
            angle = math.radians(rot)
            ca, sa = math.cos(angle), math.sin(angle)
            hpwl_total = np.zeros((w_cells, h_cells), dtype=np.float32)
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
                        nslw_old = 0.0
                    else:
                        hpwl_old = (ex_maxx - ex_minx) + (ex_maxy - ex_miny)
                        nslw_old = hpwl_old * math.log(1.0 + d0)

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
                        nslw_new = 0.0
                    else:
                        hpwl_new = (new_maxx - new_minx) + (new_maxy - new_miny)
                        nslw_new = hpwl_new * math.log(1.0 + d_new)

                    hpwl_total += np.asarray(hpwl_new - hpwl_old, dtype=np.float32)
                    nslw_total += np.asarray(nslw_new - nslw_old, dtype=np.float32)

            hpwl_delta[ri] = hpwl_total
            nslw_delta[ri] = nslw_total

        total = (
            self.objective_cfg.hpwl_weight * hpwl_delta
            + self.objective_cfg.nslw_weight * nslw_delta
        ).astype(np.float32)
        return {
            'hpwl': hpwl_delta,
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
            bmap = region_bonus + soft_edge_bonus + self.alignment_bonus * align_score + anchor_bonus + boundary_order_bonus
            bias[ri] = bmap.astype(np.float32)

        return mask, bias

    def step(self, action: Tuple[int, int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """action: (rot_index, x_index, y_index)"""
        if self.done():
            return self.observe(), 0.0, True, {}

        rix, ix, iy = int(action[0]), int(action[1]), int(action[2])
        ref = self.sequence[self.t]
        w, h = self.grid_shape()
        if rix < 0 or rix >= len(self.rotations) or ix < 0 or ix >= w or iy < 0 or iy >= h:
            self.terminated = True
            self.t = len(self.sequence)
            return self.observe(), -100.0, True, {"illegal": True, "reason": "index_oob"}

        mask, _ = self.action_mask_and_bias(ref)
        if mask[rix, ix, iy] < 0.5:
            self.terminated = True
            self.t = len(self.sequence)
            return self.observe(), -100.0, True, {"illegal": True, "reason": "masked"}

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
            return self.observe(), -100.0, True, {"illegal": True, "reason": "postcheck"}

        self.placed[ref] = (x, y, rot)
        self.placed_order.append(ref)
        self.occupied.append(bb)
        self.t += 1

        obj = self._objective()
        reward = self.prev_obj - obj  # improvement
        self.prev_obj = obj
        done = self.done()
        return self.observe(), reward, done, {"obj": obj}

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
        nslw = 0.0
        for net, pts in pins.items():
            if len(pts) <= 1:
                continue
            hpwl += hpwl_from_pins(pts)
            nslw += nslw_from_pins(pts)

        region_pen = 0.0
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
            + self.objective_cfg.nslw_weight * nslw
            + self.objective_cfg.region_weight * region_pen
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
