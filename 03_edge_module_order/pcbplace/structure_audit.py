from __future__ import annotations

"""Audit helpers for generated PCB training structures."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

BBox = Tuple[float, float, float, float]


def _as_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _area(bb: BBox | None) -> float:
    if bb is None:
        return 0.0
    x0, y0, x1, y1 = bb
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def _iou(a: BBox | None, b: BBox | None) -> float:
    inter = _intersection(a, b)
    if inter <= 0.0:
        return 0.0
    denom = _area(a) + _area(b) - inter
    return float(inter / max(1e-9, denom))


def _outside_board(bb: BBox | None, board: BBox | None, eps: float = 1e-6) -> bool:
    if bb is None or board is None:
        return True
    x0, y0, x1, y1 = bb
    bx0, by0, bx1, by1 = board
    return bool(x0 < bx0 - eps or y0 < by0 - eps or x1 > bx1 + eps or y1 > by1 + eps)


def _ref_prefix(ref: str) -> str:
    m = re.match(r"[A-Za-z_]+", str(ref or ""))
    return (m.group(0) if m else str(ref or "")).upper()


def _looks_clock_source(comp: Dict[str, Any]) -> bool:
    ref = str(comp.get('ref') or '')
    prefix = _ref_prefix(ref)
    if prefix in {'Y', 'X', 'XO', 'OSC'}:
        return True
    text = ' '.join(str(comp.get(k, '') or '') for k in ('ref', 'type', 'footprint', 'semantic_class', 'functional_group')).upper()
    return any(tok in text for tok in ('CLOCK SOURCE', 'CLK SOURCE', 'CRYSTAL', 'XTAL', 'OSCILLATOR', ' TCXO', ' VCXO'))


def _prior_bbox(module: Dict[str, Any]) -> BBox | None:
    prior = module.get('prior_region') if isinstance(module.get('prior_region'), dict) else {}
    return _as_bbox(prior.get('bbox_mm') or module.get('region_bbox_mm'))


def _prior_conf(module: Dict[str, Any]) -> float:
    prior = module.get('prior_region') if isinstance(module.get('prior_region'), dict) else {}
    try:
        return float(prior.get('confidence', module.get('region_confidence', 0.0)) or 0.0)
    except Exception:
        return 0.0


def _expert_region_leak(data: Dict[str, Any], *, mode: str = 'train') -> Dict[str, Any]:
    meta = data.get('meta') if isinstance(data.get('meta'), dict) else {}
    policy = meta.get('region_policy') if isinstance(meta.get('region_policy'), dict) else {}
    graph_policy = (data.get('graph') or {}).get('module_region_policy') if isinstance(data.get('graph'), dict) else {}
    leaks: List[str] = []
    if bool(policy.get('expert_region_as_input', False)):
        leaks.append('meta.region_policy.expert_region_as_input=true')
    if isinstance(graph_policy, dict) and bool(graph_policy.get('expert_region_as_input', False)):
        leaks.append('graph.module_region_policy.expert_region_as_input=true')
    for m in data.get('modules') or []:
        if not isinstance(m, dict):
            continue
        mid = str(m.get('module_id') or '')
        ex = m.get('expert_region_label') if isinstance(m.get('expert_region_label'), dict) else None
        if ex and bool(ex.get('use_as_input', False)):
            leaks.append(f'{mid}.expert_region_label.use_as_input=true')
        if mode == 'infer' and 'expert_region_heatmap' in m:
            leaks.append(f'{mid}.expert_region_heatmap_present_in_infer_output')
        hint = m.get('shape_hint') if isinstance(m.get('shape_hint'), dict) else {}
        source = str(hint.get('source') or '').lower()
        if 'expert' in source and m.get('region_bbox_mm') == hint.get('region_bbox_mm'):
            leaks.append(f'{mid}.active_shape_hint_looks_expert_derived')
    return {'leaked': bool(leaks), 'details': leaks[:50]}


def build_structure_audit(data: Dict[str, Any], *, mode: str = 'train', overlap_iou_threshold: float = 0.35) -> Dict[str, Any]:
    board = _as_bbox((data.get('board') or {}).get('bbox_mm'))
    components = [c for c in data.get('components') or [] if isinstance(c, dict)]
    modules = [m for m in data.get('modules') or [] if isinstance(m, dict)]
    refs = {str(c.get('ref') or '') for c in components if c.get('ref')}

    module_members: Dict[str, str] = {}
    for m in modules:
        mid = str(m.get('module_id') or '')
        for r in m.get('members') or []:
            rr = str(r)
            if rr:
                module_members[rr] = mid

    phase_counts = {}
    graph = data.get('graph') if isinstance(data.get('graph'), dict) else {}
    raw_counts = graph.get('sequence_phase_counts') if isinstance(graph.get('sequence_phase_counts'), dict) else {}
    for k, v in raw_counts.items():
        try:
            phase_counts[str(k)] = int(v)
        except Exception:
            phase_counts[str(k)] = v
    phases = graph.get('sequence_phases') if isinstance(graph.get('sequence_phases'), dict) else {}
    for key in ('hard_boundary', 'module_anchor', 'critical_local', 'module_member_frontier', 'bridge', 'weak_tail'):
        if key not in phase_counts and isinstance(phases.get(key), list):
            phase_counts[key] = len(phases[key])

    comp_without_module = [str(c.get('ref')) for c in components if not str(c.get('module_id') or (c.get('module') or {}).get('module_id') or '').strip()]
    orphan_refs = sorted(refs - set(module_members))
    needs_review_refs = [str(c.get('ref')) for c in components if bool((c.get('semantic_review') or {}).get('needs_review', False))]
    clock_refs = [str(c.get('ref')) for c in components if _looks_clock_source(c)]

    low_conf_modules = []
    prior_outside = []
    prior_bboxes: List[Tuple[str, BBox]] = []
    for m in modules:
        mid = str(m.get('module_id') or '')
        bb = _prior_bbox(m)
        conf = _prior_conf(m)
        if bb is not None:
            prior_bboxes.append((mid, bb))
        if conf < 0.45 or bb is None:
            low_conf_modules.append({'module_id': mid, 'confidence': round(conf, 4), 'has_prior_bbox': bb is not None})
        if _outside_board(bb, board):
            prior_outside.append({'module_id': mid, 'bbox_mm': list(bb) if bb else None})

    overlaps = []
    for i, (a_mid, a_bb) in enumerate(prior_bboxes):
        for b_mid, b_bb in prior_bboxes[i + 1:]:
            val = _iou(a_bb, b_bb)
            if val >= overlap_iou_threshold:
                overlaps.append({'a': a_mid, 'b': b_mid, 'iou': round(float(val), 4)})
    overlaps.sort(key=lambda x: (-x['iou'], x['a'], x['b']))

    prior_heatmap_modules = sum(1 for m in modules if isinstance(m.get('prior_region_heatmap'), dict))
    expert_heatmap_modules = sum(1 for m in modules if isinstance(m.get('expert_region_heatmap'), dict))
    comp_prior_heatmaps = sum(
        1 for c in components
        if isinstance((c.get('prior') or {}).get('region_heatmap'), dict)
        or isinstance((c.get('module') or {}).get('prior_region_heatmap'), dict)
    )
    comp_expert_heatmaps = sum(
        1 for c in components
        if isinstance((c.get('expert') or {}).get('region_heatmap'), dict)
        or isinstance(c.get('expert_region_heatmap'), dict)
    )

    leak = _expert_region_leak(data, mode=mode)
    summary = {
        'mode': mode,
        'component_count': len(components),
        'module_count': len(modules),
        'orphan_component_count': len(orphan_refs),
        'orphan_components_sample': orphan_refs[:50],
        'components_without_module_count': len(comp_without_module),
        'components_without_module_sample': comp_without_module[:50],
        'sequence_phase_counts': phase_counts,
        'hard_boundary_count': int(phase_counts.get('hard_boundary', 0) or 0),
        'bridge_count': int(phase_counts.get('bridge', 0) or 0),
        'weak_tail_count': int(phase_counts.get('weak_tail', 0) or 0),
        'clock_source_count': len(clock_refs),
        'clock_source_refs_sample': clock_refs[:50],
        'low_confidence_module_count': len(low_conf_modules),
        'low_confidence_modules': low_conf_modules[:50],
        'prior_region_overlap_count': len(overlaps),
        'prior_region_overlaps': overlaps[:50],
        'prior_bbox_outside_board_count': len(prior_outside),
        'prior_bbox_outside_board': prior_outside[:50],
        'semantic_needs_review_count': len(needs_review_refs),
        'semantic_needs_review_sample': needs_review_refs[:50],
        'expert_region_input_leak': leak,
        'region_heatmap_counts': {
            'modules_with_prior_region_heatmap': prior_heatmap_modules,
            'modules_with_expert_region_heatmap': expert_heatmap_modules,
            'components_with_prior_region_heatmap': comp_prior_heatmaps,
            'components_with_expert_region_heatmap': comp_expert_heatmaps,
        },
        'semantic_review_status_counts': dict(Counter(((c.get('semantic_review') or {}).get('review_status') or 'untracked') for c in components)),
    }
    return summary
