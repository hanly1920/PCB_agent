from __future__ import annotations

"""Unified train/infer structure generation pipeline.

This module owns the leakage boundary between training labels and inference
inputs.  It coordinates semantic annotation, module inference, prior/expert
region export, six-phase sequence rebuilding, and per-board audits.
"""

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from .dataset import task_from_json
from .json_schema import validate_task_json
from .module_partition import export_task_modules_to_json
from .semantic_labels import annotate_board_train, annotate_board_infer_safe
from .structure_audit import build_structure_audit
from .region_prior import gaussian_heatmap_from_bbox
from .utils import load_json, save_json, drop_mounting_holes_from_task_json

SEQUENCE_POLICY = 'six_phase_anchor_frontier_v1'

_DERIVED_SEMANTIC_FIELDS = {
    'semantic_class', 'region_type', 'functional_group', 'side_preference',
    'align_group', 'anchor_ref', 'subzone', 'same_side_group',
    'boundary_order', 'critical_nets', 'critical_neighbors', 'placement_role',
}

_DERIVED_COMPONENT_MODULE_FIELDS = {
    'module_region_bbox', 'module_region_confidence', 'module_region_source',
    'module_shape_hint', 'module_subregion', 'expert_module_region_bbox',
}

_DERIVED_MODULE_REGION_FIELDS = {
    'region_bbox_mm', 'region_confidence', 'shape_hint', 'module_subregion',
    'expert_bbox_mm', 'bbox_mm', 'prior_region', 'region_candidates',
    'expert_region_label',
}


def _write_temp_json(data: Dict[str, Any], source_hint: str = 'board') -> str:
    safe_hint = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in str(source_hint))[-80:]
    tmp = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix=f'.{safe_hint}.json', delete=False)
    with tmp:
        json.dump(data, tmp, ensure_ascii=False)
    return tmp.name


def _rebuild_modules_regions_sequence(data: Dict[str, Any], *, source_hint: str = 'board', load_expert_labels: bool = False) -> Dict[str, Any]:
    tmp_path = _write_temp_json(data, source_hint=source_hint)
    try:
        task = task_from_json(tmp_path, load_expert=bool(load_expert_labels))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    data = export_task_modules_to_json(data, task)
    meta = getattr(task, 'sequence_meta', {}) or {}
    graph = data.setdefault('graph', {})
    graph['sequence'] = list(task.sequence)
    graph['sequence_policy'] = SEQUENCE_POLICY
    graph['sequence_source'] = SEQUENCE_POLICY
    graph['sequence_phase_counts'] = dict(meta.get('phase_counts', {}))
    graph['sequence_phases'] = dict(meta.get('phases', {}))
    graph.pop('sequence_config', None)
    return data


def sanitize_infer_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove train-only/expert-derived fields before infer-safe generation."""
    out = copy.deepcopy(data)
    out.pop('expert_region_label', None)
    graph = out.get('graph') if isinstance(out.get('graph'), dict) else {}
    if isinstance(graph, dict):
        # graph.sequence will be rebuilt; old module bboxes/phase metadata must
        # not influence inference-safe labels.
        for key in ('sequence', 'sequence_policy', 'sequence_source', 'sequence_phase_counts', 'sequence_phases', 'sequence_config'):
            graph.pop(key, None)
        if isinstance(graph.get('module_region_policy'), dict):
            graph['module_region_policy']['expert_region_as_input'] = False
    meta = out.get('meta') if isinstance(out.get('meta'), dict) else {}
    if isinstance(meta, dict):
        if isinstance(meta.get('region_policy'), dict):
            meta['region_policy']['expert_region_as_input'] = False
        # Do not carry train semantic claims into infer mode.
        meta.pop('semantic_annotation', None)

    for comp in out.get('components') or []:
        if not isinstance(comp, dict):
            continue
        expert = comp.get('expert') if isinstance(comp.get('expert'), dict) else None
        if expert is not None:
            for key in ('xy_mm', 'rot', 'module_region_bbox_mm', 'region_heatmap'):
                expert.pop(key, None)
            if not expert:
                comp.pop('expert', None)
        for key in _DERIVED_COMPONENT_MODULE_FIELDS:
            comp.pop(key, None)
        comp.pop('expert_region_heatmap', None)
        comp.pop('prior_region_heatmap', None)
        for key in _DERIVED_SEMANTIC_FIELDS:
            comp.pop(key, None)
        sem = comp.get('semantic') if isinstance(comp.get('semantic'), dict) else None
        if sem is not None:
            for key in _DERIVED_SEMANTIC_FIELDS:
                sem.pop(key, None)
            if not sem:
                comp.pop('semantic', None)
        comp.pop('semantic_review', None)
        prior = comp.get('prior') if isinstance(comp.get('prior'), dict) else None
        if prior is not None:
            for key in ('region_bbox_mm', 'region_confidence', 'region_source', 'leakage_safe', 'region_heatmap'):
                prior.pop(key, None)
            if not prior:
                comp.pop('prior', None)
        module = comp.get('module') if isinstance(comp.get('module'), dict) else None
        if module is not None:
            # Preserve only non-spatial module identity hints.  If none exist,
            # drop the nested module object so stale spatial hints cannot leak.
            kept = {
                k: module.get(k)
                for k in ('module_id', 'anchor_ref', 'module_role')
                if module.get(k) not in (None, '')
            }
            if kept:
                comp['module'] = kept
            else:
                comp.pop('module', None)

    clean_modules = []
    for module in out.get('modules') or out.get('module_annotations') or []:
        if not isinstance(module, dict):
            continue
        row = {
            k: copy.deepcopy(module.get(k))
            for k in ('module_id', 'module_type', 'anchor_ref', 'members', 'prefix_counts', 'module_order', 'source')
            if k in module
        }
        if row.get('members'):
            clean_modules.append(row)
    if clean_modules:
        out['modules'] = clean_modules
        out.setdefault('graph', {})['modules'] = copy.deepcopy(clean_modules)
    else:
        out.pop('modules', None)
        if isinstance(out.get('graph'), dict):
            out['graph'].pop('modules', None)
    out.pop('module_annotations', None)
    return out


def finalize_infer_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee no train-only region labels are exposed as inference input."""
    out = data
    for module in out.get('modules') or []:
        if not isinstance(module, dict):
            continue
        module.pop('expert_region_label', None)
        module.pop('expert_region_heatmap', None)
        hint = module.get('shape_hint') if isinstance(module.get('shape_hint'), dict) else None
        if hint and 'expert' in str(hint.get('source') or '').lower():
            module.pop('shape_hint', None)
    graph_modules = (out.get('graph') or {}).get('modules') if isinstance(out.get('graph'), dict) else None
    if isinstance(graph_modules, list):
        for module in graph_modules:
            if not isinstance(module, dict):
                continue
            module.pop('expert_region_label', None)
            module.pop('expert_region_heatmap', None)
            hint = module.get('shape_hint') if isinstance(module.get('shape_hint'), dict) else None
            if hint and 'expert' in str(hint.get('source') or '').lower():
                module.pop('shape_hint', None)
    for comp in out.get('components') or []:
        if not isinstance(comp, dict):
            continue
        expert = comp.get('expert') if isinstance(comp.get('expert'), dict) else None
        if expert is not None:
            for key in ('xy_mm', 'rot', 'module_region_bbox_mm', 'region_heatmap'):
                expert.pop(key, None)
            if not expert:
                comp.pop('expert', None)
        comp.pop('expert_module_region_bbox', None)
        comp.pop('expert_region_heatmap', None)
        hint = comp.get('module_shape_hint') if isinstance(comp.get('module_shape_hint'), dict) else None
        if hint and 'expert' in str(hint.get('source') or '').lower():
            comp.pop('module_shape_hint', None)
        module = comp.get('module') if isinstance(comp.get('module'), dict) else None
        if module is not None:
            module.pop('expert_region_label', None)
            module.pop('expert_region_heatmap', None)
            mhint = module.get('shape_hint') if isinstance(module.get('shape_hint'), dict) else None
            if mhint and 'expert' in str(mhint.get('source') or '').lower():
                module.pop('shape_hint', None)
    policy = {
        'active_region_source': 'prior_region',
        'expert_region_as_input': False,
        'leakage_checked': True,
        'prior_generator': 'rule_v2_fine_modules_heatmap',
        'mode': 'infer',
    }
    out.setdefault('meta', {})['region_policy'] = dict(policy)
    out.setdefault('graph', {})['module_region_policy'] = dict(policy)
    return out


def _bbox_valid(bb: Any) -> bool:
    return isinstance(bb, (list, tuple)) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb) and float(bb[2]) > float(bb[0]) and float(bb[3]) > float(bb[1])


def _heatmap_payload(board_bbox: Tuple[float, float, float, float], bbox_mm: Any, confidence: float, *, grid_x: int, grid_y: int, sigma_cells: float, source: str) -> Dict[str, Any]:
    bb = tuple(float(v) for v in bbox_mm)
    values = gaussian_heatmap_from_bbox(board_bbox, bb, int(grid_x), int(grid_y), float(sigma_cells), float(confidence)).reshape(int(grid_x), int(grid_y)).tolist()
    return {
        'grid_x': int(grid_x),
        'grid_y': int(grid_y),
        'sigma_cells': float(sigma_cells),
        'values': values,
        'source': str(source),
    }


def refresh_region_heatmaps(data: Dict[str, Any], *, mode: str, grid_x: int, grid_y: int, sigma_cells: float) -> Dict[str, Any]:
    """Refresh prior/expert heatmaps with the same grid used by training.

    prior_region_heatmap is inference-safe and present in both train/infer.
    expert_region_heatmap is supervision-only and kept only in train mode.
    """
    board = data.get('board') or {}
    bb0 = board.get('bbox_mm') or [0.0, 0.0, 1.0, 1.0]
    board_bbox = tuple(float(v) for v in bb0)
    module_by_id: Dict[str, Dict[str, Any]] = {}
    for module in data.get('modules') or []:
        if not isinstance(module, dict):
            continue
        mid = str(module.get('module_id', '') or '')
        if mid:
            module_by_id[mid] = module
        prior = module.get('prior_region') if isinstance(module.get('prior_region'), dict) else {}
        prior_bbox = prior.get('bbox_mm') or module.get('region_bbox_mm')
        conf = prior.get('confidence', module.get('region_confidence', 0.35))
        try:
            conf = float(conf)
        except Exception:
            conf = 0.35
        if _bbox_valid(prior_bbox):
            module['prior_region_heatmap'] = _heatmap_payload(board_bbox, prior_bbox, conf, grid_x=grid_x, grid_y=grid_y, sigma_cells=sigma_cells, source=str(prior.get('source') or 'prior_region_bbox_gaussian'))
        if mode == 'train':
            ex = module.get('expert_region_label') if isinstance(module.get('expert_region_label'), dict) else {}
            ex_bbox = ex.get('bbox_mm')
            if _bbox_valid(ex_bbox):
                module['expert_region_heatmap'] = _heatmap_payload(board_bbox, ex_bbox, 1.0, grid_x=grid_x, grid_y=grid_y, sigma_cells=sigma_cells, source=str(ex.get('source') or 'expert_region_bbox_gaussian'))
        else:
            module.pop('expert_region_heatmap', None)

    graph_modules = (data.get('graph') or {}).get('modules') if isinstance(data.get('graph'), dict) else None
    if isinstance(graph_modules, list):
        for gm in graph_modules:
            if not isinstance(gm, dict):
                continue
            mid = str(gm.get('module_id', '') or '')
            src = module_by_id.get(mid)
            if src is None:
                continue
            if 'prior_region_heatmap' in src:
                gm['prior_region_heatmap'] = copy.deepcopy(src['prior_region_heatmap'])
            if mode == 'train' and 'expert_region_heatmap' in src:
                gm['expert_region_heatmap'] = copy.deepcopy(src['expert_region_heatmap'])
            else:
                gm.pop('expert_region_heatmap', None)

    for comp in data.get('components') or []:
        if not isinstance(comp, dict):
            continue
        mid = str(comp.get('module_id') or (comp.get('module') or {}).get('module_id') or '')
        module = module_by_id.get(mid)
        if module and isinstance(module.get('prior_region_heatmap'), dict):
            comp.setdefault('prior', {})['region_heatmap'] = copy.deepcopy(module['prior_region_heatmap'])
            comp.setdefault('module', {})['prior_region_heatmap'] = copy.deepcopy(module['prior_region_heatmap'])
        if mode == 'train' and module and isinstance(module.get('expert_region_heatmap'), dict):
            comp.setdefault('expert', {})['region_heatmap'] = copy.deepcopy(module['expert_region_heatmap'])
        else:
            expert = comp.get('expert') if isinstance(comp.get('expert'), dict) else None
            if expert is not None:
                expert.pop('region_heatmap', None)
                if not expert:
                    comp.pop('expert', None)
            comp.pop('expert_region_heatmap', None)
    data.setdefault('meta', {})['region_heatmap_config'] = {
        'grid_x': int(grid_x),
        'grid_y': int(grid_y),
        'sigma_cells': float(sigma_cells),
        'prior_region_heatmap_is_inference_safe_input': True,
        'expert_region_heatmap_is_train_only_supervision': mode == 'train',
    }
    data.setdefault('graph', {}).setdefault('module_region_policy', {})['region_heatmap_config'] = copy.deepcopy(data['meta']['region_heatmap_config'])
    return data


def generate_structure(data: Dict[str, Any], *, mode: str, source_hint: str = 'board', region_grid_x: int = 6, region_grid_y: int = 6, region_heatmap_sigma_cells: float = 0.85) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if mode not in {'train', 'infer'}:
        raise ValueError("mode must be 'train' or 'infer'")
    work = copy.deepcopy(data)
    if mode == 'infer':
        work = sanitize_infer_input(work)
        annotator = annotate_board_infer_safe
    else:
        annotator = annotate_board_train

    # Pass 1: base semantic seed -> module/prior/sequence.
    work = annotator(work)
    work = _rebuild_modules_regions_sequence(work, source_hint=f'{source_hint}.pass1', load_expert_labels=(mode == 'train'))
    work = refresh_region_heatmaps(work, mode=mode, grid_x=int(region_grid_x), grid_y=int(region_grid_y), sigma_cells=float(region_heatmap_sigma_cells))

    # Pass 2: semantic enrichment with module context -> final sequence.
    work = annotator(work)
    work = _rebuild_modules_regions_sequence(work, source_hint=f'{source_hint}.pass2', load_expert_labels=(mode == 'train'))
    work = refresh_region_heatmaps(work, mode=mode, grid_x=int(region_grid_x), grid_y=int(region_grid_y), sigma_cells=float(region_heatmap_sigma_cells))

    if mode == 'infer':
        work = finalize_infer_output(work)

    validate_task_json(work, source=source_hint)
    audit = build_structure_audit(work, mode=mode)
    work.setdefault('meta', {})['training_structure'] = {
        'generator': 'scripts/generate_training_structure.py',
        'mode': mode,
        'sequence_policy': SEQUENCE_POLICY,
        'graph_sequence_rebuilt': True,
        'semantic_passes': 2,
        'prior_region_is_model_input': True,
        'prior_region_heatmap_is_model_input': True,
        'region_heatmap_grid_x': int(region_grid_x),
        'region_heatmap_grid_y': int(region_grid_y),
        'region_heatmap_sigma_cells': float(region_heatmap_sigma_cells),
        'expert_region_label_is_supervision_only': mode == 'train',
        'infer_mode_strips_expert_layout': mode == 'infer',
    }
    work.setdefault('meta', {})['structure_audit_summary'] = {
        'module_count': audit.get('module_count'),
        'semantic_needs_review_count': audit.get('semantic_needs_review_count'),
        'expert_region_input_leak': audit.get('expert_region_input_leak', {}).get('leaked'),
        'sequence_phase_counts': audit.get('sequence_phase_counts'),
    }
    return work, audit


def generate_structure_file(input_path: str | Path, output_path: str | Path, *, mode: str, audit_out: str | Path | None = None, region_grid_x: int = 6, region_grid_y: int = 6, region_heatmap_sigma_cells: float = 0.85) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    data = drop_mounting_holes_from_task_json(load_json(str(input_path)))
    structured, audit = generate_structure(data, mode=mode, source_hint=str(input_path), region_grid_x=int(region_grid_x), region_grid_y=int(region_grid_y), region_heatmap_sigma_cells=float(region_heatmap_sigma_cells))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(structured, str(output_path))
    if audit_out is None:
        audit_out = output_path.with_suffix(output_path.suffix + '.audit.json')
    audit_path = Path(audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(audit, str(audit_path))
    return structured, audit
