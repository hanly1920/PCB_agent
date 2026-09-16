#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcbplace.training_structure import generate_structure_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Generate the unified PCB training/inference structure: semantic labels, '
            'module annotations, leakage-safe prior regions, train-only expert region labels, '
            'six-phase sequence, and an audit report.'
        )
    )
    parser.add_argument('--input', required=True, help='Input board JSON')
    parser.add_argument('--output', required=True, help='Output structured board JSON')
    parser.add_argument('--mode', required=True, choices=['train', 'infer'], help='train may use expert layout labels; infer is leakage-safe')
    parser.add_argument('--audit-out', default='', help='Optional audit JSON path. Default: <output>.audit.json')
    parser.add_argument('--region_grid_x', type=int, default=6, help='Region heatmap grid width used for generated prior/expert heatmaps.')
    parser.add_argument('--region_grid_y', type=int, default=6, help='Region heatmap grid height used for generated prior/expert heatmaps.')
    parser.add_argument('--region_heatmap_sigma_cells', type=float, default=0.85, help='Gaussian smoothing sigma in region-grid cells.')
    args = parser.parse_args()

    structured, audit = generate_structure_file(
        args.input,
        args.output,
        mode=args.mode,
        audit_out=args.audit_out or None,
        region_grid_x=args.region_grid_x,
        region_grid_y=args.region_grid_y,
        region_heatmap_sigma_cells=args.region_heatmap_sigma_cells,
    )
    graph = structured.get('graph') or {}
    summary = {
        'output': args.output,
        'mode': args.mode,
        'components': audit.get('component_count'),
        'modules': audit.get('module_count'),
        'sequence_len': len(graph.get('sequence') or []),
        'sequence_policy': graph.get('sequence_source'),
        'needs_review': audit.get('semantic_needs_review_count'),
        'expert_region_input_leak': audit.get('expert_region_input_leak', {}).get('leaked'),
        'region_grid': [args.region_grid_x, args.region_grid_y],
        'audit_out': args.audit_out or str(Path(args.output).with_suffix(Path(args.output).suffix + '.audit.json')),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
