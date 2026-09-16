#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.semantic_labels import iter_dataset_json_files, load_json
from pcbplace.json_schema import validate_task_json


def summarize_evidence(comp: dict) -> str:
    review = comp.get('semantic_review') or {}
    ev = review.get('evidence') or {}
    pieces = []
    if ev.get('semantic_rule_hits'):
        pieces.append('semantic=' + '|'.join(ev['semantic_rule_hits']))
    if ev.get('functional_group_rule_hits'):
        pieces.append('group=' + '|'.join(ev['functional_group_rule_hits']))
    if ev.get('region_rule_hits'):
        pieces.append('region=' + '|'.join(ev['region_rule_hits']))
    if ev.get('ambiguity_flags'):
        pieces.append('ambiguity=' + '|'.join(ev['ambiguity_flags']))
    return '; '.join(pieces)


def main() -> int:
    ap = argparse.ArgumentParser(description='Export inspectable/editable semantic label review CSVs.')
    ap.add_argument('--root', type=str, default='.', help='Project root')
    ap.add_argument('--out_dir', type=str, default='reviews', help='Output review directory')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / 'semantic_review_all.csv'
    queue_csv = out_dir / 'semantic_review_queue.csv'

    fields = [
        'task_relpath', 'split', 'ref', 'type', 'footprint', 'size_mm', 'allowed_sides', 'must_touch_boundary',
        'semantic_class', 'functional_group', 'region_type', 'side_preference', 'align_group',
        'anchor_ref', 'subzone', 'same_side_group', 'boundary_order',
        'critical_nets', 'critical_neighbors', 'label_source', 'auto_confidence', 'needs_review',
        'review_priority', 'review_status', 'evidence_summary', 'candidate_semantic_class',
        'candidate_functional_group', 'candidate_region_type', 'manual_semantic_class', 'manual_must_touch_boundary',
        'manual_functional_group', 'manual_region_type', 'manual_side_preference',
        'manual_align_group', 'manual_anchor_ref', 'manual_subzone', 'manual_same_side_group', 'manual_boundary_order',
        'manual_critical_nets', 'manual_critical_neighbors',
        'manual_review_status', 'manual_review_notes'
    ]

    rows = []
    for path in iter_dataset_json_files(root):
        data = load_json(path)
        validate_task_json(data, source=str(path))
        task_relpath = str(path.relative_to(root)).replace('\\', '/')
        split = task_relpath.split('/')[1] if '/' in task_relpath else 'unknown'
        for comp in data.get('components', []) or []:
            review = comp.get('semantic_review') or {}
            cands = review.get('candidate_labels') or {}
            row = {
                'task_relpath': task_relpath,
                'split': split,
                'ref': comp.get('ref', ''),
                'type': comp.get('type', ''),
                'footprint': comp.get('footprint', ''),
                'size_mm': json.dumps(comp.get('size_mm', []), ensure_ascii=False),
                'allowed_sides': json.dumps(comp.get('allowed_sides', []), ensure_ascii=False),
                'must_touch_boundary': comp.get('must_touch_boundary', (comp.get('semantic') or {}).get('must_touch_boundary', '')),
                'semantic_class': comp.get('semantic_class', ''),
                'functional_group': comp.get('functional_group', ''),
                'region_type': comp.get('region_type', ''),
                'side_preference': comp.get('side_preference', ''),
                'align_group': comp.get('align_group', ''),
                'anchor_ref': comp.get('anchor_ref', ''),
                'subzone': comp.get('subzone', ''),
                'same_side_group': comp.get('same_side_group', ''),
                'boundary_order': comp.get('boundary_order', ''),
                'critical_nets': '|'.join(comp.get('critical_nets', []) or []),
                'critical_neighbors': '|'.join(comp.get('critical_neighbors', []) or []),
                'label_source': (review.get('label_source') or (comp.get('semantic') or {}).get('label_source') or ''),
                'auto_confidence': review.get('auto_confidence', ''),
                'needs_review': review.get('needs_review', False),
                'review_priority': review.get('review_priority', ''),
                'review_status': review.get('review_status', ''),
                'evidence_summary': summarize_evidence(comp),
                'candidate_semantic_class': '|'.join(cands.get('semantic_class', []) or []),
                'candidate_functional_group': '|'.join(cands.get('functional_group', []) or []),
                'candidate_region_type': '|'.join(cands.get('region_type', []) or []),
                'manual_semantic_class': '',
                'manual_must_touch_boundary': '',
                'manual_functional_group': '',
                'manual_region_type': '',
                'manual_side_preference': '',
                'manual_align_group': '',
                'manual_anchor_ref': '',
                'manual_subzone': '',
                'manual_same_side_group': '',
                'manual_boundary_order': '',
                'manual_critical_nets': '',
                'manual_critical_neighbors': '',
                'manual_review_status': '',
                'manual_review_notes': '',
            }
            rows.append(row)

    rows.sort(key=lambda r: (r['task_relpath'], r['ref']))
    with all_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    queue_rows = [r for r in rows if str(r['needs_review']).lower() in {'true', '1', 'yes'} or r['review_priority'] in {'high', 'medium'}]
    queue_rows.sort(key=lambda r: ({'high': 0, 'medium': 1, 'low': 2}.get(r['review_priority'], 3), r['task_relpath'], r['ref']))
    with queue_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(queue_rows)

    notes = out_dir / 'README.md'
    notes.write_text(
        '# Semantic review workflow\n\n'
        '- `semantic_review_all.csv`: full editable review sheet for all components.\n'
        '- `semantic_review_queue.csv`: priority queue filtered from the full sheet.\n\n'
        'Edit the `manual_*` columns and then run:\n\n'
        '```bash\n'
        'python scripts/apply_semantic_review.py --root . --review_csv reviews/semantic_review_all.csv\n'
        '```\n',
        encoding='utf-8'
    )

    print(f'Wrote {all_csv}')
    print(f'Wrote {queue_csv}')
    print(f'Total rows: {len(rows)} | Queue rows: {len(queue_rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
