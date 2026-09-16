#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.semantic_labels import load_json, save_json, refresh_semantic_meta
from pcbplace.json_schema import validate_task_json


def _split_pipe(s: str):
    s = str(s or '').strip()
    if not s:
        return []
    return [x.strip() for x in s.split('|') if x.strip()]


def _parse_optional_bool(s: str):
    s = str(s or '').strip().lower()
    if not s:
        return None
    if s in {'1', 'true', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'no', 'n'}:
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description='Apply manual / semi-manual semantic review overrides back into dataset JSON files.')
    ap.add_argument('--root', type=str, default='.', help='Project root')
    ap.add_argument('--review_csv', type=str, required=True, help='Edited semantic review CSV')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    review_csv = Path(args.review_csv).resolve()

    rows_by_file: Dict[str, list[dict[str, Any]]] = {}
    with review_csv.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            rows_by_file.setdefault(row['task_relpath'], []).append(row)

    updated_files = 0
    updated_components = 0
    for relpath, rows in rows_by_file.items():
        path = root / relpath
        if not path.exists():
            print(f'[warn] Missing task file: {path}')
            continue
        data = load_json(path)
        validate_task_json(data, source=str(path))
        comp_by_ref = {str(c.get('ref', '')): c for c in data.get('components', []) or []}
        file_changed = False

        for row in rows:
            ref = str(row.get('ref', ''))
            comp = comp_by_ref.get(ref)
            if comp is None:
                continue
            review = comp.setdefault('semantic_review', {})
            semantic = comp.setdefault('semantic', {})
            overrides = {}

            mapping = [
                ('manual_semantic_class', 'semantic_class'),
                ('manual_functional_group', 'functional_group'),
                ('manual_region_type', 'region_type'),
                ('manual_side_preference', 'side_preference'),
                ('manual_align_group', 'align_group'),
                ('manual_anchor_ref', 'anchor_ref'),
                ('manual_subzone', 'subzone'),
                ('manual_same_side_group', 'same_side_group'),
                ('manual_boundary_order', 'boundary_order'),
            ]
            for manual_key, field_name in mapping:
                val = str(row.get(manual_key, '') or '').strip()
                if val:
                    comp[field_name] = val
                    semantic[field_name] = val
                    overrides[field_name] = val

            crit_nets = _split_pipe(row.get('manual_critical_nets', ''))
            if crit_nets:
                comp['critical_nets'] = crit_nets
                semantic['critical_nets'] = crit_nets
                overrides['critical_nets'] = crit_nets

            crit_nbs = _split_pipe(row.get('manual_critical_neighbors', ''))
            if crit_nbs:
                comp['critical_neighbors'] = crit_nbs
                semantic['critical_neighbors'] = crit_nbs
                overrides['critical_neighbors'] = crit_nbs

            manual_touch_boundary = _parse_optional_bool(row.get('manual_must_touch_boundary', ''))
            if manual_touch_boundary is not None:
                comp['must_touch_boundary'] = bool(manual_touch_boundary)
                semantic['must_touch_boundary'] = bool(manual_touch_boundary)
                overrides['must_touch_boundary'] = bool(manual_touch_boundary)

            review_status = str(row.get('manual_review_status', '') or '').strip()
            notes = str(row.get('manual_review_notes', '') or '').strip()
            if overrides or review_status or notes:
                review['review_status'] = review_status or 'approved'
                review['review_notes'] = notes
                review['needs_review'] = False if (review_status or overrides) else bool(review.get('needs_review', False))
                review['manual_overrides'] = {**(review.get('manual_overrides') or {}), **overrides}
                review['label_source'] = 'manual_review_v1' if overrides else review.get('label_source', 'auto_seed_semantic_v2')
                semantic['label_source'] = review['label_source']
                file_changed = True
                updated_components += 1

        if file_changed:
            refresh_semantic_meta(data)
            validate_task_json(data, source=str(path))
            save_json(path, data)
            updated_files += 1

    print(f'Updated files: {updated_files}')
    print(f'Updated components: {updated_components}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
