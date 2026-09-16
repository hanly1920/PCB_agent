#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.semantic_labels import populate_layout_semantic_fields


def iter_json_files(root: Path):
    for path in root.rglob('*.json'):
        s = str(path).replace('\\', '/')
        if '/data/' not in s:
            continue
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'components' in data and 'board' in data:
                yield path, data
        except Exception:
            continue


def main() -> int:
    ap = argparse.ArgumentParser(description='Populate layout-aware semantic fields (anchor_ref, subzone, same_side_group, boundary_order).')
    ap.add_argument('--root', type=str, default='.', help='Project root')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing layout-aware semantic fields.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    count = 0
    for path, data in iter_json_files(root):
        data = populate_layout_semantic_fields(data, overwrite=bool(args.overwrite))
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        count += 1
    print(f'Updated {count} JSON files under {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
