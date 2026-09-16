#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.semantic_labels import annotate_board


def iter_json_files(root: Path):
    for path in root.rglob('*.json'):
        try:
            if '/data/' not in str(path).replace('\\', '/'):
                continue
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'components' in data and 'board' in data:
                yield path, data
        except Exception:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description='Annotate PCB dataset JSON files with semantic labels.')
    parser.add_argument('--root', type=str, default='.', help='Project root directory')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    count = 0
    for path, data in iter_json_files(root):
        annotated = annotate_board(data)
        with path.open('w', encoding='utf-8') as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)
            f.write('\n')
        count += 1
    print(f'Annotated {count} JSON files under {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
