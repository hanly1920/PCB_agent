#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description='Seed semantic labels with review metadata and export review CSVs.')
    ap.add_argument('--root', type=str, default='.', help='Project root')
    ap.add_argument('--out_dir', type=str, default='reviews', help='Review output directory')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts' / 'annotate_semantic_labels.py'), '--root', str(root)])
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'scripts' / 'export_semantic_review.py'), '--root', str(root), '--out_dir', args.out_dir])
    print('Bootstrap complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
