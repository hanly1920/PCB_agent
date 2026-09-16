#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcbplace.semantic_labels import annotate_board_infer_safe
from pcbplace.training_structure import sanitize_infer_input
from pcbplace.utils import load_json, save_json


def _out_path(path: Path, out_dir: Path | None) -> Path:
    return out_dir / path.name if out_dir else path


def _annotate(path: Path) -> dict:
    return annotate_board_infer_safe(sanitize_infer_input(load_json(path)))


def main() -> int:
    parser = argparse.ArgumentParser(description='Annotate semantic labels for INFER mode. No expert xy/bbox/expert-derived hint is used.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Single input JSON')
    group.add_argument('--glob', help='Input JSON glob')
    parser.add_argument('--output', help='Single output JSON, required with --input unless updating in place')
    parser.add_argument('--out-dir', help='Output directory for --glob. Omit to update in place.')
    args = parser.parse_args()

    if args.input:
        inp = Path(args.input)
        out = Path(args.output) if args.output else inp
        save_json(_annotate(inp), out)
        print(f'[OK] infer-safe semantic: {inp} -> {out}')
        return 0

    paths = [Path(p) for p in sorted(glob.glob(args.glob, recursive=True))]
    if not paths:
        raise SystemExit('No JSON files matched.')
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for path in paths:
        try:
            out = _out_path(path, out_dir)
            save_json(_annotate(path), out)
            ok += 1
            print(f'[OK] {path} -> {out}')
        except Exception as exc:
            print(f'[FAIL] {path}: {exc}')
    print(f'done ok={ok}/{len(paths)}')
    return 0 if ok == len(paths) else 1


if __name__ == '__main__':
    raise SystemExit(main())
