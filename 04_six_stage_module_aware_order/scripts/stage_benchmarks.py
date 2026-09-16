#!/usr/bin/env python
"""
Stage PCB Bookshelf benchmark boards into the JSON format expected by step4_infer.py.

The Bookshelf files are retained for external legality/routing evaluation, while
this script copies their matching semantic JSON task from data/infer or data/train.
No expert placement is used by inference: infer.py strips expert fields defensively.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_root", required=True,
                    help="Path to extracted benchmarks directory containing _meta/")
    ap.add_argument("--project_root", default=".",
                    help="Project root containing data/infer and data/train")
    ap.add_argument("--out_root", default="runs/benchmark_inputs",
                    help="Output root for staged JSON files and manifest")
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    project_root = Path(args.project_root)
    out_root = Path(args.out_root)

    meta_dir = bench_root / "_meta"
    if not meta_dir.is_dir():
        raise SystemExit(f"Missing benchmark metadata directory: {meta_dir}")

    for split in ("heldout_inrange", "heldout_oversize", "train_seen"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    manifest = {
        "bench_root": str(bench_root),
        "project_root": str(project_root),
        "splits": {"heldout_inrange": [], "heldout_oversize": [], "train_seen": [], "unmapped": []},
    }

    for map_path in sorted(meta_dir.glob("*.map.json")):
        meta = json.loads(map_path.read_text(encoding="utf-8"))
        name = str(meta["name"])
        components = int(meta.get("pcb", {}).get("num_nodes", 0))

        infer_src = project_root / "data" / "infer" / f"{name}.json"
        train_src = project_root / "data" / "train" / f"{name}.json"

        if infer_src.exists():
            split = "heldout_inrange" if components <= 114 else "heldout_oversize"
            src = infer_src
        elif train_src.exists():
            split = "train_seen"
            src = train_src
        else:
            manifest["splits"]["unmapped"].append({
                "name": name,
                "components": components,
                "map": str(map_path),
                "reason": "No matching data/infer or data/train JSON in this project",
            })
            continue

        dst = out_root / split / src.name
        shutil.copy2(src, dst)
        manifest["splits"][split].append({
            "name": name,
            "components": components,
            "source": str(src),
            "staged": str(dst),
            "bookshelf_dir": str(bench_root / "pcb_bookshelf" / name),
        })

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    for split, items in manifest["splits"].items():
        print(f"{split}: {len(items)}")


if __name__ == "__main__":
    main()
