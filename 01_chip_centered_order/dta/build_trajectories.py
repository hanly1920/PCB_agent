"""Build and cache expert trajectories from expert folders.

Usage (PowerShell):
    python dta/build_trajectories.py --root "dta/expert_traj (2)/expert_traj" --cache output/dataset_cache --grid-size 128 --placement-mode layout --merge

This script will create per-trajectory cache files using
`datasets.expert_dataset.ExpertTrajectoryDataset` and optionally merge
all trajectories into a single `trajectories.pt` file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from datasets.expert_dataset import ExpertTrajectoryDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder containing expert*_traj dirs")
    parser.add_argument("--cache", type=str, default="output/dataset_cache", help="Cache dir for per-trajectory .pt files")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--placement-mode", type=str, choices=["heuristic", "layout"], default="heuristic")
    parser.add_argument("--merge", action="store_true", help="Also merge all trajectories into a single trajectories.pt file")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, only process this many samples (for quick check)")
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading expert trajectories from: {root}")
    dataset = ExpertTrajectoryDataset(
        root_dir=root,
        grid_size=args.grid_size,
        margin=args.margin,
        cache_dir=cache_dir,
        preload=False,
        placement_mode=args.placement_mode,
    )

    n = len(dataset)
    print(f"Found {n} trajectories (split={dataset.split}).")

    stats = {
        "total": n,
        "processed": 0,
        "lengths": [],
    }

    merged = []
    limit = args.max_samples if args.max_samples > 0 else n

    for idx in range(min(n, limit)):
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"Error processing idx={idx}: {e}")
            continue
        stats["processed"] += 1
        length = sample.states.size(0)
        stats["lengths"].append(length)
        print(f"Processed {idx+1}/{limit}: length={length}")
        if args.merge:
            merged.append(sample)

    summary = {
        "total_found": n,
        "processed": stats["processed"],
        "avg_length": float(sum(stats["lengths"]) / stats["processed"]) if stats["processed"] else 0,
    }

    summary_path = cache_dir / "trajectories_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote summary to {summary_path}")

    if args.merge:
        merged_path = cache_dir / "trajectories.pt"
        print(f"Saving merged trajectories ({len(merged)}) to {merged_path}")
        torch.save(merged, merged_path)

    print("Done.")


if __name__ == "__main__":
    main()
