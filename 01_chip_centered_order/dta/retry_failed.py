"""Retry failed expert trajectories using heuristic placement or larger margin.

Usage (PowerShell):
  $env:PYTHONPATH='c:/Users/24663/Desktop/my (2)/my'
  python dta/retry_failed.py --paths "dta/expert_traj (2)/expert_traj/expert875_traj" "dta/expert_traj (2)/expert_traj/expert679_traj"

The script will try placement_mode='layout' first (to preserve layout). If that
fails it will retry with placement_mode='heuristic' and then with an increased
margin. On success it saves a cached tensor file under `output/dataset_cache`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch

import sys
# Ensure project root is on sys.path so local `datasets` and `DT` packages import correctly
proj_root = os.getcwd()
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from datasets.expert_dataset import ExpertTrajectoryProcessor


def try_process(traj_dir: Path, out_cache_dir: Path, grid_size: int = 128):
    print(f"Processing {traj_dir}")
    # Try layout first (preserve exact expert positions)
    try:
        proc = ExpertTrajectoryProcessor(traj_dir, grid_size=grid_size, margin=2, placement_mode="layout")
        tensors = proc.rollout()
        cache_name = f"{traj_dir.name}_gs{grid_size}_m2.pt"
        out_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, out_cache_dir / cache_name)
        print(f"Success (layout). Saved cache: {out_cache_dir / cache_name}")
        return True, "layout"
    except Exception as e:
        print(f"Layout attempt failed: {e}")

    # Retry heuristic
    try:
        proc = ExpertTrajectoryProcessor(traj_dir, grid_size=grid_size, margin=2, placement_mode="heuristic")
        tensors = proc.rollout()
        cache_name = f"{traj_dir.name}_gs{grid_size}_m2_heuristic.pt"
        out_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, out_cache_dir / cache_name)
        print(f"Success (heuristic). Saved cache: {out_cache_dir / cache_name}")
        return True, "heuristic"
    except Exception as e:
        print(f"Heuristic attempt failed: {e}")

    # Retry with larger margin (to reduce tight boundary conflicts)
    for margin in (4, 6, 8):
        try:
            proc = ExpertTrajectoryProcessor(traj_dir, grid_size=grid_size, margin=margin, placement_mode="heuristic")
            tensors = proc.rollout()
            cache_name = f"{traj_dir.name}_gs{grid_size}_m{margin}_heuristic.pt"
            out_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(tensors, out_cache_dir / cache_name)
            print(f"Success (heuristic, margin={margin}). Saved cache: {out_cache_dir / cache_name}")
            return True, f"heuristic_m{margin}"
        except Exception as e:
            print(f"Heuristic+margin={margin} failed: {e}")

    # As a last resort, try using pcbagent_repro.train_expert.build_expert_traj
    try:
        # Load pcbagent_repro/train_expert.py by file path to avoid name conflicts
        import importlib.util

        pkg_dir = os.path.join(os.getcwd(), "pcbagent_repro", "pcbagent_repro")
        train_expert_path = os.path.join(pkg_dir, "train_expert.py")
        if not os.path.exists(train_expert_path):
            raise FileNotFoundError(f"train_expert.py not found at {train_expert_path}")

        spec = importlib.util.spec_from_file_location("pcbagent_train_expert", train_expert_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "build_expert_traj"):
            raise ImportError("build_expert_traj not found in loaded train_expert module")
        build_expert_traj = module.build_expert_traj

        # load raw task/layout json
        base_name = traj_dir.name.replace("_traj", "")
        task_path = traj_dir / f"{base_name}_task.json"
        layout_path = traj_dir / f"{base_name}_layout.json"
        task = json.load(open(task_path, "r", encoding="utf-8"))
        layout = json.load(open(layout_path, "r", encoding="utf-8"))

        steps, traj, (wl, slw) = build_expert_traj(task, layout)
        out_path = out_cache_dir / f"{traj_dir.name}_pcbenv_steps.json"
        out_cache_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"steps_len": len(steps), "hpwl": float(wl), "slw": float(slw), "traj": traj}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Fallback pcbagent_repro.build_expert_traj succeeded; wrote {out_path}")
        return True, "pcbenv_build"
    except Exception as e:
        print(f"Fallback pcbagent_repro.build_expert_traj failed: {e}")

    return False, "all_attempts_failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, nargs='+', required=True, help="One or more trajectory folder paths to retry")
    ap.add_argument("--out-cache", type=str, default="output/dataset_cache", help="Cache output dir")
    ap.add_argument("--grid-size", type=int, default=128)
    args = ap.parse_args()

    out_cache_dir = Path(args.out_cache)
    results = {"success": [], "failures": []}

    for p in args.paths:
        traj_dir = Path(p)
        if not traj_dir.exists():
            print(f"Path not found: {traj_dir}")
            results["failures"].append({"path": str(traj_dir), "reason": "not_found"})
            continue
        ok, mode = try_process(traj_dir, out_cache_dir, grid_size=args.grid_size)
        if ok:
            results["success"].append({"path": str(traj_dir), "mode": mode})
        else:
            results["failures"].append({"path": str(traj_dir), "reason": mode})

    summary_path = out_cache_dir / "retry_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"Done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
