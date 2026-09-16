#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace, root: Path) -> list[str]:
    return [
        sys.executable, str(root / "scripts" / "step3_train_masked.py"),
        "--train_glob", args.train_glob,
        "--device", args.device,
        "--steps", str(args.steps),
        "--warmup", "0",
        "--kmax_final", str(args.kmax_final),
        "--lr", str(args.lr),
        "--teacher_topk", str(args.teacher_topk),
        "--teacher_tau", str(args.teacher_tau),
        "--teacher_metric_weight", str(args.teacher_metric_weight),
        "--teacher_lambda_region_prior", str(args.teacher_lambda_region_prior),
        "--teacher_lambda_prior_region_heatmap", str(args.teacher_lambda_prior_region_heatmap),
        "--expert_snap_radius", str(args.expert_snap_radius),
        "--expert_snap_global_fallback",
        "--no_replay_finetune",
        "--save_path", args.save_path,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Iteratively run a smoke training check and move failing board JSON files to a bad directory.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Project root. Default: repository root.")
    parser.add_argument("--train_glob", default="data/train/*.json", help="Training JSON glob passed to step3_train_masked.py.")
    parser.add_argument("--bad_dir", default="data/bad_auto", help="Directory where detected bad JSON files are moved.")
    parser.add_argument("--max_rounds", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--kmax_final", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher_topk", type=int, default=16)
    parser.add_argument("--teacher_tau", type=float, default=0.8)
    parser.add_argument("--teacher_metric_weight", type=float, default=0.03)
    parser.add_argument("--teacher_lambda_region_prior", type=float, default=0.05)
    parser.add_argument("--teacher_lambda_prior_region_heatmap", type=float, default=0.05)
    parser.add_argument("--expert_snap_radius", type=int, default=12)
    parser.add_argument("--save_path", default="runs/fullcheck/fullcheck_model.pt")
    parser.add_argument("--dry_run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    bad_dir = (root / args.bad_dir).resolve()
    bad_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cmd = build_command(args, root)
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    bad_pattern = re.compile(r"file=(data[\\/]+train[\\/][^\s]+?\.json)")
    moved: list[str] = []
    for round_id in range(1, int(args.max_rounds) + 1):
        train_files = sorted((root / "data" / "train").glob("*.json"))
        print(f"\n========== Fullcheck round {round_id} ==========")
        print(f"Remaining train JSON: {len(train_files)}")
        if not train_files:
            print("No training JSON left.")
            return 1

        proc = subprocess.run(cmd, cwd=str(root), env=env, text=True, capture_output=True, encoding="utf-8", errors="replace")
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        if proc.returncode == 0:
            print("\nFULLCHECK OK")
            print(f"Moved bad boards: {len(moved)}")
            for name in moved:
                print(" -", name)
            print(f"Model saved to {args.save_path}")
            return 0

        output = proc.stdout + "\n" + proc.stderr
        match = bad_pattern.search(output)
        if not match:
            print("\nTraining failed, but no bad JSON filename was found in the error output.")
            print("Return code:", proc.returncode)
            return int(proc.returncode)

        rel = match.group(1).replace("\\", "/")
        bad_file = root / rel
        if not bad_file.exists():
            print("\nDetected bad file, but it does not exist:", bad_file)
            return 1
        dest = bad_dir / bad_file.name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            k = 2
            while (bad_dir / f"{stem}_{k}{suffix}").exists():
                k += 1
            dest = bad_dir / f"{stem}_{k}{suffix}"
        shutil.move(str(bad_file), str(dest))
        moved.append(bad_file.name)
        print(f"\nMoved bad board:\n  {bad_file}\n  -> {dest}")

    print("Too many rounds; stop.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
