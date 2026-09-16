from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.semantic_metrics import evaluate_task_file, summarize_semantic_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_glob", required=True, help="Task JSON glob, e.g. data/seq/*.json")
    ap.add_argument("--pred_dir", default=None, help="Directory containing *.infer.json files. If omitted, evaluate embedded expert layouts.")
    ap.add_argument("--pred_suffix", default=".infer.json", help="Prediction filename suffix inside pred_dir.")
    ap.add_argument("--out", default=None, help="Optional output JSON path.")
    ap.add_argument("--legacy_zone_edge_ratio", type=float, default=0.12)
    ap.add_argument("--legacy_zone_core_ratio", type=float, default=0.28)
    args = ap.parse_args()

    task_paths = sorted(glob.glob(args.task_glob))
    if not task_paths:
        raise SystemExit(f"No task files matched: {args.task_glob}")

    per_task = []
    missing_preds = []
    for task_path in task_paths:
        pred_path = None
        if args.pred_dir:
            pred_candidate = Path(args.pred_dir) / (Path(task_path).stem + args.pred_suffix)
            if pred_candidate.exists():
                pred_path = pred_candidate
            else:
                missing_preds.append(str(pred_candidate))
                continue
        per_task.append(
            evaluate_task_file(
                task_path,
                pred_path=pred_path,
                legacy_zone_edge_ratio=args.legacy_zone_edge_ratio,
                legacy_zone_core_ratio=args.legacy_zone_core_ratio,
            )
        )

    summary = summarize_semantic_metrics(per_task)
    payload = {
        "summary": summary,
        "per_task": per_task,
        "missing_predictions": missing_preds,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing_preds:
        print(f"[warn] missing predictions for {len(missing_preds)} tasks")


if __name__ == "__main__":
    main()
