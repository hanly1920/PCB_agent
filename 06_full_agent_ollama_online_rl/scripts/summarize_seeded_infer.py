from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "objective",
    "complete",
    "placed_count",
    "expected_count",
    "metrics.hpwl_ratio",
    "metrics.connector_side_accuracy",
    "metrics.same_side_order_accuracy",
    "metrics.module_overlap_ratio",
]


def _parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _get_nested(data: dict[str, Any], path: str) -> float | None:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, bool):
        return 1.0 if cur else 0.0
    if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
        return float(cur)
    return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize per-board infer JSONs across random seeds.")
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--seeds", default="7,13,23")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    seeds = _parse_seeds(args.seeds)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    by_board: dict[str, dict[str, list[float]]] = {}
    seen_files: dict[str, list[str]] = {}

    for seed in seeds:
        infer_dir = runs_dir / f"seed_{seed}" / "infer"
        for path in sorted(infer_dir.glob("*.infer.json")):
            board = path.name.removesuffix(".infer.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            by_board.setdefault(board, {m: [] for m in metrics})
            seen_files.setdefault(board, []).append(str(path))
            for metric in metrics:
                value = _get_nested(data, metric)
                if value is not None:
                    by_board[board][metric].append(value)

    rows: list[dict[str, Any]] = []
    for board in sorted(by_board):
        row: dict[str, Any] = {
            "board": board,
            "seed_count": len(seen_files.get(board, [])),
        }
        for metric in metrics:
            mean, std = _mean_std(by_board[board][metric])
            row[f"{metric}.mean"] = mean
            row[f"{metric}.std"] = std
        rows.append(row)

    summary = {
        "runs_dir": str(runs_dir),
        "seeds": seeds,
        "metrics": metrics,
        "boards": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["board", "seed_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
