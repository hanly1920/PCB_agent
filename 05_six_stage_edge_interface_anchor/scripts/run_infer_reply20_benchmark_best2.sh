#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="runs/checkpoints/train-floorplan-mixed-b3_3-reply20.pt"

INPUT_DIR="data/pcb_benchmark_infer_json"
if [[ ! -d "$INPUT_DIR" ]]; then
    INPUT_DIR="pcb_benchmark_infer_json"
fi

RUN_ROOT="runs/benchmark_infer_b3_3_reply20_best2"

R1_OUT="$RUN_ROOT/round1_greedy/infer_out"
R1_LOG="$RUN_ROOT/round1_greedy/infer_logs"

R2_OUT="$RUN_ROOT/round2_beam2/infer_out"
R2_LOG="$RUN_ROOT/round2_beam2/infer_logs"

BEST_OUT="$RUN_ROOT/best/infer_out"

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] checkpoint ²»´æÔÚ: $CKPT"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] ÊäÈëÄ¿Â¼²»´æÔÚ: $INPUT_DIR"
    exit 1
fi

rm -rf "$RUN_ROOT"
mkdir -p "$R1_OUT" "$R1_LOG" \
         "$R2_OUT" "$R2_LOG" \
         "$BEST_OUT"

mapfile -d '' FILES < <(
    find "$INPUT_DIR" -maxdepth 1 -type f \
        -name '*.json' ! -name '*.audit.json' \
        -print0 | sort -z
)

TOTAL="${#FILES[@]}"

if [[ "$TOTAL" -eq 0 ]]; then
    echo "[ERROR] $INPUT_DIR ÏÂÃ»ÓÐÕÒµ½ JSON"
    exit 1
fi

echo "========================================"
echo "[INFO] checkpoint: $CKPT"
echo "[INFO] input:      $INPUT_DIR"
echo "[INFO] boards:     $TOTAL"
echo "[INFO] run root:   $RUN_ROOT"
echo "========================================"


run_round() {
    local round_name="$1"
    local beam_width="$2"
    local beam_topk="$3"
    local out_dir="$4"
    local log_dir="$5"

    local ok=0
    local fail=0
    local index=0

    echo
    echo "========================================"
    echo "[$round_name] beam_width=$beam_width"
    echo "[$round_name] beam_topk=$beam_topk"
    echo "========================================"

    for file in "${FILES[@]}"; do
        index=$((index + 1))
        stem="$(basename "${file%.json}")"
        log_file="$log_dir/$stem.log"

        echo "[$round_name][$index/$TOTAL] $stem"

        if python -u scripts/step4_infer.py \
            --test_glob "$file" \
            --ckpt "$CKPT" \
            --out_dir "$out_dir" \
            --device cuda \
            --sequence_policy checkpoint \
            --layout_preset checkpoint \
            --beam_width "$beam_width" \
            --beam_topk "$beam_topk" \
            > "$log_file" 2>&1
        then
            ok=$((ok + 1))
            echo "[OK] $stem"
            tail -n 1 "$log_file" || true
        else
            fail=$((fail + 1))
            echo "[FAIL] $stem"
            tail -n 20 "$log_file" || true
        fi
    done

    echo
    echo "[$round_name SUMMARY] ok=$ok fail=$fail total=$TOTAL"
}


# µÚÒ»ÂÖ£º¿ìËÙ objective-aware greedy
run_round \
    "ROUND1" \
    1 \
    16 \
    "$R1_OUT" \
    "$R1_LOG"

# µÚ¶þÂÖ£ºÍ¬Ò» objective ÏÂ½øÐÐ beam search
run_round \
    "ROUND2" \
    2 \
    8 \
    "$R2_OUT" \
    "$R2_LOG"


echo
echo "========================================"
echo "[SELECT] selecting per-board best result"
echo "========================================"

python - "$R1_OUT" "$R2_OUT" "$BEST_OUT" "$RUN_ROOT/best_summary.csv" <<'PY'
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

round1_dir = Path(sys.argv[1])
round2_dir = Path(sys.argv[2])
best_dir = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

best_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def finite_float(value, default=float("inf")):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rank(data):
    """
    ÅÅÐòÓÅÏÈ¼¶£º
    1. ÍêÕû²¼¾ÖÓÅÓÚ²»ÍêÕû²¼¾Ö£»
    2. Á½ÕßÍêÕûÊ±£¬objective Ô½Ð¡Ô½ºÃ£»
    3. Á½Õß²»ÍêÕûÊ±£¬placed_count Ô½¶àÔ½ºÃ£»
    4. placed_count ÏàÍ¬Ê±£¬objective_partial Ô½Ð¡Ô½ºÃ¡£
    """
    if data is None:
        return (3, float("inf"), float("inf"))

    complete = bool(data.get("complete"))
    objective = finite_float(data.get("objective"))
    partial = finite_float(data.get("objective_partial"))
    placed = integer(data.get("placed_count"))

    if complete:
        return (0, objective, 0.0)

    return (1, -placed, partial)


names = sorted(
    {p.name for p in round1_dir.glob("*.infer.json")}
    | {p.name for p in round2_dir.glob("*.infer.json")}
)

rows = []
chosen_count = {"round1": 0, "round2": 0}
complete_count = 0

for name in names:
    p1 = round1_dir / name
    p2 = round2_dir / name

    d1 = read_json(p1) if p1.exists() else None
    d2 = read_json(p2) if p2.exists() else None

    if d1 is None and d2 is None:
        continue

    if rank(d2) < rank(d1):
        selected_round = "round2"
        selected_path = p2
        selected = d2
    else:
        selected_round = "round1"
        selected_path = p1
        selected = d1

    shutil.copy2(selected_path, best_dir / name)
    chosen_count[selected_round] += 1

    if bool(selected.get("complete")):
        complete_count += 1

    rows.append({
        "board": name.removesuffix(".infer.json"),
        "selected_round": selected_round,
        "complete": bool(selected.get("complete")),
        "placed_count": integer(selected.get("placed_count")),
        "expected_count": integer(selected.get("expected_count")),
        "objective_selected": finite_float(
            selected.get("objective"), default=1.0e30
        ),
        "round1_complete": bool(d1.get("complete")) if d1 else False,
        "round1_objective": (
            finite_float(d1.get("objective"), default=1.0e30)
            if d1 else ""
        ),
        "round2_complete": bool(d2.get("complete")) if d2 else False,
        "round2_objective": (
            finite_float(d2.get("objective"), default=1.0e30)
            if d2 else ""
        ),
    })

fieldnames = [
    "board",
    "selected_round",
    "complete",
    "placed_count",
    "expected_count",
    "objective_selected",
    "round1_complete",
    "round1_objective",
    "round2_complete",
    "round2_objective",
]

with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[SELECT SUMMARY] total={len(rows)}")
print(f"[SELECT SUMMARY] complete={complete_count}/{len(rows)}")
print(f"[SELECT SUMMARY] round1_selected={chosen_count['round1']}")
print(f"[SELECT SUMMARY] round2_selected={chosen_count['round2']}")
print(f"[SELECT SUMMARY] best_out={best_dir}")
print(f"[SELECT SUMMARY] csv={summary_path}")
PY

echo
echo "========================================"
echo "Reply20 benchmark inference finished."
echo "========================================"
echo "[ROUND1] $R1_OUT"
echo "[ROUND2] $R2_OUT"
echo "[BEST]   $BEST_OUT"
echo "[CSV]    $RUN_ROOT/best_summary.csv"
