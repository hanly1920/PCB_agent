#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-runs/checkpoints/train-floorplan-mixed-b3_2-spread-edge-align-bs8_final.pt}"
TRAIN_DIR="${TRAIN_DIR:-data/train}"
RUN="${RUN:-runs/train_infer_b3_2_final}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
TIMEOUT_MIN="${TIMEOUT_MIN:-30}"

OUT="$RUN/infer_out"
LOG="$RUN/infer_logs"

mkdir -p "$OUT" "$LOG"

test -f "$CKPT" || { echo "[ERR] missing checkpoint: $CKPT"; exit 1; }
test -d "$TRAIN_DIR" || { echo "[ERR] missing train dir: $TRAIN_DIR"; exit 1; }

echo "[INFO] checkpoint: $CKPT"
echo "[INFO] train_dir:  $TRAIN_DIR"
echo "[INFO] output:     $OUT"
echo "[INFO] gpu:        $GPU"
echo "[INFO] boards:     $(find "$TRAIN_DIR" -maxdepth 1 -type f -name "*.json" ! -name "*.audit.json" | wc -l)"

for f in "$TRAIN_DIR"/*.json; do
  [ -f "$f" ] || continue
  [[ "$f" == *.audit.json ]] && continue

  base="$(basename "$f" .json)"
  echo "[RUN] $base"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU" timeout "${TIMEOUT_MIN}m" \
    python -u scripts/step4_infer.py \
      --test_glob "$f" \
      --ckpt "$CKPT" \
      --out_dir "$OUT" \
      --device cuda \
      --beam_width 1 \
      > "$LOG/$base.log" 2>&1
  code=$?
  set -e

  if [ "$code" -eq 124 ]; then
    echo "[TIMEOUT] $base"
  elif [ "$code" -ne 0 ]; then
    echo "[FAIL] $base code=$code log=$LOG/$base.log"
  else
    echo "[OK] $base"
  fi
done

python - "$TRAIN_DIR" "$OUT" "$RUN/infer_summary.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

train_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
summary = Path(sys.argv[3])

rows = []

for task in sorted(train_dir.glob("*.json")):
    if task.name.endswith(".audit.json"):
        continue

    stem = task.stem
    candidates = [
        out_dir / f"{stem}.infer.json",
        out_dir / f"{stem}.json",
        out_dir / stem / "infer.json",
        out_dir / stem / f"{stem}.infer.json",
    ]
    pred = next((p for p in candidates if p.exists()), None)

    row = {
        "board": stem,
        "complete": "",
        "placed": "",
        "expected": "",
        "failure": "",
        "objective": "",
        "json_path": str(pred) if pred else "",
    }

    if pred:
        d = json.loads(pred.read_text(encoding="utf-8"))
        row.update({
            "complete": d.get("complete"),
            "placed": d.get("placed_count"),
            "expected": d.get("expected_count"),
            "failure": d.get("failure_reason"),
            "objective": d.get("objective"),
        })

    rows.append(row)

with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

complete = sum(r["complete"] is True for r in rows)
partial = sum(bool(r["json_path"]) and r["complete"] is not True for r in rows)
missing = sum(not r["json_path"] for r in rows)

print(f"[SUMMARY] complete={complete} partial={partial} missing={missing} total={len(rows)}")
print(f"[SUMMARY] {summary}")
PY
