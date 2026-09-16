#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-runs/checkpoints/train-floorplan-mixed-b3_2-spread-edge-align-bs8_final.pt}"
SRC="${SRC:-data/benchmark_infer}"
RUN="${RUN:-runs/testinfer_b3_2_final_all74}"
ROUNDS="${ROUNDS:-6}"        # round00 + retry01~retry05
TIMEOUT_MIN="${TIMEOUT_MIN:-30}"

IN="$RUN/infer_input_clean"

test -f "$CKPT" || { echo "[ERR] missing checkpoint: $CKPT"; exit 1; }
test -d "$SRC" || { echo "[ERR] missing benchmark dir: $SRC"; exit 1; }

mkdir -p "$IN"
rm -f "$IN"/*.json

find "$SRC" -maxdepth 1 -type f -name "*.json" ! -name "*.audit.json" -print0 \
  | sort -z \
  | while IFS= read -r -d '' f; do
      cp -f "$f" "$IN/$(basename "$f")"
    done

N=$(find "$IN" -maxdepth 1 -type f -name "*.json" | wc -l)
echo "[INFO] checkpoint: $CKPT"
echo "[INFO] source:     $SRC"
echo "[INFO] input:      $IN"
echo "[INFO] boards:     $N"
echo "[INFO] rounds:     $ROUNDS"

python -m py_compile pcbplace/infer.py scripts/step4_infer.py

for idx in $(seq 0 $((ROUNDS - 1))); do
  TAG=$(printf "%02d" "$idx")
  OUT="$RUN/infer_out_round${TAG}"
  LOG="$RUN/infer_logs_round${TAG}"
  TODO="$RUN/todo_round${TAG}.txt"

  mkdir -p "$OUT" "$LOG"

  python - "$RUN" "$IN" "$idx" "$TODO" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
idx = int(sys.argv[3])
todo_path = Path(sys.argv[4])

def find_json(out, stem):
    candidates = [
        out / f"{stem}.infer.json",
        out / f"{stem}.json",
        out / stem / "infer.json",
        out / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

def is_complete_before(stem):
    for j in range(idx):
        out = run / f"infer_out_round{j:02d}"
        p = find_json(out, stem)
        if not p:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("complete") is True:
            return True
    return False

boards = [p.stem for p in sorted(inp.glob("*.json"))]
todo = []
for b in boards:
    if idx == 0 or not is_complete_before(b):
        todo.append(b)

todo_path.write_text("\n".join(todo) + ("\n" if todo else ""), encoding="utf-8")
print(f"[TODO] round{idx:02d}: {len(todo)}/{len(boards)}")
PY

  if [ ! -s "$TODO" ]; then
    echo "[DONE] all boards already complete before round${TAG}"
    break
  fi

  echo "[INFO] round${TAG} output=$OUT"

  while IFS= read -r base; do
    [ -n "$base" ] || continue
    f="$IN/$base.json"

    echo "[RUN][round${TAG}] $base"

    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" timeout "${TIMEOUT_MIN}m" \
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
      echo "[TIMEOUT][round${TAG}] $base"
    elif [ "$code" -ne 0 ]; then
      echo "[FAIL][round${TAG}] $base code=$code, see $LOG/$base.log"
    else
      echo "[OK][round${TAG}] $base"
    fi
  done < "$TODO"

  python - "$RUN" "$IN" "$((idx + 1))" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
nrounds = int(sys.argv[3])

def find_json(out, stem):
    candidates = [
        out / f"{stem}.infer.json",
        out / f"{stem}.json",
        out / stem / "infer.json",
        out / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

boards = [p.stem for p in sorted(inp.glob("*.json"))]
complete = 0
partial = 0
missing = 0

for b in boards:
    best = None
    for j in range(nrounds):
        p = find_json(run / f"infer_out_round{j:02d}", b)
        if not p:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        placed = int(d.get("placed_count") or 0)
        row = (bool(d.get("complete")), placed, d.get("expected_count"), j)
        if best is None:
            best = row
        elif (row[0] and not best[0]) or (row[0] == best[0] and row[1] > best[1]):
            best = row

    if best is None:
        missing += 1
    elif best[0]:
        complete += 1
    else:
        partial += 1

print(f"[ROUND_SUMMARY] complete={complete} partial={partial} missing={missing} total={len(boards)}")
PY

done

python - "$RUN" "$IN" "$ROUNDS" <<'PY'
import csv, json, sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
nrounds = int(sys.argv[3])

def find_json(out, stem):
    candidates = [
        out / f"{stem}.infer.json",
        out / f"{stem}.json",
        out / stem / "infer.json",
        out / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

rows = []
boards = [p.stem for p in sorted(inp.glob("*.json"))]

for b in boards:
    best = None
    all_runs = []

    for j in range(nrounds):
        p = find_json(run / f"infer_out_round{j:02d}", b)
        if not p:
            continue

        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        row = {
            "board": b,
            "best_round": f"round{j:02d}",
            "complete": d.get("complete"),
            "placed": int(d.get("placed_count") or 0),
            "expected": d.get("expected_count"),
            "failure": d.get("failure_reason"),
            "objective": d.get("objective"),
            "objective_partial": d.get("objective_partial"),
            "json_path": str(p),
        }

        all_runs.append(row)

        if best is None:
            best = row
        else:
            cur_complete = bool(row["complete"])
            best_complete = bool(best["complete"])
            if (cur_complete and not best_complete) or (
                cur_complete == best_complete and int(row["placed"]) > int(best["placed"])
            ):
                best = row

    if best is None:
        best = {
            "board": b,
            "best_round": "",
            "complete": False,
            "placed": 0,
            "expected": "",
            "failure": "missing_json",
            "objective": "",
            "objective_partial": "",
            "json_path": "",
        }

    rows.append(best)

summary = run / "best_summary.csv"
fields = ["board", "best_round", "complete", "placed", "expected", "failure", "objective", "objective_partial", "json_path"]
with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

complete = sum(1 for r in rows if r["complete"] is True)
partial = sum(1 for r in rows if r["complete"] is not True and r["json_path"])
missing = sum(1 for r in rows if not r["json_path"])

print(f"[FINAL] complete={complete} partial={partial} missing={missing} total={len(rows)}")
print(f"[SUMMARY] {summary}")

print("[INCOMPLETE]")
for r in rows:
    if r["complete"] is not True:
        print(f"{r['board']:<60} best={r['best_round']:<8} placed={r['placed']}/{r['expected']} failure={r['failure']}")
PY
