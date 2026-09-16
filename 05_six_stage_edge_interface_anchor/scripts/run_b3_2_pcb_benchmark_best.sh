#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-runs/checkpoints/train-floorplan-mixed-b3_2-spread-edge-align-bs8_final.pt}"
SRC="${SRC:-data/pcb_benchmark_json}"
RUN="${RUN:-runs/infer_b3_2_pcb_benchmark_best}"

ROUNDS="${ROUNDS:-8}"
TIMEOUT_MIN="${TIMEOUT_MIN:-30}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

IN="$RUN/infer_input_clean"

test -f "$CKPT" || { echo "[ERR] missing checkpoint: $CKPT"; exit 1; }
test -d "$SRC" || { echo "[ERR] missing source directory: $SRC"; exit 1; }

mkdir -p "$IN"
rm -f "$IN"/*.json

find "$SRC" -maxdepth 1 -type f -name "*.json" ! -name "*.audit.json" -print0 |
  sort -z |
  while IFS= read -r -d '' f; do
    cp -f "$f" "$IN/$(basename "$f")"
  done

TOTAL=$(find "$IN" -maxdepth 1 -type f -name "*.json" | wc -l)

echo "[INFO] checkpoint: $CKPT"
echo "[INFO] source:     $SRC"
echo "[INFO] run root:   $RUN"
echo "[INFO] boards:     $TOTAL"
echo "[INFO] rounds:     $ROUNDS"
echo "[INFO] timeout:    ${TIMEOUT_MIN}m"
echo "[INFO] gpu:        $GPU"

python -m py_compile pcbplace/infer.py scripts/step4_infer.py

for IDX in $(seq 0 $((ROUNDS - 1))); do
  TAG=$(printf "%02d" "$IDX")
  OUT="$RUN/infer_out_round${TAG}"
  LOG="$RUN/infer_logs_round${TAG}"
  TODO="$RUN/todo_round${TAG}.txt"

  mkdir -p "$OUT" "$LOG"

  python - "$RUN" "$IN" "$IDX" "$TODO" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
current_round = int(sys.argv[3])
todo_path = Path(sys.argv[4])

def find_result(out_dir: Path, stem: str):
    candidates = [
        out_dir / f"{stem}.infer.json",
        out_dir / f"{stem}.json",
        out_dir / stem / "infer.json",
        out_dir / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

def completed_before(stem: str):
    for idx in range(current_round):
        p = find_result(run / f"infer_out_round{idx:02d}", stem)
        if not p:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("complete") is True:
            return True
    return False

boards = [p.stem for p in sorted(inp.glob("*.json"))]
todo = [b for b in boards if current_round == 0 or not completed_before(b)]

todo_path.write_text(
    "\n".join(todo) + ("\n" if todo else ""),
    encoding="utf-8",
)

print(f"[TODO] round{current_round:02d}: {len(todo)}/{len(boards)}")
PY

  if [ ! -s "$TODO" ]; then
    echo "[DONE] all boards completed before round${TAG}"
    break
  fi

  while IFS= read -r BASE; do
    [ -n "$BASE" ] || continue

    INPUT="$IN/$BASE.json"
    echo "[RUN][round${TAG}] $BASE"

    set +e
    CUDA_VISIBLE_DEVICES="$GPU" timeout "${TIMEOUT_MIN}m" \
      python -u scripts/step4_infer.py \
        --test_glob "$INPUT" \
        --ckpt "$CKPT" \
        --out_dir "$OUT" \
        --device cuda \
        --beam_width 1 \
        > "$LOG/$BASE.log" 2>&1
    CODE=$?
    set -e

    if [ "$CODE" -eq 124 ]; then
      echo "[TIMEOUT][round${TAG}] $BASE"
    elif [ "$CODE" -ne 0 ]; then
      echo "[FAIL][round${TAG}] $BASE code=$CODE log=$LOG/$BASE.log"
    else
      echo "[OK][round${TAG}] $BASE"
    fi
  done < "$TODO"

  python - "$RUN" "$IN" "$((IDX + 1))" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
round_count = int(sys.argv[3])

def find_result(out_dir: Path, stem: str):
    candidates = [
        out_dir / f"{stem}.infer.json",
        out_dir / f"{stem}.json",
        out_dir / stem / "infer.json",
        out_dir / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

complete = partial = missing = 0

for task in sorted(inp.glob("*.json")):
    board = task.stem
    best = None

    for idx in range(round_count):
        p = find_result(run / f"infer_out_round{idx:02d}", board)
        if not p:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        cur = (
            bool(data.get("complete") is True),
            int(data.get("placed_count") or 0),
        )

        if best is None or cur > best:
            best = cur

    if best is None:
        missing += 1
    elif best[0]:
        complete += 1
    else:
        partial += 1

print(
    f"[ROUND SUMMARY] complete={complete} "
    f"partial={partial} missing={missing} "
    f"total={complete + partial + missing}"
)
PY

done

python - "$RUN" "$IN" "$ROUNDS" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
inp = Path(sys.argv[2])
max_rounds = int(sys.argv[3])

best_dir = run / "infer_out_best"
best_dir.mkdir(parents=True, exist_ok=True)

for p in best_dir.glob("*.json"):
    p.unlink()

def find_result(out_dir: Path, stem: str):
    candidates = [
        out_dir / f"{stem}.infer.json",
        out_dir / f"{stem}.json",
        out_dir / stem / "infer.json",
        out_dir / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

rows = []

for task in sorted(inp.glob("*.json")):
    board = task.stem
    best = None

    for idx in range(max_rounds):
        p = find_result(run / f"infer_out_round{idx:02d}", board)
        if not p:
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        candidate = {
            "board": board,
            "best_round": f"round{idx:02d}",
            "complete": data.get("complete") is True,
            "placed": int(data.get("placed_count") or 0),
            "expected": int(data.get("expected_count") or 0),
            "failure": data.get("failure_reason") or "",
            "objective": data.get("objective"),
            "objective_partial": data.get("objective_partial"),
            "source_json": str(p),
        }

        if best is None:
            best = candidate
        else:
            better = False

            if candidate["complete"] and not best["complete"]:
                better = True
            elif candidate["complete"] == best["complete"]:
                if candidate["placed"] > best["placed"]:
                    better = True
                elif candidate["placed"] == best["placed"]:
                    # Í¬ÑùÍêÕû¡¢Í¬Ñù placed Ê±£¬ÓÅÏÈ objective ¸üÐ¡¡£
                    ca = candidate["objective"]
                    cb = best["objective"]
                    if ca is not None and (cb is None or float(ca) < float(cb)):
                        better = True

            if better:
                best = candidate

    if best is None:
        best = {
            "board": board,
            "best_round": "",
            "complete": False,
            "placed": 0,
            "expected": 0,
            "failure": "missing_json",
            "objective": "",
            "objective_partial": "",
            "source_json": "",
        }
    else:
        src = Path(best["source_json"])
        dst = best_dir / f"{board}.infer.json"
        shutil.copy2(src, dst)
        best["best_json"] = str(dst)

    rows.append(best)

summary = run / "best_summary.csv"
fields = [
    "board",
    "best_round",
    "complete",
    "placed",
    "expected",
    "failure",
    "objective",
    "objective_partial",
    "source_json",
    "best_json",
]

with summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})

complete = [r for r in rows if r["complete"]]
partial = [r for r in rows if not r["complete"] and r.get("source_json")]
missing = [r for r in rows if not r.get("source_json")]

print()
print(f"[FINAL] total={len(rows)}")
print(f"[FINAL] complete={len(complete)}")
print(f"[FINAL] partial={len(partial)}")
print(f"[FINAL] missing={len(missing)}")
print(f"[BEST DIR] {best_dir}")
print(f"[SUMMARY] {summary}")

if partial or missing:
    print()
    print("[INCOMPLETE]")
    for r in partial + missing:
        print(
            f"{r['board']:<60} "
            f"best={r['best_round']:<8} "
            f"placed={r['placed']}/{r['expected']} "
            f"failure={r['failure']}"
        )
PY
