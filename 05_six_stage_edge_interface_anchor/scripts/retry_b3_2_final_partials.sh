#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="runs/checkpoints/train-floorplan-mixed-b3_2-spread-edge-align-bs8_final.pt"
IN="runs/testinfer_b3_2_final_bench/infer_input_clean"
RUN="runs/testinfer_b3_2_final_bench"

PARTIALS=(
  "5__kitspace_Driverino-Shield"
  "AnalogThermometer"
  "IR-Transponder-ATTiny85-v2_Transponder_v2"
  "ISO-port_ch340-usb-serial-isolated"
  "cvbs-mipi-bridge"
  "esp8266_wi07_3_adapter_esp"
)

test -f "$CKPT" || { echo "[ERR] missing checkpoint: $CKPT"; exit 1; }

for attempt in 01 02 03; do
  OUT="$RUN/infer_out_retry${attempt}"
  LOG="$RUN/infer_logs_retry${attempt}"
  mkdir -p "$OUT" "$LOG"

  echo "[INFO] retry attempt=$attempt"
  echo "[INFO] output=$OUT"

  for base in "${PARTIALS[@]}"; do
    f="$IN/$base.json"
    test -f "$f" || { echo "[ERR] missing input: $f"; exit 1; }

    echo "[RUN][retry$attempt] $base"

    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" timeout 30m \
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
      echo "[TIMEOUT][retry$attempt] $base"
    elif [ "$code" -ne 0 ]; then
      echo "[FAIL][retry$attempt] $base code=$code, see $LOG/$base.log"
    else
      echo "[OK][retry$attempt] $base"
    fi
  done
done

python - <<'PY'
import csv, json
from pathlib import Path

run = Path("runs/testinfer_b3_2_final_bench")
boards = [
    "5__kitspace_Driverino-Shield",
    "AnalogThermometer",
    "IR-Transponder-ATTiny85-v2_Transponder_v2",
    "ISO-port_ch340-usb-serial-isolated",
    "cvbs-mipi-bridge",
    "esp8266_wi07_3_adapter_esp",
]

dirs = [
    ("base", run / "infer_out_timeout30m"),
    ("retry01", run / "infer_out_retry01"),
    ("retry02", run / "infer_out_retry02"),
    ("retry03", run / "infer_out_retry03"),
]

def find_json(out, stem):
    candidates = [
        out / f"{stem}.infer.json",
        out / f"{stem}.json",
        out / stem / "infer.json",
        out / stem / f"{stem}.infer.json",
    ]
    return next((p for p in candidates if p.exists()), None)

rows = []
for board in boards:
    best = None
    for tag, out in dirs:
        p = find_json(out, board)
        row = {
            "board": board,
            "run": tag,
            "complete": "",
            "placed": -1,
            "expected": "",
            "failure": "",
            "json_path": str(p) if p else "",
        }
        if p:
            d = json.loads(p.read_text(encoding="utf-8"))
            row.update({
                "complete": d.get("complete"),
                "placed": int(d.get("placed_count") or 0),
                "expected": d.get("expected_count"),
                "failure": d.get("failure_reason"),
            })
        rows.append(row)

        if best is None:
            best = row
        else:
            cur_complete = bool(row["complete"])
            best_complete = bool(best["complete"])
            if (cur_complete and not best_complete) or (
                cur_complete == best_complete and int(row["placed"]) > int(best["placed"])
            ):
                best = row

    print(
        f"[BEST] {board:<45} "
        f"run={best['run']:<7} complete={best['complete']} "
        f"placed={best['placed']}/{best['expected']} "
        f"path={best['json_path']}"
    )

summary = run / "infer_retry_compare_summary.csv"
fields = ["board", "run", "complete", "placed", "expected", "failure", "json_path"]
with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("[SUMMARY]", summary)
PY
