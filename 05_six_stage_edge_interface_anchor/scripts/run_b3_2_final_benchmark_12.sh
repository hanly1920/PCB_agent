#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="runs/checkpoints/train-floorplan-mixed-b3_2-spread-edge-align-bs8_final.pt"
SRC="data/benchmark_infer"

RUN="runs/testinfer_b3_2_final_bench"
IN="$RUN/infer_input_clean"
OUT="$RUN/infer_out_timeout30m"
LOG="$RUN/infer_logs_timeout30m"
SUMMARY="$RUN/infer_timeout30m_summary.csv"

mkdir -p "$IN" "$OUT" "$LOG"

test -f "$CKPT" || { echo "[ERR] missing checkpoint: $CKPT"; exit 1; }
test -d "$SRC" || { echo "[ERR] missing benchmark dir: $SRC"; exit 1; }

python -m py_compile pcbplace/infer.py scripts/step4_infer.py

rm -f "$IN"/*.json

for name in \
  KiCad-Like-a-Pro-Tutorial_proj2-sevensegment.json \
  AnalogThermometer.json \
  9__RF_SIGNAL_GENERATOR_HARDWARE_RF_Signal_Generator.json \
  Usb_dac_dac.json \
  cvbs-mipi-bridge.json \
  1__analog_esr_meter_esr_meter_rev_a.json \
  5__kitspace_Driverino-Shield.json \
  6__kitspace_training_board_v02_-_.json \
  BreadboardBasics_RevA.json \
  ISO-port_ch340-usb-serial-isolated.json \
  esp8266_wi07_3_adapter_esp.json \
  IR-Transponder-ATTiny85-v2_Transponder_v2.json
do
  test -f "$SRC/$name" || { echo "[ERR] missing benchmark json: $SRC/$name"; exit 1; }
  cp -f "$SRC/$name" "$IN/$name"
done

echo "[INFO] checkpoint: $CKPT"
echo "[INFO] source:     $SRC"
echo "[INFO] input:      $IN"
echo "[INFO] output:     $OUT"

for f in "$IN"/*.json; do
  base="$(basename "$f" .json)"
  echo "[RUN] $base"

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
    echo "[TIMEOUT] $base"
  elif [ "$code" -ne 0 ]; then
    echo "[FAIL] $base code=$code, see $LOG/$base.log"
  else
    echo "[OK] $base"
  fi
done

python - <<'PY'
import csv, json
from pathlib import Path

run = Path("runs/testinfer_b3_2_final_bench")
inp = run / "infer_input_clean"
out = run / "infer_out_timeout30m"
log = run / "infer_logs_timeout30m"
summary = run / "infer_timeout30m_summary.csv"

rows = []
for task in sorted(inp.glob("*.json")):
    stem = task.stem

    candidates = [
        out / f"{stem}.infer.json",
        out / f"{stem}.json",
        out / stem / "infer.json",
        out / stem / f"{stem}.infer.json",
    ]
    pred = next((p for p in candidates if p.exists()), None)

    logp = log / f"{stem}.log"

    row = {
        "board": stem,
        "has_json": bool(pred),
        "json_path": str(pred) if pred else "",
        "complete": "",
        "placed": "",
        "expected": "",
        "failure": "",
        "objective": "",
        "objective_partial": "",
        "hpwl_ratio": "",
        "module_centroid_error": "",
        "module_bbox_iou_with_original": "",
        "connector_side_accuracy": "",
        "same_side_order_accuracy": "",
        "pitch_cv": "",
        "post_changed": "",
        "status": "missing_json",
    }

    if pred:
        d = json.loads(pred.read_text(encoding="utf-8"))
        m = d.get("metrics") or {}
        row.update({
            "complete": d.get("complete"),
            "placed": d.get("placed_count"),
            "expected": d.get("expected_count"),
            "failure": d.get("failure_reason"),
            "objective": d.get("objective"),
            "objective_partial": d.get("objective_partial"),
            "hpwl_ratio": m.get("hpwl_ratio"),
            "module_centroid_error": m.get("module_centroid_error"),
            "module_bbox_iou_with_original": m.get("module_bbox_iou_with_original"),
            "connector_side_accuracy": m.get("connector_side_accuracy"),
            "same_side_order_accuracy": m.get("same_side_order_accuracy"),
            "pitch_cv": m.get("pitch_cv"),
            "post_changed": d.get("postprocess_changed_count"),
            "status": "ok" if d.get("complete") else "partial",
        })
    elif logp.exists():
        txt = logp.read_text(errors="ignore").lower()
        if "timeout" in txt or "timed out" in txt:
            row["status"] = "timeout"
        elif "traceback" in txt or "error" in txt or "exception" in txt:
            row["status"] = "error"

    rows.append(row)

fields = list(rows[0].keys()) if rows else []
with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"[SUMMARY] {summary}")
for r in rows:
    print(
        f"{r['board']:<55} "
        f"status={r['status']:<12} "
        f"complete={r['complete']} "
        f"placed={r['placed']}/{r['expected']} "
        f"hpwl_ratio={r['hpwl_ratio']}"
    )
PY
