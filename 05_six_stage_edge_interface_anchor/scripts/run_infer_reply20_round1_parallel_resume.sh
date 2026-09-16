#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:?Usage: $0 GPU_ID WORKER_NAME}"
WORKER_NAME="${2:-gpu${GPU_ID}}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="runs/checkpoints/train-floorplan-mixed-b3_3-reply20.pt"

INPUT_DIR="data/pcb_benchmark_infer_json"
if [[ ! -d "$INPUT_DIR" ]]; then
    INPUT_DIR="pcb_benchmark_infer_json"
fi

RUN_ROOT="runs/benchmark_infer_b3_3_reply20_best2"
OUT_DIR="$RUN_ROOT/round1_greedy/infer_out"
LOG_DIR="$RUN_ROOT/round1_greedy/infer_logs"

STATE_DIR="$RUN_ROOT/round1_parallel_state"
CLAIM_DIR="$STATE_DIR/claims"
ATTEMPT_DIR="$STATE_DIR/attempted"

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] checkpoint ²»´æÔÚ: $CKPT"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] ÊäÈëÄ¿Â¼²»´æÔÚ: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CLAIM_DIR" "$ATTEMPT_DIR"

mapfile -d '' FILES < <(
    find "$INPUT_DIR" -maxdepth 1 -type f \
        -name '*.json' ! -name '*.audit.json' \
        -print0 | sort -z
)

TOTAL="${#FILES[@]}"

valid_output() {
    local output_path="$1"

    python - "$output_path" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)

try:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    raise SystemExit(1)

valid = (
    isinstance(data, dict)
    and "complete" in data
    and (
        "placed_count" in data
        or "placed" in data
        or "placements" in data
    )
)

raise SystemExit(0 if valid else 1)
PY
}

echo "========================================"
echo "[WORKER]       $WORKER_NAME"
echo "[PHYSICAL GPU] $GPU_ID"
echo "[VISIBLE GPU]  cuda:0"
echo "[CHECKPOINT]   $CKPT"
echo "[INPUT]        $INPUT_DIR"
echo "[OUTPUT]       $OUT_DIR"
echo "[TOTAL]        $TOTAL"
echo "========================================"

processed=0
skipped=0
failed=0

while true; do
    claimed=0

    for file in "${FILES[@]}"; do
        stem="$(basename "${file%.json}")"
        output_file="$OUT_DIR/$stem.infer.json"
        log_file="$LOG_DIR/$stem.log"
        claim_path="$CLAIM_DIR/$stem.lock"
        attempted_path="$ATTEMPT_DIR/$stem"

        # ÒÑ¾­ÓÐºÏ·¨½á¹û£¬°üÀ¨ complete=False µÄÕý³£ÍÆÀí½á¹û¡£
        if valid_output "$output_file"; then
            continue
        fi

        # ±¾´Î²¢ÐÐÔËÐÐÖÐÒÑ¾­³¢ÊÔ¹ý£¬±ÜÃâ´íÎóÈÎÎñÎÞÏÞÖØÊÔ¡£
        if [[ -e "$attempted_path" ]]; then
            continue
        fi

        # mkdir ÊÇÔ­×Ó²Ù×÷£¬±£Ö¤Í¬Ò»¿é°åÖ»±»Ò»ÕÅ¿¨ÁìÈ¡¡£
        if ! mkdir "$claim_path" 2>/dev/null; then
            continue
        fi

        # »ñµÃËøºóÔÙ´Î¼ì²é£¬·ÀÖ¹¼ì²éÓë¼ÓËøÖ®¼äÁíÒ»½ø³ÌÒÑÍê³É¡£
        if valid_output "$output_file" || [[ -e "$attempted_path" ]]; then
            rmdir "$claim_path" 2>/dev/null || true
            continue
        fi

        claimed=1
        rm -f "$output_file"

        echo
        echo "[$WORKER_NAME][RUN] $stem"

        if python -u scripts/step4_infer.py \
            --test_glob "$file" \
            --ckpt "$CKPT" \
            --out_dir "$OUT_DIR" \
            --device cuda \
            --sequence_policy checkpoint \
            --layout_preset checkpoint \
            --beam_width 1 \
            --beam_topk 16 \
            > "$log_file" 2>&1
        then
            processed=$((processed + 1))
            echo "[$WORKER_NAME][OK] $stem"
            tail -n 1 "$log_file" || true
        else
            failed=$((failed + 1))
            echo "[$WORKER_NAME][FAIL] $stem  log=$log_file"
            tail -n 20 "$log_file" || true
        fi

        touch "$attempted_path"
        rmdir "$claim_path" 2>/dev/null || true

        # Ã¿´ÎÖ»ÁìÈ¡Ò»¿é£¬Íê³ÉºóÖØÐÂÉ¨Ãè¹²Ïí¶ÓÁÐ¡£
        break
    done

    if [[ "$claimed" -eq 0 ]]; then
        break
    fi
done

for file in "${FILES[@]}"; do
    stem="$(basename "${file%.json}")"
    if valid_output "$OUT_DIR/$stem.infer.json"; then
        skipped=$((skipped + 1))
    fi
done

echo
echo "========================================"
echo "[$WORKER_NAME SUMMARY]"
echo "newly_processed=$processed"
echo "failed=$failed"
echo "currently_valid_outputs=$skipped/$TOTAL"
echo "========================================"
