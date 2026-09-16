#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH=. python scripts/step4_infer_remaining_fast_cached.py \
  --test_glob "data/pcb_benchmark_infer_json/*.json" \
  --ckpt runs/full_cuda_old_20260419_141236/last.ckpt \
  --out_dir runs/infer_pcb_benchmark_full_cuda_old_20260419_141236 \
  --device cuda

PYTHONPATH=. python scripts/step4_infer_remaining_fast_cached.py \
  --test_glob "data/pcb_benchmark_infer_json/*.json" \
  --ckpt runs/full_cuda_newdata/last.ckpt \
  --out_dir runs/infer_pcb_benchmark_full_cuda_newdata \
  --device cuda
