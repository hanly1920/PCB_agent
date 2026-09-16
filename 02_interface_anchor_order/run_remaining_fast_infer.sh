#!/usr/bin/env bash
set -euo pipefail

cd /u3disk/fuyq24/pcb_autoplace1

# 只补缺失输出；不会覆盖已有 .infer.json。
# 若想强制重新跑剩下/全部，加 --overwrite。

PYTHONPATH=. python scripts/step4_infer_remaining_fast_cached.py \
  --test_glob "data/benchmark_infer/*.json" \
  --ckpt runs/full_cuda_old_20260419_141236/last.ckpt \
  --out_dir runs/infer_full_cuda_old_20260419_141236 \
  --device cuda

PYTHONPATH=. python scripts/step4_infer_remaining_fast_cached.py \
  --test_glob "data/benchmark_infer/*.json" \
  --ckpt runs/full_cuda_newdata/last.ckpt \
  --out_dir runs/infer_full_cuda_newdata \
  --device cuda

# 如果只想跑排序后的第 50-74 个，也可以在上面两条命令后面加：
#   --start_index 50 --end_index 74
#
# 如果优先追求速度、可接受跳过 CPU 后处理，加：
#   --no_postprocess
