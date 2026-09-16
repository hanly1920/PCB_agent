# Infer Speed Patch B4.1

## Main changes

1. Added process-level inference model cache in `pcbplace/infer.py`.

   `scripts/step4_infer.py` calls `infer_layout()` once per board. Previously each call reloaded the checkpoint, rebuilt the model, and moved weights to CUDA. Now the model bundle is cached by `(checkpoint path, checkpoint mtime, device, strict_state_dict)`.

2. Reused CUDA grid cache for `edge_hard` candidate masking.

   `_apply_edge_hard_candidate_mask()` no longer allocates fresh `arange`/`meshgrid` tensors every placement step. It reuses `_cached_grid_maps_cuda()`.

## Recommended fast infer command

```bash
python -u scripts/step4_infer.py \
  --test_glob "data/test/*.json" \
  --ckpt runs/checkpoints/train-floorplan-mixed-b4-edgehard-groupalign_final.pt \
  --out_dir runs/infer_b4_fast \
  --device cuda \
  --beam_width 1 \
  --max_tokens 96 \
  --no_metrics \
  --no_postprocess
```

For final quality evaluation, re-enable metrics and postprocess after validating speed.
