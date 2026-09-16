# Prefix / Mixed Curriculum Patch

This patched project adds the connector/front-k training changes discussed in chat.

## Main changes

1. `pcbplace/train.py`
   - Added `curriculum_window_mode` with modes:
     - `suffix`: old behavior, trains the final k actions.
     - `prefix`: trains the first k actions from an empty board.
     - `random`: trains a random k-step window.
     - `mixed`: 50% prefix, 25% suffix, 25% random.
   - Added `window_starts` / `window_steps` to `rollout_suffix_loss_batched_boards()`.
   - Added `_BoardRuntimeCache.make_env_at_prefix()` for arbitrary prefix snapshots.
   - Checkpoints now save/restore `curriculum_window_mode`.

2. `scripts/step3_train_masked.py`
   - Added CLI argument:
     `--curriculum_window_mode {suffix,prefix,random,mixed}`

3. `pcbplace/module_region_proposer.py`
   - Unknown-side interface modules no longer receive a center Gaussian prior.
   - Their active bbox becomes full board, and heatmap becomes a four-edge mixture.

4. `pcbplace/module_partition.py`
   - Added soft interface anchors so J/P/JP/CN/CON/RJ/X style connectors can enter phase 0 ordering
     without becoming hard-boundary legal constraints.

## Suggested run

Prefix repair:

```bash
python scripts/step3_train_masked.py \
  --train_glob "data/train/*.json" \
  --checkpoint_path runs/checkpoints/train-floorplan-scratch-bs8_resume_latest.pt \
  --resume \
  --save_path runs/checkpoints/train-floorplan-prefix-repair_latest.pt \
  --batch_size 8 \
  --adaptive_k \
  --fixed_k_steps 750 \
  --curriculum_window_mode prefix \
  --expert_mix_start 1.0 \
  --expert_mix_end 1.0 \
  --expert_mix_anneal_steps 20000 \
  --max_main_steps 8000 \
  --no_replay_finetune
```

Mixed follow-up:

```bash
python scripts/step3_train_masked.py \
  --train_glob "data/train/*.json" \
  --checkpoint_path runs/checkpoints/train-floorplan-prefix-repair_latest.pt \
  --resume \
  --save_path runs/checkpoints/train-floorplan-mixed_latest.pt \
  --batch_size 8 \
  --adaptive_k \
  --fixed_k_steps 750 \
  --curriculum_window_mode mixed \
  --expert_mix_start 1.0 \
  --expert_mix_end 0.7 \
  --expert_mix_anneal_steps 30000 \
  --max_main_steps 15000 \
  --no_replay_finetune
```
