## Step 11.1 (patch)

- checkpoint `env_config` now persists `nslw_weight`, `region_weight`, and `conn_weight`, and inference restores `nslw_weight` from the checkpoint.
- removed `data/seq_k15_20_train/expert311.json` from the training split.
- orientation consistency now uses absolute rotations (`0/90/180/270`) instead of only axis-equivalent classes (`0~180`, `90~270`). Side-specific fallback targets are left=90, right=270, top=0, bottom=180.

# Modification Summary

This package was updated to follow the provided objective-design document.

## Main changes

1. Unified placement objective in `pcbplace/env.py`
   - Replaced scattered reward shaping with a unified objective composed of:
     - HPWL / NSLW
     - region semantic term
     - connectivity proximity term
     - alignment term
     - grouping compactness term
   - Added `objective_delta_mask()` so each candidate action can be scored by unified `Delta J`.
   - Reward remains `J(P_{t-1}) - J(P_t)`.

2. Teacher and loss redesign in `pcbplace/train.py`
   - Teacher distribution now uses unified `Delta J(a)` instead of old wire+bias energy.
   - Added metric loss `E_pi[Delta J]` weighted by `teacher.metric_weight`.
   - Replay fine-tuning uses the same unified target.

3. Inference alignment in `pcbplace/infer.py`
   - Rollout gating now uses unified `Delta J`.
   - Backward-compatible checkpoint loading retained where possible.

4. CLI updates in `scripts/step3_train_masked.py`
   - Added new arguments for unified objective weights and metric loss weight.
   - Replaced old teacher wire/bias threshold args with objective-based args.

## Default objective weights

- HPWL: 1.0
- Region: 0.9
- Connectivity: 0.7
- Alignment: 0.25
- Group: 0.20

These defaults were chosen to match the target document's guidance.

- 2026-04 objective sync fix: teacher objective delta now matches env reward objective exactly, keeping NSLW in both; removed stale --teacher_lambda_wire/--teacher_lambda_bias shell args.


## Step 2: semantic auxiliary labels
- Replaced the old auxiliary supervision based on expert heatmaps and edge/mid/core zones.
- Auxiliary heads now predict:
  - `semantic_class` (for example `core`, `power`, `interface`, `passive`, `rf`)
  - `region_type` (one of `edge_top`, `edge_bottom`, `edge_left`, `edge_right`, `core`, `free`)
- The learned soft prior used during teacher guidance and rollout gating now comes from the predicted `region_type` logits instead of the old zone head.
- Checkpoint metadata now stores `region_type_names` and `semantic_class_names`.


## Step 3 - fully semantic-driven training objective

- `pcbplace/env.py`
  - Objective terms now read semantic labels directly from each component:
    - `region_type` / `side_preference` for region loss
    - `critical_neighbors` / `critical_nets` for connection loss
    - `align_group` for alignment loss
    - `functional_group` for group compactness loss
  - Removed the old objective dependence on inferred type-prefix grouping for `region / conn / align / group`.
  - Action bias is now semantic-region + semantic-alignment driven instead of old edge-preferred heuristics.
- `pcbplace/dataset.py`
  - Task loading now carries semantic labels into the runtime `Component` objects so env / reward / teacher use the dataset semantics directly.


## Step 4 - NSLW retained at lower weight

- Kept NSLW in the unified semantic objective, but lowered its default weight from `0.3` to `0.2`.
- Added `--objective_nslw_weight` to `scripts/step3_train_masked.py` and threaded it into `pcbplace/train.py`.
- Updated provided run scripts to pass `--objective_nslw_weight 0.2` explicitly.


## Step 5 - Lower late expert mix

- Lowered the default late-stage expert imitation ratio from the old higher values to `expert_mix_end = 0.40`.
- Updated `pcbplace/train.py` and `scripts/step3_train_masked.py` defaults accordingly.
- Updated provided training shell scripts to pass `--expert_mix_end 0.40` explicitly.
- Kept `L_teacher` and `L_metric` active in both masked suffix training and replay fine-tuning.


## Step 6 - Independent validation metrics

Added independent semantic evaluation for:
- edge hit rate
- core center rate
- critical connection average distance
- align group average deviation
- functional group compactness
- HPWL

New files:
- `pcbplace/semantic_metrics.py`
- `scripts/eval_semantic_metrics.py`

Usage examples:
- `python scripts/eval_semantic_metrics.py --task_glob "data/seq/*.json"`
- `python scripts/eval_semantic_metrics.py --task_glob "data/seq_test/*.json" --pred_dir runs/infer --out runs/semantic_metrics.json`


## Step 7 - Reviewable / semi-manual semantic labels

- Upgraded semantic labels from pure rule-seeded fields to a reviewable workflow.
- Added per-component `semantic_review` metadata with confidence, evidence, candidate labels, review priority, and manual override slots.
- Added CSV export and apply-back scripts for human / semi-human review.
- Generated `reviews/semantic_review_all.csv` and `reviews/semantic_review_queue.csv`.


## Step 9 - Review-aware training

Training now uses `semantic_review.review_status` and `semantic_review.auto_confidence` directly:

- `load_region_targets_for_task()` exports `review_status`, `auto_confidence`, `needs_review`, and a derived `review_weight`.
- Main suffix training scales semantic trust per component:
  - reviewed labels (`approved` / `edited`) keep full semantic weight
  - seeded labels use a softer weight derived from `auto_confidence`
  - `needs_review=true` downweights semantic supervision further
- `review_weight` is applied to:
  - teacher region prior contribution
  - `L_teacher` / `L_metric` trust balance (via effective expert mix + metric scaling)
  - semantic auxiliary losses (`semantic_class`, `region_type`)
- Replay fine-tuning also uses the same `review_weight`, so on-policy updates do not overfit uncertain semantic labels.


## Step 10 - layout-aware semantic control

Added first-batch layout-quality controls focused on semantic structure and post-process polishing:
- `anchor_ref` + `subzone` fields for anchor-centric grouping around major components.
- `same_side_group` + `boundary_order` fields for edge-device same-side and ordered strip constraints.
- objective extensions in `pcbplace/env.py` for anchor/subzone and boundary-order penalties.
- lightweight inference post-process in `pcbplace/infer.py` for edge-band snap, local alignment, gentle anchor nudging, and orientation cleanup.
- review/export/apply scripts now support the new manual columns.


## Step11: uniform pitch + orientation consistency
- Added `objective_pitch_weight` and `objective_orientation_weight` into the unified objective.
- Added pitch regularization for `same_side_group` / `align_group` members to reduce uneven spacing.
- Added orientation consistency regularization and post-process orientation unification.
- Extended post-process with group-level uniform pitch smoothing.
