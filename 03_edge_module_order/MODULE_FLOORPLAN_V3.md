# Module-level Floorplan Objective V3

This patch adds an opt-in module-level floorplan objective for PCB placement training and inference.

## What changed

New objective term:

```text
module_floorplan
```

It is different from the existing `module_region` term:

- `module_region` scores each candidate component against its module prior region / shape hint.
- `module_floorplan` builds the **partial bbox of the whole module** and scores module-level relationships.

The new term penalizes:

1. Cross-module bbox overlap.
2. Missing soft routing channel/gap between unrelated modules.
3. Excessive module bbox growth / scattered module members.
4. Whole-module bbox leakage outside prior module region.

It softens separation pressure for strongly connected or critical-neighbor cross-module pairs, so support/decoupling modules are not pushed too far away.

## New CLI parameters

```bash
--objective_module_floorplan_weight 0.36 \
--module_floorplan_separation_mm 2.5 \
--module_floorplan_overlap_scale 1.10 \
--module_floorplan_channel_scale 0.80 \
--module_floorplan_compact_scale 0.24 \
--module_floorplan_region_scale 0.70
```

Defaults keep the behavior backward-compatible:

```bash
--objective_module_floorplan_weight 0.0
```

## Recommended route-aware values

```bash
--objective_module_region_weight 0.38 \
--objective_module_floorplan_weight 0.36 \
--module_floorplan_separation_mm 2.5 \
--module_floorplan_overlap_scale 1.10 \
--module_floorplan_channel_scale 0.80 \
--module_floorplan_compact_scale 0.24 \
--module_floorplan_region_scale 0.70 \
--objective_boundary_group_weight 0.38 \
--objective_conn_weight 0.65 \
--objective_anchor_weight 0.30 \
--objective_edge_clearance_weight 0.60
```

## Files touched

- `pcbplace/env.py`
- `pcbplace/env_cuda.py`
- `pcbplace/train.py`
- `pcbplace/infer.py`
- `scripts/step3_train_masked.py`

## Smoke checks run

```bash
python -m py_compile pcbplace/env.py pcbplace/env_cuda.py pcbplace/train.py pcbplace/infer.py scripts/step3_train_masked.py
PYTHONPATH=. pytest -q tests/test_module_refinement_v2.py tests/test_inference_action_scoring_metadata.py
PYTHONPATH=. pytest -q tests/test_cuda_objective_shape_guard.py
```

