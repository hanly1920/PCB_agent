# Fixed Components / KiCad Locked Footprints / PCBAgent Integration

## What changed

- `Component` now has runtime fixed fields: `fixed`, `fixed_xy`, `fixed_rot`, `fixed_source`.
- `task_from_json()` accepts fixed placements from:
  - `fixed: true` + `fixed_xy_mm` / `fixed_position_mm` / `locked_xy_mm`
  - `fixed: {xy_mm: [...], rot: ...}` or compact `fixed: [x, y, rot]`
  - `placement.xy_mm`, `layout.xy_mm`, or `expert.xy_mm` only when the component is explicitly fixed.
- `PlacementEnv.reset()` pre-places fixed components, inserts their bboxes into `occupied`, and removes them from the rollout `sequence`.
- CUDA mask/objective/context paths automatically see fixed components through `env.occupied`, `env.placed`, and `env.placed_order`.
- Inference completion accounting now separates:
  - `expected_count` / `placed_count`: all components, including fixed
  - `dynamic_expected_count` / `dynamic_placed_count`: model-placed components only
  - `fixed_count` / `fixed_refs`: fixed components
- Inference postprocess treats fixed components as frozen anchors and will not move or rotate them.
- Training/expert snapping uses `env.sequence`, so fixed components are skipped during teacher action generation and curriculum replay.
- KiCad `(locked)` footprints are parsed and exported as fixed components with `fixed_xy_mm`, `fixed_rot`, and `fixed_source: kicad_locked`.
- The PCBAgent package is included and its placement tool now applies DSL `locked_refs` / `lock|fix|freeze` constraints to a generated `.agent_fixed.json` task before calling `pcbplace.infer_layout()`.

## Interface notes

A fixed component must have a coordinate source. Valid examples:

```json
{"ref": "U1", "fixed": true, "fixed_xy_mm": [12.5, 8.0], "fixed_rot": 90}
```

```json
{"ref": "U1", "fixed": {"xy_mm": [12.5, 8.0], "rot": 90}}
```

For DSL locks via PCBAgent, the referenced component must already have `fixed_xy_mm`, `placement.xy_mm`, `layout.xy_mm`, or `expert.xy_mm`; otherwise the tool fails early with a clear message rather than silently moving the component.

## Validation run

```bash
PYTHONPATH=. pytest -q \
  tests/test_fixed_components.py \
  tests/test_expert_leakage_guard.py \
  tests/test_inference_action_scoring_metadata.py \
  tests_agent/test_orchestrator_smoke.py \
  tests_agent/test_dsl_compiler.py \
  tests_agent/test_kicad_roundtrip.py
```

Result: `12 passed`.
