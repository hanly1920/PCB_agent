# Module Floorplan V4 interface check

This version keeps the V3 module-level floorplan objective and fixes inference CLI parity.

## Interface status

- Train: `scripts/step3_train_masked.py` exposes `--objective_module_floorplan_weight` and all `--module_floorplan_*` knobs, passes them into `train_cuda`, stores/restores them via checkpoint env config.
- Replay: replay uses the same `env_kwargs` / `PlacementEnv` objective configuration as main training, so module-floorplan terms are active during rollout scoring and replay fine-tune when the weight is non-zero.
- Infer: `scripts/step4_infer.py` now exposes direct overrides for module-region, module-floorplan, density/spacing scales, and the `routeaware` preset.

## Recommended inference preset

Use `--layout_preset routeaware` to match the route-aware/module-floorplan objective at inference time.
