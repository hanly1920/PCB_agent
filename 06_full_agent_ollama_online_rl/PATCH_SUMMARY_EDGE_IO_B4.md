# Edge IO B4 Patch Summary

This patch implements the edge interface semantics requested for PCB auto-placement while preserving checkpoint compatibility.

## Implemented

1. **Train JSON interface classification**
   - Adds `external_io_role`: `edge_hard`, `edge_soft`, `internal_connector`, or `none`.
   - Uses expert/manual component center and bbox proximity to the board edge.
   - Uses per-component edge band:
     `edge_band_mm = max(2.5, 0.5 * max(component_w, component_h) + 1.0)`.
   - Internal headers, jumpers, and test points are kept free/internal when not genuinely edge-near.

2. **Role-aware training/inference semantics**
   - Adds `external_io_role` to the dataset loader and `Component` dataclass.
   - Applies role-specific `semantic_strength`, `constraint_level`, and `placement_role`.
   - Train mode keeps `edge_hard` as `constraint_level=soft`.
   - Infer-safe annotation marks hard external footprints as `edge_hard`; side-specific hard masking is only applied when `side_preference` is known.

3. **Grouped alignment / pitch**
   - Alignment and pitch groups are now generated as:
     `module_id :: footprint_bucket :: external_io_role :: side_preference`.
   - `left/right` edge groups align on `x` and pitch/order along `y`.
   - `top/bottom` edge groups align on `y` and pitch/order along `x`.
   - Old broad/global connector alignment is cleared for internal connectors and singletons.

4. **Inference edge-hard candidate mask**
   - `pcbplace/infer.py` now applies an approximate hard mask for `edge_hard + constraint_level=hard`.
   - Candidates are first restricted to the `side_preference` edge band when known.
   - If side is unknown, any board edge band is used as the first-pass mechanical prior.
   - If no legal candidate exists in the band, inference falls back to all legal candidates.

5. **CUDA / CPU consistency**
   - CUDA alignment delta and bias now respect `align_axis` instead of rewarding arbitrary global x/y alignment.
   - CUDA edge-band feature uses the per-component band where available.

## Modified files

- `pcbplace/semantic_labels.py`
- `pcbplace/dataset.py`
- `pcbplace/env.py`
- `pcbplace/env_cuda.py`
- `pcbplace/infer.py`
- `PATCH_SUMMARY_EDGE_IO_B4.md`

## Smoke checks run

```bash
python -m py_compile pcbplace/semantic_labels.py pcbplace/dataset.py pcbplace/env.py pcbplace/env_cuda.py pcbplace/infer.py
PYTHONPATH=. pytest -q tests/test_inference_action_scoring_metadata.py tests/test_mounting_hole_filter.py tests/test_cuda_objective_shape_guard.py
```

Result: selected tests passed.
