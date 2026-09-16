# pcb_autoplace_clean (Scheme A)

A clean, end-to-end pipeline for:
1) KiCad `.kicad_pcb` -> training-ready JSON
2) Generate ONE sequence source of truth (`graph.sequence`) via heuristic (no GNN)
3) Explicit suffix-masked autoregressive training (k grows to full mask)
4) Inference on test set with hard-constraint action masks, grid alignment, and HPWL+NSLW objective (NSLW weight = 0.2)

## Folder layout
- `scripts/step1_kicad_to_json.py` : parse KiCad into minimal task JSON
- `scripts/step2_generate_sequence.py` : generate `graph.sequence`
- `scripts/step3_train_masked.py` : train masked policy (requires expert placements on training set)
- `scripts/step4_infer.py` : inference on test set
- `pcbplace/` : core modules (parser, heuristic, env, model, train, infer)

## JSON requirements
For training set JSON, each component MUST include:
```json
"expert": {"xy_mm":[x,y], "rot":0}
```
and task MUST include:
```json
"graph": {"sequence":[...]}
```

## Notes
- This is a clean baseline: no keepouts/polygons/complex footprints. Add them incrementally.
- Hard constraints:
  - inside board bbox
  - no overlap + `min_spacing_mm`
  - interface-on-boundary (type==interface or allowed_sides non-empty)
- Alignment/clean placement:
  - discrete grid placement + alignment bias to existing centers
- Objective:
  - minimize `HPWL + 0.2 * NSLW_proxy`
  - reward is improvement in that objective during rollout

## Quick start
```bash
python scripts/step1_kicad_to_json.py --kicad_pcb demo.kicad_pcb --out_json data/task0.json --grid_mm 1.0
python scripts/step2_generate_sequence.py --in_json data/task0.json --out_json data/task0.seq.json --bfs_depth 3
# add expert placements to training jsons before step3
python scripts/step3_train_masked.py --train_glob "data/train/*.seq.json" --save_path model.pt
python scripts/step4_infer.py --test_glob "data/test/*.seq.json" --ckpt model.pt --out_dir data/infer_out
```


## Training schedule update

- Default `expert_mix_end` is now `0.40`.
- `L_teacher` and `L_metric` remain enabled in both suffix training and replay fine-tuning.


## Independent semantic validation metrics

Evaluate either embedded expert layouts or model predictions:

```bash
python scripts/eval_semantic_metrics.py --task_glob "data/seq/*.json"
python scripts/eval_semantic_metrics.py --task_glob "data/seq_test/*.json" --pred_dir runs/infer --out runs/semantic_metrics.json
```

Reported summary metrics:
- edge_hit_rate_mean
- core_center_rate_mean
- critical_connection_avg_distance_mean
- align_group_avg_deviation_mean
- functional_group_compactness_mean
- hpwl_mean


## Human / semi-human semantic label curation

To move from pure rule-generated labels toward curated labels, use:

```bash
python scripts/bootstrap_semantic_review.py --root . --out_dir reviews
python scripts/apply_semantic_review.py --root . --review_csv reviews/semantic_review_all.csv
```

The editable review sheets live under `reviews/`.


## Review-aware semantic training

The training loop now consumes `semantic_review.review_status` and `semantic_review.auto_confidence`:

- `approved` / `edited` semantic labels are treated as high-trust supervision.
- `seeded` labels are weighted by their `auto_confidence`.
- `needs_review=true` reduces semantic supervision weight further.

This trust weight is used in:
- teacher region prior
- effective expert-vs-teacher balance
- metric loss scaling
- semantic auxiliary losses
