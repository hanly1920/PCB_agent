# Semantic Label Annotation and Review Workflow

This package contains dataset JSON files enriched with semantic labels for each component.

Added per-component fields:

- `semantic_class`: high-level semantic role such as `interface`, `core`, `power`, `passive`, `ui`, `mechanical`, `rf`
- `region_type`: preferred target region such as `edge_top`, `edge_bottom`, `edge_left`, `edge_right`, `core`, `free`
- `functional_group`: functional grouping such as `power_supply`, `control_processing`, `interface_io`, `user_interface`, `timing`, `mechanics`
- `side_preference`: nearest or intended board side for edge-oriented parts
- `align_group`: coarse alignment group identifier for repeated parts that are likely to be arranged in rows or columns
- `critical_nets`: top weighted nets connected to the component
- `critical_neighbors`: top neighboring components derived from weighted shared nets
- `semantic`: nested copy of the main semantic fields plus `label_source`

Added board-level metadata:

- `meta.semantic_annotation`: version, description, field list, label distribution summary, and top critical net weights

Generation method:

- deterministic heuristic annotation based on component type, reference designator, connected nets, and expert placement side
- no training logic or loss functions were changed in this step

Entry point used to regenerate labels:

- `scripts/annotate_semantic_labels.py`


## Step 2 training note
The auxiliary supervision has been upgraded from geometric weak labels to semantic labels:
- `semantic_class` is used as the component-class auxiliary target.
- `region_type` is used as the semantic placement-region auxiliary target.
The previous heatmap and edge/mid/core zone supervision are no longer the main auxiliary targets.


## Human / semi-human review upgrade

This package now upgrades the previous auto labels into a reviewable workflow. Each component includes a `semantic_review` block with confidence, evidence, candidate labels, and review status.

New review assets:

- `reviews/semantic_review_all.csv`
- `reviews/semantic_review_queue.csv`
- `scripts/export_semantic_review.py`
- `scripts/apply_semantic_review.py`
- `SEMANTIC_REVIEW_WORKFLOW.md`

The intended flow is:

1. inspect the queue CSV
2. edit the `manual_*` columns
3. apply edits back into JSON
4. retrain using corrected top-level semantic fields


## Step 10 - layout-aware semantic control

Added first-batch layout-quality controls focused on semantic structure and post-process polishing:
- `anchor_ref` + `subzone` fields for anchor-centric grouping around major components.
- `same_side_group` + `boundary_order` fields for edge-device same-side and ordered strip constraints.
- objective extensions in `pcbplace/env.py` for anchor/subzone and boundary-order penalties.
- lightweight inference post-process in `pcbplace/infer.py` for edge-band snap, local alignment, gentle anchor nudging, and orientation cleanup.
- review/export/apply scripts now support the new manual columns.
