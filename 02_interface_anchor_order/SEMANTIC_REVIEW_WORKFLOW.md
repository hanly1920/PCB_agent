# Semantic review workflow (human / semi-human curation)

This package upgrades semantic labels from a pure rule-generated seed into a **reviewable and correctable labeling workflow**.

## What is new

Each component now includes:

- `semantic_review.schema_version`
- `semantic_review.auto_confidence`
- `semantic_review.needs_review`
- `semantic_review.review_priority`
- `semantic_review.review_status`
- `semantic_review.evidence`
- `semantic_review.candidate_labels`
- `semantic_review.manual_overrides`

This means you can inspect *why* a label was suggested, decide whether it needs manual review, and then write corrected labels back into the dataset.

## Files added for review

- `reviews/semantic_review_all.csv`
  - full editable sheet covering all components
- `reviews/semantic_review_queue.csv`
  - filtered priority queue for manual checking

## Recommended workflow

### 1) Review the priority queue first

Open:

- `reviews/semantic_review_queue.csv`

Focus on rows with:

- `review_priority = high`
- `needs_review = true`
- low `auto_confidence`
- evidence strings showing ambiguity

### 2) Edit the `manual_*` columns

Most important editable columns are:

- `manual_semantic_class`
- `manual_functional_group`
- `manual_region_type`
- `manual_align_group`
- `manual_critical_nets`
- `manual_critical_neighbors`
- `manual_review_status`
- `manual_review_notes`

### 3) Apply the manual edits back to JSON

Run:

```bash
python scripts/apply_semantic_review.py --root . --review_csv reviews/semantic_review_all.csv
```

This updates the per-component top-level semantic fields and also stores the override history inside `semantic_review.manual_overrides`.

### 4) Re-export the review sheet if needed

After applying edits, regenerate the review tables:

```bash
python scripts/export_semantic_review.py --root . --out_dir reviews
```

## Notes

- The current labels are still **seed labels**, not guaranteed ground truth.
- The point of this upgrade is to make the labels **inspectable, auditable, and editable**.
- Training reads the top-level semantic fields, so once overrides are written back, later training will consume the corrected labels directly.
