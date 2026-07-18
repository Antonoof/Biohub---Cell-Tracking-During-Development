# Biohub Division Data Catalog

This catalog prevents repeated extraction and records which assets are safe to
join.  Paths are local WSL paths unless noted otherwise.

## Canonical assets

| Asset | Rows/videos | Purpose | Join key / alignment | Status |
|---|---:|---|---|---|
| `/home/tweak/bio/ab_proposals_export/biohub_ab_proposals` | 199 videos | Raw shared A+B detections and per-member detection probabilities | Per-video proposal row | Reuse; do not re-export |
| `/home/tweak/bio/ab_edge_probs_v16` | 199 videos, about 1.1 GB | Fused and per-model A+B edge probabilities | Per-video proposal source/target rows; graph nodes require spatial matching | Reuse; do not re-export |
| `/home/tweak/bio/division_training_cache_v1.npz` | 127,398 sources; 1,095,856 pairs | V1 source/pair features and sparse-safe labels | `pair_source_idx`; row-aligned with the tube and V16 edge caches | Canonical V1 training cache |
| `/home/tweak/bio/division_event_tubes_v14.npz` | 127,398 sources; 1,095,856 pairs | Tracklet/tube and node-token mapping | Exact source/pair row alignment with V1 cache | Canonical tube mapping |
| `/home/tweak/bio/division_edge_features_v16/edge_x.npy` | 1,095,856 x 29 | A+B alternative-edge evidence | Exact pair-row alignment with V1 cache | Canonical pair edge evidence |
| `/home/tweak/bio/division_v3_full_population_cache` | 199 videos; 4,995,373 transitions | Full-population transition inputs and sparse-safe labels | Per-video `source_id`, `source_t`, `source_tube` | Canonical all-cell validation population |
| `/home/tweak/bio/division_pseudo_candidates_60_v1/candidates_full` | 20 videos, about 0.22 GB | Feature-retaining full-runtime source tables | Per-video graph node IDs and pair node IDs | Feasibility tests only; generated with deploy V2 and must be rescored by corrected V1. These files retain only one preselected pair per paired source, so they cannot validate a daughter-pair ranker. |
| `/home/tweak/bio/division_gbm_deploy_v1_crossfit` | 2 models | Corrected V1 78-to-123 cascade | Feature names in `deploy_spec.json` | Proven corrected teacher |

## Verified invariants

- V1 `pair_x` and V16 `edge_x` both contain exactly 1,095,856 rows.
- V1 source/pair rows align exactly with `division_event_tubes_v14.npz`.
- The full-population cache contains 149 annotated positive transitions,
  94,624 confirmed continuations, and 4,900,600 unknown transitions.
- Unknown transitions are never supervised as negatives.
- Raw A+B proposal indices are **not** graph node IDs.  Any join from graph
  nodes to A+B proposals must use frame-local spatial matching, as implemented
  by `build_division_edge_features_v16.py`.
- The prior V16 controlled result used a GT-informed parent shortlist.  Its
  pair transaction is useful evidence, but its parent-selection metric is not
  deployable.  New parent gates must be evaluated on the complete population.

## Missing data

Two potentially useful signals are not already stored:

1. Contextual A/B transformer node embeddings.  Center-sampled UNet detector
   embeddings were tested on 20 videos and failed the promotion gate.
2. Full-population multi-pair V1 teacher tables.  The retained 20 runtime files
   contain source features but only one preselected pair per paired source.

Both are deliberately deferred.  The first distillation attempt uses the
existing A/B edge probabilities and graph/trajectory features.  New extraction
should be narrowly scoped only after that honest image-free student is measured.

## Current experiment

`train_division_v1_edge_student_20.py` is a no-new-extraction feasibility gate:

1. Rescore the 20 retained runtime caches with corrected V1.
2. Spatially join graph nodes to the existing A+B edge exports.
3. Train an image-free pair and source student with grouped whole-video OOF.
4. Measure teacher parent-selection fidelity.  Pair fidelity is reported only
   when a source has multiple retained pairs; one-pair sources are not counted.
5. Do not package or modify inference unless the frozen promotion gates pass.

### Feasibility result (2026-07-16)

The 20-video grouped OOF run did **not** promote:

- V1 teacher selections: 1,693
- Student selections: 1,635
- Exact parent-selection TP/FP/FN: 244 / 1,391 / 1,449
- Parent-selection precision: 0.1492
- Parent-selection recall: 0.1441
- Parent-selection Jaccard: 0.0791

Conclusion: geometry plus exported A/B edge probabilities can reproduce the
number of forks but not which parent/transition V1 selects.  The missing signal
is parent appearance/internal representation, not another edge-probability
aggregation.  No inference integration was made.

### Center-embedding distillation result (2026-07-16)

A narrowly scoped export sampled the 32-channel A and B UNet feature maps at
each retained source in both temporal contexts `(previous,current)` and
`(current,next)`:

- Export: `/home/tweak/bio/ab_source_embeddings_20_v1`
- Videos: 20
- Rows: 592,947
- Storage: 149.7 MB
- Student inputs: 297 image-free/embedding features
- Validation: grouped whole-video OOF with disjoint grouped calibration

The embedding student also did **not** promote:

- Parent-selection TP/FP/FN: 185 / 1,312 / 1,508
- Precision: 0.1236
- Recall: 0.1093
- Jaccard: 0.0616

A diagnostic that kept every video represented while holding out complete
tracklets reached only 0.1835 precision and 0.1768 recall.  Therefore the failure
is not merely cross-video calibration: center-sampled detector embeddings do not
contain enough of V1's patch-level parent appearance signal.  The export was not
expanded to 199 videos and no inference integration was made.

Scripts:

- `export_ab_source_embeddings_20.py`
- `train_division_v1_embedding_student_20.py`
