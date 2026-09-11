# Validation record

## Validation rules

The system uses several distinct validation types. They must not be combined or
reported as if they were the same measurement.

- Grouped movie OOF: each movie is scored by a fold model not fitted on that
  movie.
- Held-20: 20 complete movies are separated from the 175-movie development
  pool for a saved validation panel.
- Practice-four: four named movies used for end-to-end execution and metric
  checks. They are not treated as a universal transfer estimate.
- Exact graph replay: the official graph metric is run on complete movie
  outputs after the named stage order.
- Hidden leaderboard: the final Kaggle score. This is the promotion authority
  for the current production lineage.

Random edge-row cross-validation is not accepted as graph validation because
rows from the same movie and trajectory are dependent.

## Current production result

- Notebook: `fork-of-fork-of-division-focused.ipynb`
- Hidden score: `0.957`
- Previous production parent: `0.954`
- Displayed improvement: `+0.003`
- Promoted change: wider-geometry division before Multi-UniGRAFT

## Component evidence

### P1/P2 base models

The current graph uses frozen hash-verified checkpoints. The P2 snapshot records
best epoch 381 with internal score `0.9779747766`, but it was trained on all 199
movies and is not an unseen transfer result. P1's repackaged artifact does not
retain a complete split manifest. End-to-end production validation therefore
comes from complete graph replay and hidden results, not the checkpoint's local
classifier score alone.

### Model C combined decoder

- V2-feature control grouped OOF division Jaccard: `0.1809`
- V2 + Model C grouped OOF division Jaccard: `0.3812`
- V2-feature control held-20 division Jaccard: `0.0741`
- V2 + Model C held-20 division Jaccard: `0.1905`

These are matched division-selection diagnostics. They are not hidden scores.

### Source-cardinality head

Exact held-20 graph result on its matching substrate:

- adjusted edge Jaccard: `0.905229`
- division TP/FP/FN: `7 / 8 / 10`
- division Jaccard: `0.280000`
- composite: `0.933229`

Threshold `0.40` was selected on grouped train-video OOF before held scoring.

### Independent P1/P2 UG2 head

Frozen held-20 result:

- division TP/FP/FN: `6 / 4 / 11`
- division Jaccard: `0.285714`
- frozen threshold: `0.51`

### Live V2 runtime specialist

Matched current-order validation on the 42 train movies whose topology changed:

- division delta: `+8 TP / 0 FP / -8 FN`
- changed-subset score delta: `+0.009063`

This package does not claim an absolute all-175 result because the saved
validation record is a matched changed-movie comparison.

### Full-population ownership

- candidate rows: `2,089,875`
- five grouped video folds
- OOF selected sources: `654`
- OOF winner keys reproduced exactly by the packaged models
- unknown sources excluded from fit and threshold selection

### EdgeGRAFT V3

Parent-ranker grouped OOF:

- targets: `86,864`
- baseline correct ownership: `83,838` (`0.965164`)
- model correct ownership: `85,284` (`0.981811`)
- net correct: `+1,446`

Exact replay on the saved training substrate:

- adjusted edge Jaccard: `0.907468 -> 0.925228`
- gain: `+0.017760`

The transaction gate was trained on 3,883 non-neutral labeled replacements;
metric-neutral rows were excluded.

### CandidateGRAFT

- grouped OOF ROC AUC: `0.760766`
- grouped OOF average precision: `0.938969`
- frozen threshold: `0.90`
- exact OOF control: `0.972210`
- exact grouped score: `0.973942`
- final-fit replay: `0.974107`
- final-fit delta: `+0.001897`
- division result unchanged: `96 / 60 / 35`, Jaccard `0.502618`

### Motion Corrector V1

Historical 9-movie assignment validation:

- geometric baseline: TP `5,629`, FP `391`, FN `364`, Jaccard `0.881736`
- production checkpoint: TP `5,661`, FP `355`, FN `332`, Jaccard `0.891777`
- assignment Jaccard gain: approximately `+0.01004`

These are motion-assignment metrics on the historical A+B proposal population,
not the end-to-end competition score.

### DeepCenter

- model-validation selection: lowest validation loss
- deployed best checkpoint: epoch `2`
- best validation loss: `0.0450031`
- sparse gate diagnostic: 240 frames, 7.0 um match radius

Sparse-label precision and recall are calibration diagnostics, not complete-cell
metrics. Production uses DeepCenter only as a targeted confirmation model.

### Wider-geometry division

This is deterministic and has no fitting split. Its promotion evidence is the
hidden production change `0.954 -> 0.957` when placed after gap recovery and
before Multi-UniGRAFT. The current notebook records proposed pairs, pairs
retained after UniGRAFT, and pairs retained in the final graph.

## End-to-end acceptance checks

A run is accepted only when all of these are true:

1. Every dependency, manifest, helper module, and checkpoint hash passes
   preflight.
2. Every requested movie has one raw graph and all required evidence files.
3. Every movie receives one combined CPU graph upgrade or an explicitly logged
   validated deadline recovery.
4. Every final edge has existing endpoints and advances exactly one frame.
5. Final in-degree is at most 1 and out-degree is at most 2.
6. Protected ownership and daughter-pair contracts remain intact.
7. Every movie has one validated CSV shard.
8. The final submission contains every required dataset.

## Reporting rule

Always name the validation population next to a score. A local, OOF, held-20,
practice-four, exact replay, and hidden leaderboard value are different claims.

