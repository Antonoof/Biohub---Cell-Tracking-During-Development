# Biohub production Motion Corrector V1 — teammate handoff

Prepared 2026-08-31 from the current `.953` production notebook and the exact local training artifacts.

## Bottom line

The current production motion corrector **was trained on the historical A+B proposal population**, not on the current P1/P2 production graph.

More precisely, the V1 trainer read 199 cached A+B detector-proposal files from:

`/home/tweak/bio/ab_proposals_export/biohub_ab_proposals`

Those files contain the A+B fused node population (`coords`, fused detection probability, and both member-model probabilities). The trainer then generated candidate continuation pairs geometrically and supervised them by matching the proposal nodes to Biohub GT. It was **not trained from finalized P1/P2 ILP graphs**, and it did not learn from the current P1/P2 native edge probabilities.

The exact checkpoint loaded by production is included at:

`CURRENT_PRODUCTION/motion_corrector_best.pt`

SHA256:

`57b2fa4f0585c5915f46c2c99e9b48690b74654ea5652f19577963c474cf2ae5`

That hash matches both the local canonical runtime-safe checkpoint and the Kaggle dataset copy.

## What the model does

The corrector is a small MLP that predicts a bounded residual correction to a geometric continuation cost. It does not detect cells and does not create divisions. For each geometrically gated source/target candidate it receives 23 features, predicts a scalar residual bounded to `[-2, +2]`, and production subtracts that residual from the relinking assignment cost before the Hungarian solve.

Architecture:

`Linear(23,64) -> SiLU -> Dropout(0.05) -> Linear(64,32) -> SiLU -> Linear(32,1)`

Production equation:

`assignment_cost = motion_distance + 0.05 * raw_distance - 1.0 * P1/P2_edge_probability - learned_residual`

The exact feature order, normalization arrays, weights, residual scale, and validation metrics are stored inside the checkpoint. See `CURRENT_PRODUCTION/checkpoint_inspection.json`.

## Exact V1 training population

- Dataset: the 199 real Biohub training movies.
- Proposal source: historical A+B fused proposal cache.
- Split: 190 training movies / 9 validation movies from `TRAINING_V1/splits_ensembleB.json`.
- Practice-four movies were part of the 190-video training side of this historical split.
- Cached training rows: 354,255 candidate pairs; 120,028 positive labels; 18,810 frame-transition groups.
- Cached validation rows: 540,613 candidate pairs; 5,882 positive labels; 891 frame-transition groups.
- Candidate geometry: registered distance at most 9.5 um.
- Assignment evaluation: tight pass 6.2 um, relaxed pass 9.5 um.
- Velocity weight: 0.52.
- Base training cost: `motion_distance + 0.05 * registered_distance`.
- Negative mining: closest hard negatives, up to 20 negatives per positive with a minimum pool of 64 per transition.
- Seed: 2028.
- Optimizer: AdamW, learning rate 0.002, weight decay 0.0001.
- Batch size: 8192.
- Maximum epochs: 40; early-stopping patience: 7.
- Loss: focal-weighted binary cross entropy on `logit = 2.5 - base_cost + residual`.

The exact cached row matrices are included in `TRAINING_V1/training_cache/`. They make it possible to reproduce the original fit without rebuilding rows. The complete 199-file A+B proposal population is also included in `TRAINING_V1/ab_proposals/` for provenance and for rebuilding the cache against Biohub source data.

## V1 validation result

Geometric motion baseline on the historical 9-video validation split:

- TP 5,629 / FP 391 / FN 364
- Jaccard 0.881736

Exact production runtime-safe checkpoint:

- TP 5,661 / FP 355 / FN 332
- precision 0.940991
- recall 0.944602
- Jaccard 0.891777

This is an improvement of approximately +0.01004 Jaccard on that historical validation assignment task. These are motion-assignment metrics, not the official end-to-end competition score.

## Important train/serve differences in current production

The checkpoint is real and useful, but the present `.953` notebook no longer serves it under the exact conditions used for V1 training:

| Item | V1 training | Current `.953` production |
|---|---:|---:|
| Node/proposal population | historical A+B fused proposals | P1/P2 production graph |
| Tight gate | 6.2 um | 6.0 um |
| Relaxed gate | 9.5 um | 10.0 um |
| Velocity | scalar 0.52 | per-axis `[0.0, 0.45, 0.47]` |
| Frame registration | used in candidate/cache construction | enabled, weight 1.0 |
| Learned edge probability in assignment cost | absent | P1/P2 probability bonus 1.0 |
| Hard maximum match cost | absent in original trainer | 7.5 um, judged before learned bonus |
| Raw-distance coefficient | training base uses registered distance | production uses raw distance, coefficient 0.05 |

This is the main reason an updated corrector is a credible improvement avenue: retrain on the exact current P1/P2 pre-motion population and optimize the exact production decision rule, rather than merely fine-tuning the existing A+B-trained MLP.

## Later V2 attempt

`REFERENCE_ONLY_NONPRODUCTION_V2/` contains a later trainer and checkpoint explicitly designed to reduce the train/serve mismatch by training on raw-ILP graphs and including edge probability in the assignment cost. It was a legitimate experiment but it is **not the current production model** and must not be silently substituted.

Its saved summary reports:

- held-20 V1 motion Jaccard: 0.845852
- held-20 V2 motion Jaccard: 0.846933
- untouched practice-four V1: 0.898260
- untouched practice-four V2: 0.899107

Those small local gains did not establish it as the hidden-scored production replacement. Use the V2 code as a design reference, not as an approved swap.

## Recommended clean retraining experiment

1. Export the exact current P1/P2 pre-motion node graph and native edge probabilities movie-by-movie.
2. Use grouped movie OOF splits; do not use random edge rows across the same movie in train and validation.
3. Rebuild candidates with the exact production gates, frame registration, per-axis velocity, refusal cap, and edge-probability bonus.
4. Train a residual or ownership scorer on those exact candidates.
5. Compare full-movie official metrics, with the current V1 checkpoint frozen as control.
6. Preserve the current checkpoint and runtime as rollback. Do not judge replacement only by pairwise AUC.

## Reproducing the historical V1 fit from the bundled cache

From an environment with PyTorch, NumPy, SciPy, and tqdm:

```bash
python TRAINING_V1/train_motion_cost_corrector.py \
  --data /path/to/biohub/train \
  --proposals TRAINING_V1/ab_proposals \
  --splits TRAINING_V1/splits_ensembleB.json \
  --cache TRAINING_V1/training_cache \
  --output /path/to/new_motion_v1 \
  --device cuda:0
```

Because `train.npz` and `val.npz` are included, the trainer reuses those exact cached rows unless `--rebuild-cache` is supplied. The raw Biohub training data is therefore only required if the cache is rebuilt.

## Directory guide

- `CURRENT_PRODUCTION/`: exact production checkpoint, checkpoint metadata, exact metrics.
- `TRAINING_V1/`: original trainer, dependency, split, training cache, A+B proposal population, and historical log.
- `PRODUCTION_RUNTIME/`: current `.953` runtime contract and the relevant source excerpt.
- `REFERENCE_ONLY_NONPRODUCTION_V2/`: later mismatch-aware experiment, explicitly not production.
- `MANIFEST.json`: file inventory and provenance.
- `SHA256SUMS.txt`: integrity hashes for every packaged file except itself.

## Provenance caution

The historical console log ends with the first V1 output directory and Jaccard 0.891041. The production runtime-safe checkpoint was then written minutes later with the five non-runtime features removed and Jaccard 0.891777. No separate runtime-safe console log was found; the exact checkpoint metadata and `metrics.json` are the authoritative records for the checkpoint actually used in production.
