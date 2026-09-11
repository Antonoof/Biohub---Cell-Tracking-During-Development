# HOCT Higher-Order Division Experiment

## Objective

Evaluate the competition host's pretrained Higher-Order Cell Tracking
Transformer (HOCT) without replacing the proven A+B tracker or corrected
Division GBM V1.  The intended integration is HOCT as a division-candidate
feature expert, not as the submission's detector or Gurobi solver.

## Fixed production fallback

- Primary tracker: proven A+B `.907` pipeline.
- Division correction: corrected cross-fitted Division GBM V1.
- Scheduling: streaming/parallel anytime notebook with atomic per-video A+B
  fallback.
- Best hidden score remains `.920`.

No HOCT experiment modifies that path until it passes grouped validation and
the four-practice-video exact graph replay.

## Official model

- Repository: `historical\hoct`
- Checkpoint: `historical\hoct\models\general_v0.pt`
- Size: 25,510,490 bytes
- SHA256: `024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a`
- Forward outputs: edge logits, 288-dimensional contextual node features,
  288-dimensional contextual edge features, and orphan logits.

The official point-only graph constructor is currently unimplemented.  The
pretrained model expects 19 segmentation-derived node features, so A+B point
detections require a segmentation/morphology adapter.

## V1 adapter

The first adapter computed adaptive local ellipsoid morphology and normalized
intensity statistics around each A+B/post-processed detection.  HOCT was run
only on the V1 broad division-candidate subgraph.  Gurobi was not invoked.

Operational result:

- 20 practice-excluded videos, balanced 10/10 across embryos.
- 193,160 mapped candidate pairs.
- 220 positive pairs.
- 71.8 MB feature panel.
- Approximately 100 seconds per video locally.
- Pair-edge mapping generally 99.8-100%.

## Grouped-video OOF result

| Metric | A+B geometry/edge baseline | Baseline + HOCT | Delta |
|---|---:|---:|---:|
| Pair average precision | 0.011781 | 0.004162 | -0.007619 |
| Positive-source top-1 pair accuracy | 0.408284 | 0.355030 | -0.053254 |

Decision: **not promoted**.  The scalar HOCT similarities and 288-dimensional
edge representation are not submission-safe under the local-ellipsoid adapter.

## Interpretation and next gate

This result rejects the adapter, not necessarily HOCT.  The host explicitly
states that tracking quality depends on segmentation-mask quality.  The next
credible gate is marker-controlled 3D watershed segmentation:

1. Use all A+B detections in a frame as watershed markers.
2. Segment the normalized fluorescent nuclei jointly rather than independently.
3. Compute HOCT's exact region properties from those masks.
4. Re-extract the same practice-held-out 20-video candidate panel.
5. Promote only if both AP and top-1 selection beat the identical baseline.

If the watershed adapter also fails, stop the pretrained-HOCT integration and
retain corrected V1 streaming as the primary division solution.

## Reproducibility

- `scripts/audit_hoct_ab_onevideo.py`
- `scripts/export_hoct_division_features.py`
- `scripts/run_hoct_division_feature_panel.py`
- `scripts/train_hoct_pair_panel.py`
- Feature panel: `data/hoct_division_features_panel_v1`
- OOF summary: `data/hoct_pair_panel_v1/summary.json`
