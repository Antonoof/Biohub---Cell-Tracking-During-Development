# Multi-frame contextual linker experiment (not promoted)

Production source of truth remains:

`C:\Kaggle\division-gbm-w-updated-model.ipynb`

Production SHA256 before and after this experiment:

`9DD9B748B9150FB8115D77EB9AF9D265D1921A9FADEC1F3A8DCC25946C7045CB`

## Why this experiment was run

The project had previously tested several linker-like corrections, but none was
a true multi-frame contextual edge model:

- The residual edge corrector used 29 single-transition features.
- The learned motion-cost corrector used one predecessor velocity.
- Proposal-aware B2 fine-tuning did not improve its fused validation baseline.
- HOCT was already tested on a 20-video, practice-excluded panel and worsened
  pair ranking (AP `0.01178 -> 0.00416`, top-1 `0.4083 -> 0.3550`).

The retained A+B proposal and edge exports made it possible to test a real
multi-frame linker without rerunning either neural network.

## Reused assets

- `/home/tweak/bio/ab_proposals_export/biohub_ab_proposals`
- `/home/tweak/bio/ab_edge_probs_v16`
- Sparse GT under `/home/tweak/bio/train`
- Existing official metric implementation and cached audit graphs

No model inference or 199-video image extraction was run.

## Model and validation

For each safely supervised `t -> t+1` candidate edge, the contextual model
added best-path evidence over approximately `t-2 .. t+3`:

- previous and future best-edge probabilities;
- two-step path depth;
- displacement, speed-ratio, acceleration and direction consistency;
- detection-confidence and local-density trends;
- frozen-transition context.

Unknown proposals were never labelled negative.

The four repeatedly used practice clips were excluded. Validation used 195
videos with grouped whole-video OOF folds. Each held fold used a threshold
selected on a separate grouped calibration fold and frozen before held scoring.
The essential control compared the contextual model with the same GBM family
using only local, single-transition features.

## OOF result

Dataset:

- 195 videos
- 2,128,571 safely supervised candidate edges
- 125,221 positive candidate edges
- 126,756 annotated next-frame GT edges

Metrics:

| Model | Edge Jaccard |
|---|---:|
| Frozen fused A+B probability | 0.756407 |
| Local-only GBM | 0.858018 |
| Multi-frame GBM | 0.859122 |

The contextual increment over the equally capable local control was
`+0.001104`. All five held folds improved. `6bba` improved, while `44b6` was
effectively flat (`-0.000218`). This passed the offline feature gate but was too
small to integrate directly.

## Paired cached-ILP replay

The retained proposal export does not reconstruct the exact `.921` graph, so
the replay was used only as a paired comparison. To isolate temporal context
and prevent full-population distribution shift, the replay applied only:

`A+B logit + strength * clip(logit(context) - logit(local), -0.5, 0.5)`

It did not deploy either GBM's full probability.

Initial two-video screen:

- Half strength adjusted-edge delta: `+0.005473`
- Full strength adjusted-edge delta: `+0.004730`
- No division regression

Independent four-video size-stratified holdout at half strength:

- Baseline adjusted edge: `0.933016`
- Context adjusted edge: `0.932582`
- Delta: `-0.000434`
- Per-video outcome: 2 improved, 2 regressed
- No division change

The dense `44b6` case regressed enough to erase the smaller improvements.

## Decision

Do not integrate or submit this multi-frame linker.

The experiment proves that temporal context contains a small transferable
row-level edge signal, but the residual is not stable after global ILP
optimization. More threshold or residual-strength tuning on this small graph
panel would be overfitting. Revisit only if a future linker is trained and
validated against the structured graph objective itself rather than candidate
edge Jaccard.

## Artifacts

- Training script: `scripts/train_multiframe_context_linker.py`
- Paired replay: `scripts/replay_multiframe_context_linker_panel.py`
- Full OOF output: `/home/tweak/bio/multiframe_context_linker_v1`
- Initial panel: `/home/tweak/bio/multiframe_context_linker_panel_smoke`
- Independent holdout: `/home/tweak/bio/multiframe_context_linker_panel_holdout_v1`
