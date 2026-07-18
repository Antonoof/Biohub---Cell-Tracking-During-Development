# Structured Association V1

## Decision

The `.921` production notebook remains frozen. A separate candidate notebook
adds a structured A+B association head before the shared ILP, while preserving
the proven `.921` post-processing, learned motion cost, registration-aware
logic, streaming CPU division workers, and Division Parent Gate V2.

This experiment addresses a base-model weakness rather than adding another
post-ILP repair rule.

## Why this was built

The full 199-video audit showed that the detector/proposal stage has strong
coverage, but final association is not close to perfect:

- GT next-frame links recoverable from A+B proposals: `125,221 / 126,756`
  (`98.8%`).
- Final audited edge counts: TP `122,415`, FP `6,581`, FN `6,468`.
- Aggregate edge Jaccard: `0.9037`.
- Videos below adjusted edge `0.80`: `32 / 199`.
- Videos below adjusted edge `0.90`: `81 / 199`.
- The worst 40 videos contain approximately `60.4%` of residual association
  errors.

Therefore A+B usually detects the necessary nodes and candidate links, but its
pointwise edge fusion and shared ILP confidence do not reliably select the
correct topology. The existing motion/gap/division helpers compensate for this
weakness but do not replace a stronger association core.

## Model

Structured Association V1 is a small DeepSets-style residual head anchored to
the fused A+B edge logit.

For each candidate edge it uses 77 image-free features:

- A, B, and fused edge probabilities;
- member disagreement and detection confidence;
- physical displacement and distance;
- source-child and target-parent ranks/margins/counts;
- local 15 um density and Z-boundary context;
- best-path context over approximately `t-2 ... t+3`;
- motion residuals, direction cosines, speed ratios, and path confidence.

It jointly scores:

- up to 16 candidate children for each source;
- up to 32 candidate parents for each target.

The output is a bounded logit correction blended with the original A+B logit.
The production ILP edge threshold remains fixed at `0.54`.

## Sparse-safe training

- Training source: retained A+B top-16 proposals within 20 um.
- Training videos: 195.
- Excluded from all training: the four practice clips.
- Validation: five grouped whole-video folds, stratified by embryo.
- Each fold model uses three folds for fitting, one for calibration, and one
  held out; unseen inference averages all five fold logits.
- Unknown/unannotated edges are never treated as pseudo-negatives.
- Listwise loss is used only where sparse GT identifies exactly one positive
  competitor; division-parent source groups are not mislabeled as single-child
  groups.
- Pointwise BCE preserves absolute confidence at the fixed `0.54` threshold.
- No held-video threshold refitting.

## Validation ladder

### Row-level whole-video OOF

| Model | TP | FP | FN | Jaccard |
|---|---:|---:|---:|---:|
| Fused A+B | 120,687 | 32,797 | 6,069 | 0.7564 |
| Structured V1 | 118,323 | 11,034 | 8,433 | 0.8587 |

This showed a large precision gain, but row-level metrics were not accepted as
the promotion gate.

### Four excluded practice clips, paired graph + ILP replay

All variants used identical nodes, candidate proposal construction, threshold,
ILP, and evaluator.

| Variant | Edge Jaccard | Adjusted edge | Delta adjusted |
|---|---:|---:|---:|
| Baseline cached A+B | 0.8636 | 0.8459 | — |
| Structured strength 0.5 | 0.8824 | 0.8653 | +0.0193 |
| Structured strength 1.0 | 0.8848 | 0.8744 | **+0.0285** |

- 3/4 clips improved.
- Both embryo families improved in aggregate.
- Division result was unchanged.

### Twelve-video whole-video OOF graph panel

The panel sampled weak, middle, and already-strong videos from both embryos.
Each video used only its held-fold model.

| Variant | Edge Jaccard | Adjusted edge | Delta adjusted | Videos up/down |
|---|---:|---:|---:|---:|
| Baseline cached A+B | 0.8214 | 0.8033 | — | — |
| Structured strength 0.5 | 0.8368 | 0.8195 | +0.0162 | 9 / 3 |
| Structured strength 1.0 | 0.8390 | 0.8284 | **+0.0251** | **10 / 2** |

Mean per-video adjusted deltas at full strength:

- `44b6`: `+0.0135`
- `6bba`: `+0.0199`

The weakest selected video improved by `+0.0857`. The two regressions were
approximately `-0.017` each. Already-strong videos were not systematically
damaged.

## Runtime implementation

The candidate notebook now retains the same top-16, <=20 um A+B candidate set
used during training instead of discarding every edge below `0.54` immediately.
After all frames of one video are available it:

1. Builds the 77 image-free features.
2. Scores complete source and target competitor sets with five tiny fold heads.
3. Averages logits and applies a strength-1.0, +/-2.0-logit bounded correction.
4. Sends only corrected probabilities >=`0.54` into the same shared ILP.
5. Writes the same atomic ready graph consumed by the existing `.921` pipeline.

The five tiny heads run on `cuda:0` by default while Model A is idle. Image-free
feature assembly briefly uses CPU, but its spatial query is single-threaded so
the four streaming Division V2 workers retain nearly all CPU capacity. The
stage does not rerun A or B and does not extract image patches.

Runtime parity was verified on `44b6_0113de3b`:

- feature matrix difference vs the training builder: exactly zero;
- corrected probability difference vs the validation replay: exactly zero.

## Files

- Frozen source notebook: `C:\Kaggle\division-gbm-w-updated-model.ipynb`
- Candidate notebook: `C:\Kaggle\division-gbm-structured-association.ipynb`
- Upload artifact: `C:\Kaggle\biohub-structured-association-v1.zip`
- Artifact SHA256:
  `6F38DFE52F40FF7CC00E8B1B16B01F19679D9BF2ECA1802726F36B26952FF5DE`
- Training script:
  `scripts/train_structured_association.py`
- Four-practice replay:
  `scripts/replay_structured_association_panel.py`
- Twelve-video OOF graph replay:
  `scripts/replay_structured_association_oof_panel.py`
- Runtime artifact source:
  `artifacts/biohub-structured-association-v1/structured_association_runtime.py`

## Full integrated four-practice result

The full-strength candidate subsequently completed the full `.921`
post-processing and Division V2 stack. It confirmed the association signal but
failed the combined promotion gate:

| Metric | Frozen V2 baseline | Structured strength 1.0 | Delta |
|---|---:|---:|---:|
| Edge Jaccard | 0.932579 | 0.931705 | -0.000874 |
| Adjusted edge Jaccard | 0.929390 | 0.933585 | +0.004195 |
| Division Jaccard | 0.666667 | 0.333333 | -0.333334 |
| Combined local proxy | 0.996056 | 0.966919 | -0.029137 |

The correction reduced nodes from 142,494 to 130,017 and broke the scored
division component at source `63001217`, frame 62. Full strength is therefore
**not promoted**, even though adjusted edge improved.

Strength `0.50` remains the only active integrated candidate because it had a
smaller positive OOF graph delta. It must preserve the V2 division outcome and
improve the combined score before any Kaggle submission.

## Honest limitations and promotion rule

- Cached graph replays are paired comparisons, not byte-identical
  reconstructions of the full `.921` submission and not Kaggle score claims.
- Strength `1.0` failed the full four-practice combined gate and must not be
  submitted unchanged.
- Strength `0.50` still must complete the full post-processing/division stack
  before any submission.
- The `.921` notebook is the rollback baseline and must not be overwritten.
- Promote only if the generated log confirms five structured models loaded,
  all videos produced atomic ready shards, the final submission is complete,
  and the full local evaluator does not show an adjusted-edge regression.
