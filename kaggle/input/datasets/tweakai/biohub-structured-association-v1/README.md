# Biohub Structured Association V1

Inference-only pre-ILP association corrector for TWEAK's frozen A+B detector.

## Purpose

The A+B detector has high proposal coverage, but its pointwise edge fusion
leaves substantial association error. This head scores complete competing
source-child and target-parent candidate sets using local confidence, member
disagreement, geometry, density, and multi-frame motion context.

## Training and validation

- 195 Biohub training videos; the four practice clips were excluded.
- Five grouped whole-video folds, stratified by embryo.
- Unknown/unannotated edges were not used as pseudo-negatives.
- Fixed production edge threshold: `0.54`; no held-video threshold fitting.
- The five fold heads are averaged on unseen inference videos.

## Evidence

- 195-video row-level OOF: fused A+B Jaccard `0.7564` -> structured `0.8587`.
- Four excluded practice graph/ILP replay: adjusted edge `0.8459` -> `0.8744`
  (`+0.0285`), 3/4 clips improved, both embryos positive in aggregate.
- Twelve-video weakness-stratified whole-video OOF graph replay: adjusted edge
  `0.8033` -> `0.8284` (`+0.0251`), 10/12 videos improved; mean per-video
  deltas were `+0.0135` for 44b6 and `+0.0199` for 6bba.
- Division output was unchanged in both graph gates.

These are paired cached-proposal graph comparisons, not a claimed Kaggle score.
The production notebook remains frozen until the full `.921` pipeline smoke
test passes.

## Runtime contract

- Candidate construction: each source's top 16 fused A+B edges within 20 um.
- Required models: `fold_0_model.pt` through `fold_4_model.pt`.
- Feature schema: 77 columns embedded in `structured_association_runtime.py`.
- Source group width: 16; target group width: 32.
- Blend strength: `1.0`; residual clip: `2.0` logits.
- ILP admission threshold remains `0.54`.
- Runtime is image-free after A+B inference. Feature assembly is CPU-light and
  single-threaded in its expensive spatial query; the neural heads default to
  `cuda:0`, when Model A is idle, to preserve CPU capacity for Division V2.

`structured_association_runtime.py` was parity-tested against the training
feature builder and replay implementation: feature and output differences were
exactly zero on `44b6_0113de3b`.
