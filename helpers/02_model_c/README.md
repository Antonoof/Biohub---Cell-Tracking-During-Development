# Biohub V2 + Native Model C Combined-Primary Division Artifact

This Kaggle artifact accompanies
`division-gbm-model-c-combined-primary-hidden.ipynb`.

It contains:

- the unchanged Division Parent Gate V2 runtime and models;
- the correct division-balanced Model C checkpoint;
- the leakage-safe Model-C pair and parent decoder;
- a runtime that applies the jointly trained V2+Model-C decoder as the primary
  division selector.

Model C is not a third fused ensemble member. It has zero influence on A+B
detections, ordinary continuation edges, and the initial ILP graph. Legacy V2
does not select or lock events first; it runs only if the combined decoder
fails.

## Frozen inference contract

- Model C detection threshold: `0.99`
- Model C pool radius: `3.0 um`
- Model C detection TTA: original, X flip, Y flip, XY flip
- Evidence search: `20.0 um`, top 16 targets/source, probability >= `0.01`
- Native-C to A+B mapping: nearest candidate within `6.0 um`
- Decoder pair width: `102`
- Decoder source width: `144`
- Decoder source threshold: `0.96`
- Selection: one event per linked lineage
- Priority: combined V2+Model-C decoder; existing A+B/ILP forks remain protected
- Failure fallback: legacy Division V2
- Graph constraints: in-degree <= 1, out-degree <= 2

## Clean diagnostic evidence

The decoder used 175 training videos and a disjoint 20-video validation split.
At the frozen `0.96` threshold:

- V2-feature control grouped OOF division Jaccard: `0.1809`
- V2 + Model C grouped OOF division Jaccard: `0.3812`
- V2-feature control held-20 division Jaccard: `0.0741`
- V2 + Model C held-20 division Jaccard: `0.1905`

These are matched-event selection diagnostics, not official hidden-test scores.
The four practice clips were not used for training or threshold selection.

## Integrity

Correct Model C checkpoint SHA-256:

`3a4cfd682cfea77ec225af2f0e910a752dbf5b43ebe57cc40b0adc9fc4214cda`
