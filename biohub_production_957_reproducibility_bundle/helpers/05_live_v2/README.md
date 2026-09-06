# Biohub live V2 global division bundle v2

Corrected CPU-only missed-division specialist for the frozen `.952` V17
UG1/UG2 production notebook.

Inference order:

`UG1+UG2 -> specialist -> EdgeGRAFT V3 -> V17 cleanup`

The helper reuses the already materialized UG2 128-pair V2 population and
native Model-C/P1/P2 evidence. It performs no image inference or additional
GPU pass. It contains no GT, video/source decision table, or cached hidden
decisions. Existing forks are protected, changes are atomic, and a helper
failure returns the completed UG1/UG2 graph.

Matched current-order validation on all 42 structurally changed train videos:

- division: 22/36/31 -> 30/36/23;
- exact delta: +8 TP, 0 FP, -8 FN;
- adjusted edge: 0.891539 -> 0.891613 on the changed-video panel;
- composite proxy: +0.009063 on that changed-video panel.

No absolute all-175 V17 Jaccard is claimed. The older 0.455497 -> 0.502618
result belongs to a different boundary-finalized UG2+UG3 substrate and is not
the validation basis for this package.
