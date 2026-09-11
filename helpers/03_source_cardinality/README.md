# Biohub Public-.934 Source Cardinality Head v2

This artifact is a low-capacity grouped option head for native division
selection. It does not detect cells and does not replace the public P1+P2
continuation graph.

For every V2 source transition it jointly normalizes one `CONTINUE` option and
all retained `DIVIDE(daughter_a, daughter_b)` options. The pair feature contract
is:

1. V2 geometry: 78 columns;
2. Model C native competition evidence: 24 columns;
3. public P1 native competition evidence: 24 columns;
4. public P2 native competition evidence: 24 columns.

The source input is the frozen 41-column V2 source schema. The OOF-frozen
threshold is `0.40`. Training and serving both retain at most 128 pair options
per source in stable V2 geometry order. This cap covers every supervised source
in the training data (observed maximum: 96), eliminating the provisional v1
train/serve mismatch.

Tube NMS, child-collision handling, and graph replacement must remain atomic.
Unknown sources were not supervised as negatives, and no NULL daughter class
exists.

The checkpoint must be deployed with the same P1/P2 evidence extraction used
during training: four-view detection TTA, `0.96875` detection threshold,
`5.0 um` pooling, identity-view edge features, `20 um` evidence radius, and up
to 16 targets per source. A different evidence path is train/serve skew.

Exact held-20 host-patched graph result on the matching current `.934` graph:

- adjusted edge Jaccard: `0.905229`;
- division TP/FP/FN: `7 / 8 / 10`;
- division Jaccard: `0.280000`;
- composite: `0.933229` (`+0.016265` versus the exact deployed decoder).

The production `.934` notebook remains unchanged. This artifact is for a
separate Kaggle candidate with the exact native-evidence contract.

The same frozen head was also rescored and replayed on the graph-preserving
P1+P2 plus add-only-gap substrate. Exact held-20 host-patched results were:

- adjusted edge Jaccard: `0.912867`;
- division TP/FP/FN: `6 / 10 / 11`;
- division Jaccard: `0.222222`;
- composite: `0.935089` (`+0.018125` versus the current P1+P2 `.934`
  motion/gap graph plus exact decoder).

The historical `0.190476` division Jaccard (`4 / 4 / 13`) was measured on the
original `.931` A+B graph. The graph-matched P1+P2 `.934` control used by this
artifact is `0.125000` (`3 / 7 / 14`).

This is the strongest complete held-20 candidate. The isolated gains were not
fully additive: division Jaccard was lower than cardinality alone, while the
preserved continuation graph raised the final composite.

Runtime files:

- `source_cardinality_runtime.py`: grouped CONTINUE/DIVIDE scorer;
- `source_cardinality_graph_runtime.py`: tube NMS and atomic graph transaction;
- `export_model_c_native_division_evidence.py`: exact four-view P1/P2 evidence
  exporter used by the first parity-focused Kaggle candidate.

Matching notebook:
`public-934-source-cardinality-v2.ipynb`

Preferred combined notebook:
`public-934-combined-preserve-cardinality-v2.ipynb`

Checkpoint SHA-256:
`18AFB6E10B0D551354240DF50497B5C9B5330D2DC08A5BF6FEF570F09CCB2CC3`
