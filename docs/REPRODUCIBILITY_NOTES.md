# Reproducibility scope and retained limitations

This document separates what can be reproduced directly from this source
release from what requires external data or model artifacts. The repository
contains executable training and runtime code, but it does not duplicate large
image volumes, graph banks, caches, or checkpoints that are stored separately.

## Reproduction levels

1. **Source inspection:** the trainer, feature schema, split, configuration,
   runtime contract, and validation procedure can be reviewed.
2. **Artifact reproduction:** the referenced data and candidate populations
   are available, allowing the trainer to be rerun.
3. **Inference reproduction:** the frozen artifact passes its checksum and the
   production notebook completes with the documented inputs.
4. **Metric reproduction:** the official evaluator is run on the exact final
   graph with the same population, coordinate convention, and matching radius.

Every scientific or performance claim should state which level was achieved.

## Known limitations

1. The P1 package retains its checkpoint hash and architecture, but not a full
   historical train/validation movie split. Its checkpoint metric must not be
   described as unseen validation.
2. P2 was trained on all 199 recorded training movies. Its best-epoch history
   is a training-run selection record rather than an independent generalization
   estimate.
3. Motion Corrector V1 was trained on the historical A+B proposal population.
   It is served on the later registered, z-damped P1/P2 graph while preserving
   the original one-step feature contract. This train/serve difference is
   intentional, observable, and documented.
4. EdgeGRAFT, CandidateGRAFT, ownership, and source-cardinality are
   graph-dependent. Replacing a detector or materially changing candidate
   generation requires honest revalidation and may require retraining.
5. Unannotated graph candidates are unknown. They must not be silently
   converted to negative examples.
6. Scores on the four practice movies are execution diagnostics, not
   substitutes for grouped OOF, held-movie validation, or hidden evaluation.
7. The recorded hidden score is a reference deployment measurement. It is not
   a training target and does not establish performance on a new laboratory,
   microscope, phenotype, or acquisition protocol.

## Reproducing the original release

Use the frozen production notebook and the exact artifact hashes recorded in
the manifests. Reconstruct external inputs using `DATA_LAYOUT.md`, then follow
the training and serving order in the architecture PDF. Do not silently replace
an artifact, alter the graph order, or retune a threshold while claiming exact
reproduction.

## Adapting the pipeline to a new detector

First produce a frozen detector graph and evidence bank. Then rebuild and
revalidate, in serving order:

1. source-cardinality and pair-evidence mappings;
2. full-population ownership;
3. EdgeGRAFT parent ranking and transaction gate;
4. CandidateGRAFT continuation gate.

The geometric portions of UniGRAFT can be reused as candidate generators, but
their learned scoring heads must not be assumed calibrated on a changed graph.
Motion Corrector can first be evaluated with its retained feature contract; a
retrain is warranted when grouped held-movie evidence shows a material shift.

## Reporting a reproduction

A reproducibility report should record:

- repository commit and release tag;
- data and checkpoint identifiers plus hashes;
- complete-movie train, OOF, validation, and test partitions;
- software environment and accelerator configuration;
- random seeds and deterministic settings;
- trainer arguments and selected epoch;
- frozen feature order and decision thresholds;
- graph population at each helper's input and output;
- component metrics and final end-to-end metrics;
- deviations from the reference configuration.

This information is necessary to distinguish an exact reproduction from a
scientific adaptation of the architecture.
