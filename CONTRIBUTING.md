# Contributing

Contributions that improve reproducibility, portability, validation, or
scientific clarity are welcome after the repository is publicly released.

## Before proposing a change

1. Read the architecture PDF and `TRAINING_AND_GRAPH_PROVENANCE.md`.
2. Identify the exact graph population and serving position affected.
3. Preserve complete-movie grouping and sparse-label safety.
4. Separate source changes from regenerated data, caches, and checkpoints.
5. Run `python tools/verify_repository.py`.

## Reproducibility information to include

- base commit and component version;
- data and artifact identifiers plus hashes;
- complete-movie train, OOF, validation, and test partitions;
- command line, configuration, random seed, and selected epoch;
- feature schema and frozen decision threshold;
- input and output graph counts;
- component-level and end-to-end validation results;
- any deviation from the reference serving order.

## Graph safety

Every proposed graph mutation must preserve the production contract:

- both edge endpoints exist;
- each edge advances exactly one frame;
- node in-degree is at most one;
- node out-degree is at most two;
- a transaction is committed atomically or not at all.

## Data and generated files

Do not commit raw movies, labels with restricted redistribution terms, model
checkpoints, generated caches, credentials, or personal filesystem paths.
Document how authorized users can obtain or regenerate required artifacts.

## Scientific reporting

Label training, grouped OOF, held-movie, practice-movie, exact-replay, and
hidden measurements separately. Do not characterize practice or leaderboard
measurements as independent biological validation.
