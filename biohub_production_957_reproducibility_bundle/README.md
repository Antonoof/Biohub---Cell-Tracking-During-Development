# Biohub production .957 reproducibility bundle

This bundle documents the current production system and points directly to the
code used to train, fit, validate, package, and serve every attached helper.
It is organized so a teammate can start here, identify a dataset, and then find
the relevant source without searching the full project history.

## Current production identity

- Notebook: `production/fork-of-fork-of-division-focused.ipynb`
- Hidden score: `0.957`
- Notebook SHA-256: `0C141C83CE67FB24A4C30FCC65B06856331F2AA17119BAC8F36502FB337F7EC2`
- Production change over the previous `0.954` parent: wider-geometry division
  runs after gap recovery and before Multi-UniGRAFT.

The notebook is the authoritative inference implementation. It contains the
exact order, thresholds, model paths, dependency checks, scheduler, graph
helpers, and final validation logic.

## Read these files first

1. `docs/biohub_production_957_full_technical_architecture.pdf` - complete
   architecture, flowcharts, helper descriptions, and active constants.
2. `docs/EXPERIMENTS.md` - current production status and completed validation
   history.
3. `TRAINING_AND_GRAPH_PROVENANCE.md` - what trained each learned component and
   which graph or candidate population it used.
4. `VALIDATION.md` - validation protocol and the recorded evidence for each
   production component.
5. `FILE_INDEX.md` - directory-by-directory map of the included source files.

## Required attached datasets

The production notebook resolves these exact Kaggle dataset IDs:

1. `pilkwang/biohub-tracking-support-pack-50ep-v1`
2. `pilkwang/biohub-temporal-unet3d-seed314159-v1`
3. `tweakai/biohub-model-c-combined-primary-v1`
4. `tweakai/biohub-public934-source-cardinality-v2`
5. `tweakai/biohub-unigraft-p1p2-independent-v1`
6. `tweakai/biohub-live-v2-global-bundle-v2`
7. `tweakai/biohub-ownership-exact-v2`
8. `tweakai/biohub-edgegraft-v3-full-population-v1`
9. `tweakai/biohub-candidategraft-direct-v1`
10. `tweakai/biohub-motion-corrector-v1`
11. `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`

The large model checkpoints and image datasets are not duplicated in this ZIP.
They remain in the attached Kaggle datasets above. This bundle includes their
available manifests, configuration, split records, training summaries, and
source code. Checkpoint hashes are recorded in the notebook and manifests.

## What is trained and what is not

Trained or fitted components:

- P1 and P2 3D UNet plus node-transformer models
- Model C and its division pair/source models
- source-cardinality head
- independent P1/P2 cardinality head
- Motion Corrector V1
- DeepCenter 3D center-prior model
- full-population ownership model
- EdgeGRAFT parent ranker and transaction gate
- CandidateGRAFT direct-edge classifier

Runtime-only components with no training script:

- wider-geometry division pass
- Live V2 transaction bundle
- sub-voxel peak refinement
- retention guard and guarded ownership restoration
- raw edge sanitation and single-parent repair
- one-frame gap transaction logic
- isolated-node pruning, short-track filtering, boundary rescue, node budgets,
  and line-fit smoothing
- graph validation, streaming scheduler, and atomic shard promotion

These runtime-only components are implemented inside the production notebook or
in the included deployment modules. They should not be described as learned
models.

## Reproduction sequence

1. Attach the competition input and the 11 datasets listed above.
2. Run the production notebook from a clean Kaggle session.
3. Confirm that dependency and checksum preflight passes before inference.
4. Confirm every movie receives a combined CPU upgrade and one validated shard.
5. Confirm final topology has no invalid endpoints, nonconsecutive edges,
   in-degree above 1, or out-degree above 2.
6. Use `VALIDATION.md` to distinguish grouped OOF, held-movie, practice-movie,
   exact replay, and hidden leaderboard results.

Training scripts are preserved exactly as found. Some retain historical
absolute paths in their defaults. Those paths identify the original graph bank
or feature cache; command-line arguments should be redirected when reproducing
the fit in a new environment. Do not silently change the candidate population,
feature order, fold grouping, or unknown-label policy.

## Sparse-label rule

For the graph heads, an unannotated candidate is not automatically a false
biological event. The recorded trainers either exclude unknown rows from the
supervised loss or explicitly mark metric-neutral transactions. This policy is
part of the training contract and must be preserved.

## Integrity

`SHA256SUMS.csv` lists every file in the bundle with its SHA-256 hash. Use it to
confirm that a copied trainer or runtime is byte-identical to this handoff.

