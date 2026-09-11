# Portable data layout

Run training commands from the repository root. The retained defaults use the
following relative layout; every major trainer also exposes command-line path
arguments for a different environment.

```text
data/
|-- train/                                      raw `.zarr` and labeled `.geff`
|-- division_candidate_audit_v1/
|   |-- pre_safe_graphs/                        graph substrate
|   `-- registration_shifts/                    per-frame physical shifts
|-- division_official_event_cache_v2_parts/     sparse-safe division events
|-- division_v3_full_population_cache/          complete proposal population
|-- model_c_native_division_evidence_*/         Model C evidence by panel
|-- public914_backbone_matched_v1/              historical graph bank identifier
|-- ug12_ownership_geometry_bank_v2/            ownership candidate table
|-- edgegraft_current_labels_v2/                EdgeGRAFT parent decisions
|-- edgegraft_v2_rich_oof_decisions/            transaction-gate decisions
|-- native_endpoint_candidate_graft_v2/         CandidateGRAFT population
|-- ab_proposals_export/biohub_ab_proposals/    historical motion proposals
`-- splits_ensembleB.json                       motion train/validation split

external/
|-- bio_track_repo/                             base tracking source tree
`-- kaggle-cell-tracking-competition-patched/   metric/evaluator source tree

artifacts/                                      generated checkpoints/packages
runs/                                           logs and intermediate outputs
```

Names such as `public914` and `ug12` are retained historical graph identities.
They identify the exact population used for a fitted artifact. Renaming a graph
bank without recording its origin makes a result less reproducible.

## Required raw inputs

- Biohub image volumes in Zarr format.
- Corresponding GEFF graphs for supervised movies.
- Physical voxel spacing and per-frame registration information.
- Frozen detector evidence when training a downstream graph helper.

## Generated inputs

Several graph helpers train on candidate populations rather than directly on
raw images. Generate those populations with the extraction or materialization
scripts stored beside the relevant trainer. Never substitute a later graph
without recording the change: a helper's input graph is part of its model
definition.

## Paths and secrets

No personal workstation paths or credentials belong in this repository.
Kaggle dataset identifiers are retained only as artifact provenance. Keep API
tokens in the platform credential store or an ignored local `.env` file.

