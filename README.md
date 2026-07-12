# Biohub Cell Tracking Team

Reproducible code for the Biohub Cell Tracking During Development competition.

## Repository layout

- `notebooks/` — Kaggle inference notebooks.
- `scripts/` — proposal export and learned motion-cost training.
- `validation/` — local metrics and run-comparison utilities (validation branch).
- `visualization/` — prediction/GT visualization utilities (validation branch).
- `docs/` — workflow and experiment notes.

## Current stable pipeline

The `main` branch contains the A+B true ensemble with registration-aware
post-processing and a learned residual cost inside the sequential Hungarian
motion relinker.

Model checkpoints and competition data are intentionally excluded. See
`models/README.md` for the expected external artifacts.

## Environment

The code targets Python 3.12 and the offline dependency bundle used by the
Kaggle notebook. Paths are configured through the `BIOHUB_*` environment
variables documented in the notebook.

## Collaboration

Use `main` for submission-ready code. Visualization and local-validation work
lives on the `validation-visualization` branch and should be merged only after
it is reproducible and does not change inference behavior.
