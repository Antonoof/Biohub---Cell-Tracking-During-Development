# Repository history retained on this branch

This release branch is based on the repository's `main` branch and preserves
its earlier notebooks, scripts, models documentation, and research notes.

- `notebooks/` contains earlier competition notebooks retained from `main`.
- `scripts/` contains earlier proposal-export and motion-training utilities.
- `models/` documents the external artifacts expected by those earlier files.
- `docs/DEVELOPMENT_EXPERIMENTS.md` is the experiment registry that previously
  occupied `docs/EXPERIMENTS.md` on `main`.
- `docs/EXPERIMENTS.md` is the experiment record belonging to the `.957`
  reproducibility release.

The `helpers/`, `production/`, provenance, validation, and architecture files
form the reproducibility release described by the root README. Retaining the
earlier material preserves Git history without presenting it as the current
reference pipeline.
