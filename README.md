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

## Visualizing predictions

`visualization/app.py` is a FastAPI service that steps through a
`submission.csv` frame by frame, overlaying detections and short trailing
tracks on the Z-max-intensity projection of the matching `.zarr` volume.

1. Replace `submission.csv` at the repo root with your own submission
   (same `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` format),
   and put the corresponding `{dataset}.zarr` volumes in `results/`.
   Both locations can be overridden with the `BIOHUB_SUBMISSION_CSV` and
   `BIOHUB_RESULTS_DIR` environment variables.
2. Install the viewer's dependencies:
   ```bash
   python -m pip install -r requirements-validation.txt
   ```
3. Start the service:
   ```bash
   uvicorn visualization.app:app --reload --port 8000
   ```
4. Open `http://localhost:8000` and pick a dataset. Each color is a stable
   lineage id, so a sudden color change on a trailing line usually means an
   identity swap. See `docs/VALIDATION.md` for more on what to look for.

## Environment

The code targets Python 3.12 and the offline dependency bundle used by the
Kaggle notebook. Paths are configured through the `BIOHUB_*` environment
variables documented in the notebook.

## Collaboration

Use `main` for submission-ready code. Visualization and local-validation work
lives on the `validation-visualization` branch and should be merged only after
it is reproducible and does not change inference behavior.
