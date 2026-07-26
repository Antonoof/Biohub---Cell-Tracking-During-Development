# Biohub Cell Tracking Vizualization

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
