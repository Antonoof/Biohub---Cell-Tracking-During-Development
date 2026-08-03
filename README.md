# Biohub Cell Tracking Vizualization

## Visualizing predictions

`visualization/app.py` is a FastAPI service that steps through a
`submission.csv` frame by frame, overlaying detections and short trailing
tracks on the Z-max-intensity projection of the matching `.zarr` volume.

1. Replace `submission.csv` at the repo root with your own submission
   (same `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` format),
   and put the corresponding `{dataset}.zarr` volumes in `files_zarr/`.
   Both locations can be overridden with the `BIOHUB_SUBMISSION_CSV` and
   `BIOHUB_ZARR_DIR` environment variables.
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

### Saving and comparing runs

**💾 Save** (top-left) copies the submission currently on screen into
`save_files/{name}.csv` under a name you choose — a snapshot of the run so it
survives the next time `submission.csv` is regenerated.

**⇄ Compare** puts a saved submission under the live one on the same page, as
a 3×2 grid: the top row is the three projections of submission **A** (the one
you are viewing), the bottom row the same three projections of **B**, the file
you pick from `save_files/`. Both rows share the frame slider, play button,
trail length and zoom/pan, so the same cells are on screen in both at once, and
the score card reports the metrics for each row separately.

Saved runs live in `save_files/` (git-ignored, since a submission is large);
`BIOHUB_SAVE_DIR` moves them elsewhere.

### Score

`
score = adjusted_edge_jaccard + 0.1 * division_jaccard
`

The score card shows both terms: the adjusted edge Jaccard with its TP/FP/FN
and node penalty, and the division Jaccard with its own TP/FP/FN plus the
number of GT divisions and predicted forks.
