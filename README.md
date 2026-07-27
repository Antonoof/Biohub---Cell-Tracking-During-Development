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
<<<<<<< HEAD

### Score card

If the ground-truth lineages for a dataset are available as
`labels_geff/{dataset}.geff` (override with `BIOHUB_LABELS_DIR`), selecting it
also scores it. The score sits in a chip in the top-right corner; click it or
press `m` to expand the card. Besides the score itself it shows what the score
is made of, which is the part you can act on:

- **edge TP/FP/FN, precision and recall** — whether the sample is losing more
  to links it invented or to links it never made;
- **detected GT cells** — the detection ceiling every link sits under;
- **missed GT links, split by cause** — `never detected` is a segmentation
  problem, `linked to another cell` is a tracking problem, `track just stops`
  is a gap-closing problem. These three usually point at three different fixes;
- **worst frames** — the timepoints carrying the most FP+FN, as buttons that
  jump straight there, so a bad number can be looked at instead of guessed at.

The `/api/metrics/{dataset}` JSON carries the same numbers plus the full
per-timepoint error series and the division term.

`visualization/metrics.py` implements the full competition score locally:

    score = adjusted_edge_jaccard + 0.1 * division_jaccard

- **Adjusted edge Jaccard** — per-timepoint optimal node assignment within 7 µm
  of scaled centroid distance, edge TP/FP/FN against the sparse ground truth
  (an edge is only a FP where the GT contradicts it), scaled by a penalty on
  over-predicting the node count.
- **Division Jaccard** — every GT node with ≥ 2 successors against every
  predicted node with ≥ 2 successors, judged inside a local
  grandparent → parent → children → grandchildren window so a fork one
  timepoint early or late still counts, then paired by maximum-cardinality
  bipartite matching.

Aggregation is micro: the edge term is weight-averaged by TP+FP+FN, the
division term is one Jaccard over the summed counts.

`python -m visualization.test_metrics` runs its synthetic checks. Scored
against seven submissions with known leaderboard results, the local edge term
tracks them at Pearson 0.99 / MAE 0.02 — the residual is the split difference
(the leaderboard runs on the public videos, these labels are the test ones), so
treat it as a strong relative signal rather than an exact leaderboard preview.

The division term is computed and included in the score but is not shown in the
card: only one of the four local samples annotates any division at all (three
of them), so locally it is noise. The place to watch it is
`/api/metrics/{dataset}` → `division.n_pred_forks` against `n_gt_divisions`.

## Environment

The code targets Python 3.12 and the offline dependency bundle used by the
Kaggle notebook. Paths are configured through the `BIOHUB_*` environment
variables documented in the notebook.

## Collaboration

Use `main` for submission-ready code. Visualization and local-validation work
lives on the `validation-visualization` branch and should be merged only after
it is reproducible and does not change inference behavior.
=======
>>>>>>> 4094e2dac3aaf57d048d574142e9d164ae791d27
