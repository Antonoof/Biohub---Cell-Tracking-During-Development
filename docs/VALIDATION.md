# Validation and visualization workflow

Install the optional plotting dependencies outside Kaggle with:

```bash
python -m pip install -r requirements-validation.txt
```

## Local metric

The competition-aware evaluator is `validation/biohub_local_gt_eval.py`. In a
Kaggle run it reads the generated submission and the four practice GT graphs,
then writes `local_gt_eval.csv`. Its adjusted score is a proxy, not the hidden
leaderboard metric, so compare candidates one change at a time.

Key gates for the current V8 reference:

```text
adjusted_edge_jaccard_proxy > 0.92542
edge_tp                     >= 2061
edge_fp_metric              <= 91
edge_fn_metric              <= 66
```

## Compare experiments

```bash
python validation/compare_runs.py \
  results/v8-baseline results/candidate \
  --output results/comparison.csv
```

## Visual inspection

`visualization/app.py` is a small FastAPI service that steps through a
submission frame by frame, overlaying detections and short trailing tracks on
the Z-max-intensity projection of the matching `.zarr` volume.

```bash
export BIOHUB_SUBMISSION_CSV=submission.csv   # default
export BIOHUB_RESULTS_DIR=results             # dir containing {dataset}.zarr
uvicorn visualization.app:app --reload --port 8000
```

Open `http://localhost:8000`, pick a dataset, and use the ◀ / ▶ buttons,
slider, or arrow keys / space to play through frames. Each color is a stable
lineage id (union-find over the edge graph), so a track's color should not
change frame to frame — a sudden switch is an identity swap.

Inspect at least one sparse and one dense sample. Look for identity swaps,
long jumps, boundary flicker, broken divisions, and short isolated fragments.

## Branch policy

- `main`: submission-ready pipeline only.
- `validation-visualization`: evaluators, plots, diagnostics, and experiments.
- Never commit competition data, model weights, credentials, or generated
  submissions.
