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

Render detections and a short trajectory window:

```bash
python visualization/visualize_submission.py \
  --submission results/candidate/submission.csv \
  --dataset 6bba_05db0fb1 \
  --frame 50 --window 6 \
  --zarr /path/to/6bba_05db0fb1.zarr \
  --output results/candidate/trajectory_t50.png
```

Inspect at least one sparse and one dense sample. Look for identity swaps,
long jumps, boundary flicker, broken divisions, and short isolated fragments.

## Branch policy

- `main`: submission-ready pipeline only.
- `validation-visualization`: evaluators, plots, diagnostics, and experiments.
- Never commit competition data, model weights, credentials, or generated
  submissions.
