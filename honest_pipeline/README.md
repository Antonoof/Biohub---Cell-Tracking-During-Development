# Honest Biohub pipeline

**Start here:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — commands, GPUs, ETA, metric paths.

## Contract

- **5-fold GroupKFold by movie** on `train175` (same folds for every stage)
- Save **OOF + logs + weights** every run under `runs/<stage>/`
- Thresholds / blends only via **mean-over-folds on OOF** (never held20/practice)
- **P2 alltrain banned**; Motion default `proposal_nn` (no GT teacher forcing); DeepCenter via `--splits-json`

## Quick start on stage-h200-node2

```bash
export BIO=/data/projects/ryzhichkin/biohub
export HP=$BIO/honest_pipeline
export PY=$BIO/.venv/bin/python
cd $HP && $PY scripts/verify_splits.py

# P1 GKF5 on GPUs 3-7
GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 FOLD_FLAG=--split TAG=p1_gkf5_ep50 \
./scripts/launch_gkf5_parallel.sh -- \
  $PY stages/01_p1_p2/launch_detector_train.py \
    --scheme gkf_movie --epochs 50 --tag p1_gkf5_ep50 --stage 01_p1_detector
```

Audit: [`docs/AUDIT_LEAKS.md`](docs/AUDIT_LEAKS.md)
