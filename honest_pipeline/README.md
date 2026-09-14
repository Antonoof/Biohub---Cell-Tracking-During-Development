# Honest Biohub pipeline

**Start here:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — commands, GPUs, ETA, metric paths.

## Contract

- **5-fold GroupKFold by movie** on `train175` (same folds for every stage)
- Save **OOF + logs + weights** every run under `runs/<stage>/`
- Thresholds / blends only via **mean-over-folds on OOF** (never held20/practice)
- **Detector = public 0_917 Exp203 stack** (`helpers/12_classical_unet3d/run_exp203_to_geff.py`). Old P1/P2 unet-transformer dropped.
- Motion default `proposal_nn` (no GT teacher forcing); DeepCenter via `--splits-json`

## Quick start on stage-h200-node2

```bash
export BIO=/data/projects/ryzhichkin/biohub
export HP=$BIO/honest_pipeline
export PY=$BIO/.venv/bin/python
cd $HP && $PY scripts/verify_splits.py

# Freeze 0_917 + DeepCenter OOF gate
$PY scripts/freeze_0917_stack_oof.py

# Export 0_917 nodes as motion proposals
$PY scripts/export_exp203_proposals.py \
  --geff-dir $HP/runs/p1_candidate_compare/kaggle_train_all/classical_exp203 \
  --data-dir $BIO/kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --out-dir $BIO/data/exp203_0917_proposals
```

Audit: [`docs/AUDIT_LEAKS.md`](docs/AUDIT_LEAKS.md)
