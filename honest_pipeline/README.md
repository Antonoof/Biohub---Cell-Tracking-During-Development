# Honest Biohub pipeline

**Start here:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — commands, GPUs, ETA, metric paths.

## Contract

- **5-fold GroupKFold by movie** on `train175` (same folds for every stage)
- Save **OOF + logs + weights** every run under `runs/<stage>/`
- Thresholds / blends only via **mean-over-folds on OOF** (never held20/practice)
- **P1 = public Support Pack**, **P2 = public 0_917 Exp203**. Old honest unet-transformer P1/P2 dropped.
- Motion default `proposal_nn` (no GT teacher forcing); DeepCenter via `--splits-json`. Proposals = fused SP∪0_917.

## Quick start on stage-h200-node2

```bash
export BIO=/data/projects/ryzhichkin/biohub
export HP=$BIO/honest_pipeline
export PY=$BIO/.venv/bin/python
cd $HP && $PY scripts/verify_splits.py

# Freeze P1=Support Pack + P2=0_917 + DeepCenter OOF gate
$PY scripts/assemble_public_detector_oof.py
$PY scripts/freeze_sp_0917_oof.py

# Export fused A/B proposals for motion
$PY scripts/export_ab_geff_proposals.py \
  --p1-geff-dir $HP/runs/p1_candidate_compare/kaggle_train_all/support_pack \
  --p2-geff-dir $HP/runs/p1_candidate_compare/kaggle_train_all/classical_exp203 \
  --data-dir $BIO/kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --out-dir $BIO/data/ab_proposals_sp_0917
```

Audit: [`docs/AUDIT_LEAKS.md`](docs/AUDIT_LEAKS.md)
