#!/usr/bin/env bash
set -euo pipefail
REPO=external/bio_track_repo
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/train_motion_corrector_${STAMP}.log"
cd "$REPO"
export PYTHONPATH=src:scripts
echo "Starting learned motion-relink cost training"
echo "Log: $LOG"
python scripts/train_motion_cost_corrector.py \
  --device cuda:0 \
  2>&1 | tee "$LOG"
echo "Checkpoint: external/bio_track_repo/weights/motion_cost_corrector/motion_corrector_best.pt"
