#!/usr/bin/env bash
set -euo pipefail

REPO="/home/tweak/bio_track_repo"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/train_edge_corrector_${STAMP}.log"

cd "$REPO"
export PYTHONPATH=src:scripts

echo "Starting A+B residual edge-corrector cache + training"
echo "Log: $LOG"

/home/tweak/venv-track/bin/python scripts/train_residual_edge_corrector.py \
  --device cuda:0 \
  --edge-threshold 0.54 \
  2>&1 | tee "$LOG"

echo "Corrector: /home/tweak/bio_track_repo/weights/residual_edge_corrector/edge_corrector_best.pt"
