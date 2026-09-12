#!/usr/bin/env bash
# Launch one command per GKF fold on separate GPUs, wait, then optionally assemble.
# Usage:
#   GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 \
#   ./scripts/launch_gkf5_parallel.sh --tag p1_gkf5 -- \
#     python stages/01_p1_p2/launch_detector_train.py --scheme gkf_movie --epochs 50 --tag p1
#
# The fold index is appended as: ... --split $FOLD   (override with FOLD_FLAG=--fold)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-3,4,5,6,7}"
FOLDS="${FOLDS:-0,1,2,3,4}"
FOLD_FLAG="${FOLD_FLAG:---split}"
TAG="${TAG:-run}"
STAGE_DIR="${STAGE_DIR:-runs/_parallel}"
ASSEMBLE_CMD="${ASSEMBLE_CMD:-}"

mkdir -p "$STAGE_DIR/$TAG"
LOGDIR="$STAGE_DIR/$TAG"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a FOLD_ARR <<< "$FOLDS"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 -- cmd with fold placeholder" >&2
  exit 2
fi
# strip leading --
if [[ "$1" == "--" ]]; then shift; fi
BASE_CMD=("$@")

PIDS=()
i=0
for fold in "${FOLD_ARR[@]}"; do
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  logfile="$LOGDIR/fold_${fold}_gpu${gpu}.log"
  echo "[launch] fold=$fold gpu=$gpu log=$logfile"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "CMD: ${BASE_CMD[*]} $FOLD_FLAG $fold"
    "${BASE_CMD[@]}" $FOLD_FLAG "$fold"
  ) >"$logfile" 2>&1 &
  PIDS+=($!)
  i=$((i + 1))
done

ec=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[error] pid $pid failed" >&2
    ec=1
  fi
done

if [[ $ec -ne 0 ]]; then
  echo "One or more folds failed. Logs in $LOGDIR" >&2
  exit $ec
fi

echo "[ok] all folds finished. Logs: $LOGDIR"
if [[ -n "$ASSEMBLE_CMD" ]]; then
  echo "[assemble] $ASSEMBLE_CMD"
  eval "$ASSEMBLE_CMD"
fi
