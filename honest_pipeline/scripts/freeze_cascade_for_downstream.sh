#!/usr/bin/env bash
# Freeze P1/P2/DC/motion serve, apply OOF motion GEFFs, bank graphs for Model C.
set -euo pipefail
BIO="${BIO:-/data/projects/ryzhichkin/biohub}"
HP="${HP:-$BIO/honest_pipeline}"
PY="${PY:-$BIO/.venv/bin/python}"
TRAIN="${TRAIN:-$BIO/kaggle/input/competitions/biohub-cell-tracking-during-development/train}"
WILLIAM="${WILLIAM:-$BIO/william-duckworth-reproducible-training-pipeline}"
PACK="$BIO/public_models/Biohub Tracking Support Pack"
export PYTHONPATH="${PACK}/repo/src:${BIO}/helpers/01_p1_p2_base/shared_repo/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
cd "$HP"

echo "== freeze serve P1/P2/DC =="
$PY scripts/freeze_sp_0917_oof.py

echo "== P1 graph bank + manifest =="
$PY scripts/prepare_downstream_oof.py

echo "== apply OOF motion (CPU) =="
$PY scripts/apply_motion_oof.py --workers "${WORKERS:-6}" --resume

echo "== re-select graph (P1 vs motion adj) =="
$PY scripts/prepare_downstream_oof.py

echo "== Model C audit layout =="
$PY scripts/materialize_model_c_audit.py

AUDIT="$HP/runs/oof_graphs/model_c_audit"
CACHE="$AUDIT/division_training_cache_v1.npz"
if [[ "${BUILD_EVENT_CACHE:-0}" == "1" ]]; then
  echo "== event cache from selected GEFFs =="
  MC="$BIO/helpers/02_model_c/train_division_pair_model.py"
  [[ -f "$MC" ]] || MC="$WILLIAM/helpers/02_model_c/train_division_pair_model.py"
  $PY "$MC" \
    --data "$TRAIN" \
    --division-audit "$AUDIT/division_candidate_audit_v1" \
    --cache "$CACHE" \
    --cache-only --rebuild-cache
fi

echo "== done =="
cat "$HP/runs/oof_graphs/manifest.json"
