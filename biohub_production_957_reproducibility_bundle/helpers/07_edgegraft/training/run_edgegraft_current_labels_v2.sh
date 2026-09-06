#!/usr/bin/env bash
set -euo pipefail
WORKSPACE=/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c
PY=/home/tweak/venv-track/bin/python
OUTPUT=/home/tweak/bio/edgegraft_current_labels_v2
cd "$WORKSPACE"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing EdgeGRAFT v2 label cache: $OUTPUT" >&2
  exit 2
fi
$PY -m py_compile scripts/build_edgegraft_current_labels_v2.py
$PY -u scripts/build_edgegraft_current_labels_v2.py --output "$OUTPUT"
echo "DONE: $OUTPUT/manifest.json"
