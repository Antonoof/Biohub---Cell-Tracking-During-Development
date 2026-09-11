#!/usr/bin/env bash
set -euo pipefail
WORKSPACE=.
PY=python
OUTPUT=data/edgegraft_current_labels_v2
cd "$WORKSPACE"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing EdgeGRAFT v2 label cache: $OUTPUT" >&2
  exit 2
fi
$PY -m py_compile scripts/build_edgegraft_current_labels_v2.py
$PY -u scripts/build_edgegraft_current_labels_v2.py --output "$OUTPUT"
echo "DONE: $OUTPUT/manifest.json"
