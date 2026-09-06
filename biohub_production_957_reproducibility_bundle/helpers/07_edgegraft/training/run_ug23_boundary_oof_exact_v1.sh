#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c
PY=/home/tweak/venv-track/bin/python
ROOT=/home/tweak/bio/ug23_boundary_oof_exact_v1
CONTROL=/home/tweak/bio/ug3_joint_prelock_oof_v1/prefinal_joint
CANDIDATE=/home/tweak/bio/ug23_oof_exact_v1/prefinal
NOTEBOOK=/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c/best-951-boundary-ug23-ownership-v1.ipynb
PATCHED=/home/tweak/bio/kaggle-cell-tracking-competition-patched

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing output: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"

for name in control candidate; do
  if [[ "$name" == control ]]; then input="$CONTROL"; else input="$CANDIDATE"; fi
  "$PY" -u scripts/finalize_public951_ug3_graphs_v1.py \
    --input "$input" \
    --output "$ROOT/final_$name" \
    --notebook "$NOTEBOOK" \
    --data /home/tweak/bio/train
  "$PY" -u scripts/evaluate_exact_patched_graph_panel.py \
    --pred-dir "$ROOT/final_$name" \
    --data-dir /home/tweak/bio/train \
    --patched-repo "$PATCHED" \
    --output "$ROOT/exact_$name"
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/home/tweak/bio/ug23_boundary_oof_exact_v1")
result = {
    name: json.loads((root / f"exact_{name}" / "summary.json").read_text())["summary"]
    for name in ("control", "candidate")
}
keys = (
    "score", "adj_edge_jaccard", "division_jaccard",
    "division_tp", "division_fp", "division_fn",
)
compact = {name: {key: row[key] for key in keys} for name, row in result.items()}
compact["delta"] = {
    key: compact["candidate"][key] - compact["control"][key]
    for key in keys[:3]
}
(root / "comparison.json").write_text(json.dumps(compact, indent=2) + "\n")
print(json.dumps(compact, indent=2))
PY

echo "DONE $ROOT/comparison.json"
