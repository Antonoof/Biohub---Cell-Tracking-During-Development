#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=.
PY=python
ROOT=data/ownership_full_population_after_specialist_v1

cd "$WORKSPACE"
if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite: $ROOT" >&2
  exit 2
fi

echo "STEP 1/3: materialize complete full-population OOF ownership policy"
$PY -u scripts/materialize_ownership_full_population_after_specialist_v1.py --output "$ROOT"

echo "STEP 2/3: exact host-patched metric on every changed video"
$PY -u scripts/evaluate_exact_patched_graph_panel.py \
  --pred-dir "$ROOT/graphs" \
  --data-dir data/train \
  --patched-repo external/kaggle-cell-tracking-competition-patched \
  --output "$ROOT/exact"

echo "STEP 3/3: exact global delta from the complete specialist-final control"
$PY - <<'PY'
import json
from pathlib import Path

bio=Path('data')
root=bio/'ownership_full_population_after_specialist_v1'
candidate=json.loads((root/'exact/summary.json').read_text())
stems={str(row['dataset']) for row in candidate['rows']}
control_full=json.loads((bio/'live_v2_global_bundle_ug12_matched_v1/train175/exact/summary.json').read_text())
control_rows=[row for row in control_full['rows'] if str(row['dataset']) in stems]

def agg(rows):
 tp=sum(int(r['division_tp']) for r in rows); fp=sum(int(r['division_fp']) for r in rows); fn=sum(int(r['division_fn']) for r in rows)
 return {'n':len(rows),'tp':tp,'fp':fp,'fn':fn,'j':tp/max(tp+fp+fn,1),'edge':sum(float(r['adj_edge_jaccard']) for r in rows)/len(rows)}

c=agg(control_rows); a=agg(candidate['rows']); g=control_full['summary']
dtp=a['tp']-c['tp']; dfp=a['fp']-c['fp']; dfn=a['fn']-c['fn']
tp=int(g['division_tp'])+dtp; fp=int(g['division_fp'])+dfp; fn=int(g['division_fn'])+dfn
j=tp/max(tp+fp+fn,1)
edge_delta=(a['edge']-c['edge'])*len(stems)/int(g['n_adj'])
out={'control_changed':c,'candidate_changed':a,'global_delta':{'tp':dtp,'fp':dfp,'fn':dfn,'division_j':j-float(g['division_jaccard']),'edge':edge_delta,'composite':edge_delta+.1*(j-float(g['division_jaccard']))},'global_candidate':{'tp':tp,'fp':fp,'fn':fn,'division_j':j,'edge':float(g['adj_edge_jaccard'])+edge_delta}}
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY

echo "DONE output=$ROOT"
