#!/usr/bin/env python3
"""Compare Biohub local_gt_eval/run_stats outputs across experiment folders."""
from __future__ import annotations
import argparse,csv
from pathlib import Path

METRICS=("pred_nodes","pred_edges","edge_tp","edge_fp_metric","edge_fn_metric",
         "metric_edge_jaccard_proxy","adjusted_edge_jaccard_proxy",
         "aggregate_adjusted_edge_jaccard_debug","pred_division_sources")

def total(path:Path):
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    return next(r for r in rows if r.get("dataset")=="TOTAL")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("runs",nargs="+",help="Directories containing local_gt_eval.csv")
    p.add_argument("--baseline",type=int,default=0,help="Run index used for deltas")
    p.add_argument("--output",type=Path,default=Path("run_comparison.csv"))
    a=p.parse_args();rows=[]
    for value in a.runs:
        root=Path(value);record=total(root/"local_gt_eval.csv")
        rows.append({"run":root.name,**{k:record.get(k,"") for k in METRICS}})
    baseline=rows[a.baseline]
    for row in rows:
        for key in METRICS:
            try: row[f"delta_{key}"]=float(row[key])-float(baseline[key])
            except (TypeError,ValueError): row[f"delta_{key}"]=""
    fields=list(rows[0]);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f"{'run':34s} {'AdjJ':>9s} {'dAdjJ':>9s} {'TP':>6s} {'FP':>5s} {'FN':>5s} {'nodes':>8s}")
    for r in rows:
        print(f"{r['run'][:34]:34s} {float(r['adjusted_edge_jaccard_proxy']):9.6f} "
              f"{float(r['delta_adjusted_edge_jaccard_proxy']):+9.6f} {int(float(r['edge_tp'])):6d} "
              f"{int(float(r['edge_fp_metric'])):5d} {int(float(r['edge_fn_metric'])):5d} "
              f"{int(float(r['pred_nodes'])):8d}")
    print("wrote",a.output)
if __name__=="__main__":main()
