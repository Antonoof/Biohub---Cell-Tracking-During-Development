#!/usr/bin/env python3
"""Train a sparse-safe OOF gate for EdgeGRAFT v2 atomic transactions.

Only transactions whose exact ordinary-continuation TP count changes are used
as supervision. GT-neutral/unknown transactions are never labeled negative.
The assignment ranker scores are already grouped-video OOF; this second stage
learns whether a proposed complete rewire is likely to have positive net edge
value after accounting for every removed edge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import audit_edgegraft_component_oracle_v1 as edgebase


FEATURES = [
    "top_score", "runner_up_score", "top_margin", "current_present",
    "current_score_filled", "advantage_filled", "remove_count",
    "topology_2to1", "embryo_6bba",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=Path("data/edgegraft_v2_oof_decisions"))
    parser.add_argument("--baseline", type=Path, default=Path("data/ug23_boundary_oof_exact_v1/final_candidate"))
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("data/edgegraft_v2_transaction_gate"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--match-distance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["current_score_filled"] = result.current_score.fillna(-1.0)
    result["advantage_filled"] = result.advantage.fillna(result.top_score + 1.0)
    result["topology_2to1"] = (result.topology == "2to1").astype(np.float32)
    result["embryo_6bba"] = result.dataset.str.startswith("6bba").astype(np.float32)
    return result


def best_threshold(frame: pd.DataFrame, score_column: str) -> dict:
    known = frame[frame.delta_tp != 0].copy()
    values = np.unique(np.r_[0.0, np.quantile(known[score_column], np.linspace(0, 1, 501)), 1.0])
    rows = []
    for threshold in values:
        selected = known[known[score_column] >= threshold]
        rows.append({
            "threshold": float(threshold), "selected": len(selected),
            "wins": int((selected.delta_tp > 0).sum()),
            "losses": int((selected.delta_tp < 0).sum()),
            "net_tp": int(selected.delta_tp.sum()),
        })
    rows.sort(key=lambda row: (row["net_tp"], -row["selected"]), reverse=True)
    return rows[0]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    sys.path.insert(0, "external/kaggle-cell-tracking-competition-patched/src")
    sys.path.insert(0, "external/bio_track_repo/src")
    from biohub_tracking.io import open_dataset

    frames = []
    stems = sorted(path.stem for path in args.baseline.glob("*.geff"))
    for index, stem in enumerate(stems, 1):
        path = args.decisions / f"{stem}.parquet"
        decision = pd.read_parquet(path)
        if decision.empty:
            continue
        graph = edgebase.load_graph(args.baseline / f"{stem}.geff")
        gt = edgebase.load_graph(args.data / f"{stem}.geff")
        dataset = open_dataset(args.data / f"{stem}.zarr", require_tracks=False, load_image=False, device="cpu")
        truth = edgebase.matched_truth(graph, gt, np.asarray(dataset.scale, np.float64), args.match_distance)
        delta = []
        added_truth = []
        removed_truth = []
        for row in decision.itertuples(index=False):
            added = int((int(row.source), int(row.target)) in truth)
            removed = int((int(row.current_source), int(row.target)) in truth)
            if int(row.source_current_target) >= 0:
                removed += int((int(row.source), int(row.source_current_target)) in truth)
            added_truth.append(added)
            removed_truth.append(removed)
            delta.append(added - removed)
        decision["added_truth"] = added_truth
        decision["removed_truth"] = removed_truth
        decision["delta_tp"] = delta
        frames.append(decision)
        if index % 25 == 0 or index == len(stems):
            print(f"labeled {index}/{len(stems)}", flush=True)
    frame = enrich(pd.concat(frames, ignore_index=True))
    frame = frame[~frame.protected].copy()
    known = frame[frame.delta_tp != 0].copy()
    x = known[FEATURES].to_numpy(np.float32)
    y = (known.delta_tp > 0).to_numpy(np.int8)
    weight = np.abs(known.delta_tp.to_numpy(np.float32))
    gate_oof = np.full(len(known), np.nan, np.float32)
    fold_reports = []
    for fold in range(args.folds):
        train = known.fold.to_numpy() != fold
        held = ~train
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2.0,
            random_state=args.seed + fold,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        joblib.dump(model, args.output / f"fold_{fold}.joblib", compress=3)
        gate_oof[held] = model.predict_proba(x[held])[:, 1]
        part = known.loc[held].copy()
        part["gate_oof"] = gate_oof[held]
        fold_reports.append({"fold": fold, "known": len(part), "ungated_net": int(part.delta_tp.sum())})
    if not np.isfinite(gate_oof).all():
        raise RuntimeError("Gate OOF scoring incomplete")
    known["gate_oof"] = gate_oof
    global_best = best_threshold(known, "gate_oof")
    fold_thresholds = []
    for fold in range(args.folds):
        train_best = best_threshold(known[known.fold != fold], "gate_oof")
        held = known[known.fold == fold]
        selected = held[held.gate_oof >= train_best["threshold"]]
        fold_thresholds.append({
            "fold": fold, "threshold": train_best["threshold"],
            "selected": len(selected), "wins": int((selected.delta_tp > 0).sum()),
            "losses": int((selected.delta_tp < 0).sum()), "net_tp": int(selected.delta_tp.sum()),
        })
    threshold = float(np.median([row["threshold"] for row in fold_thresholds]))
    selected = known[known.gate_oof >= threshold]
    known.to_parquet(args.output / "known_transaction_oof.parquet", index=False)
    report = {
        "version": "edgegraft-v2-transaction-gate",
        "features": FEATURES, "known_transactions": len(known),
        "known_wins": int((known.delta_tp > 0).sum()),
        "known_losses": int((known.delta_tp < 0).sum()),
        "ungated_net_tp": int(known.delta_tp.sum()),
        "global_oracle_threshold": global_best,
        "fold_thresholds": fold_thresholds,
        "frozen_median_threshold": threshold,
        "median_threshold_result": {
            "selected": len(selected), "wins": int((selected.delta_tp > 0).sum()),
            "losses": int((selected.delta_tp < 0).sum()), "net_tp": int(selected.delta_tp.sum()),
        },
        "warning": "GT-neutral transactions were excluded from supervision, not labeled negative.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
