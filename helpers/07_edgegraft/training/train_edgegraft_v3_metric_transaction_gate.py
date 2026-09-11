#!/usr/bin/env python3
"""Train a grouped-OOF EdgeGRAFT gate on official edge-Jaccard utility.

The earlier transaction label counted only recovered truth edges.  The metric
also rewards removal of valid false-positive edges.  Here each transaction is
labeled by its exact local change in matched TP and valid predicted-edge count
under the frozen node matching.  Unmatched/metric-neutral transactions remain
outside supervision; they are not declared biological negatives.
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

import edgegraft_metric_utils as edgebase
SUMMARY_FEATURES = [
    "top_score", "runner_up_score", "top_margin", "current_present",
    "current_score_filled", "advantage_filled", "remove_count",
    "topology_2to1", "embryo_6bba",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=Path("data/edgegraft_v2_rich_oof_decisions"))
    parser.add_argument("--baseline", type=Path, default=Path("data/ug23_boundary_oof_exact_v1/final_candidate"))
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--baseline-exact", type=Path, default=Path("data/ug23_boundary_oof_exact_v1/exact_candidate/per_video.csv"))
    parser.add_argument("--rich-report", type=Path, default=Path("data/edgegraft_v2_rich_transaction_gate/report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/edgegraft_v3_metric_transaction_gate"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--match-distance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def edge_valid(edge: tuple[int, int], pred_to_gt: dict[int, int], gt_out: dict[int, int], gt_in: dict[int, int]) -> int:
    source, target = edge
    source_gt = pred_to_gt.get(int(source), -1)
    target_gt = pred_to_gt.get(int(target), -1)
    return int((source_gt >= 0 and gt_out.get(source_gt, 0) > 0)
               or (target_gt >= 0 and gt_in.get(target_gt, 0) > 0))


def label_transactions(args: argparse.Namespace) -> pd.DataFrame:
    sys.path.insert(0, "external/kaggle-cell-tracking-competition-patched/src")
    sys.path.insert(0, "external/bio_track_repo/src")
    from biohub_tracking.io import open_dataset
    from tracking_cellmot.division_metrics import _match_full

    exact = pd.read_csv(args.baseline_exact).set_index("dataset")
    frames = []
    paths = sorted(args.decisions.glob("*.parquet"))
    for index, path in enumerate(paths, 1):
        stem = path.stem
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        graph = edgebase.load_graph(args.baseline / f"{stem}.geff")
        gt = edgebase.load_graph(args.data / f"{stem}.geff")
        dataset = open_dataset(args.data / f"{stem}.zarr", require_tracks=False, load_image=False, device="cpu")
        scale = np.asarray(dataset.scale, np.float64)
        matched = _match_full(graph, gt, scale, args.match_distance)
        pred_to_gt = {
            int(row["node_id"]): int(row["match_node_id"])
            for row in matched.node_attrs().to_dicts()
            if int(row["match_node_id"]) >= 0
        }
        gt_out = {int(node): int(gt.out_degree(int(node))) for node in gt.node_ids()}
        gt_in = {int(node): int(gt.in_degree(int(node))) for node in gt.node_ids()}
        truth = edgebase.matched_truth(graph, gt, scale, args.match_distance)
        base_tp = int(exact.loc[stem, "edge_tp"])
        denominator = int(exact.loc[stem, "edge_tp"] + exact.loc[stem, "edge_fp"] + exact.loc[stem, "edge_fn"])
        delta_tp = []
        delta_valid = []
        delta_fp = []
        utility = []
        metric_delta = []
        for row in frame.itertuples(index=False):
            add = (int(row.source), int(row.target))
            remove = {(int(row.current_source), int(row.target))}
            if int(row.source_current_target) >= 0:
                remove.add((int(row.source), int(row.source_current_target)))
            dtp = int(add in truth) - sum(int(edge in truth) for edge in remove)
            dvalid = edge_valid(add, pred_to_gt, gt_out, gt_in) - sum(
                edge_valid(edge, pred_to_gt, gt_out, gt_in) for edge in remove
            )
            dfp = dvalid - dtp
            # Exact sign of the per-video Jaccard change with fixed nodes.
            util = int(dtp * denominator - base_tp * dfp)
            before = base_tp / denominator if denominator else 0.0
            after_denom = denominator + dfp
            after = (base_tp + dtp) / after_denom if after_denom > 0 else 0.0
            delta_tp.append(dtp)
            delta_valid.append(dvalid)
            delta_fp.append(dfp)
            utility.append(util)
            metric_delta.append(after - before)
        frame["metric_delta_tp"] = delta_tp
        frame["metric_delta_valid"] = delta_valid
        frame["metric_delta_fp"] = delta_fp
        frame["metric_utility"] = utility
        frame["metric_delta"] = metric_delta
        frames.append(frame)
        if index % 25 == 0 or index == len(paths):
            print(f"labeled {index}/{len(paths)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def best_threshold(frame: pd.DataFrame, score_column: str) -> dict:
    values = np.unique(np.r_[0.0, np.quantile(frame[score_column], np.linspace(0, 1, 501)), 1.0])
    rows = []
    for threshold in values:
        selected = frame[frame[score_column] >= threshold]
        rows.append({
            "threshold": float(threshold), "selected": len(selected),
            "wins": int((selected.metric_utility > 0).sum()),
            "losses": int((selected.metric_utility < 0).sum()),
            "utility": int(selected.metric_utility.sum()),
            "metric_delta_sum": float(selected.metric_delta.sum()),
        })
    rows.sort(key=lambda row: (row["utility"], -row["selected"]), reverse=True)
    return rows[0]


def feature_families(rich_features: list[str]) -> dict[str, list[str]]:
    new_only = [feature for feature in rich_features if not feature.startswith("old_") and not feature.startswith("delta_")]
    return {
        "summary": list(SUMMARY_FEATURES),
        "summary_new": new_only,
        "summary_contrast": rich_features,
    }


def train_family(frame: pd.DataFrame, features: list[str], args: argparse.Namespace, family: str):
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y = (frame.metric_utility > 0).to_numpy(np.int8)
    raw_weight = np.abs(frame.metric_delta.to_numpy(np.float64))
    median = float(np.median(raw_weight[raw_weight > 0]))
    weight = np.clip(raw_weight / max(median, 1e-12), 0.25, 12.0).astype(np.float32)
    oof = np.full(len(frame), np.nan, np.float32)
    models = []
    for fold in range(args.folds):
        train = frame.fold.to_numpy() != fold
        held = ~train
        model = HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=5.0,
            random_state=args.seed + 100 + fold,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        oof[held] = model.predict_proba(x[held])[:, 1]
        models.append(model)
    work = frame.copy()
    work["gate_oof"] = oof
    folds = []
    thresholds = []
    for fold in range(args.folds):
        fit_best = best_threshold(work[work.fold != fold], "gate_oof")
        held = work[work.fold == fold]
        selected = held[held.gate_oof >= fit_best["threshold"]]
        thresholds.append(float(fit_best["threshold"]))
        folds.append({
            "fold": fold, "threshold": float(fit_best["threshold"]),
            "selected": len(selected), "wins": int((selected.metric_utility > 0).sum()),
            "losses": int((selected.metric_utility < 0).sum()),
            "utility": int(selected.metric_utility.sum()),
            "metric_delta_sum": float(selected.metric_delta.sum()),
        })
    threshold = float(np.median(thresholds))
    selected = work[work.gate_oof >= threshold]
    report = {
        "family": family, "feature_count": len(features), "frozen_median_threshold": threshold,
        "selected": len(selected), "wins": int((selected.metric_utility > 0).sum()),
        "losses": int((selected.metric_utility < 0).sum()),
        "utility": int(selected.metric_utility.sum()),
        "metric_delta_sum": float(selected.metric_delta.sum()),
        "folds": folds, "positive_folds": int(sum(row["utility"] > 0 for row in folds)),
        "minimum_fold_utility": int(min(row["utility"] for row in folds)),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report, models, oof


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    all_rows = label_transactions(args)
    all_rows.to_parquet(args.output / "all_transaction_metric_labels.parquet", index=False)
    frame = all_rows[(~all_rows.protected) & (all_rows.metric_utility != 0)].copy()
    rich_features = list(json.loads(args.rich_report.read_text())["features"])
    results = []
    artifacts = {}
    for family, features in feature_families(rich_features).items():
        report, models, oof = train_family(frame, features, args, family)
        results.append(report)
        artifacts[family] = (features, models, oof)
    stable = [row for row in results if row["positive_folds"] == args.folds]
    pool = stable if stable else results
    best = max(pool, key=lambda row: (row["utility"], row["minimum_fold_utility"], -row["feature_count"]))
    features, models, oof = artifacts[best["family"]]
    for fold, model in enumerate(models):
        joblib.dump(model, args.output / f"fold_{fold}.joblib", compress=3)
    keep = ["dataset", "fold", "source", "target", "current_source", "source_current_target",
            "topology", "metric_delta_tp", "metric_delta_fp", "metric_utility", "metric_delta"]
    frame[keep].assign(gate_oof=oof).to_parquet(args.output / "metric_transaction_oof.parquet", index=False)
    payload = {
        "version": "edgegraft-v3-metric-transaction-gate",
        "all_transactions": len(all_rows), "metric_non_neutral": len(frame),
        "metric_positive": int((frame.metric_utility > 0).sum()),
        "metric_negative": int((frame.metric_utility < 0).sum()),
        "families": results, "selected_family": best["family"],
        "features": features, "frozen_median_threshold": best["frozen_median_threshold"],
        "warning": "Metric-neutral/unmatched transactions excluded from supervision, not biological negatives.",
    }
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selected_family": best["family"], "utility": best["utility"],
                      "threshold": best["frozen_median_threshold"],
                      "non_neutral": len(frame)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
