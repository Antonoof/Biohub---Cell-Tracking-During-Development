#!/usr/bin/env python3
"""Grouped-video EdgeGRAFT v2 assignment-ranker mechanism test.

This stage answers only whether a current-population learner can rank the true
parent above competing P1/P2 candidates more often than the frozen graph.  It
uses five grouped-video folds and writes OOF scores.  It does not mutate or
promote a graph; exact atomic component replay is a later gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


METADATA = {
    "dataset", "partition", "fold", "video_fold", "component", "group",
    "source", "target", "y",
}

RELATIVE_MAX = [
    "p1_probability", "p2_probability", "model_probability_max",
    "model_probability_mean", "raw_edge_probability",
    "ctx_p1_path_min", "ctx_p1_path_mean", "ctx_p1_path_geomean",
    "ctx_p2_path_min", "ctx_p2_path_mean", "ctx_p2_path_geomean",
]
RELATIVE_MIN = [
    "distance_um", "velocity_residual_um", "geom_distance_um_exact",
    "geom_incoming_residual_um", "geom_outgoing_residual_um",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/edgegraft_current_labels_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/edgegraft_current_ranker_v2"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def add_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby(["dataset", "target"], sort=False)
    for column in RELATIVE_MAX:
        if column not in result:
            continue
        maximum = groups[column].transform("max")
        result[f"rankrel_{column}_to_max"] = result[column] - maximum
        result[f"rankrel_{column}_rank"] = groups[column].rank(method="average", ascending=False)
    for column in RELATIVE_MIN:
        if column not in result:
            continue
        minimum = groups[column].transform("min")
        result[f"rankrel_{column}_from_min"] = result[column] - minimum
        result[f"rankrel_{column}_rank"] = groups[column].rank(method="average", ascending=True)
    return result


def load_labels(root: Path) -> pd.DataFrame:
    frames = []
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No label files found: {root}")
    for index, path in enumerate(paths, 1):
        frame = pd.read_parquet(path)
        if len(frame):
            frames.append(frame)
        if index % 25 == 0 or index == len(paths):
            print(f"loaded {index}/{len(paths)}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result = add_relative_features(result)
    return result


def feature_names(frame: pd.DataFrame) -> list[str]:
    output = []
    for column in frame.columns:
        if column in METADATA:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            output.append(column)
    return output


def target_weights(frame: pd.DataFrame) -> np.ndarray:
    # Every known target contributes equal total weight. Positive and all
    # competing negatives receive half of that target's mass each.
    negative_count = frame.groupby(["dataset", "target"], sort=False).y.transform(lambda x: max(int((x == 0).sum()), 1))
    return np.where(frame.y.to_numpy() == 1, 0.5, 0.5 / negative_count.to_numpy()).astype(np.float32)


def target_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, int | float]:
    work = frame[["dataset", "target", "source", "y", "is_current_parent"]].copy()
    work["score"] = score
    chosen = work.loc[work.groupby(["dataset", "target"], sort=False).score.idxmax()].copy()
    targets = len(chosen)
    model_correct = int(chosen.y.sum())
    base_correct = int(
        work.loc[(work.y == 1) & (work.is_current_parent == 1), ["dataset", "target"]]
        .drop_duplicates().shape[0]
    )
    chosen["base_correct"] = chosen.set_index(["dataset", "target"]).index.map(
        work.loc[(work.y == 1) & (work.is_current_parent == 1)]
        .set_index(["dataset", "target"]).index.unique().__contains__
    )
    recovered = int(((chosen.y == 1) & ~chosen.base_correct).sum())
    lost = int(((chosen.y == 0) & chosen.base_correct).sum())
    return {
        "targets": targets,
        "baseline_correct": base_correct,
        "baseline_accuracy": base_correct / max(targets, 1),
        "model_correct": model_correct,
        "model_accuracy": model_correct / max(targets, 1),
        "net_correct": model_correct - base_correct,
        "recovered": recovered,
        "lost": lost,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    frame = load_labels(args.labels)
    features = feature_names(frame)
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y = frame.y.to_numpy(np.int8)
    weights = target_weights(frame)
    oof = np.full(len(frame), np.nan, np.float32)
    fold_reports = []
    for fold in range(args.folds):
        train = frame.video_fold.to_numpy() != fold
        held = ~train
        model = HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_iter=args.iterations,
            max_leaf_nodes=args.leaves,
            l2_regularization=args.l2,
            min_samples_leaf=40,
            max_bins=255,
            random_state=args.seed + fold,
        )
        model.fit(x[train], y[train], sample_weight=weights[train])
        oof[held] = model.predict_proba(x[held])[:, 1]
        metrics = target_metrics(frame.loc[held], oof[held])
        metrics["fold"] = fold
        fold_reports.append(metrics)
        print(f"fold {fold}: {json.dumps(metrics, sort_keys=True)}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF scoring incomplete")
    overall = target_metrics(frame, oof)
    frame[["dataset", "video_fold", "component", "source", "target", "y", "is_current_parent"]].assign(oof_score=oof).to_parquet(
        args.output / "oof_scores.parquet", index=False,
    )
    report = {
        "version": "edgegraft-current-ranker-v2",
        "rows": len(frame), "features": features, "folds": fold_reports,
        "overall": overall,
        "warning": "Mechanism test only; requires atomic component and exact graph replay before promotion.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
