#!/usr/bin/env python3
"""Persist grouped-video EdgeGRAFT rankers for exact OOF graph replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import train_edgegraft_current_ranker_v2 as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("/home/tweak/bio/edgegraft_current_labels_v2"))
    parser.add_argument("--output", type=Path, default=Path("/home/tweak/bio/edgegraft_current_ranker_v3"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    frame = base.load_labels(args.labels)
    features = base.feature_names(frame)
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y = frame.y.to_numpy(np.int8)
    weights = base.target_weights(frame)
    oof = np.full(len(frame), np.nan, np.float32)
    reports = []
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
        joblib.dump(model, args.output / f"fold_{fold}.joblib", compress=3)
        oof[held] = model.predict_proba(x[held])[:, 1]
        report = base.target_metrics(frame.loc[held], oof[held])
        report["fold"] = fold
        reports.append(report)
        print(f"fold {fold}: {json.dumps(report, sort_keys=True)}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF scoring incomplete")
    columns = ["dataset", "video_fold", "component", "source", "target", "y", "is_current_parent"]
    frame[columns].assign(oof_score=oof).to_parquet(args.output / "oof_scores.parquet", index=False)
    report = {
        "version": "edgegraft-current-ranker-v3",
        "features": features,
        "folds": reports,
        "overall": base.target_metrics(frame, oof),
        "model_format": "sklearn HistGradientBoostingClassifier mechanism-test artifact",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
