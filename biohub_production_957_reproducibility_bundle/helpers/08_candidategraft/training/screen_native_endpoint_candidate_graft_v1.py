#!/usr/bin/env python3
"""Grouped-video OOF screen for native CandidateGRAFT and EndpointGRAFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


BIO = Path("/home/tweak/bio")
DIRECT_FEATURES = [
    "p1", "p2", "pmax", "pmean", "present", "winner_count", "rank_min", "distance",
]
LEAF_FEATURES = DIRECT_FEATURES + [
    "nearest_final_same_frame_um", "density10", "is_missing_target",
]
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population", type=Path,
        default=BIO / "native_endpoint_candidate_graft_v2",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "native_endpoint_candidate_graft_screen_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=271828)
    return parser.parse_args()


def fold_for(dataset: str, folds: int) -> int:
    digest = hashlib.sha1(dataset.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def make_model(seed: int, positives: int, negatives: int) -> HistGradientBoostingClassifier:
    minimum = max(4, min(16, (positives + negatives) // 20))
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=minimum,
        l2_regularization=2.0,
        random_state=seed,
    )


def threshold_rows(frame: pd.DataFrame, score: np.ndarray) -> list[dict[str, object]]:
    known = frame.label_known.to_numpy(bool)
    positive = frame.label_positive.fillna(False).to_numpy(bool)
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        selected = score >= threshold
        tp = int(np.sum(selected & known & positive))
        fp = int(np.sum(selected & known & ~positive))
        known_positive = int(np.sum(known & positive))
        rows.append({
            "threshold": threshold,
            "selected_all": int(selected.sum()),
            "selected_videos": int(frame.loc[selected, "dataset"].nunique()),
            "known_tp": tp,
            "known_fp": fp,
            "known_precision": tp / max(tp + fp, 1),
            "known_recall": tp / max(known_positive, 1),
        })
    return rows


def crossfit(frame: pd.DataFrame, features: list[str], folds: int, seed: int) -> tuple[np.ndarray, dict]:
    frame = frame.copy()
    frame["fold"] = frame.dataset.map(lambda value: fold_for(str(value), folds))
    known = frame.label_known.to_numpy(bool)
    labels = frame.label_positive.fillna(False).to_numpy(np.int8)
    score = np.zeros(len(frame), np.float32)
    fold_rows = []
    for fold in range(folds):
        fit = known & (frame.fold.to_numpy() != fold)
        validation = frame.fold.to_numpy() == fold
        y = labels[fit]
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"Fold {fold} training labels contain one class: {np.bincount(y)}")
        model = make_model(seed + fold, int(y.sum()), int((1 - y).sum()))
        model.fit(matrix(frame.loc[fit], features), y)
        score[validation] = model.predict_proba(matrix(frame.loc[validation], features))[:, 1]
        fold_rows.append({
            "fold": fold,
            "fit_known": int(fit.sum()),
            "fit_positive": int(y.sum()),
            "validation_rows": int(validation.sum()),
            "validation_known": int((validation & known).sum()),
        })
    known_score = score[known]
    known_label = labels[known]
    summary = {
        "rows": len(frame),
        "known": int(known.sum()),
        "positive": int(known_label.sum()),
        "negative": int(len(known_label) - known_label.sum()),
        "roc_auc": float(roc_auc_score(known_label, known_score)),
        "average_precision": float(average_precision_score(known_label, known_score)),
        "folds": fold_rows,
        "thresholds": threshold_rows(frame, score),
    }
    return score, summary


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    direct = pd.read_parquet(args.population / "direct_edges.parquet")
    leaf = pd.read_parquet(args.population / "endpoint_leaves.parquet")
    leaf["is_missing_target"] = (leaf.transaction_type == "missing_target").astype(np.float32)
    direct_score, direct_summary = crossfit(direct, DIRECT_FEATURES, args.folds, args.seed)
    leaf_score, leaf_summary = crossfit(leaf, LEAF_FEATURES, args.folds, args.seed + 100)
    direct["oof_score"] = direct_score
    leaf["oof_score"] = leaf_score
    direct.to_parquet(args.output / "direct_edges_scored.parquet", index=False)
    leaf.to_parquet(args.output / "endpoint_leaves_scored.parquet", index=False)
    summary = {
        "version": "native-endpoint-candidate-graft-screen-v1",
        "contract": "label-free candidate construction; known-only grouped-video OOF labels",
        "direct": direct_summary,
        "leaf": leaf_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, value in (("direct", direct_summary), ("leaf", leaf_summary)):
        print(f"{name}: AUC={value['roc_auc']:.5f} AP={value['average_precision']:.5f}")
        print(pd.DataFrame(value["thresholds"]).to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
