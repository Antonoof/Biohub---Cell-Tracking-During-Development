#!/usr/bin/env python3
"""Ownership OOF with canonical GKF5 folds + inner-fold threshold + RunLogger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from biohub_cv.logging_utils import RunLogger, new_run_dir  # noqa: E402
from biohub_cv.oof import save_oof_parquet, select_threshold_on_inner_oof  # noqa: E402
from biohub_cv.splits import load_canonical_splits  # noqa: E402

# Reuse William feature lists if available.
WILLIAM_FEAT = (
    ROOT.parent
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers"
    / "06_full_population_ownership"
)
LOCAL_FEAT = ROOT.parent / "helpers" / "06_full_population_ownership"
FEAT_ROOT = WILLIAM_FEAT if WILLIAM_FEAT.exists() else LOCAL_FEAT
sys.path.insert(0, str(FEAT_ROOT))
from benchmark_ownership_feature_groups_v3 import RANK, TOP  # noqa: E402


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame[[*TOP, *RANK]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(np.float32)
    )


def jaccard_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / max(tp + fp + fn, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--tag", default="ownership_gkf5")
    args = p.parse_args()

    splits = load_canonical_splits(ROOT / "splits" / "canonical_splits.json")
    run_dir = new_run_dir(ROOT / "runs", "08_ownership", args.tag)
    logger = RunLogger(run_dir, stage="08_ownership", config={"bank": str(args.bank)})

    full = pd.read_parquet(args.bank)
    if "panel" in full.columns:
        train = full.loc[full.panel.eq("train175")].copy()
    else:
        train = full.loc[full.dataset.astype(str).isin(splits.train175)].copy()
    known = train.loc[train.source_label.notna()].copy().reset_index(drop=True)
    known["target"] = (
        known.source_label.eq(1) & known.best_pair_label.eq(1)
    ).astype(np.int8)

    parts = []
    models = []
    for fold_i, fold in enumerate(splits.gkf5_train175):
        fit = known.loc[known.dataset.astype(str).isin(fold.train_movies)]
        score_movies = set(fold.val_movies)
        local = train.loc[train.dataset.astype(str).isin(score_movies)].copy()
        if fit.empty or local.empty:
            raise RuntimeError(f"{fold.fold_id}: empty fit/score")
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=20,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=324459 + fold_i,
        )
        model.fit(matrix(fit), fit.target.to_numpy(np.int8))
        models.append(model)
        local = local.copy()
        local["y_score"] = model.predict_proba(matrix(local))[:, 1]
        local["fold_id"] = fold.fold_id
        local["scheme"] = "gkf_movie"
        local["movie"] = local.dataset.astype(str)
        if "target" not in local.columns:
            local["y_true"] = np.where(
                local.source_label.notna(),
                (local.source_label.eq(1) & local.best_pair_label.eq(1)).astype(np.float32),
                np.nan,
            )
        else:
            local["y_true"] = local["target"]
        parts.append(local)
        joblib.dump(model, logger.path("folds") / f"fold_{fold_i}.joblib")
        logger.log(f"{fold.fold_id}: fit={len(fit)} scored={len(local)}")

    oof = pd.concat(parts, axis=0)
    oof_path = logger.path("oof_gkf_movie.parquet")
    save_oof_parquet(
        oof_path,
        oof[["movie", "fold_id", "scheme", "y_score", "y_true"]].copy(),
        stage="08_ownership",
        scheme="gkf_movie",
    )
    oof.to_parquet(logger.path("all_source_scores.parquet"), index=False)

    labeled = oof.dropna(subset=["y_true"])
    thr = select_threshold_on_inner_oof(
        labeled.rename(columns={"y_true": "y_true"}),
        metric_fn=jaccard_binary,
    )
    (logger.path("threshold_sweep.json")).write_text(json.dumps(thr, indent=2) + "\n")
    logger.write_summary(
        {
            "n_oof": len(oof),
            "n_labeled": len(labeled),
            "threshold": thr["threshold"],
            "mean_metric": thr["mean_metric"],
            "oof_path": str(oof_path),
        }
    )
    logger.log(f"threshold={thr['threshold']} mean_jaccard={thr['mean_metric']:.6f}")


if __name__ == "__main__":
    main()
