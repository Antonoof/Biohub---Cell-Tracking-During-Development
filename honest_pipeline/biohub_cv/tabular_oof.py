#!/usr/bin/env python3
"""Generic grouped-movie OOF fit for sklearn-like tabular stages.

Usage pattern for cardinality / ownership / CandidateGRAFT-style heads:
  - load feature table with a `movie` column and labels
  - call fit_oof_sklearn(...)
  - write run via RunLogger
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from biohub_cv.logging_utils import RunLogger
from biohub_cv.oof import save_oof_parquet, select_threshold_on_inner_oof
from biohub_cv.splits import CanonicalSplits, FoldSpec, embryo_of


def _subset(df: pd.DataFrame, movies: Sequence[str], movie_col: str) -> pd.DataFrame:
    return df.loc[df[movie_col].astype(str).isin(set(movies))].copy()


def fit_oof_sklearn(
    df: pd.DataFrame,
    *,
    folds: Sequence[FoldSpec],
    feature_cols: Sequence[str],
    label_col: str,
    movie_col: str,
    model_factory: Callable[[int], Any],
    predict_proba: bool = True,
    unknown_mask: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, list[Any]]:
    """Fit one model per fold; score only that fold's val movies (true OOF).

    Rows with ``unknown_mask`` True are scored but never used in ``fit``.
    """
    if unknown_mask is None:
        unknown_mask = np.zeros(len(df), dtype=bool)
    else:
        unknown_mask = np.asarray(unknown_mask, dtype=bool)
    known = df.loc[~unknown_mask].copy()
    parts: list[pd.DataFrame] = []
    models: list[Any] = []

    for fold_i, fold in enumerate(folds):
        train_df = _subset(known, fold.train_movies, movie_col)
        # Score all rows (known+unknown) whose movie is in val.
        score_df = _subset(df, fold.val_movies, movie_col)
        if train_df.empty:
            raise RuntimeError(f"{fold.fold_id}: empty train")
        if score_df.empty:
            raise RuntimeError(f"{fold.fold_id}: empty val score set")

        model = model_factory(fold_i)
        x_tr = train_df[list(feature_cols)].to_numpy(np.float32)
        y_tr = train_df[label_col].to_numpy()
        model.fit(x_tr, y_tr)
        models.append(model)

        x_va = score_df[list(feature_cols)].to_numpy(np.float32)
        if predict_proba:
            scores = model.predict_proba(x_va)[:, 1]
        else:
            scores = np.asarray(model.predict(x_va), dtype=np.float64)

        out = score_df[[movie_col]].copy()
        out = out.rename(columns={movie_col: "movie"})
        out["fold_id"] = fold.fold_id
        out["scheme"] = fold.scheme
        out["y_score"] = scores
        if label_col in score_df.columns:
            out["y_true"] = score_df[label_col].to_numpy()
        out["embryo"] = out["movie"].map(embryo_of)
        # Keep index alignment for joins.
        out.index = score_df.index
        parts.append(out)

    oof = pd.concat(parts, axis=0).sort_index()
    return oof, models


def run_tabular_stage(
    *,
    stage: str,
    df: pd.DataFrame,
    splits: CanonicalSplits,
    feature_cols: Sequence[str],
    label_col: str,
    movie_col: str,
    model_factory: Callable[[int], Any],
    run_dir: Path,
    scheme: str = "gkf_movie",
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    unknown_mask: np.ndarray | pd.Series | None = None,
    extra_config: dict | None = None,
) -> dict:
    """End-to-end: OOF fit on train175 GKF (or LOEO), log, save parquet, freeze threshold."""
    logger = RunLogger(run_dir, stage=stage, config=extra_config or {})
    if scheme == "gkf_movie":
        folds = list(splits.gkf5_train175)
        pool = _subset(df, splits.train175, movie_col)
    elif scheme == "loeo":
        folds = list(splits.loeo_folds)
        pool = _subset(df, splits.train175, movie_col)
    else:
        raise ValueError(f"Unsupported scheme {scheme}")

    # Align unknown mask to pool if provided on full df.
    if unknown_mask is not None:
        unknown_mask = np.asarray(unknown_mask, dtype=bool)
        if len(unknown_mask) != len(df):
            raise ValueError("unknown_mask length must match df")
        unknown_mask = unknown_mask[pool.index.to_numpy()]

    logger.log(f"stage={stage} scheme={scheme} rows={len(pool)} folds={len(folds)}")
    oof, models = fit_oof_sklearn(
        pool.reset_index(drop=True),
        folds=folds,
        feature_cols=feature_cols,
        label_col=label_col,
        movie_col=movie_col,
        model_factory=model_factory,
        unknown_mask=unknown_mask,
    )

    oof_path = logger.path(f"oof_{scheme}.parquet")
    save_oof_parquet(oof_path, oof, stage=stage, scheme=scheme)

    summary: dict[str, Any] = {
        "stage": stage,
        "scheme": scheme,
        "n_oof_rows": len(oof),
        "n_oof_movies": int(oof["movie"].nunique()),
        "oof_path": str(oof_path),
    }

    if metric_fn is not None and "y_true" in oof.columns:
        labeled = oof.dropna(subset=["y_true"])
        thr = select_threshold_on_inner_oof(
            labeled,
            metric_fn=metric_fn,
        )
        (logger.path("threshold_sweep.json")).write_text(json.dumps(thr, indent=2) + "\n")
        summary["threshold"] = thr["threshold"]
        summary["threshold_mean_metric"] = thr["mean_metric"]
        logger.log(f"frozen threshold={thr['threshold']} mean_metric={thr['mean_metric']:.6f}")

    # Persist fold models if they support joblib.
    try:
        import joblib

        model_dir = logger.path("folds")
        for i, model in enumerate(models):
            joblib.dump(model, model_dir / f"fold_{i}.joblib")
        summary["n_models"] = len(models)
    except Exception as exc:  # noqa: BLE001
        logger.log(f"model save skipped: {exc}")

    logger.write_summary(summary)
    return summary
