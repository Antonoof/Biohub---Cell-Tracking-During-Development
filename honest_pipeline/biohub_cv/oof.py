"""Durable OOF prediction I/O and nested threshold selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


REQUIRED_OOF_COLUMNS = (
    "movie",
    "fold_id",
    "scheme",
    "y_true",
    "y_pred",
    "y_score",
)


def save_oof_parquet(
    path: Path | str,
    frame: pd.DataFrame,
    *,
    stage: str,
    scheme: str,
    extra_meta: dict | None = None,
) -> Path:
    """Write OOF predictions. Requires movie + fold_id + scores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in ("movie", "fold_id") if c not in frame.columns]
    if missing:
        raise ValueError(f"OOF frame missing columns: {missing}")
    if "scheme" not in frame.columns:
        frame = frame.copy()
        frame["scheme"] = scheme
    frame.to_parquet(path, index=False)
    meta = {
        "stage": stage,
        "scheme": scheme,
        "n_rows": int(len(frame)),
        "n_movies": int(frame["movie"].nunique()),
        "folds": sorted(frame["fold_id"].astype(str).unique().tolist()),
        "columns": list(frame.columns),
        **(extra_meta or {}),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def append_oof_rows(
    parts: list[pd.DataFrame],
    *,
    movie: Sequence[str],
    fold_id: str,
    scheme: str,
    y_score: np.ndarray | Sequence[float],
    y_true: np.ndarray | Sequence[float] | None = None,
    y_pred: np.ndarray | Sequence[float] | None = None,
    extra: dict[str, Sequence] | None = None,
) -> None:
    n = len(movie)
    row = {
        "movie": list(movie),
        "fold_id": [fold_id] * n,
        "scheme": [scheme] * n,
        "y_score": np.asarray(y_score, dtype=np.float64),
    }
    if y_true is not None:
        row["y_true"] = np.asarray(y_true)
    if y_pred is not None:
        row["y_pred"] = np.asarray(y_pred)
    if extra:
        for k, v in extra.items():
            row[k] = list(v) if not hasattr(v, "__array__") else np.asarray(v)
    parts.append(pd.DataFrame(row))


def select_threshold_on_inner_oof(
    oof: pd.DataFrame,
    *,
    score_col: str = "y_score",
    label_col: str = "y_true",
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    thresholds: Sequence[float] | None = None,
    higher_is_better: bool = True,
    fold_col: str = "fold_id",
) -> dict:
    """Pick a global threshold using only OOF rows (never held/practice).

    For each candidate threshold, score each fold separately then average.
    This is still slightly optimistic vs true nested CV, but far better than
    picking the threshold that maximizes the pooled OOF headline metric after
    seeing all folds. Prefer calling this on *inner* OOF only.
    """
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 4).tolist()
    folds = sorted(oof[fold_col].astype(str).unique())
    best_t = None
    best_score = -np.inf if higher_is_better else np.inf
    sweep = []
    for t in thresholds:
        fold_scores = []
        for f in folds:
            sub = oof.loc[oof[fold_col].astype(str).eq(f)]
            if label_col not in sub.columns:
                raise KeyError(f"Missing {label_col} for threshold selection")
            y = sub[label_col].to_numpy()
            s = sub[score_col].to_numpy()
            pred = (s >= t).astype(np.int8)
            fold_scores.append(float(metric_fn(y, pred)))
        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        sweep.append({"threshold": float(t), "mean": mean, "std": std, "per_fold": fold_scores})
        better = mean > best_score if higher_is_better else mean < best_score
        if better:
            best_score = mean
            best_t = float(t)
    return {
        "threshold": best_t,
        "mean_metric": best_score,
        "n_folds": len(folds),
        "sweep": sweep,
        "policy": "mean_over_inner_folds",
    }
