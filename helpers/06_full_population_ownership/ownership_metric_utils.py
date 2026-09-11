"""Small metric helpers shared by ownership training scripts."""

from __future__ import annotations

import numpy as np


def best_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    """Choose the Jaccard-optimal threshold, resolving ties conservatively."""
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    true_positive = np.cumsum(sorted_y)
    false_positive = np.cumsum(1 - sorted_y)
    total = int(y.sum())
    false_negative = total - true_positive
    jaccard = true_positive / np.maximum(
        true_positive + false_positive + false_negative, 1
    )
    index = int(np.flatnonzero(jaccard == jaccard.max())[0])
    threshold = float(score[order[index]])
    selected = score >= threshold
    return {
        "threshold": threshold,
        "selected": int(selected.sum()),
        "tp": int(y[selected].sum()),
        "fp": int(selected.sum() - y[selected].sum()),
        "fn": int(total - y[selected].sum()),
        "jaccard": float(jaccard[index]),
    }

