#!/usr/bin/env python3
"""Score the full UG1/UG2 ownership population for an open-recall sweep.

This deliberately separates *known false* from *unknown*.  Models are fit only
on graph-matched annotated rows.  Every unannotated source is scored at serving
time but never enters the negative loss or threshold diagnostics.

Each train-175 video is scored only by a model that excluded the complete
video.  The output retains every source, rather than only the old zero-FP
threshold, so exact graph replays can test recall doses without rerunning the
GPU producers or regenerating V2 geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
DEFAULT_BANK = BIO / "ug12_ownership_geometry_bank_v2/ownership_geometry.parquet"
DEFAULT_OUTPUT = BIO / "ownership_open_recall_oof_v3"

sys.path.insert(0, str(WORKSPACE / "scripts"))
from benchmark_ownership_feature_groups_v3 import RANK, TOP  # noqa: E402


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame[[*TOP, *RANK]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(np.float32)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    full = pd.read_parquet(args.bank)
    train = full.loc[full.panel.eq("train175")].copy()
    known = train.loc[train.source_label.notna()].copy().reset_index(drop=True)
    known["target"] = (
        known.source_label.eq(1) & known.best_pair_label.eq(1)
    ).astype(np.int8)
    known["known_key"] = (
        known.dataset.astype(str) + ":" + known.source.astype(str)
    )

    x_known = matrix(known)
    y = known.target.to_numpy(np.int8)
    groups = known.dataset.astype(str).to_numpy()
    splits = list(GroupKFold(n_splits=5).split(x_known, y, groups))

    keep = [
        "dataset", "panel", "source", "a", "b", "current_child",
        "source_label", "best_pair_label", "source_score", "v2_source_score_raw",
        "agreement4", "agreement5", "source_time",
    ]
    parts: list[pd.DataFrame] = []
    models: list[ExtraTreesClassifier] = []
    fold_rows: list[dict[str, int]] = []
    for fold, (fit, held) in enumerate(splits):
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=20,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=324459 + fold,
        )
        model.fit(x_known[fit], y[fit])
        models.append(model)
        held_videos = set(known.iloc[held].dataset.astype(str).unique())
        local = train.loc[train.dataset.astype(str).isin(held_videos), keep].copy()
        local["score"] = model.predict_proba(matrix(train.loc[local.index]))[:, 1]
        local["fold"] = np.int8(fold)
        parts.append(local)
        fold_rows.append(
            {
                "fold": fold,
                "fit_known_rows": int(len(fit)),
                "held_known_rows": int(len(held)),
                "held_videos": int(len(held_videos)),
                "full_rows_scored": int(len(local)),
            }
        )
        print(
            f"fold {fold}: held_videos={len(held_videos)} "
            f"rows={len(local):,}",
            flush=True,
        )

    # A video can have no annotated source rows at all. Such a video never
    # entered any model fit, so score it with the mean of all five grouped-OOF
    # models. It remains wholly unlabeled and cannot affect thresholds.
    covered_videos = {str(value) for part in parts for value in part.dataset.unique()}
    all_videos = set(train.dataset.astype(str).unique())
    unlabeled_only_videos = sorted(all_videos - covered_videos)
    if unlabeled_only_videos:
        local = train.loc[
            train.dataset.astype(str).isin(unlabeled_only_videos), keep
        ].copy()
        x_local = matrix(train.loc[local.index])
        local["score"] = np.mean(
            [model.predict_proba(x_local)[:, 1] for model in models], axis=0
        )
        local["fold"] = np.int8(-1)
        if local.source_label.notna().any():
            raise RuntimeError("Unlabeled-only routing unexpectedly contains labels")
        parts.append(local)
        fold_rows.append(
            {
                "fold": -1,
                "fit_known_rows": int(len(known)),
                "held_known_rows": 0,
                "held_videos": int(len(unlabeled_only_videos)),
                "full_rows_scored": int(len(local)),
            }
        )
        print(
            f"unlabeled-only ensemble: videos={len(unlabeled_only_videos)} "
            f"rows={len(local):,}",
            flush=True,
        )

    scored = pd.concat(parts, ignore_index=True)
    if len(scored) != len(train):
        raise RuntimeError(f"OOF coverage mismatch: {len(scored)} != {len(train)}")
    if scored[["dataset", "source"]].duplicated().any():
        raise RuntimeError("Expected exactly one best-pair row per source")

    scored["known_target"] = (
        scored.source_label.eq(1) & scored.best_pair_label.eq(1)
    )
    scored.to_parquet(args.output / "all_source_scores.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / "folds.csv", index=False)

    positive_scores = np.sort(
        scored.loc[scored.known_target, "score"].to_numpy(np.float64)
    )
    thresholds: set[float] = {
        0.9470820974745776,
        0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10,
    }
    for recall in (0.50, 0.65, 0.75, 0.85, 0.95, 1.00):
        needed = max(1, int(np.ceil(recall * len(positive_scores))))
        thresholds.add(float(positive_scores[-needed]))

    records: list[dict[str, float | int]] = []
    known_mask = scored.source_label.notna().to_numpy(bool)
    target = scored.known_target.to_numpy(bool)
    unknown_mask = ~known_mask
    scores = scored.score.to_numpy(np.float64)
    for threshold in sorted(thresholds, reverse=True):
        selected = scores >= threshold
        tp = int(np.sum(selected & target))
        fp = int(np.sum(selected & known_mask & ~target))
        fn = int(np.sum(target & ~selected))
        records.append(
            {
                "threshold": threshold,
                "selected_sources": int(selected.sum()),
                "selected_videos": int(scored.loc[selected, "dataset"].nunique()),
                "known_tp": tp,
                "known_fp": fp,
                "known_fn": fn,
                "known_recall": tp / max(tp + fn, 1),
                "known_precision": tp / max(tp + fp, 1),
                "known_jaccard": tp / max(tp + fp + fn, 1),
                "unknown_selected": int(np.sum(selected & unknown_mask)),
            }
        )
    sweep = pd.DataFrame(records).sort_values("threshold", ascending=False)
    sweep.to_csv(args.output / "threshold_sweep.csv", index=False)
    payload = {
        "version": "ownership-open-recall-oof-v3",
        "contract": {
            "fit": "annotated rows only",
            "unknown_sources_used_as_negatives": False,
            "score": "grouped-video OOF on every train-175 source",
            "unlabeled_only_videos": (
                "mean of five OOF models; never used for fit or threshold selection"
            ),
            "pair_population": "one production V2/UG best pair per source",
        },
        "rows": int(len(scored)),
        "videos": int(scored.dataset.nunique()),
        "known_rows": int(known_mask.sum()),
        "known_positive_sources": int(target.sum()),
        "unknown_rows": int(unknown_mask.sum()),
        "folds": fold_rows,
        "threshold_rows": records,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(sweep.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
