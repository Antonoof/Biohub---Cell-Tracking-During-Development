#!/usr/bin/env python3
"""Grouped-video benchmark of compact ownership-specific feature families."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ownership_metric_utils as common


BIO = Path("data")
BANK = BIO / "ug12_ownership_geometry_bank_v2/ownership_geometry.parquet"
OUTPUT = BIO / "ownership_feature_groups_v3"

RANK = [
    "source_frame_percentile", "source_frame_log_margin",
    "v2_frame_percentile", "v2_frame_log_margin", "agreement4", "agreement5",
    "time_norm",
]
IDENTITY = [
    "current_parent_um", "alternate_parent_um", "alternate_minus_current_um",
    "alternate_over_current", "incumbent_exists", "incumbent_back_depth",
    "incumbent_to_alternate_um", "source_to_incumbent_um",
    "steal_distance_delta_um", "steal_distance_ratio",
    "current_edge_prob", "incumbent_edge_prob",
    "incumbent_edge_prob_minus_current",
]
MOTION = [
    "source_previous_step_um", "current_step_um", "alternate_step_um",
    "parent_current_cosine", "parent_alternate_cosine",
    "parent_split_axis_abs_cosine", "predicted_midpoint_error_um",
    "incumbent_previous_step_um", "incumbent_continuation_cosine",
    "current_forward_depth", "alternate_forward_depth", "forward_depth_difference",
    "current_next_step_um", "alternate_next_step_um",
    "current_forward_cosine", "alternate_forward_cosine",
]
TOP = [
    "steal_distance_delta_um", "alternate_minus_current_um",
    "incumbent_edge_prob", "alternate_parent_um", "steal_distance_ratio",
    "incumbent_back_depth", "predicted_midpoint_error_um",
]

GROUPS = {
    "identity": IDENTITY,
    "identity_rank": [*IDENTITY, *RANK],
    "top": TOP,
    "top_rank": [*TOP, *RANK],
    "motion_rank": [*MOTION, *RANK],
    "identity_motion": [*IDENTITY, *MOTION],
}


def factories():
    result = {
        "logistic": lambda seed: make_pipeline(
            StandardScaler(), LogisticRegression(
                C=.25, class_weight="balanced", max_iter=2000, random_state=seed,
            )
        )
    }
    for depth in (2, 3, 4, 5):
        for leaf in (8, 16):
            result[f"extra_d{depth}_l{leaf}"] = lambda seed, d=depth, l=leaf: ExtraTreesClassifier(
                n_estimators=500, max_depth=d, min_samples_leaf=l,
                max_features=.8, class_weight="balanced", n_jobs=-1,
                random_state=seed,
            )
    for leaves in (3, 7):
        result[f"hist_l{leaves}"] = lambda seed, l=leaves: HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, max_leaf_nodes=l, max_depth=3,
            min_samples_leaf=20, l2_regularization=2., random_state=seed,
        )
    return result


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    full = pd.read_parquet(BANK)
    known = full.loc[full.source_label.notna()].copy().reset_index(drop=True)
    known["target"] = (known.source_label.eq(1) & known.best_pair_label.eq(1)).astype(np.int8)
    y = known.target.to_numpy(np.int8)
    groups = known.dataset.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(known, y, groups))
    records = []
    predictions = known[["dataset", "source", "target"]].copy()
    for group_index, (group_name, features) in enumerate(GROUPS.items()):
        x = known[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        for model_index, (model_name, factory) in enumerate(factories().items()):
            oof = np.zeros(len(known), np.float64)
            for fold, (fit, held) in enumerate(splits):
                model = factory(161803 + group_index * 10000 + model_index * 100 + fold)
                if model_name.startswith("hist_"):
                    weight = np.where(
                        y[fit] == 1,
                        max((y[fit] == 0).sum() / max((y[fit] == 1).sum(), 1), 1), 1.,
                    )
                    model.fit(x[fit], y[fit], sample_weight=weight)
                else:
                    model.fit(x[fit], y[fit])
                oof[held] = model.predict_proba(x[held])[:, 1]
            summary = common.best_threshold(y, oof)
            key = f"{group_name}__{model_name}"
            summary.update({
                "model": key, "feature_group": group_name,
                "roc_auc": float(roc_auc_score(y, oof)),
                "pr_auc": float(average_precision_score(y, oof)),
                "positive_videos_selected": int(
                    known.loc[(oof >= summary["threshold"]) & known.target.eq(1), "dataset"].nunique()
                ),
            })
            records.append(summary)
            predictions[key] = oof
        print(f"completed {group_name}", flush=True)
    result = pd.DataFrame(records).sort_values(
        ["jaccard", "tp", "fp"], ascending=[False, False, True], kind="stable"
    )
    result.to_csv(OUTPUT / "summary.csv", index=False)
    predictions.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
    payload = {
        "version": "ownership-feature-groups-v3",
        "known_rows": int(len(known)), "known_positive_rows": int(y.sum()),
        "groups": GROUPS, "raw_refit_probabilities_used": False,
        "unknown_sources_used_as_negatives": False,
        "best": result.iloc[0].to_dict(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(result.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
