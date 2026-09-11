#!/usr/bin/env python3
"""Fit the final known-only CandidateGRAFT direct-edge gate.

Candidate construction is label-free.  Only graph-matched annotated rows are
used for fitting; unknown/unannotated rows are never treated as negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from screen_native_endpoint_candidate_graft_v1 import DIRECT_FEATURES, make_model, matrix


BIO = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population", type=Path,
        default=BIO / "native_endpoint_candidate_graft_v2/direct_edges.parquet",
    )
    parser.add_argument(
        "--oof-report", type=Path,
        default=BIO / "native_endpoint_candidate_graft_screen_v1/summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "candidategraft_direct_v1",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=271828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    frame = pd.read_parquet(args.population)
    known = frame.label_known.to_numpy(bool)
    labels = frame.label_positive.fillna(False).to_numpy(np.int8)
    fit = frame.loc[known].copy()
    y = labels[known]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("Known fitting rows must contain both classes")
    model = make_model(args.seed, positives, negatives)
    model.fit(matrix(fit, DIRECT_FEATURES), y)
    joblib.dump(model, args.output / "candidategraft_direct.joblib", compress=3)
    oof = json.loads(args.oof_report.read_text())
    report = {
        "version": "candidategraft-direct-v1",
        "contract": (
            "label-free full native P1/P2 candidate population; final fit uses "
            "annotated known rows only; unannotated rows are never negatives"
        ),
        "features": DIRECT_FEATURES,
        "threshold": float(args.threshold),
        "fit_rows": int(len(fit)),
        "fit_positive": positives,
        "fit_negative": negatives,
        "population_rows": int(len(frame)),
        "oof": oof["direct"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
