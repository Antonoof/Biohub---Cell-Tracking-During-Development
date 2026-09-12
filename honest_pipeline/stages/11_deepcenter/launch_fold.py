#!/usr/bin/env python3
"""Honest DeepCenter fold launcher (GKF5 or LOEO)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from biohub_cv.logging_utils import RunLogger, new_run_dir  # noqa: E402

TRAIN = (
    ROOT.parent
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers"
    / "10_deepcenter"
    / "train_full_frame_center_detector.py"
)
LOCAL = ROOT.parent / "_honest_helpers" / "10_deepcenter" / "train_full_frame_center_detector.py"
LOCAL2 = ROOT.parent / "helpers" / "10_deepcenter" / "train_full_frame_center_detector.py"
DATA = Path(
    "/data/projects/ryzhichkin/biohub/kaggle/input/competitions/"
    "biohub-cell-tracking-during-development/train"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["gkf_movie", "loeo"], default="gkf_movie")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tag", default="deepcenter")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    splits = (
        ROOT / "splits" / "dataset_splits_gkf5_train175.json"
        if args.scheme == "gkf_movie"
        else ROOT / "splits" / "dataset_splits_loeo.json"
    )
    train_script = TRAIN if TRAIN.exists() else (LOCAL if LOCAL.exists() else LOCAL2)
    run_dir = new_run_dir(ROOT / "runs", "11_deepcenter", f"{args.tag}_fold{args.fold}")
    logger = RunLogger(
        run_dir,
        stage="11_deepcenter",
        config={
            "scheme": args.scheme,
            "fold": args.fold,
            "epochs": args.epochs,
            "splits": str(splits),
            "train_script": str(train_script),
        },
    )
    out = run_dir / "weights"
    cmd = [
        sys.executable,
        str(train_script),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--splits-json",
        str(splits),
        "--fold",
        str(args.fold),
        "--overwrite",
        *args.extra,
    ]
    logger.log(" ".join(cmd))
    env = os.environ.copy()
    rc = subprocess.run(cmd, env=env).returncode
    logger.write_summary({"returncode": rc, "fold": args.fold, "scheme": args.scheme})
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
