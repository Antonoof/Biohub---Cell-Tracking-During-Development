#!/usr/bin/env python3
"""Honest Motion Corrector fold launcher (GKF5, proposal_nn parents)."""

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
    / "09_motion_corrector"
    / "TRAINING_V1"
    / "train_motion_cost_corrector.py"
)
LOCAL = (
    ROOT.parent
    / "helpers"
    / "09_motion_corrector"
    / "TRAINING_V1"
    / "train_motion_cost_corrector.py"
)
DATA = Path(
    "/data/projects/ryzhichkin/biohub/kaggle/input/competitions/"
    "biohub-cell-tracking-during-development/train"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--tag", default="motion_gkf5")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--proposals", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--parent-mode", default="proposal_nn", choices=["proposal_nn", "gt"])
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    splits = ROOT / "splits" / "dataset_splits_gkf5_train175.json"
    train_script = TRAIN if TRAIN.exists() else LOCAL
    run_dir = new_run_dir(ROOT / "runs", "04_motion_corrector", f"{args.tag}_fold{args.fold}")
    logger = RunLogger(
        run_dir,
        stage="04_motion_corrector",
        config={
            "fold": args.fold,
            "epochs": args.epochs,
            "parent_mode": args.parent_mode,
            "proposals": str(args.proposals),
            "splits": str(splits),
        },
    )
    cache = run_dir / "cache"
    out = run_dir / "weights"
    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        str(args.data_dir),
        "--proposals",
        str(args.proposals),
        "--splits",
        str(splits),
        "--fold",
        str(args.fold),
        "--cache",
        str(cache),
        "--output",
        str(out),
        "--epochs",
        str(args.epochs),
        "--parent-mode",
        args.parent_mode,
        "--rebuild-cache",
        "--device",
        "cuda:0",
        *args.extra,
    ]
    logger.log(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(train_script.parent), env=os.environ.copy()).returncode
    logger.write_summary({"returncode": rc, "fold": args.fold, "parent_mode": args.parent_mode})
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
