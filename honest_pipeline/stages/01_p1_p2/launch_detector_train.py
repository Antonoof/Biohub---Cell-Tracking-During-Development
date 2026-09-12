#!/usr/bin/env python3
"""Launch P1/P2 detector training under the honest split contract.

Wraps William's train_unet_transformer.py with:
  - canonical GKF5 or LOEO splits (never alltrain)
  - RunLogger under runs/01_p1_detector or runs/02_p2_detector
  - CUDA_VISIBLE_DEVICES selection

Example (remote):
  CUDA_VISIBLE_DEVICES=3 \\
  python stages/01_p1_p2/launch_detector_train.py \\
    --scheme gkf_movie --split 0 --epochs 50 --tag p1_gkf5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biohub_cv.logging_utils import RunLogger, new_run_dir  # noqa: E402
from biohub_cv.splits import load_canonical_splits  # noqa: E402

WILLIAM_TRAIN = (
    ROOT.parent
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers"
    / "01_p1_p2_base"
    / "shared_repo"
    / "scripts"
    / "train_unet_transformer.py"
)
# Local mirror fallback.
LOCAL_TRAIN = (
    ROOT.parent
    / "helpers"
    / "01_p1_p2_base"
    / "shared_repo"
    / "scripts"
    / "train_unet_transformer.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["gkf_movie", "loeo"], default="gkf_movie")
    p.add_argument("--split", type=str, default="0", help="Fold index or 'all'")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--tag", type=str, default="detector")
    p.add_argument("--stage", choices=["01_p1_detector", "02_p2_detector"], default="01_p1_detector")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/data/projects/ryzhichkin/biohub/kaggle/input/competitions/"
            "biohub-cell-tracking-during-development/train"
        ),
    )
    p.add_argument("--train-script", type=Path, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_canonical_splits(ROOT / "splits" / "canonical_splits.json")
    split_file = (
        ROOT / "splits" / "dataset_splits_gkf5_train175.json"
        if args.scheme == "gkf_movie"
        else ROOT / "splits" / "dataset_splits_loeo.json"
    )
    train_script = args.train_script
    if train_script is None:
        train_script = WILLIAM_TRAIN if WILLIAM_TRAIN.exists() else LOCAL_TRAIN
    if not train_script.exists():
        raise SystemExit(f"Missing train script: {train_script}")

    method = args.method or f"honest_{args.stage}_{args.scheme}"
    run_dir = new_run_dir(ROOT / "runs", args.stage, f"{args.tag}_split{args.split}")
    logger = RunLogger(
        run_dir,
        stage=args.stage,
        config={
            "scheme": args.scheme,
            "split": args.split,
            "epochs": args.epochs,
            "data_dir": str(args.data_dir),
            "splits_file": str(split_file),
            "train_script": str(train_script),
            "method": method,
            "protocol": splits.to_jsonable()["protocol"],
            "banned": ["alltrain", "held20_threshold_tuning", "practice_in_fit"],
            "notes": [
                "Honest detector CV launch. Do not use alltrain checkpoints in cascade OOF.",
                "After folds finish, export per-movie OOF detections before downstream stages.",
            ],
            "extra": args.extra,
        },
    )

    cmd = [
        sys.executable,
        str(train_script),
        "--data-dir",
        str(args.data_dir),
        "--splits",
        str(split_file),
        "--split",
        str(args.split),
        "--epochs",
        str(args.epochs),
        "--method",
        method,
        *args.extra,
    ]
    logger.log("cmd: " + " ".join(cmd))
    env = os.environ.copy()
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    env["BIOHUB_HONEST_RUN_DIR"] = str(run_dir)
    env["BIOHUB_WEIGHTS_DIR"] = str(weights_dir)
    env["BIOHUB_DATA_DIR"] = str(args.data_dir)
    # Force math SDPA (flash/mem-efficient break on H200 for this temporal MHA).
    env.setdefault("PYTORCH_SDP_BACKEND", "math")
    src = str(train_script.parent.parent / "src")
    scripts = str(train_script.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [src, scripts, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    proc = subprocess.run(cmd, cwd=str(train_script.parent), env=env)

    summary = {
        "returncode": proc.returncode,
        "scheme": args.scheme,
        "split": args.split,
        "method": method,
        "splits_file": str(split_file),
    }
    logger.write_summary(summary)
    (run_dir / "launch_cmd.json").write_text(json.dumps(cmd, indent=2) + "\n")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    logger.log("done")


if __name__ == "__main__":
    main()
