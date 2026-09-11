#!/usr/bin/env python3
"""Print the portable metadata stored in a Motion Corrector checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path("motion_corrector_best.pt"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "features": list(checkpoint["features"]),
        "mean": np.asarray(checkpoint["mean"]).tolist(),
        "std": np.asarray(checkpoint["std"]).tolist(),
        "residual_scale": float(checkpoint["residual_scale"]),
        "metrics": checkpoint["metrics"],
        "state_shapes": {key: list(value.shape) for key, value in checkpoint["model"].items()},
        "parameter_count": sum(value.numel() for value in checkpoint["model"].values()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
