#!/usr/bin/env python3
"""Quick P1 train microbench on one fold (node2).

Reports per-iter data/forward/backward and peak VRAM for a few batches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    root = Path("/data/projects/ryzhichkin/biohub")
    william = root / "william-duckworth-reproducible-training-pipeline"
    scripts = william / "helpers/01_p1_p2_base/shared_repo/scripts"
    src = william / "helpers/01_p1_p2_base/shared_repo/src"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(scripts))

    os.environ.setdefault("BIOHUB_WEIGHTS_DIR", str(root / "honest_pipeline/runs/_bench/weights"))
    Path(os.environ["BIOHUB_WEIGHTS_DIR"]).mkdir(parents=True, exist_ok=True)

    from train_unet_transformer import train  # noqa: E402

    data_dir = root / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
    splits = root / "honest_pipeline/splits/dataset_splits_gkf5_train175.json"

    t0 = time.perf_counter()
    train(
        data_dir=data_dir,
        fold=args.split,
        splits_file=splits,
        method=f"bench_chunked_bs{args.batch_size}",
        n_epochs=1,
        lr=1e-4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_iters=args.iters,
        data_parallel=False,
    )
    elapsed = time.perf_counter() - t0
    out = {
        "iters": args.iters,
        "batch_size": args.batch_size,
        "elapsed_s": elapsed,
        "sec_per_iter": elapsed / max(args.iters, 1),
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        ),
    }
    print("BENCH_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
