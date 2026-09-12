#!/usr/bin/env python3
"""Assemble per-fold detector run dirs into a single OOF bank + summary.

Expects fold run dirs under runs/<stage>/ matching --glob, each with
weights/.../split_<k>/edge_predictor_best.pth OR a fold marker in the name.

For P1/P2 the honest contract is: movie M is scored only by the fold model
whose val set contains M. This script writes:
  assembled/oof_movies.json   — movie -> fold_id / weight path
  assembled/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biohub_cv.splits import load_canonical_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="e.g. 01_p1_detector")
    p.add_argument("--tag", required=True, help="run tag substring, e.g. p1_gkf5")
    p.add_argument("--scheme", choices=["gkf_movie", "loeo"], default="gkf_movie")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_canonical_splits(ROOT / "splits" / "canonical_splits.json")
    folds = list(splits.gkf5_train175 if args.scheme == "gkf_movie" else splits.loeo_folds)
    stage_root = ROOT / "runs" / args.stage
    run_dirs = sorted(
        [p for p in stage_root.glob(f"*{args.tag}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise SystemExit(f"No run dirs matching *{args.tag}* under {stage_root}")

    # Map fold index -> run dir (prefer name containing splitN)
    fold_runs: dict[int, Path] = {}
    for rd in run_dirs:
        for i in range(len(folds)):
            if f"split{i}" in rd.name or f"split_{i}" in rd.name or f"fold{i}" in rd.name:
                fold_runs[i] = rd
                break
        else:
            # fallback: look for weights/split_k
            for i in range(len(folds)):
                hits = list(rd.glob(f"weights/**/split_{i}/edge_predictor_best.pth"))
                if hits:
                    fold_runs[i] = rd

    movie_to_fold: dict[str, dict] = {}
    for i, fold in enumerate(folds):
        rd = fold_runs.get(i)
        weight = None
        if rd is not None:
            hits = list(rd.glob(f"weights/**/split_{i}/edge_predictor_best.pth"))
            if not hits:
                hits = list(rd.glob("weights/**/edge_predictor_best.pth"))
            weight = str(hits[0]) if hits else None
        for m in fold.val_movies:
            movie_to_fold[m] = {
                "fold_index": i,
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "run_dir": str(rd) if rd else None,
                "weight": weight,
            }

    missing_folds = [i for i in range(len(folds)) if i not in fold_runs]
    missing_weights = [m for m, info in movie_to_fold.items() if not info["weight"]]
    out = args.out or (stage_root / f"assembled_{args.tag}_{args.scheme}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof_movies.json").write_text(json.dumps(movie_to_fold, indent=2) + "\n")
    summary = {
        "stage": args.stage,
        "tag": args.tag,
        "scheme": args.scheme,
        "n_movies": len(movie_to_fold),
        "n_fold_runs_found": len(fold_runs),
        "missing_folds": missing_folds,
        "n_movies_missing_weight": len(missing_weights),
        "fold_runs": {str(k): str(v) for k, v in fold_runs.items()},
        "ready_for_oof_predict": len(missing_folds) == 0 and len(missing_weights) == 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if missing_folds:
        raise SystemExit(f"Missing fold runs: {missing_folds}")


if __name__ == "__main__":
    main()
