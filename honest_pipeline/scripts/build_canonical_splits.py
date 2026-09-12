#!/usr/bin/env python3
"""Build canonical_splits.json from the teammate 175/20/4 panel (+ optional disk check)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biohub_cv.splits import build_canonical_splits, embryo_of  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--panel",
        type=Path,
        default=ROOT.parent
        / "helpers/02_model_c/division_balanced_175_20_split.json",
        help="Source 175/20/4 panel JSON (William lineage).",
    )
    p.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional path to train/ with .zarr/.geff to verify all movies exist.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "splits" / "canonical_splits.json",
    )
    p.add_argument("--seed", type=int, default=20260326)
    return p.parse_args()


def movies_on_disk(train_dir: Path) -> set[str]:
    names = set()
    for p in train_dir.iterdir():
        if p.suffix in {".zarr", ".geff"} or p.name.endswith(".zarr") or p.name.endswith(".geff"):
            names.add(p.name.replace(".zarr", "").replace(".geff", ""))
        elif p.is_dir() and not p.name.startswith("."):
            names.add(p.name.replace(".zarr", "").replace(".geff", ""))
    return names


def main() -> None:
    args = parse_args()
    panel = json.loads(args.panel.read_text())
    train175 = panel["train"]
    held20 = panel["held"]
    practice4 = panel["practice"]

    notes = [
        "Outer honesty gate: leave-one-embryo-out on train175 only (held20/practice untouched).",
        "Inner development CV: 5-fold GroupKFold by movie on train175.",
        "held20 is score-once transfer; practice4 is smoke only — never in fold train/val.",
        "Never train-on-all for any checkpoint used inside a CV cascade.",
        "Thresholds/blends freeze on inner OOF only; never on held/practice.",
        "Inherited panel membership from William division_balanced_175_20_split.json.",
    ]
    splits = build_canonical_splits(
        train175,
        held20,
        practice4,
        seed=args.seed,
        source_panel=str(args.panel),
        notes=notes,
    )

    if args.train_dir is not None:
        disk = movies_on_disk(args.train_dir)
        missing = [m for m in splits.all_labeled if m not in disk]
        if missing:
            raise SystemExit(f"Missing {len(missing)} movies on disk, e.g. {missing[:5]}")
        extra_note = f"Verified {len(splits.all_labeled)} labeled movies under {args.train_dir}"
        notes = list(splits.notes) + [extra_note]
        splits = build_canonical_splits(
            train175,
            held20,
            practice4,
            seed=args.seed,
            source_panel=str(args.panel),
            notes=notes,
        )

    splits.save(args.out)
    print(f"Wrote {args.out}")
    print(f"  train175={len(splits.train175)} held20={len(splits.held20)} practice4={len(splits.practice4)}")
    print(f"  LOEO folds: {[f.fold_id for f in splits.loeo_folds]}")
    for f in splits.loeo_folds:
        print(
            f"    {f.fold_id}: train={len(f.train_movies)} "
            f"({ {embryo_of(m) for m in f.train_movies} }) "
            f"val={len(f.val_movies)} ({f.meta.get('val_embryo')})"
        )
    print(f"  GKF5 train175 folds: {len(splits.gkf5_train175)}")
    for f in splits.gkf5_train175:
        print(f"    {f.fold_id}: train={len(f.train_movies)} val={len(f.val_movies)} {f.meta.get('val_by_embryo')}")
    print(f"  Companion: dataset_splits_gkf5_train175.json, dataset_splits_loeo.json, dataset_splits_nested_loeo_gkf5.json")


if __name__ == "__main__":
    main()
