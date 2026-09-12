#!/usr/bin/env python3
"""Self-check: no train∩val leaks, panels disjoint, LOEO covers both embryos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biohub_cv.splits import embryo_of, load_canonical_splits  # noqa: E402


def main() -> None:
    path = ROOT / "splits" / "canonical_splits.json"
    s = load_canonical_splits(path)
    s.assert_disjoint_panels()

    errors: list[str] = []
    for fold in list(s.loeo_folds) + list(s.gkf5_train175):
        try:
            s.assert_no_train_val_leak(fold)
        except AssertionError as exc:
            errors.append(str(exc))

    # LOEO: val is exactly one embryo; train has zero of that embryo.
    for fold in s.loeo_folds:
        val_emb = {embryo_of(m) for m in fold.val_movies}
        train_emb = {embryo_of(m) for m in fold.train_movies}
        if len(val_emb) != 1:
            errors.append(f"{fold.fold_id}: val embryos={val_emb}")
        if val_emb & train_emb:
            errors.append(f"{fold.fold_id}: embryo leak {val_emb & train_emb}")

    # GKF: union of val movies == train175; pairwise val disjoint.
    covered = []
    for fold in s.gkf5_train175:
        covered.extend(fold.val_movies)
    if sorted(covered) != sorted(s.train175):
        errors.append("GKF5 val union != train175")
    for i, a in enumerate(s.gkf5_train175):
        for b in s.gkf5_train175[i + 1 :]:
            inter = set(a.val_movies) & set(b.val_movies)
            if inter:
                errors.append(f"GKF val overlap {a.fold_id}/{b.fold_id}: {len(inter)}")

    # Nested: inner never touches outer val embryo movies.
    for outer, inner in s.iter_nested_gkf_inside_loeo():
        if set(inner.val_movies) & set(outer.val_movies):
            errors.append(f"nested leak {outer.fold_id}/{inner.fold_id}")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("OK")
    print(f"  panels train175={len(s.train175)} held20={len(s.held20)} practice4={len(s.practice4)}")
    print(f"  loeo={len(s.loeo_folds)} gkf5={len(s.gkf5_train175)}")
    n_nested = sum(1 for _ in s.iter_nested_gkf_inside_loeo())
    print(f"  nested inner folds={n_nested}")


if __name__ == "__main__":
    main()
