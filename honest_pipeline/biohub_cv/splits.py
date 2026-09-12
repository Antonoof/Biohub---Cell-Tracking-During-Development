"""Canonical split contract shared by every learned stage.

Protocol
--------
Outer honesty gate
    Leave-one-embryo-out (LOEO) on the two labeled embryos ``44b6`` / ``6bba``.
    Use LOEO metrics (and LOEO-frozen thresholds) for promotion decisions.

Inner development CV
    5-fold GroupKFold by movie inside the development pool (train175 by
    default, or inside each outer LOEO train set when nesting).
    Use inner OOF for architecture / HPO / stacking features.

Frozen panels (score-once, never for threshold search after peek)
    - held20: balanced 10+10 embryo transfer panel
    - practice4: E2E smoke only

Deploy fit
    Optional all-labeled refit only AFTER outer LOEO + held20 are frozen.
    Deploy ≠ reported CV. Document every deploy-fit artifact as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

EMBRYOS = ("44b6", "6bba")
DEFAULT_N_MOVIE_FOLDS = 5
DEFAULT_SEED = 20260326


def embryo_of(movie: str) -> str:
    return str(movie).split("_", 1)[0]


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    scheme: str  # "loeo" | "gkf_movie" | "held20" | "practice4" | "deploy_all"
    train_movies: tuple[str, ...]
    val_movies: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "scheme": self.scheme,
            "train_movies": list(self.train_movies),
            "val_movies": list(self.val_movies),
            "meta": self.meta,
        }


@dataclass
class CanonicalSplits:
    """Single source of truth for movie membership across stages."""

    train175: tuple[str, ...]
    held20: tuple[str, ...]
    practice4: tuple[str, ...]
    all_labeled: tuple[str, ...]
    loeo_folds: tuple[FoldSpec, ...]
    gkf5_train175: tuple[FoldSpec, ...]
    gkf5_per_embryo: dict[str, tuple[FoldSpec, ...]]
    seed: int = DEFAULT_SEED
    n_movie_folds: int = DEFAULT_N_MOVIE_FOLDS
    source_panel: str = ""
    notes: tuple[str, ...] = ()

    def assert_disjoint_panels(self) -> None:
        t, h, p = set(self.train175), set(self.held20), set(self.practice4)
        if t & h:
            raise AssertionError(f"train175 ∩ held20 = {sorted(t & h)[:5]}")
        if t & p:
            raise AssertionError(f"train175 ∩ practice4 = {sorted(t & p)[:5]}")
        if h & p:
            raise AssertionError(f"held20 ∩ practice4 = {sorted(h & p)[:5]}")
        if set(self.all_labeled) != t | h | p:
            missing = (t | h | p) - set(self.all_labeled)
            extra = set(self.all_labeled) - (t | h | p)
            raise AssertionError(f"all_labeled mismatch missing={missing} extra={extra}")

    def assert_no_train_val_leak(self, fold: FoldSpec) -> None:
        leak = set(fold.train_movies) & set(fold.val_movies)
        if leak:
            raise AssertionError(f"{fold.fold_id} train∩val leak: {sorted(leak)[:10]}")

    def iter_loeo(self) -> Iterator[FoldSpec]:
        for fold in self.loeo_folds:
            self.assert_no_train_val_leak(fold)
            yield fold

    def iter_gkf_train175(self) -> Iterator[FoldSpec]:
        for fold in self.gkf5_train175:
            self.assert_no_train_val_leak(fold)
            yield fold

    def iter_nested_gkf_inside_loeo(self) -> Iterator[tuple[FoldSpec, FoldSpec]]:
        """Yield (outer_loeo, inner_gkf) with inner groups drawn only from outer train."""
        for outer in self.iter_loeo():
            for inner in _movie_group_folds(
                movies=outer.train_movies,
                n_splits=self.n_movie_folds,
                seed=self.seed,
                fold_id_prefix=f"{outer.fold_id}__inner",
            ):
                self.assert_no_train_val_leak(inner)
                # Inner val must be subset of outer train; never touch outer val embryo.
                assert set(inner.val_movies).isdisjoint(outer.val_movies)
                assert set(inner.train_movies) | set(inner.val_movies) == set(outer.train_movies)
                yield outer, inner

    def organizer_style_splits_list(self, scheme: str = "gkf_movie") -> list[dict]:
        """Format accepted by train_unet_transformer.py: list of {train,test}."""
        folds = self.gkf5_train175 if scheme == "gkf_movie" else self.loeo_folds
        return [
            {
                "train": list(fold.train_movies),
                "test": list(fold.val_movies),
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "meta": fold.meta,
            }
            for fold in folds
        ]

    def organizer_style_splits_dict(self) -> dict:
        """Deprecated wrapper kept for callers; prefer organizer_style_splits_list."""
        return {
            "splits": self.organizer_style_splits_list("gkf_movie"),
            "n_folds": len(self.gkf5_train175),
            "seed": self.seed,
        }

    def loeo_organizer_style_splits_dict(self) -> dict:
        return {
            "splits": self.organizer_style_splits_list("loeo"),
            "n_folds": len(self.loeo_folds),
            "seed": self.seed,
        }

    def to_jsonable(self) -> dict:
        return {
            "protocol": {
                "outer": "leave_one_embryo_out",
                "inner": f"{self.n_movie_folds}_fold_group_kfold_by_movie",
                "held_panel": "held20_score_once",
                "practice_panel": "practice4_smoke_only",
                "seed": self.seed,
            },
            "source_panel": self.source_panel,
            "n_movie_folds": self.n_movie_folds,
            "notes": list(self.notes),
            "panels": {
                "train175": list(self.train175),
                "held20": list(self.held20),
                "practice4": list(self.practice4),
                "all_labeled": list(self.all_labeled),
            },
            "counts": {
                "train175": len(self.train175),
                "held20": len(self.held20),
                "practice4": len(self.practice4),
                "all_labeled": len(self.all_labeled),
                "train175_by_embryo": _count_by_embryo(self.train175),
                "held20_by_embryo": _count_by_embryo(self.held20),
                "all_by_embryo": _count_by_embryo(self.all_labeled),
            },
            "loeo_folds": [f.as_dict() for f in self.loeo_folds],
            "gkf5_train175": [f.as_dict() for f in self.gkf5_train175],
            "gkf5_per_embryo": {
                emb: [f.as_dict() for f in folds]
                for emb, folds in self.gkf5_per_embryo.items()
            },
        }

    def save(self, path: Path) -> None:
        self.assert_disjoint_panels()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2) + "\n")
        out_dir = path.parent
        # William/organizer trainers index folds[i] on a raw list.
        (out_dir / "dataset_splits_gkf5_train175.json").write_text(
            json.dumps(self.organizer_style_splits_list("gkf_movie"), indent=2) + "\n"
        )
        (out_dir / "dataset_splits_loeo.json").write_text(
            json.dumps(self.organizer_style_splits_list("loeo"), indent=2) + "\n"
        )
        nested = {}
        for outer, inner in self.iter_nested_gkf_inside_loeo():
            nested.setdefault(outer.fold_id, {"outer": outer.as_dict(), "inner_folds": []})
            nested[outer.fold_id]["inner_folds"].append(inner.as_dict())
        (out_dir / "dataset_splits_nested_loeo_gkf5.json").write_text(
            json.dumps(nested, indent=2) + "\n"
        )


def _count_by_embryo(movies: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in movies:
        e = embryo_of(m)
        out[e] = out.get(e, 0) + 1
    return out


def _stable_movie_hash(movie: str, seed: int) -> int:
    """Platform-stable 64-bit mix (not Python's randomized hash())."""
    import hashlib

    h = hashlib.blake2b(f"{seed}:{movie}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def _movie_group_folds(
    movies: Sequence[str],
    n_splits: int,
    seed: int,
    fold_id_prefix: str,
) -> tuple[FoldSpec, ...]:
    movies = tuple(sorted(set(movies)))
    if len(movies) < n_splits:
        raise ValueError(f"Need >= {n_splits} movies, got {len(movies)}")
    # Deterministic order by stable hash, then GroupKFold on that order.
    ordered = tuple(sorted(movies, key=lambda m: (_stable_movie_hash(m, seed), m)))
    X = np.zeros(len(ordered))
    y = np.zeros(len(ordered))
    groups = np.array(ordered)
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[FoldSpec] = []
    for i, (tr, va) in enumerate(splitter.split(X, y, groups)):
        train_m = tuple(sorted(groups[tr].tolist()))
        val_m = tuple(sorted(groups[va].tolist()))
        folds.append(
            FoldSpec(
                fold_id=f"{fold_id_prefix}_{i}",
                scheme="gkf_movie",
                train_movies=train_m,
                val_movies=val_m,
                meta={
                    "fold_index": i,
                    "n_train": len(train_m),
                    "n_val": len(val_m),
                    "val_by_embryo": _count_by_embryo(val_m),
                },
            )
        )
    return tuple(folds)


def build_loeo_folds(all_labeled: Sequence[str]) -> tuple[FoldSpec, ...]:
    by_emb: dict[str, list[str]] = {e: [] for e in EMBRYOS}
    for m in sorted(set(all_labeled)):
        e = embryo_of(m)
        if e not in by_emb:
            raise ValueError(f"Unexpected embryo in {m}")
        by_emb[e].append(m)
    folds: list[FoldSpec] = []
    for val_emb in EMBRYOS:
        train_emb = [e for e in EMBRYOS if e != val_emb]
        train_movies = tuple(sorted(m for e in train_emb for m in by_emb[e]))
        val_movies = tuple(sorted(by_emb[val_emb]))
        folds.append(
            FoldSpec(
                fold_id=f"loeo_val_{val_emb}",
                scheme="loeo",
                train_movies=train_movies,
                val_movies=val_movies,
                meta={
                    "val_embryo": val_emb,
                    "train_embryos": train_emb,
                    "n_train": len(train_movies),
                    "n_val": len(val_movies),
                },
            )
        )
    return tuple(folds)


def build_canonical_splits(
    train175: Sequence[str],
    held20: Sequence[str],
    practice4: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    n_movie_folds: int = DEFAULT_N_MOVIE_FOLDS,
    source_panel: str = "",
    notes: Iterable[str] = (),
) -> CanonicalSplits:
    train175 = tuple(sorted(set(train175)))
    held20 = tuple(sorted(set(held20)))
    practice4 = tuple(sorted(set(practice4)))
    all_labeled = tuple(sorted(set(train175) | set(held20) | set(practice4)))

    # Outer honesty gate is LOEO on the development pool only. held20/practice4
    # stay out of every fold so they remain score-once panels.
    loeo = build_loeo_folds(train175)
    gkf5 = _movie_group_folds(train175, n_movie_folds, seed, "gkf5_train175")
    per_emb: dict[str, tuple[FoldSpec, ...]] = {}
    for emb in EMBRYOS:
        emb_movies = [m for m in train175 if embryo_of(m) == emb]
        # Use fewer folds if an embryo has few movies; still group by movie.
        n = min(n_movie_folds, max(2, len(emb_movies) // 5))
        n = min(n, len(emb_movies))
        if n < 2:
            continue
        per_emb[emb] = _movie_group_folds(
            emb_movies, n, seed + sum(ord(c) for c in emb), f"gkf_train175_{emb}"
        )

    splits = CanonicalSplits(
        train175=train175,
        held20=held20,
        practice4=practice4,
        all_labeled=all_labeled,
        loeo_folds=loeo,
        gkf5_train175=gkf5,
        gkf5_per_embryo=per_emb,
        seed=seed,
        n_movie_folds=n_movie_folds,
        source_panel=source_panel,
        notes=tuple(notes),
    )
    splits.assert_disjoint_panels()
    # Guarantees for score-once panels.
    held_set, prac_set = set(held20), set(practice4)
    for fold in list(loeo) + list(gkf5):
        if set(fold.train_movies) & held_set or set(fold.val_movies) & held_set:
            raise AssertionError(f"{fold.fold_id} touches held20")
        if set(fold.train_movies) & prac_set or set(fold.val_movies) & prac_set:
            raise AssertionError(f"{fold.fold_id} touches practice4")
    return splits


def load_canonical_splits(path: Path | str) -> CanonicalSplits:
    raw = json.loads(Path(path).read_text())
    panels = raw["panels"]

    def _fold(d: dict) -> FoldSpec:
        return FoldSpec(
            fold_id=d["fold_id"],
            scheme=d["scheme"],
            train_movies=tuple(d["train_movies"]),
            val_movies=tuple(d["val_movies"]),
            meta=dict(d.get("meta") or {}),
        )

    return CanonicalSplits(
        train175=tuple(panels["train175"]),
        held20=tuple(panels["held20"]),
        practice4=tuple(panels["practice4"]),
        all_labeled=tuple(panels["all_labeled"]),
        loeo_folds=tuple(_fold(f) for f in raw["loeo_folds"]),
        gkf5_train175=tuple(_fold(f) for f in raw["gkf5_train175"]),
        gkf5_per_embryo={
            k: tuple(_fold(f) for f in v) for k, v in raw.get("gkf5_per_embryo", {}).items()
        },
        seed=int(raw.get("protocol", {}).get("seed", DEFAULT_SEED)),
        n_movie_folds=int(raw.get("n_movie_folds", DEFAULT_N_MOVIE_FOLDS)),
        source_panel=str(raw.get("source_panel") or ""),
        notes=tuple(raw.get("notes") or ()),
    )


def movie_fold_indices(
    movies: Sequence[str],
    folds: Sequence[FoldSpec],
) -> np.ndarray:
    """Map each row's movie id to its validation fold index (for GKF OOF)."""
    movie_to_fold: dict[str, int] = {}
    for i, fold in enumerate(folds):
        for m in fold.val_movies:
            movie_to_fold[m] = i
    out = np.empty(len(movies), dtype=np.int16)
    for i, m in enumerate(movies):
        if m not in movie_to_fold:
            raise KeyError(f"Movie {m} not in any fold val set")
        out[i] = movie_to_fold[m]
    return out
