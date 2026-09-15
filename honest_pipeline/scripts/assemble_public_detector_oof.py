#!/usr/bin/env python3
"""Assemble OOF indices for public P1=Support Pack and P2=0_917 classical.

These detectors use public all-train weights (no fold-specific checkpoints).
OOF here means: every GKF5 val movie has a GEFF, and we report per-fold
adj_edge_jaccard already scored on those val movies. Inference is not rerun
if GEFFs exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF = ROOT / "runs/p1_candidate_compare/kaggle_train_all"
SPLITS = ROOT / "splits/dataset_splits_gkf5_train175.json"
P1_GEFF = BAKEOFF / "support_pack"
P2_GEFF = BAKEOFF / "classical_exp203"


def _stems(d: Path) -> set[str]:
    return {p.name.replace(".geff", "") for p in d.glob("*.geff")}


def _fold_movies(splits: list[dict]) -> list[tuple[int, str, list[str]]]:
    out = []
    for i, s in enumerate(splits):
        movies = list(s.get("test") or s.get("val") or [])
        out.append((i, str(s.get("fold_id", f"fold{i}")), movies))
    return out


def assemble(stage: str, tag: str, geff_dir: Path, metrics_path: Path, splits: list[dict]) -> dict:
    present = _stems(geff_dir)
    movie_to_fold: dict[str, dict] = {}
    missing: list[str] = []
    fold_cov = []
    for i, fold_id, movies in _fold_movies(splits):
        n_ok = 0
        for m in movies:
            geff = geff_dir / f"{m}.geff"
            ok = geff.exists()
            n_ok += int(ok)
            movie_to_fold[m] = {
                "fold_index": i,
                "fold_id": fold_id,
                "scheme": "gkf_movie",
                "geff": str(geff) if ok else None,
                "weight": "public_alltrain",
            }
            if not ok:
                missing.append(m)
        fold_cov.append({"fold_index": i, "fold_id": fold_id, "n_val": len(movies), "n_geff": n_ok})

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    out = ROOT / "runs" / stage / f"assembled_{tag}_gkf_movie"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof_movies.json").write_text(json.dumps(movie_to_fold, indent=2) + "\n")
    summary = {
        "stage": stage,
        "tag": tag,
        "scheme": "gkf_movie",
        "detector": tag,
        "geff_dir": str(geff_dir),
        "n_movies": len(movie_to_fold),
        "n_geff_on_disk": len(present),
        "missing_oof_movies": missing,
        "fold_coverage": fold_cov,
        "kaggle_train_199": {
            "mean_adj_edge_jaccard": metrics.get("mean_adj_edge_jaccard"),
            "mean_node_recall": metrics.get("mean_node_recall"),
            "n_scored": metrics.get("n_scored"),
        },
        "gkf5": metrics.get("gkf5"),
        "ready_for_oof_predict": len(missing) == 0,
        "note": (
            "Public all-train weights. GEFFs already predicted on full kaggle train. "
            "OOF reporting = GKF5 val movies / mean-of-fold-means. Not fold-OOF weights."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "gkf5"}, indent=2))
    return summary


def main() -> None:
    splits = json.loads(SPLITS.read_text())
    p1 = assemble("01_p1_detector", "support_pack", P1_GEFF, P1_GEFF / "metrics.json", splits)
    p2 = assemble("02_p2_detector", "0917_classical", P2_GEFF, P2_GEFF / "metrics.json", splits)
    report = {
        "p1": "support_pack",
        "p2": "0917_classical",
        "p1_ready": p1["ready_for_oof_predict"],
        "p2_ready": p2["ready_for_oof_predict"],
        "p1_n_oof": p1["n_movies"],
        "p2_n_oof": p2["n_movies"],
        "p1_missing": p1["missing_oof_movies"],
        "p2_missing": p2["missing_oof_movies"],
        "p1_gkf5_mean_adj": (p1.get("gkf5") or {}).get("mean_of_fold_means_adj"),
        "p2_gkf5_mean_adj": (p2.get("gkf5") or {}).get("mean_of_fold_means_adj"),
        "all_ready": p1["ready_for_oof_predict"] and p2["ready_for_oof_predict"],
    }
    combo = ROOT / "runs" / "detector_ab_sp_0917"
    combo.mkdir(parents=True, exist_ok=True)
    (combo / "oof_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_ready"]:
        raise SystemExit("OOF GEFFs missing; re-run compare_p1_candidates.py --run-name kaggle_train_all")


if __name__ == "__main__":
    main()
