#!/usr/bin/env python3
"""Compare P1 candidates on a train directory: Support Pack vs exp203 classical.

Writes GEFFs, scores adj_edge_jaccard / node_recall vs sibling GT .geff,
and picks the winner into summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _resolve_pack() -> Path:
    """Prefer no-space layouts (H200 rsync mangling); fall back to local spaced name."""
    cands = [
        ROOT / "public_models" / "support_pack",
        ROOT / "public_models" / "Biohub",
        ROOT / "public_models" / "Biohub Tracking Support Pack",
    ]
    for c in cands:
        pred = c / "repo" / "scripts" / "predict_unet_transformer.py"
        w = c / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
        # weights may live next to pack root even if repo is symlinked elsewhere
        if pred.exists() and w.exists():
            return c
        if pred.exists():
            # weights under Support Pack with spaces, repo via symlink
            alt_w = (
                ROOT
                / "public_models"
                / "Biohub Tracking Support Pack"
                / "weights"
                / "unet_transformer"
                / "split_0"
                / "edge_predictor_best.pth"
            )
            if alt_w.exists():
                return c
    raise FileNotFoundError(f"Support Pack not found under {ROOT / 'public_models'}")


PACK = _resolve_pack()
PACK_SCRIPTS = PACK / "repo" / "scripts"
PACK_SRC = PACK / "repo" / "src"
CLASSICAL = ROOT / "helpers" / "12_classical_unet3d" / "run_exp203_to_geff.py"


def _support_pack_weights() -> Path:
    for base in (
        PACK,
        ROOT / "public_models" / "Biohub Tracking Support Pack",
        ROOT / "public_models" / "Biohub",
        ROOT / "public_models" / "support_pack",
    ):
        w = base / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
        if w.exists():
            return w
    raise FileNotFoundError("edge_predictor_best.pth not found")


WEIGHTS_SP = _support_pack_weights()


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_geff_dir(pred_dir: Path, data_dir: Path, stems: list[str]) -> list[dict]:
    sys.path.insert(0, str(PACK_SRC))
    sys.path.insert(0, str(ROOT / "helpers" / "01_p1_p2_base" / "shared_repo" / "src"))
    from biohub_tracking.io import open_dataset
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics

    rows = []
    for stem in stems:
        pred = pred_dir / f"{stem}.geff"
        if not pred.exists():
            rows.append({"dataset": stem, "error": "missing_pred", "adj_edge_jaccard": float("nan")})
            continue
        try:
            gt = open_dataset(data_dir / stem, load_image=False, require_tracks=True)
            pr = open_dataset(pred, load_image=False, require_tracks=True)
            # open_dataset on .geff alone: pass path with .geff stem handling
            # Prefer loading pred graph via tracksdata
            import tracksdata as td

            pred_graph = td.graph.IndexedRXGraph.from_geff(str(pred)) if hasattr(td.graph.IndexedRXGraph, "from_geff") else pr.tracks
            if pred_graph is None:
                pred_graph = pr.tracks
            gt_graph = gt.tracks
            scale = tuple(gt.scale) if gt.scale else (1.625, 0.40625, 0.40625)
            res = evaluate(pred_graph, gt_graph, scale=scale, max_distance=7.0)
            nr = node_recall(pred_graph, gt_graph, scale=scale, max_distance=7.0)
            row = per_sample_metrics(res, node_recall=nr)
            row["dataset"] = stem
            rows.append(row)
        except Exception as e:
            rows.append({"dataset": stem, "error": str(e), "adj_edge_jaccard": float("nan")})
    return rows


def score_geff_dir_v2(pred_dir: Path, data_dir: Path, stems: list[str]) -> list[dict]:
    """Score using Support Pack evaluate helpers with open_dataset on stem paths."""
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise
    import tracksdata as td

    rows = []
    for stem in stems:
        pred_path = pred_dir / f"{stem}.geff"
        gt_stem = data_dir / stem
        if not pred_path.exists():
            rows.append({"dataset": stem, "error": "missing_pred", "adj_edge_jaccard": float("nan")})
            continue
        try:
            gt = open_dataset(gt_stem, load_image=False, require_tracks=True)
            # Load prediction geff into a Dataset-like graph
            pred_ds = open_dataset(pred_path.with_suffix(""), load_image=False, require_tracks=False)
            # open_dataset looks for sibling .geff of stem — put pred next to a temp stem
            # Simpler: use geff/tracksdata read
            from biohub_tracking.io import save_graph  # noqa: F401

            # Support Pack open_dataset: if we pass pred_dir/stem and geff is there...
            # Copy isn't needed if we open_dataset(pred_dir / stem) when both don't exist as zarr.
            # Fallback: read geff via tracksdata functional
            if hasattr(td, "functional") and hasattr(td.functional, "from_geff"):
                pred_graph = td.functional.from_geff(pred_path)
            else:
                # IndexedRXGraph constructor variants
                pred_graph = td.graph.RXGraph.from_geff(pred_path) if hasattr(td.graph, "RXGraph") else None
            if pred_graph is None:
                # Write a fake pairing: open via copying path trick
                tmp = pred_dir / "_eval_tmp"
                tmp.mkdir(exist_ok=True)
                zlink = tmp / f"{stem}.zarr"
                glink = tmp / f"{stem}.geff"
                if not glink.exists():
                    if pred_path.is_dir():
                        if glink.exists() or glink.is_symlink():
                            glink.unlink()
                        glink.symlink_to(pred_path.resolve())
                    else:
                        shutil.copytree(pred_path, glink) if False else None
                        if not glink.exists():
                            os.symlink(pred_path.resolve(), glink)
                # Need a dummy zarr? open_dataset requires zarr
                # Use GT zarr path by opening GT and replacing tracks
                pred_graph = td.graph.InMemoryGraph.from_geff(str(pred_path)) if hasattr(td.graph.InMemoryGraph, "from_geff") else None
            if pred_graph is None:
                # Last resort: use geff library + rebuild
                import geff

                gdata = geff.read(pred_path)
                # Expect nodes with t,z,y,x and edges
                raise RuntimeError(f"cannot load pred graph for {stem}; geff keys={getattr(gdata,'keys',lambda:[])()}")
            scale = tuple(gt.original_scale or gt.scale or (1.625, 0.40625, 0.40625))
            res = evaluate(pred_graph, gt.tracks, scale=scale, max_distance=7.0)
            nr = float(node_recall(pred_graph, gt.tracks, scale=scale, max_distance=7.0))
            row = per_sample_metrics(res, node_recall=nr)
            row["dataset"] = stem
            rows.append(row)
        except Exception as e:
            rows.append({"dataset": stem, "error": repr(e), "adj_edge_jaccard": float("nan")})
    return rows


def load_pred_and_gt(pred_geff: Path, data_dir: Path, stem: str):
    """Load GT via open_dataset; load pred geff via tracksdata/geff."""
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset
    import tracksdata as td

    gt = open_dataset(data_dir / stem, load_image=False, require_tracks=True)
    # Prefer Graph.from_geff if available
    pred_graph = None
    for attr in ("from_geff", "read_geff"):
        for cls in (getattr(td.graph, "InMemoryGraph", None), getattr(td.graph, "IndexedRXGraph", None), getattr(td.graph, "RXGraph", None)):
            if cls is None:
                continue
            fn = getattr(cls, attr, None)
            if callable(fn):
                try:
                    pred_graph = fn(pred_geff)
                    break
                except Exception:
                    try:
                        pred_graph = fn(str(pred_geff))
                        break
                    except Exception:
                        pass
        if pred_graph is not None:
            break
    if pred_graph is None:
        # symlink into a temp dir with matching GT zarr so open_dataset works
        import tempfile

        td_dir = Path(tempfile.mkdtemp(prefix="p1cmp_"))
        os.symlink((data_dir / f"{stem}.zarr").resolve(), td_dir / f"{stem}.zarr")
        os.symlink(pred_geff.resolve(), td_dir / f"{stem}.geff")
        pred_ds = open_dataset(td_dir / stem, load_image=False, require_tracks=True)
        pred_graph = pred_ds.tracks
    return pred_graph, gt.tracks, tuple(gt.original_scale or gt.scale or (1.625, 0.40625, 0.40625))


def score_stems(pred_dir: Path, data_dir: Path, stems: list[str]) -> list[dict]:
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics
    from geff import GeffMetadata
    import tempfile
    import os

    rows = []
    for stem in stems:
        pred = pred_dir / f"{stem}.geff"
        if not pred.exists():
            rows.append({"dataset": stem, "error": "missing_pred", "adj_edge_jaccard": float("nan")})
            continue
        try:
            gt_path = data_dir / f"{stem}.geff"
            try:
                meta = GeffMetadata.read(gt_path)
                n_total = float((meta.extra or {}).get("estimated_number_of_nodes") or float("nan"))
            except Exception:
                n_total = float("nan")

            td_dir = Path(tempfile.mkdtemp(prefix="p1cmp_"))
            os.symlink((data_dir / f"{stem}.zarr").resolve(), td_dir / f"{stem}.zarr")
            os.symlink(pred.resolve(), td_dir / f"{stem}.geff")
            pred_ds = open_dataset(td_dir / stem, load_image=False, require_tracks=True)
            gt = open_dataset(data_dir / stem, load_image=False, require_tracks=True)
            scale = tuple(gt.original_scale or gt.scale or (1.625, 0.40625, 0.40625))
            res = evaluate(pred_ds.tracks, gt.tracks, scale=scale, max_distance=7.0)
            nr = float(node_recall(pred_ds.tracks, gt.tracks))
            if not (n_total == n_total) or n_total <= 0:
                n_total = float(gt.tracks.num_nodes())
            row = per_sample_metrics(res, n_total, nr)
            row["dataset"] = stem
            rows.append(row)
            shutil.rmtree(td_dir, ignore_errors=True)
        except Exception as e:
            rows.append({"dataset": stem, "error": repr(e), "adj_edge_jaccard": float("nan")})
    return rows


def mean_metric(rows: list[dict], key: str = "adj_edge_jaccard") -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
    return float(np.mean(vals)) if vals else float("nan")


DEFAULT_KAGGLE_TRAIN = (
    ROOT
    / "kaggle"
    / "input"
    / "competitions"
    / "biohub-cell-tracking-during-development"
    / "train"
)
DEFAULT_SPLITS = ROOT / "honest_pipeline" / "splits" / "dataset_splits_gkf5_train175.json"


def fold_breakdown(rows: list[dict], splits: list[dict]) -> dict:
    by_ds = {r.get("dataset"): r for r in rows if "dataset" in r}
    folds = []
    for s in splits:
        test = list(s.get("test") or s.get("val") or [])
        fold_rows = [by_ds[d] for d in test if d in by_ds]
        folds.append(
            {
                "fold_id": s.get("fold_id"),
                "n_test": len(test),
                "n_scored": sum(1 for r in fold_rows if r.get("adj_edge_jaccard") == r.get("adj_edge_jaccard")),
                "mean_adj_edge_jaccard": mean_metric(fold_rows),
                "mean_node_recall": mean_metric(fold_rows, "node_recall"),
            }
        )
    # OOF-style: mean of per-fold means (only folds with scores)
    fold_means = [f["mean_adj_edge_jaccard"] for f in folds if f["mean_adj_edge_jaccard"] == f["mean_adj_edge_jaccard"]]
    return {
        "folds": folds,
        "mean_of_fold_means_adj": float(np.mean(fold_means)) if fold_means else float("nan"),
    }


def run_classical(data_dir: Path, out_dir: Path, stems: list[str], device: str, py: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (out_dir / f"{s}.geff").exists()]
    print(f"classical todo={len(todo)} skip_existing={len(stems)-len(todo)}", flush=True)
    if not todo:
        return
    # One movie at a time so a corrupt zarr does not abort the whole run.
    for i, stem in enumerate(todo):
        print(f"[classical {i+1}/{len(todo)}] {stem}", flush=True)
        cmd = [
            py,
            str(CLASSICAL),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--device",
            device,
            "--stems",
            stem,
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"classical FAIL {stem}: {e}", flush=True)


def run_support_pack(data_dir: Path, out_dir: Path, stems: list[str], py: str) -> None:
    """Call Support Pack predict per stem with --debug-video; copy geffs to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PACK_SRC}:{PACK_SCRIPTS}:{env.get('PYTHONPATH', '')}"
    for i, stem in enumerate(stems):
        dest = out_dir / f"{stem}.geff"
        if dest.exists():
            print(f"[support_pack {i+1}/{len(stems)}] {stem} skip_existing", flush=True)
            continue
        zarr = data_dir / f"{stem}.zarr"
        print(f"[support_pack {i+1}/{len(stems)}] {stem}", flush=True)
        cmd = [
            py,
            str(PACK_SCRIPTS / "predict_unet_transformer.py"),
            "--debug-video",
            str(zarr),
            "--weights",
            str(WEIGHTS_SP),
            "--det-threshold",
            "0.99",
            "--use-ilp",
            "--method",
            "unet_transformer",
            "--unet-batch-size",
            "8",
        ]
        try:
            subprocess.check_call(cmd, cwd=str(PACK_SCRIPTS), env=env)
        except subprocess.CalledProcessError as e:
            print(f"support_pack FAIL {stem}: {e}", flush=True)
            continue
        user = env.get("USER", env.get("USERNAME", "unknown"))
        d = PACK / "repo" / "predictions" / user / "unet_transformer" / "split_0"
        cands = []
        for name in (f"{stem}.geff", f"{stem}.zarr.geff"):
            p = d / name
            if p.exists():
                cands.append(p)
        if not cands and d.exists():
            cands = sorted(d.glob(f"*{stem}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print(f"support_pack missing pred for {stem}", flush=True)
            continue
        src = cands[0]
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_KAGGLE_TRAIN)
    ap.add_argument("--out-root", type=Path, default=ROOT / "honest_pipeline" / "runs" / "p1_candidate_compare")
    ap.add_argument("--slice", default=":", help="python slice on sorted stems; default all")
    ap.add_argument("--only", choices=["both", "support_pack", "classical"], default="both")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help="gkf5 splits JSON for per-fold means (train175 test folds)",
    )
    ap.add_argument(
        "--stems-from-splits",
        action="store_true",
        help="Restrict to union of gkf5 test movies (175) instead of all zarrs in data-dir",
    )
    ap.add_argument("--run-name", default=None, help="Fixed output subdir name (resume-friendly)")
    args = ap.parse_args()

    splits = []
    if args.splits and args.splits.exists():
        splits = json.loads(args.splits.read_text())

    if args.stems_from_splits and splits:
        stems = sorted({d for s in splits for d in (s.get("test") or s.get("val") or [])})
    else:
        stems = sorted(p.name.replace(".zarr", "") for p in args.data_dir.glob("*.zarr"))
        if args.slice and args.slice != ":":
            sl = slice(*[int(x) if x else None for x in args.slice.split(":")])
            stems = stems[sl]
    print(f"n_movies={len(stems)} data={args.data_dir}", flush=True)

    stamp = args.run_name or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stems.json").write_text(json.dumps(stems, indent=2))

    device = args.device if args.device != "auto" else _device()
    results = {}

    if args.only in ("both", "classical"):
        cdir = run_dir / "classical_exp203"
        t0 = time.time()
        run_classical(args.data_dir, cdir, stems, device, args.python)
        rows = score_stems(cdir, args.data_dir, stems)
        block = {
            "mean_adj_edge_jaccard": mean_metric(rows),
            "mean_node_recall": mean_metric(rows, "node_recall"),
            "seconds": time.time() - t0,
            "n_scored": sum(1 for r in rows if r.get("adj_edge_jaccard") == r.get("adj_edge_jaccard")),
            "rows": rows,
        }
        if splits:
            block["gkf5"] = fold_breakdown(rows, splits)
        results["classical_exp203"] = block
        (cdir / "metrics.json").write_text(json.dumps({k: v for k, v in block.items() if k != "rows"}, indent=2))
        print("CLASSICAL mean adj", block["mean_adj_edge_jaccard"], "gkf5", block.get("gkf5", {}).get("mean_of_fold_means_adj"), flush=True)

    if args.only in ("both", "support_pack"):
        sdir = run_dir / "support_pack"
        t0 = time.time()
        run_support_pack(args.data_dir, sdir, stems, args.python)
        rows = score_stems(sdir, args.data_dir, stems)
        block = {
            "mean_adj_edge_jaccard": mean_metric(rows),
            "mean_node_recall": mean_metric(rows, "node_recall"),
            "seconds": time.time() - t0,
            "n_scored": sum(1 for r in rows if r.get("adj_edge_jaccard") == r.get("adj_edge_jaccard")),
            "rows": rows,
        }
        if splits:
            block["gkf5"] = fold_breakdown(rows, splits)
        results["support_pack"] = block
        (sdir / "metrics.json").write_text(json.dumps({k: v for k, v in block.items() if k != "rows"}, indent=2))
        print("SUPPORT_PACK mean adj", block["mean_adj_edge_jaccard"], "gkf5", block.get("gkf5", {}).get("mean_of_fold_means_adj"), flush=True)

    ranking = sorted(
        ((k, v["mean_adj_edge_jaccard"], v.get("mean_node_recall", float("nan"))) for k, v in results.items()),
        key=lambda x: (x[1] == x[1], x[1], x[2] == x[2], x[2]),
        reverse=True,
    )
    winner = ranking[0][0] if ranking else None
    summary = {
        "data_dir": str(args.data_dir),
        "n_movies": len(stems),
        "device": device,
        "splits": str(args.splits) if splits else None,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()},
        "per_movie": {k: v.get("rows") for k, v in results.items()},
        "winner": winner,
        "ranking": [{"name": n, "mean_adj": a, "mean_node_recall": r} for n, a, r in ranking],
        "note": (
            "Public all-train weights scored on kaggle train. "
            "gkf5 block = per-fold means on each fold's test set (comparable layout to OOF; "
            "weights themselves are NOT fold-OOF)."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_root / "latest_summary.json").write_text(json.dumps(summary, indent=2))
    print("WINNER", winner, summary["ranking"], flush=True)
    print("Wrote", run_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
