#!/usr/bin/env python3
"""Apply OOF motion corrector, write GEFFs, score adj_edge_jaccard.

Uses fused SP∪0_917 proposals (same population as training) and the frozen
geom_tight5_rel8 fold checkpoints. Val movies get their held-out fold weights;
movies outside train175 use fold 0 (held, not for selection).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HP = Path(__file__).resolve().parents[1]
BIO = HP.parent
if not (BIO / "kaggle").exists() and (Path("/data/projects/ryzhichkin/biohub") / "kaggle").exists():
    BIO = Path("/data/projects/ryzhichkin/biohub")
    HP = BIO / "honest_pipeline"
    ROOT = BIO

TRAINER_DIR = (
    BIO
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers/09_motion_corrector/TRAINING_V1"
)
if not TRAINER_DIR.exists():
    TRAINER_DIR = ROOT / "helpers/09_motion_corrector/TRAINING_V1"

SWEEP = HP / "runs/04_motion_corrector/20260914T200230Z_motion_sp_p2_0917_geom_mine_feat_loss"
P1_GEFF = HP / "runs/p1_candidate_compare/kaggle_train_all/support_pack"
PROPOSALS = BIO / "data/ab_proposals_sp_0917"
DATA = BIO / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
SPLITS = HP / "splits/dataset_splits_gkf5_train175.json"
WINNER = "geom_tight5_rel8"


def _pack_src() -> Path:
    for c in (
        ROOT / "public_models" / "support_pack" / "repo" / "src",
        ROOT / "public_models" / "Biohub" / "repo" / "src",
        ROOT / "public_models" / "Biohub Tracking Support Pack" / "repo" / "src",
        ROOT / "helpers" / "01_p1_p2_base" / "shared_repo" / "src",
    ):
        if (c / "biohub_tracking" / "io.py").exists():
            return c
    raise FileNotFoundError("biohub_tracking.io not found")


def movie_fold(splits: list[dict]) -> dict[str, int]:
    out = {}
    for i, s in enumerate(splits):
        for m in s.get("test") or s.get("val") or []:
            out[m] = i
    return out


def make_args(ckpt_meta: dict) -> SimpleNamespace:
    return SimpleNamespace(
        tight=float(ckpt_meta.get("tight", 5.0)),
        relaxed=float(ckpt_meta.get("relaxed", 8.0)),
        velocity_weight=float(ckpt_meta.get("velocity_weight", 0.52)),
        velocity_axes=str(ckpt_meta.get("velocity_axes", "iso")),
        reg_weight=0.05,
        parent_mode="proposal_nn",
        extra_feats="none",
        include_det_feats=True,
        feat_mode=str(ckpt_meta.get("feat_mode", "raw")),
        residual_scale=2.0,
        batch_size=8192,
        device="cpu",
        arch="mlp",
        hidden=0,
        dropout=0.05,
        tabm_k=8,
        seed=2028,
        negative_ratio=20,
        mine_mode="hard",
        ambiguous_margin=1.5,
        max_videos=0,
    )


def load_model(ckpt_path: Path, args):
    sys.path.insert(0, str(TRAINER_DIR))
    import train_motion_cost_corrector as tr

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_names = [str(x) for x in ckpt["features"]]
    runtime = [str(x) for x in ckpt.get("runtime_base_features") or feat_names]
    args.residual_scale = float(ckpt.get("residual_scale", 2.0))
    meta = ckpt.get("meta") or {}
    args.tight = float(meta.get("tight", args.tight))
    args.relaxed = float(meta.get("relaxed", args.relaxed))
    args.feat_mode = str(meta.get("feat_mode", args.feat_mode))
    if "include_det_feats" in meta:
        args.include_det_feats = bool(meta["include_det_feats"])
    model = tr.build_model("mlp", len(feat_names), args)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, np.asarray(ckpt["mean"], np.float32), np.asarray(ckpt["std"], np.float32), feat_names, runtime, args


def rows_to_arrays(rows, group_meta):
    if not rows:
        return None
    arrays = [np.concatenate([r[i] for r in rows]) for i in range(7)]
    return {
        "features": arrays[0],
        "groups": arrays[3],
        "src": arrays[4],
        "tgt": arrays[5],
        "registered": arrays[6],
        "group_meta": np.asarray(group_meta, np.int32),
    }


def apply_one(stem: str, fold: int, ckpt: str, out_dir: str, proposals: str, data: str) -> dict:
    sys.path.insert(0, str(TRAINER_DIR))
    sys.path.insert(0, str(_pack_src()))
    import train_motion_cost_corrector as tr
    import finetune_edge_head_on_proposals as ft
    from biohub_tracking.io import save_graph
    import tracksdata as td
    import polars as pl

    args = make_args({})
    model, mean, std, feat_names, runtime, args = load_model(Path(ckpt), args)
    v = ft.load_proposal_video(Path(data), Path(proposals), stem, 7.0)
    rows, meta, _ = tr.video_rows(v, 0, False, args)
    packed = rows_to_arrays(rows, meta)
    T = int(v.image_shape_raw[0])
    coords = v.coords
    # default: no edges if no candidates
    edges = []
    if packed is not None and len(packed["features"]):
        names = list(tr.FEATURES[: tr.BASE_FEAT_COUNT])
        x0 = packed["features"].astype(np.float32)
        keep = np.asarray([n in set(runtime) for n in names[: x0.shape[1]]])
        if keep.sum() != len(runtime):
            # align by name
            idx = {n: i for i, n in enumerate(names[: x0.shape[1]])}
            cols = [x0[:, idx[n]] for n in runtime if n in idx]
            x_run = np.column_stack(cols).astype(np.float32)
        else:
            x_run = x0[:, keep]
        x, _ = tr.transform_features(x_run, runtime, args.feat_mode)
        corr = tr.residuals(model, x, mean, std, args)
        cost = x_run[:, 0] - corr
        sel = tr.assignments(cost, packed["registered"], packed["groups"], packed["src"], packed["tgt"], packed["group_meta"], args)
        # map selected pair -> global node indices
        # groups are per t; src/tgt are local indices in that frame
        gmeta = packed["group_meta"]
        groups = packed["groups"]
        src, tgt = packed["src"], packed["tgt"]
        for gi, (gid, n0, n1, _) in enumerate(gmeta):
            t = gi  # video_rows appends one meta per frame transition in order
            a0 = int(v.offsets[t])
            b0 = int(v.offsets[t + 1])
            idx = np.flatnonzero((groups == gid) & sel)
            for k in idx:
                edges.append((a0 + int(src[k]), b0 + int(tgt[k])))

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    node_rows = [{"t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3])} for c in coords]
    gids = g.bulk_add_nodes(node_rows) if node_rows else []
    if edges and len(gids):
        erows = [{"source_id": int(gids[s]), "target_id": int(gids[d])} for s, d in edges if 0 <= s < len(gids) and 0 <= d < len(gids)]
        if erows:
            g.bulk_add_edges(erows)
    dest = Path(out_dir) / f"{stem}.geff"
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, dest, overwrite=True)
    return {"dataset": stem, "fold": fold, "n_nodes": int(g.num_nodes()), "n_edges": int(g.num_edges())}


def score_dir(pred_dir: Path, data_dir: Path, stems: list[str]) -> list[dict]:
    sys.path.insert(0, str(HP / "scripts"))
    from compare_p1_candidates import score_stems, fold_breakdown, mean_metric

    rows = score_stems(pred_dir, data_dir, stems)
    splits = json.loads(SPLITS.read_text()) if SPLITS.exists() else []
    block = {
        "mean_adj_edge_jaccard": mean_metric(rows),
        "mean_node_recall": mean_metric(rows, "node_recall"),
        "n_scored": sum(1 for r in rows if r.get("adj_edge_jaccard") == r.get("adj_edge_jaccard")),
        "gkf5": fold_breakdown(rows, splits) if splits else None,
        "rows": rows,
    }
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--winner", default=WINNER)
    ap.add_argument("--proposals", type=Path, default=PROPOSALS)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=HP / "runs/oof_graphs/motion_geom_tight5_rel8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stems", default="", help="comma-separated; empty=all")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip stems that already have a GEFF")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    splits = json.loads(SPLITS.read_text())
    fold_of = movie_fold(splits)
    stems = sorted(p.name.replace(".npz", "") for p in args.proposals.glob("*.npz"))
    if args.stems.strip():
        want = {s.strip() for s in args.stems.split(",") if s.strip()}
        stems = [s for s in stems if s in want]
    ckpts = {
        f: args.sweep / args.winner / f"fold_{f}" / "motion_corrector_best.pt" for f in range(5)
    }
    missing = [f for f, p in ckpts.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing motion ckpts for folds {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stem in stems:
        dest = args.out_dir / f"{stem}.geff"
        if args.resume and dest.exists():
            continue
        fold = int(fold_of.get(stem, 0))
        jobs.append((stem, fold, str(ckpts[fold])))
    print(f"apply {len(jobs)}/{len(stems)} movies workers={args.workers}", flush=True)
    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    apply_one,
                    stem,
                    fold,
                    ckpt,
                    str(args.out_dir),
                    str(args.proposals),
                    str(args.data_dir),
                ): stem
                for stem, fold, ckpt in jobs
            }
            for i, fut in enumerate(as_completed(futs), 1):
                stem = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {"dataset": stem, "error": repr(e)}
                    print("FAIL", stem, e, flush=True)
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {stem} {row}", flush=True)
    if args.resume:
        existing = [
            {"dataset": p.stem, "resumed": True}
            for p in sorted(args.out_dir.glob("*.geff"))
            if p.stem not in {r.get("dataset") for r in rows}
        ]
        rows = existing + rows
    (args.out_dir / "apply_manifest.json").write_text(json.dumps(rows, indent=2))
    ok_stems = [r["dataset"] for r in rows if "error" not in r]
    if args.skip_score:
        print("skip score, wrote", args.out_dir)
        return
    print("Scoring", len(ok_stems), flush=True)
    metrics = score_dir(args.out_dir, args.data_dir, ok_stems)
    slim = {k: v for k, v in metrics.items() if k != "rows"}
    (args.out_dir / "metrics.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps(slim, indent=2))
    print("Wrote", args.out_dir)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
