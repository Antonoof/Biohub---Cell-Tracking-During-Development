#!/usr/bin/env python3
"""Freeze post-hoc serve hyperparameters from honest GKF5 OOF only.

Stages covered
--------------
1) P1 / P2 detector serve knobs (det_threshold, pool_kernel_um, edge_threshold,
   edge_activation, optional ILP on/off) via predict+evaluate mean-over-folds.
2) DeepCenter gate threshold from existing per-fold gate_threshold_metrics.csv.
3) A+B blend weight from OOF detector scores (normalized adj_edge_jaccard).

Never uses held20 / practice for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIO = Path(os.environ.get("BIO", "/data/projects/ryzhichkin/biohub"))
PY = Path(os.environ.get("PY", BIO / ".venv/bin/python"))
SCRIPTS = BIO / "william-duckworth-reproducible-training-pipeline/helpers/01_p1_p2_base/shared_repo/scripts"
SRC = BIO / "william-duckworth-reproducible-training-pipeline/helpers/01_p1_p2_base/shared_repo/src"
DATA = BIO / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
SPLITS = ROOT / "splits/dataset_splits_gkf5_train175.json"
PREDICT = SCRIPTS / "predict_unet_transformer.py"

EVAL_RE = re.compile(
    r"Evaluation \((?P<n>\d+) videos\): "
    r"score=(?P<score>[\d.]+)  "
    r"edge_jaccard=(?P<edge>[\d.]+)  "
    r"adj_edge_jaccard=(?P<adj>[\d.]+).*?"
    r"node_recall=(?P<node>[\d.]+)"
)


@dataclass(frozen=True)
class ServeCfg:
    det_threshold: float
    pool_kernel_um: float
    edge_threshold: float
    edge_activation: str
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    def tag(self) -> str:
        ilp = "ilp1" if self.use_ilp else "ilp0"
        return (
            f"det{self.det_threshold:g}_pool{self.pool_kernel_um:g}_"
            f"edge{self.edge_threshold:g}_{self.edge_activation}_{ilp}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "serve_hparams")
    p.add_argument("--gpus", default="3,4,5,6,7")
    p.add_argument("--models", default="p1,p2", help="Comma list: p1,p2")
    p.add_argument("--slice", default=None, help="Optional predict --slice, e.g. :8 for coarse")
    p.add_argument("--phase", choices=["coarse", "refine", "ilp", "all"], default="all")
    p.add_argument("--metric", default="adj_edge_jaccard", choices=["adj_edge_jaccard", "score", "edge_jaccard"])
    p.add_argument("--skip-deepcenter", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def weight_paths() -> dict[str, list[Path]]:
    return {
        "p1": [
            ROOT
            / "runs/01_p1_detector"
            / f"20260912T174423Z_p1_gkf5_ep50_split{f}"
            / "weights/honest_01_p1_detector_gkf_movie"
            / f"split_{f}/edge_predictor_best.pth"
            for f in range(5)
        ],
        "p2": [
            ROOT
            / "runs/02_p2_detector"
            / f"20260913T080920Z_p2_gkf5_ep50_split{f}"
            / "weights/honest_02_p2_detector_gkf_movie_seed7"
            / f"split_{f}/edge_predictor_best.pth"
            for f in range(5)
        ],
    }


def run_one(
    *,
    model: str,
    fold: int,
    cfg: ServeCfg,
    weight: Path,
    gpu: str,
    out_root: Path,
    video_slice: str | None,
    dry_run: bool,
) -> dict:
    method = f"serve_sweep_{model}_{cfg.tag()}"
    log = out_root / "logs" / f"{model}_fold{fold}_{cfg.tag()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY),
        str(PREDICT),
        "--data-dir",
        str(DATA),
        "--splits",
        str(SPLITS),
        "--split",
        str(fold),
        "--weights",
        str(weight),
        "--method",
        method,
        "--det-threshold",
        str(cfg.det_threshold),
        "--pool-kernel-um",
        str(cfg.pool_kernel_um),
        "--edge-threshold",
        str(cfg.edge_threshold),
        "--edge-activation",
        cfg.edge_activation,
        "--evaluate",
    ]
    if cfg.use_ilp:
        cmd += [
            "--use-ilp",
            "--ilp-edge-weight",
            str(cfg.ilp_edge_weight),
            "--ilp-appearance-weight",
            str(cfg.ilp_appearance_weight),
            "--ilp-disappearance-weight",
            str(cfg.ilp_disappearance_weight),
            "--ilp-division-weight",
            str(cfg.ilp_division_weight),
        ]
    if video_slice:
        cmd += ["--slice", video_slice]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{SCRIPTS}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["BIOHUB_DATA_DIR"] = str(DATA)
    if dry_run:
        return {"model": model, "fold": fold, "cfg": asdict(cfg), "dry_run": True, "cmd": cmd}
    with log.open("w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="ignore")
    m = None
    for match in EVAL_RE.finditer(text):
        m = match
    if m is None:
        return {
            "model": model,
            "fold": fold,
            "cfg": asdict(cfg),
            "ok": False,
            "returncode": proc.returncode,
            "log": str(log),
        }
    return {
        "model": model,
        "fold": fold,
        "cfg": asdict(cfg),
        "ok": True,
        "returncode": proc.returncode,
        "score": float(m.group("score")),
        "edge_jaccard": float(m.group("edge")),
        "adj_edge_jaccard": float(m.group("adj")),
        "node_recall": float(m.group("node")),
        "n": int(m.group("n")),
        "log": str(log),
    }


def launch_grid(
    *,
    models: list[str],
    configs: list[ServeCfg],
    gpus: list[str],
    out_root: Path,
    video_slice: str | None,
    dry_run: bool,
) -> list[dict]:
    """Round-robin fold×config×model across GPUs (sequential worker per GPU)."""
    weights = weight_paths()
    jobs: list[tuple[str, int, ServeCfg, Path]] = []
    for model in models:
        for fold, w in enumerate(weights[model]):
            if not w.exists():
                raise FileNotFoundError(w)
            for cfg in configs:
                jobs.append((model, fold, cfg, w))

    # Group jobs by GPU for simple sequential execution per device via GNU-ish bash.
    # Here we run sequentially in-process but interleave GPUs with Popen batches.
    results: list[dict] = []
    pending: list[tuple[subprocess.Popen, dict, Path]] = []

    def flush_done() -> None:
        nonlocal pending
        still = []
        for proc, meta, log in pending:
            if proc.poll() is None:
                still.append((proc, meta, log))
                continue
            text = log.read_text(errors="ignore") if log.exists() else ""
            m = None
            for match in EVAL_RE.finditer(text):
                m = match
            row = dict(meta)
            row["returncode"] = proc.returncode
            if m is None:
                row["ok"] = False
            else:
                row.update(
                    {
                        "ok": True,
                        "score": float(m.group("score")),
                        "edge_jaccard": float(m.group("edge")),
                        "adj_edge_jaccard": float(m.group("adj")),
                        "node_recall": float(m.group("node")),
                        "n": int(m.group("n")),
                    }
                )
            results.append(row)
            print(
                f"[{len(results)}/{len(jobs)}] {row.get('model')} fold{row.get('fold')} "
                f"{ServeCfg(**row['cfg']).tag()} ok={row.get('ok')} "
                f"adj={row.get('adj_edge_jaccard')}",
                flush=True,
            )
        pending = still

    gpu_i = 0
    for model, fold, cfg, w in jobs:
        while len(pending) >= len(gpus):
            flush_done()
            if len(pending) >= len(gpus):
                import time

                time.sleep(5)
        gpu = gpus[gpu_i % len(gpus)]
        gpu_i += 1
        method = f"serve_sweep_{model}_{cfg.tag()}"
        log = out_root / "logs" / f"{model}_fold{fold}_{cfg.tag()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PY),
            str(PREDICT),
            "--data-dir",
            str(DATA),
            "--splits",
            str(SPLITS),
            "--split",
            str(fold),
            "--weights",
            str(w),
            "--method",
            method,
            "--det-threshold",
            str(cfg.det_threshold),
            "--pool-kernel-um",
            str(cfg.pool_kernel_um),
            "--edge-threshold",
            str(cfg.edge_threshold),
            "--edge-activation",
            cfg.edge_activation,
            "--evaluate",
        ]
        if cfg.use_ilp:
            cmd += [
                "--use-ilp",
                "--ilp-edge-weight",
                str(cfg.ilp_edge_weight),
                "--ilp-appearance-weight",
                str(cfg.ilp_appearance_weight),
                "--ilp-disappearance-weight",
                str(cfg.ilp_disappearance_weight),
                "--ilp-division-weight",
                str(cfg.ilp_division_weight),
            ]
        if video_slice:
            cmd += ["--slice", video_slice]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC}:{SCRIPTS}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["BIOHUB_DATA_DIR"] = str(DATA)
        meta = {"model": model, "fold": fold, "cfg": asdict(cfg), "log": str(log), "gpu": gpu}
        if dry_run:
            results.append({**meta, "ok": True, "dry_run": True, "adj_edge_jaccard": 0.0, "score": 0.0, "edge_jaccard": 0.0, "node_recall": 0.0, "n": 0})
            continue
        fh = log.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        pending.append((proc, meta, log))
    while pending:
        flush_done()
        if pending:
            import time

            time.sleep(5)
    return results


def aggregate(results: list[dict], metric: str) -> list[dict]:
    by_key: dict[tuple, list[dict]] = {}
    for r in results:
        if not r.get("ok"):
            continue
        cfg = ServeCfg(**r["cfg"])
        key = (r["model"], cfg.tag())
        by_key.setdefault(key, []).append(r)
    rows = []
    for (model, tag), xs in sorted(by_key.items()):
        vals = [float(x[metric]) for x in xs]
        rows.append(
            {
                "model": model,
                "tag": tag,
                "cfg": xs[0]["cfg"],
                "n_folds": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_fold": vals,
                "mean_node_recall": float(np.mean([x["node_recall"] for x in xs])),
            }
        )
    rows.sort(key=lambda d: d["mean"], reverse=True)
    return rows


def freeze_deepcenter(out_root: Path) -> dict:
    thr_scores: dict[float, list[float]] = {}
    used = []
    for fold in range(5):
        runs = sorted((ROOT / "runs/11_deepcenter").glob(f"*dc_gkf5_ep30_fold{fold}*"))
        if not runs:
            continue
        csv_path = runs[-1] / "weights" / "gate_threshold_metrics.csv"
        if not csv_path.exists():
            continue
        used.append(str(csv_path))
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                t = float(row["threshold"])
                thr_scores.setdefault(t, []).append(float(row["f1_sparse"]))
    sweep = []
    best_t, best_m = None, -1.0
    for t, xs in sorted(thr_scores.items()):
        m = float(np.mean(xs))
        sweep.append({"threshold": t, "mean_f1_sparse": m, "std": float(np.std(xs)), "per_fold": xs})
        if m > best_m:
            best_m, best_t = m, t
    # Also pick high-precision conservative gate (precision>=0.2 max recall) if available
    cons = []
    for fold in range(5):
        runs = sorted((ROOT / "runs/11_deepcenter").glob(f"*dc_gkf5_ep30_fold{fold}*"))
        if not runs:
            continue
        csv_path = runs[-1] / "weights" / "gate_threshold_metrics.csv"
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        ok = [r for r in rows if float(r["precision_sparse"]) >= 0.2]
        if ok:
            cons.append(max(ok, key=lambda r: float(r["recall_sparse"])))
    cons_thr = float(np.mean([float(r["threshold"]) for r in cons])) if cons else best_t
    out = {
        "metric": "f1_sparse_mean_over_folds",
        "best_threshold": best_t,
        "best_mean_f1_sparse": best_m,
        "conservative_threshold_mean": cons_thr,
        "peak_min_distance": 1,
        "match_radius_um": 7.0,
        "sources": used,
        "sweep": sweep,
    }
    (out_root / "deepcenter_gate.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    args = parse_args()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Phase grids
    coarse = [
        ServeCfg(det_threshold=d, pool_kernel_um=p, edge_threshold=0.5, edge_activation="softmax")
        for d, p in product([0.7, 0.9, 0.95, 0.99], [3.0, 5.0])
    ]
    # refine filled after coarse
    all_results: list[dict] = []

    if args.phase in ("coarse", "all"):
        print(f"COARSE grid n={len(coarse)} models={models} slice={args.slice}", flush=True)
        all_results += launch_grid(
            models=models,
            configs=coarse,
            gpus=gpus,
            out_root=out_root,
            video_slice=args.slice,
            dry_run=args.dry_run,
        )
        (out_root / "raw_coarse.json").write_text(json.dumps(all_results, indent=2) + "\n")

    agg = aggregate(all_results, args.metric)
    (out_root / "aggregate_coarse.json").write_text(json.dumps(agg, indent=2) + "\n")

    best_by_model: dict[str, dict] = {}
    for model in models:
        rows = [r for r in agg if r["model"] == model]
        if not rows:
            continue
        best_by_model[model] = rows[0]
        print(f"BEST coarse {model}: {rows[0]['tag']} mean={rows[0]['mean']:.4f}", flush=True)

    refine_results: list[dict] = []
    if args.phase in ("refine", "all") and best_by_model:
        for model, row in best_by_model.items():
            base = ServeCfg(**row["cfg"])
            refine_cfgs = []
            for et in (0.3, 0.5, 0.7):
                for act in ("softmax", "sigmoid"):
                    refine_cfgs.append(
                        ServeCfg(
                            det_threshold=base.det_threshold,
                            pool_kernel_um=base.pool_kernel_um,
                            edge_threshold=et,
                            edge_activation=act,
                        )
                    )
            uniq = {c.tag(): c for c in refine_cfgs}
            refine_cfgs = list(uniq.values())
            print(f"REFINE {model} grid n={len(refine_cfgs)} (full OOF)", flush=True)
            refine_results += launch_grid(
                models=[model],
                configs=refine_cfgs,
                gpus=gpus,
                out_root=out_root,
                video_slice=None if args.phase == "all" else args.slice,
                dry_run=args.dry_run,
            )
        all_results += refine_results
        (out_root / "raw_refine.json").write_text(json.dumps(refine_results, indent=2) + "\n")
        agg = aggregate(all_results, args.metric)
        for model in models:
            rows = [r for r in agg if r["model"] == model]
            if rows:
                best_by_model[model] = rows[0]
                print(f"BEST refine {model}: {rows[0]['tag']} mean={rows[0]['mean']:.4f}", flush=True)

    if args.phase in ("ilp", "all") and best_by_model:
        ilp_results: list[dict] = []
        for model, row in best_by_model.items():
            base = ServeCfg(**row["cfg"])
            ilp_cfgs = [
                ServeCfg(**{**asdict(base), "use_ilp": False}),
                ServeCfg(**{**asdict(base), "use_ilp": True}),
            ]
            print(f"ILP compare {model}", flush=True)
            ilp_results += launch_grid(
                models=[model],
                configs=ilp_cfgs,
                gpus=gpus,
                out_root=out_root,
                video_slice=None,
                dry_run=args.dry_run,
            )
        all_results += ilp_results
        agg = aggregate(all_results, args.metric)
        for model in models:
            rows = [r for r in agg if r["model"] == model]
            if rows:
                best_by_model[model] = rows[0]

    # Blend from OOF means
    blend = {"p1_weight": 0.5, "p2_weight": 0.5, "rule": "equal"}
    if "p1" in best_by_model and "p2" in best_by_model:
        s1 = max(best_by_model["p1"]["mean"], 1e-6)
        s2 = max(best_by_model["p2"]["mean"], 1e-6)
        w1 = s1 / (s1 + s2)
        w2 = s2 / (s1 + s2)
        blend = {
            "p1_weight": float(w1),
            "p2_weight": float(w2),
            "rule": "normalized_mean_adj_edge_jaccard_oof",
            "p1_mean_metric": best_by_model["p1"]["mean"],
            "p2_mean_metric": best_by_model["p2"]["mean"],
        }

    dc = None
    if not args.skip_deepcenter:
        dc = freeze_deepcenter(out_root)
        print(
            f"DeepCenter gate thr={dc['best_threshold']} "
            f"f1={dc['best_mean_f1_sparse']:.4f} cons={dc['conservative_threshold_mean']}",
            flush=True,
        )

    serve = {
        "protocol": "gkf5_mean_over_folds_oof",
        "metric": args.metric,
        "slice_used_for_selection": args.slice,
        "note": "If slice_used_for_selection is set, re-run phase=refine/ilp with --slice omitted to confirm.",
        "p1": best_by_model.get("p1"),
        "p2": best_by_model.get("p2"),
        "ab_blend": blend,
        "deepcenter_gate": dc,
        "legacy_defaults_replaced": {
            "det_threshold": 0.99,
            "pool_kernel_um_predict_default": 3.0,
            "pool_kernel_um_train_config": 5.0,
            "edge_threshold": 0.5,
        },
    }
    (out_root / "serve_config.json").write_text(json.dumps(serve, indent=2) + "\n")
    (out_root / "aggregate_all.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(serve, indent=2))


if __name__ == "__main__":
    main()
