#!/usr/bin/env python3
"""GKF5 motion sweep: P1=Support Pack + P2=0_917 members, no graph blend.

Architecture FIXED (mlp). Baseline builds the pair cache; geom/mining/extra-feats
rebuild; feat-mode and loss reuse this sweep's baseline_mlp caches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BIO = Path("/data/projects/ryzhichkin/biohub")
HP = BIO / "honest_pipeline"
PY = BIO / ".venv/bin/python"
TRAIN = (
    BIO
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers/09_motion_corrector/TRAINING_V1/train_motion_cost_corrector.py"
)
LOCAL_TRAIN = BIO / "helpers/09_motion_corrector/TRAINING_V1/train_motion_cost_corrector.py"
SPLITS = HP / "splits/dataset_splits_gkf5_train175.json"
PROPOSALS = BIO / "data/ab_proposals_sp_0917"
DATA = BIO / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"


def experiments() -> list[dict]:
    """One-at-a-time ablations. Arch always mlp. rebuild=True when pair set changes."""
    exps: list[dict] = []

    def add(name: str, rebuild: bool = False, **kw):
        kw.setdefault("include_det_feats", True)
        exps.append({"name": name, "rebuild_cache": rebuild, "arch": "mlp", **kw})

    # baseline cache for this P1/P2 proposal bank (queued first so feat/loss can reuse it)
    add("baseline_mlp", rebuild=True)

    # --- 1) geometry (pair graph / motion cost) ---
    add("geom_tight5_rel8", rebuild=True, tight=5.0, relaxed=8.0)
    add("geom_tight7_rel11", rebuild=True, tight=7.0, relaxed=11.0)
    add("geom_tight6.2_rel12", rebuild=True, tight=6.2, relaxed=12.0)
    add("geom_vw0.35", rebuild=True, velocity_weight=0.35)
    add("geom_vw0.70", rebuild=True, velocity_weight=0.70)
    add("geom_axes_prod", rebuild=True, velocity_axes="0,0.45,0.47")
    add("geom_regw0.02", rebuild=True, reg_weight=0.02)
    add("geom_regw0.15", rebuild=True, reg_weight=0.15)

    # --- 2) negative mining ---
    add("mine_random", rebuild=True, mine_mode="random")
    add("mine_ambiguous", rebuild=True, mine_mode="ambiguous")
    add("mine_hard_ambiguous", rebuild=True, mine_mode="hard_ambiguous")
    add("mine_hard_nr8", rebuild=True, mine_mode="hard", negative_ratio=8)
    add("mine_hard_nr40", rebuild=True, mine_mode="hard", negative_ratio=40)
    add("mine_amb_m0.8", rebuild=True, mine_mode="ambiguous", ambiguous_margin=0.8)
    add("mine_amb_m2.5", rebuild=True, mine_mode="ambiguous", ambiguous_margin=2.5)

    # --- 3) features (transforms reuse baseline cache; extras rebuild) ---
    for feat in ["log_dist", "quad", "interact", "log_interact", "full"]:
        add(f"feat_{feat}", feat_mode=feat)
    add("feat_no_det", include_det_feats=False)
    add("feat_extra_competition", rebuild=True, extra_feats="competition")
    add("feat_extra_both", rebuild=True, extra_feats="both")

    # --- 4) losses (same pairs) ---
    add("loss_bce", loss="bce")
    add("loss_focal_g1", loss="focal", focal_gamma=1.0)
    add("loss_focal_soft", loss="focal_soft")
    add("loss_asymmetric_pw2", loss="asymmetric", pos_weight=2.0)
    add("loss_ranking", loss="ranking")
    add("loss_margin", loss="margin")

    seen = set()
    out = []
    for e in exps:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        out.append(e)
    return out


def find_cache(fold: int, root: Path | None = None) -> Path:
    if root is not None:
        here = root / "baseline_mlp" / f"fold_{fold}" / "cache"
        if (here / "train.npz").exists():
            return here
    base = HP / "runs/04_motion_corrector"
    for pat in (
        f"*motion_sp_p2_0917*/baseline_mlp/fold_{fold}/cache",
        f"*motion_sp_p2*/baseline_mlp/fold_{fold}/cache",
    ):
        cands = sorted(base.glob(pat))
        for c in reversed(cands):
            if (c / "train.npz").exists():
                return c
    raise FileNotFoundError(f"No SP+P2 baseline cache for fold {fold}")


def build_cmd(exp: dict, fold: int, out_dir: Path, cache: Path, gpu: int) -> list[str]:
    script = TRAIN if TRAIN.exists() else LOCAL_TRAIN
    cmd = [
        str(PY),
        str(script),
        "--data",
        str(DATA),
        "--proposals",
        str(PROPOSALS),
        "--splits",
        str(SPLITS),
        "--fold",
        str(fold),
        "--cache",
        str(cache),
        "--output",
        str(out_dir),
        "--parent-mode",
        "proposal_nn",
        "--device",
        "cuda:0",
        "--exp-name",
        exp["name"],
        "--arch",
        exp.get("arch", "mlp"),
        "--feat-mode",
        exp.get("feat_mode", "raw"),
        "--loss",
        exp.get("loss", "focal"),
        "--epochs",
        str(exp.get("epochs", 40)),
        "--patience",
        str(exp.get("patience", 7)),
        "--lr",
        str(exp.get("lr", 2e-3)),
        "--weight-decay",
        str(exp.get("weight_decay", 1e-4)),
        "--dropout",
        str(exp.get("dropout", 0.05)),
        "--residual-scale",
        str(exp.get("residual_scale", 2.0)),
        "--batch-size",
        str(exp.get("batch_size", 8192)),
        "--scheduler",
        exp.get("scheduler", "none"),
        "--focal-gamma",
        str(exp.get("focal_gamma", 2.0)),
        "--pos-weight",
        str(exp.get("pos_weight", 1.0)),
        "--tight",
        str(exp.get("tight", 6.2)),
        "--relaxed",
        str(exp.get("relaxed", 9.5)),
        "--velocity-weight",
        str(exp.get("velocity_weight", 0.52)),
        "--velocity-axes",
        str(exp.get("velocity_axes", "iso")),
        "--reg-weight",
        str(exp.get("reg_weight", 0.05)),
        "--mine-mode",
        str(exp.get("mine_mode", "hard")),
        "--negative-ratio",
        str(exp.get("negative_ratio", 20)),
        "--ambiguous-margin",
        str(exp.get("ambiguous_margin", 1.5)),
        "--extra-feats",
        str(exp.get("extra_feats", "none")),
        "--seed",
        str(2028 + fold),
    ]
    if exp.get("rebuild_cache"):
        cmd.append("--rebuild-cache")
    if exp.get("include_det_feats"):
        cmd.append("--include-det-feats")
    if exp.get("learn_logit_bias"):
        cmd.append("--learn-logit-bias")
    if exp.get("hidden"):
        cmd += ["--hidden", str(exp["hidden"])]
    return cmd


def run_one(exp: dict, fold: int, root: Path, gpu: int) -> dict:
    out = root / exp["name"] / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache" if exp.get("rebuild_cache") else find_cache(fold, root)
    cmd = build_cmd(exp, fold, out, cache, gpu)
    (out / "cmd.json").write_text(json.dumps({"cmd": cmd, "gpu": gpu, "cache": str(cache)}, indent=2))
    log = out / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with log.open("w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(Path(cmd[1]).parent))
    elapsed = time.time() - t0
    summary_path = out / "summary.json"
    best = None
    if summary_path.exists():
        best = json.loads(summary_path.read_text()).get("best_jaccard")
    elif (out / "metrics.json").exists():
        hist = json.loads((out / "metrics.json").read_text())
        best = max((r["jaccard"] for r in hist), default=None)
    row = {
        "name": exp["name"],
        "fold": fold,
        "gpu": gpu,
        "returncode": rc.returncode,
        "elapsed_s": elapsed,
        "best_jaccard": best,
        "out": str(out),
    }
    (out / "job_result.json").write_text(json.dumps(row, indent=2))
    print(
        f"[{exp['name']} fold{fold} gpu{gpu}] rc={rc.returncode} jacc={best} t={elapsed:.0f}s",
        flush=True,
    )
    return row


def aggregate(root: Path, results: list[dict]) -> None:
    by = {}
    for r in results:
        by.setdefault(r["name"], []).append(r)
    rows = []
    for name, items in by.items():
        js = [x["best_jaccard"] for x in items if x.get("best_jaccard") is not None]
        rows.append(
            {
                "name": name,
                "n_folds": len(items),
                "n_ok": sum(1 for x in items if x["returncode"] == 0 and x.get("best_jaccard") is not None),
                "mean_jaccard": float(sum(js) / len(js)) if js else None,
                "std_jaccard": float((sum((j - sum(js) / len(js)) ** 2 for j in js) / len(js)) ** 0.5) if len(js) > 1 else 0.0,
                "per_fold": {str(x["fold"]): x.get("best_jaccard") for x in sorted(items, key=lambda z: z["fold"])},
                "fail": [x["fold"] for x in items if x["returncode"] != 0],
            }
        )
    rows.sort(key=lambda r: (r["mean_jaccard"] is not None, r["mean_jaccard"] or -1), reverse=True)
    (root / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    lines = ["name\tn_ok\tmean\tstd\tfolds"]
    for r in rows:
        lines.append(
            f"{r['name']}\t{r['n_ok']}\t{r['mean_jaccard']}\t{r['std_jaccard']}\t{r['per_fold']}"
        )
    (root / "leaderboard.tsv").write_text("\n".join(lines) + "\n")
    print("\n=== TOP 15 ===", flush=True)
    for r in rows[:15]:
        print(f"{r['mean_jaccard']:.4f} ± {r['std_jaccard']:.4f}  {r['name']}  folds={r['per_fold']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="3,4,5,6,7")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="motion_sp_p2_0917_geom_mine_feat_loss")
    ap.add_argument("--max-exps", type=int, default=0, help="0=all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    exps = experiments()
    if args.max_exps:
        exps = exps[: args.max_exps]

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = HP / "runs/04_motion_corrector" / f"{stamp}_{args.tag}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiments.json").write_text(json.dumps(exps, indent=2))
    print(f"Root {root}\nExperiments {len(exps)} × folds {folds} = {len(exps)*len(folds)} jobs on GPUs {gpus}")

    jobs = [(exp, fold) for exp in exps for fold in folds]

    if args.dry_run:
        print("Dry run sample:", jobs[0], "gpus", gpus)
        print("Total jobs", len(jobs), "exps", len(exps))
        return

    # One dedicated worker thread per GPU — pulls from shared queue (no double-booking).
    from queue import Queue

    q: Queue = Queue()
    for job in jobs:
        q.put(job)
    for _ in gpus:
        q.put(None)

    results: list[dict] = []
    results_lock = __import__("threading").Lock()

    def worker(gpu: int) -> None:
        while True:
            item = q.get()
            if item is None:
                return
            exp, fold = item
            row = run_one(exp, fold, root, gpu)
            with results_lock:
                results.append(row)
                (root / "all_results.json").write_text(json.dumps(results, indent=2))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = [pool.submit(worker, gpu) for gpu in gpus]
        for fut in as_completed(futs):
            fut.result()

    (root / "all_results.json").write_text(json.dumps(results, indent=2))
    aggregate(root, results)
    print(f"\nWrote {root}/leaderboard.json")


if __name__ == "__main__":
    main()
