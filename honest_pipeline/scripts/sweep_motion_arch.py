#!/usr/bin/env python3
"""Launch ~30 motion-corrector architecture/feature/train experiments on GKF5.

Reuses per-fold caches from a baseline motion run (no rebuild) so each fold is ~1–3 min.
Round-robins jobs across GPUs 3–7.
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
PROPOSALS = BIO / "data/honest_ab_proposals_oof"
DATA = BIO / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
BASE_CACHE_GLOB = "20260914T073012Z_motion_gkf5_fold{fold}"


def experiments() -> list[dict]:
    """~30 named configs. Geometry (tight/relaxed/vw) fixed → shared cache OK."""
    exps: list[dict] = []

    def add(name: str, **kw):
        exps.append({"name": name, **kw})

    # --- architectures (raw features, default train) ---
    for arch in [
        "mlp",
        "mlp_wide",
        "mlp_deep",
        "resmlp",
        "realmlp",
        "realmlp_wide",
        "tabm",
        "tabm_wide",
        "gated",
        "se_mlp",
        "highway",
    ]:
        add(f"arch_{arch}", arch=arch)

    # --- feature modes on best-ish arch family ---
    for feat in ["log_dist", "quad", "interact", "log_interact", "full"]:
        add(f"feat_{feat}_realmlp", arch="realmlp", feat_mode=feat)
        add(f"feat_{feat}_tabm", arch="tabm", feat_mode=feat)

    # --- residual scale ---
    for rs in [1.0, 3.0, 4.0, 6.0]:
        add(f"rscale_{rs:g}_realmlp", arch="realmlp", residual_scale=rs)

    # --- training ---
    add("train_lr5e4_realmlp", arch="realmlp", lr=5e-4)
    add("train_lr5e3_realmlp", arch="realmlp", lr=5e-3)
    add("train_wd1e3_realmlp", arch="realmlp", weight_decay=1e-3)
    add("train_drop02_realmlp", arch="realmlp", dropout=0.2)
    add("train_cosine_realmlp", arch="realmlp", scheduler="cosine", epochs=50, patience=10)
    add("train_bce_realmlp", arch="realmlp", loss="bce")
    add("train_asym_pw2_realmlp", arch="realmlp", loss="asymmetric", pos_weight=2.0)
    add("train_focal_g1_realmlp", arch="realmlp", loss="focal", focal_gamma=1.0)
    add("train_learn_bias_realmlp", arch="realmlp", learn_logit_bias=True)
    add("train_bs4k_realmlp", arch="realmlp", batch_size=4096)
    add("train_patience12_tabm", arch="tabm", patience=12, epochs=50)
    add("train_tabm_k16", arch="tabm", tabm_k=16)
    add("combo_full_realmlp_rs4", arch="realmlp", feat_mode="full", residual_scale=4.0, scheduler="cosine", epochs=50, patience=10)
    add("combo_interact_tabm_asym", arch="tabm", feat_mode="interact", loss="asymmetric", pos_weight=2.0, patience=10)
    add("combo_log_gated_rs3", arch="gated", feat_mode="log_dist", residual_scale=3.0, lr=1e-3, dropout=0.1)

    # Deduplicate by name (arch_mlp etc already unique)
    seen = set()
    out = []
    for e in exps:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        out.append(e)
    return out


def find_cache(fold: int) -> Path:
    base = HP / "runs/04_motion_corrector"
    # Prefer known baseline; else any fold cache
    preferred = base / BASE_CACHE_GLOB.format(fold=fold) / "cache"
    if (preferred / "train.npz").exists():
        return preferred
    cands = sorted(base.glob(f"*motion_gkf5_fold{fold}/cache"))
    for c in reversed(cands):
        if (c / "train.npz").exists():
            return c
    raise FileNotFoundError(f"No cache for fold {fold}")


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
        "--tabm-k",
        str(exp.get("tabm_k", 8)),
        "--focal-gamma",
        str(exp.get("focal_gamma", 2.0)),
        "--pos-weight",
        str(exp.get("pos_weight", 1.0)),
        "--seed",
        str(2028 + fold),
    ]
    if exp.get("learn_logit_bias"):
        cmd.append("--learn-logit-bias")
    if exp.get("hidden"):
        cmd += ["--hidden", str(exp["hidden"])]
    return cmd


def run_one(exp: dict, fold: int, root: Path, gpu: int) -> dict:
    out = root / exp["name"] / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    cache = find_cache(fold)
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
    ap.add_argument("--tag", default="motion_arch_sweep_v1")
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
