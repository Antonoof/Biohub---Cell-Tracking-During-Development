#!/usr/bin/env python3
"""Launch Model C V2 event decoder on frozen honest oof_graphs.

Requires:
  runs/oof_graphs/manifest.json
  event-cache parts from train_division_pair_model --cache-only
  full_population_cache from materialize_model_c_audit.py
  native Model-C evidence npz dirs (optional until exported)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from biohub_cv.logging_utils import RunLogger, new_run_dir  # noqa: E402

WILLIAM = (
    ROOT.parent
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers/02_model_c/train_model_c_v2_event_decoder.py"
)
LOCAL = ROOT.parent / "helpers/02_model_c/train_model_c_v2_event_decoder.py"
SPLIT = ROOT.parent / "helpers/02_model_c/division_balanced_175_20_split.json"
V2_CANDIDATES = (
    ROOT.parent / "kaggle/input/datasets/antonoof/all_files/model_c",
    ROOT.parent / "legacy_antonoof/all_files/model_c",
    ROOT.parent / "william-duckworth-reproducible-training-pipeline/helpers/02_model_c",
)


def _trainer() -> Path:
    for cand in (LOCAL, WILLIAM):
        if cand.exists() and (cand.parent / "decoder_backends.py").exists():
            return cand
    if LOCAL.exists():
        return LOCAL
    return WILLIAM


def _v2_artifact() -> Path | None:
    need = ("pair_model.joblib", "v1_source_model.joblib", "gate_model.joblib")
    for cand in V2_CANDIDATES:
        if all((cand / name).is_file() for name in need):
            return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="model_c_gkf5")
    p.add_argument(
        "--event-cache",
        type=Path,
        default=ROOT / "runs/oof_graphs/model_c_audit/division_training_cache_v1_parts",
    )
    p.add_argument(
        "--full-population-cache",
        type=Path,
        default=ROOT / "runs/oof_graphs/full_population_cache",
    )
    p.add_argument("--train-evidence", type=Path, required=True)
    p.add_argument("--held-evidence", type=Path, required=True)
    p.add_argument("--practice-evidence", type=Path, required=True)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--v2-artifact", type=Path, default=None)
    p.add_argument("--backends", default="catboost,lightgbm,tabm,realmlp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    man = json.loads((ROOT / "runs/oof_graphs/manifest.json").read_text())
    graph_dir = Path(man["selected_graph_dir"])
    script = _trainer()
    v2 = args.v2_artifact or _v2_artifact()
    if v2 is None:
        raise SystemExit("Missing Division V2 artifact (pair_model.joblib / v1_source_model.joblib / gate_model.joblib)")
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    n_pt = len(list(args.event_cache.glob("*.pt")))
    n_geff = len(list(graph_dir.glob("*.geff")))
    if n_pt < 100:
        raise SystemExit(
            f"Event cache not ready ({n_pt} .pt in {args.event_cache}). "
            "Run train_division_pair_model.py --cache-only on the materialized audit."
        )
    if n_geff < 100:
        raise SystemExit(f"Graph bank too small: {n_geff} GEFFs in {graph_dir}")

    run_dir = new_run_dir(ROOT / "runs", "05_model_c", args.tag)
    logger = RunLogger(
        run_dir,
        stage="05_model_c",
        config={
            "graph_dir": str(graph_dir),
            "event_cache": str(args.event_cache),
            "selected": man.get("selected"),
            "v2_artifact": str(v2),
            "trainer": str(script),
        },
    )
    cmd = [
        sys.executable,
        str(script),
        "--event-cache",
        str(args.event_cache),
        "--full-population-cache",
        str(args.full_population_cache),
        "--graph-dir",
        str(graph_dir),
        "--train-evidence",
        str(args.train_evidence),
        "--held-evidence",
        str(args.held_evidence),
        "--practice-evidence",
        str(args.practice_evidence),
        "--split",
        str(args.split),
        "--v2-artifact",
        str(v2),
        "--output",
        str(run_dir / "weights"),
        "--backends",
        args.backends,
        "--device",
        args.device,
        "--inner-folds",
        str(args.inner_folds),
        *extra,
    ]
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    logger.write_summary({"returncode": rc, "graph_dir": str(graph_dir)})
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
