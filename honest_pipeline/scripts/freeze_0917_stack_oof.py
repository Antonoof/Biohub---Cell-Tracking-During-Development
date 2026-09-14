#!/usr/bin/env python3
"""Freeze 0_917 stack as the detector (replaces P1 and P2) + DeepCenter OOF gate.

P1/P2 unet-transformer blend is dropped. Detector graphs are the Exp203 classical
ensemble already scored on full kaggle train. DeepCenter gate is mean-over-folds
on existing GKF5 CSVs (never held20).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_serve_hparams_oof import freeze_deepcenter  # noqa: E402

BAKEOFF = ROOT / "runs/p1_candidate_compare/kaggle_train_all"
DEFAULT_KNOBS = {
    "unet_thresh": 0.15,
    "cand_thr": 0.05,
    "nms_um": 4.0,
    "max_link_um": 10.0,
    "tight_um": 6.0,
    "repair": True,
    "gap_dt": 0,
    "short_min": 6,
    "linefit_weight": 0.8,
    "linefit_window": 2,
    "weights": [
        "unet3d_bright.pt",
        "unet3d_traintophat.pt",
        "unet3d_v2_tophat_b32.pt",
    ],
}


def main() -> None:
    out = ROOT / "runs" / "serve_hparams_0917_dc"
    out.mkdir(parents=True, exist_ok=True)
    metrics_c = BAKEOFF / "classical_exp203" / "metrics.json"
    metrics_s = BAKEOFF / "support_pack" / "metrics.json"
    classical = json.loads(metrics_c.read_text()) if metrics_c.exists() else {}
    support = json.loads(metrics_s.read_text()) if metrics_s.exists() else {}
    dc = freeze_deepcenter(out)
    serve = {
        "protocol": "public_exp203_alltrain_weights + deepcenter_gkf5_oof_gate",
        "detector": "exp203_classical_0_917_stack",
        "replaces": ["01_p1_detector", "02_p2_detector"],
        "p1": None,
        "p2": None,
        "ab_blend": None,
        "note": (
            "User-selected 0_917 stack over Support Pack despite lower adj on kaggle train. "
            "Weights are public all-train (same leak class as the bake-off). "
            "DeepCenter gate is honest GKF5 mean-over-folds. "
            "Downstream tabular/motion retune on 0_917 graphs, not P1/P2."
        ),
        "exp203_knobs": DEFAULT_KNOBS,
        "kaggle_train_199": {
            "classical_exp203": {
                "mean_adj_edge_jaccard": classical.get("mean_adj_edge_jaccard"),
                "mean_node_recall": classical.get("mean_node_recall"),
                "n_scored": classical.get("n_scored"),
                "gkf5_mean_of_fold_means_adj": (classical.get("gkf5") or {}).get("mean_of_fold_means_adj"),
            },
            "support_pack_reference_not_selected": {
                "mean_adj_edge_jaccard": support.get("mean_adj_edge_jaccard"),
                "mean_node_recall": support.get("mean_node_recall"),
            },
            "current_p1_leak_fold0_train140": 0.8084,
        },
        "graphs_dir": str(BAKEOFF / "classical_exp203"),
        "deepcenter_gate": dc,
    }
    (out / "serve_config.json").write_text(json.dumps(serve, indent=2) + "\n")
    print(json.dumps({k: serve[k] for k in serve if k != "deepcenter_gate"}, indent=2))
    print(
        "DeepCenter gate thr=",
        dc.get("best_threshold"),
        "f1=",
        dc.get("best_mean_f1_sparse"),
        "cons=",
        dc.get("conservative_threshold_mean"),
    )
    print("Wrote", out / "serve_config.json")


if __name__ == "__main__":
    main()
