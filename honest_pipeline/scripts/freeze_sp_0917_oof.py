#!/usr/bin/env python3
"""Freeze serve hparams: P1=Support Pack graph, P2=0_917 motion member, DeepCenter GKF5.

No graph blend. DeepCenter gate is reused from existing GKF5 CSVs (not retrained).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_serve_hparams_oof import freeze_deepcenter  # noqa: E402

BAKEOFF = ROOT / "runs/p1_candidate_compare/kaggle_train_all"
EXP203_KNOBS = {
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
SP_KNOBS = {
    "det_threshold": 0.99,
    "use_ilp": True,
    "method": "unet_transformer",
    "unet_batch_size": 8,
}


def main() -> None:
    out = ROOT / "runs" / "serve_hparams_sp_p2_0917"
    out.mkdir(parents=True, exist_ok=True)
    support = json.loads((BAKEOFF / "support_pack" / "metrics.json").read_text())
    classical = json.loads((BAKEOFF / "classical_exp203" / "metrics.json").read_text())
    p1_oof = (support.get("gkf5") or {}).get("mean_of_fold_means_adj")
    p2_oof = (classical.get("gkf5") or {}).get("mean_of_fold_means_adj")
    if p1_oof is None or p2_oof is None:
        raise SystemExit("Missing gkf5 mean_of_fold_means_adj in bake-off metrics")
    dc = freeze_deepcenter(out)
    serve = {
        "protocol": "p1_support_pack_graph + p2_0917_member_only + deepcenter_gkf5_oof_gate",
        "graph_blend": None,
        "p1": {
            "detector": "support_pack",
            "graphs_dir": str(BAKEOFF / "support_pack"),
            "knobs": SP_KNOBS,
            "kaggle_train_199_adj": support.get("mean_adj_edge_jaccard"),
            "kaggle_train_199_node_recall": support.get("mean_node_recall"),
            "gkf5": support.get("gkf5"),
        },
        "p2": {
            "detector": "exp203_classical_0_917",
            "graphs_dir": str(BAKEOFF / "classical_exp203"),
            "knobs": EXP203_KNOBS,
            "kaggle_train_199_adj": classical.get("mean_adj_edge_jaccard"),
            "kaggle_train_199_node_recall": classical.get("mean_node_recall"),
            "gkf5": classical.get("gkf5"),
        },
        "ab_blend": None,
        "serve_graph": "p1",
        "p2_role": "motion_member_only",
        "deepcenter_gate": dc,
        "note": (
            "P1=Support Pack is the serve/track graph. P2=0_917 is the second "
            "proposal member (A/B disagreement), not a fused tracker. "
            "Graph blend scored worse than P1 (union 0.876 / relink 0.870 vs P1 0.911). "
            "DeepCenter gate from GKF5 mean-over-folds."
        ),
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
