#!/usr/bin/env python3
"""Freeze motion winner + materialize OOF graph bank for downstream trainers.

Serve graph stays P1 Support Pack until motion GEFFs beat P1 adj_edge_jaccard.
Downstream Model C / grafts should read --graph-dir from manifest.selected_graph_dir.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF = ROOT / "runs/p1_candidate_compare/kaggle_train_all"
P1 = BAKEOFF / "support_pack"
P2 = BAKEOFF / "classical_exp203"
SWEEP = ROOT / "runs/04_motion_corrector/20260914T200230Z_motion_sp_p2_0917_geom_mine_feat_loss"
WINNER = "geom_tight5_rel8"
SERVE = ROOT / "runs/serve_hparams_sp_p2_0917"
OUT = ROOT / "runs/oof_graphs"


def main() -> None:
    serve_path = SERVE / "serve_config.json"
    serve = json.loads(serve_path.read_text()) if serve_path.exists() else {}
    ckpts = {
        str(f): str(SWEEP / WINNER / f"fold_{f}" / "motion_corrector_best.pt") for f in range(5)
    }
    missing = [k for k, v in ckpts.items() if not Path(v).exists()]
    if missing:
        raise SystemExit(f"Missing motion checkpoints: {missing}")
    lb = json.loads((SWEEP / "leaderboard.json").read_text())
    top = next(r for r in lb if r["name"] == WINNER)
    p1m = json.loads((P1 / "metrics.json").read_text())
    p2m = json.loads((P2 / "metrics.json").read_text())
    motion_metrics = None
    mot_dir = OUT / "motion_geom_tight5_rel8"
    if (mot_dir / "metrics.json").exists():
        motion_metrics = json.loads((mot_dir / "metrics.json").read_text())

    p1_adj = float(p1m.get("mean_adj_edge_jaccard") or 0)
    mot_adj = float((motion_metrics or {}).get("mean_adj_edge_jaccard") or -1)
    selected = "motion_geom_tight5_rel8" if mot_adj > p1_adj else "p1_support_pack"

    p1_bank = OUT / "p1_support_pack"
    p1_bank.mkdir(parents=True, exist_ok=True)
    n_link = 0
    for g in sorted(P1.glob("*.geff")):
        dest = p1_bank / g.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(g.resolve(), dest)
        n_link += 1

    serve["motion"] = {
        "winner": WINNER,
        "assignment_jaccard_mean": top.get("mean_jaccard"),
        "per_fold_assignment": top.get("per_fold"),
        "tight": 5.0,
        "relaxed": 8.0,
        "arch": "mlp",
        "include_det_feats": True,
        "checkpoints": ckpts,
        "applied_to_graph": motion_metrics is not None,
        "e2e_adj_edge_jaccard": (motion_metrics or {}).get("mean_adj_edge_jaccard"),
        "e2e_gkf5": (motion_metrics or {}).get("gkf5"),
    }
    serve["downstream"] = {
        "selected_graph": selected,
        "p1_adj199": p1_adj,
        "motion_adj199": None if mot_adj < 0 else mot_adj,
        "rule": "use motion graphs only if adj_edge_jaccard > P1",
    }
    SERVE.mkdir(parents=True, exist_ok=True)
    serve_path.write_text(json.dumps(serve, indent=2) + "\n")

    selected_dir = OUT / "current"
    if selected_dir.exists() or selected_dir.is_symlink():
        selected_dir.unlink()
    os.symlink((OUT / selected).resolve(), selected_dir)

    manifest = {
        "p1": {
            "detector": "support_pack",
            "graph_dir": str(p1_bank),
            "n_geff": n_link,
            "mean_adj_edge_jaccard": p1_adj,
            "gkf5": p1m.get("gkf5"),
        },
        "p2": {
            "detector": "exp203_classical_0_917",
            "role": "motion_member_only",
            "graph_dir": str(P2),
            "mean_adj_edge_jaccard": p2m.get("mean_adj_edge_jaccard"),
        },
        "motion": serve["motion"],
        "deepcenter": serve.get("deepcenter_gate"),
        "selected_graph_dir": str((OUT / selected).resolve()),
        "selected": selected,
        "model_c_graph_dir": str((OUT / selected).resolve()),
        "splits": str(ROOT / "splits/dataset_splits_gkf5_train175.json"),
        "kaggle_train": "/data/projects/ryzhichkin/biohub/kaggle/input/competitions/biohub-cell-tracking-during-development/train",
        "note": (
            "P1 is the serve/track graph. P2 is motion member only. "
            "Motion e2e adj is kept for audit but is not the serve graph unless it beats P1. "
            "Model C / cardinality / grafts should train on selected_graph_dir. "
            "Rebuild event/evidence caches from these GEFFs; do not reuse William .951 graphs."
        ),
        "next": {
            "event_cache": str(OUT / "model_c_audit/division_training_cache_v1_parts"),
            "full_population_cache": str(OUT / "full_population_cache"),
            "decoder_launch": str(ROOT / "stages/05_model_c/launch_decoder.py"),
            "needs": "native Model-C evidence npz (train/held/practice) before decoder train",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in manifest if k != "deepcenter"}, indent=2))
    print("Wrote", OUT / "manifest.json")
    print("serve", serve_path)


if __name__ == "__main__":
    main()
