#!/usr/bin/env python3
"""GKF5 OOF sweep over Model-C decoder inputs / models / losses.

Native Model-C UNet stays frozen. This only retunes the V2+C event decoder.
Thresholds: mean-over-folds on OOF. held20/practice are not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model_c_v2_event_decoder import (  # noqa: E402
    attach_v2_raw_scores,
    load_video,
    train_family,
)

CONFIGS = [
    dict(name="hgb_v2c_baseline", pair="hgb", source="hgb", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_v2c_topk2", pair="hgb", source="hgb", use_c=True, topk=2, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_v2c_topk3", pair="hgb", source="hgb", use_c=True, topk=3, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_v2c_w8", pair="hgb", source="hgb", use_c=True, topk=1, pair_cap=8, source_cap=12, v2=False),
    dict(name="hgb_v2c_noweight", pair="hgb", source="hgb", use_c=True, topk=1, pair_cap=0, source_cap=0, v2=False),
    dict(name="hgb_deep_v2c", pair="hgb_deep", source="hgb_deep", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_slow_v2c", pair="hgb_slow", source="hgb_slow", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_v2_only", pair="hgb", source="hgb", use_c=False, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="hgb_v2c_v2scores", pair="hgb", source="hgb", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=True),
    dict(name="hgb_topk2_w8", pair="hgb", source="hgb", use_c=True, topk=2, pair_cap=8, source_cap=12, v2=False),
    dict(name="cb_deep_v2c", pair="catboost_deep", source="catboost_deep", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="cb_deep_topk2", pair="catboost_deep", source="catboost_deep", use_c=True, topk=2, pair_cap=75, source_cap=100, v2=False),
    dict(name="lgb_rank_hgb", pair="lightgbm_rank", source="hgb", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="cb_rank_hgb", pair="catboost_rank", source="hgb", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False),
    dict(name="tabm_focal_v2c", pair="tabm_focal", source="tabm_focal", use_c=True, topk=1, pair_cap=75, source_cap=100, v2=False, device="cuda"),
    dict(name="hgb_topk2_v2scores", pair="hgb", source="hgb", use_c=True, topk=2, pair_cap=75, source_cap=100, v2=True),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--event-cache", type=Path, required=True)
    p.add_argument("--full-population-cache", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--train-evidence", type=Path, required=True)
    p.add_argument("--held-evidence", type=Path, required=True)
    p.add_argument("--practice-evidence", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--v2-artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--max-pair-negatives", type=int, default=180_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", default="", help="Comma-separated config names to run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.oof_only = True
    args.inner_folds = 3
    args.skip_v2_control = True
    args.public_primary_evidence = None
    args.public_secondary_evidence = None
    args.ctc_evidence = None
    split = json.loads(args.split.read_text())
    train_names = [str(x).removesuffix(".zarr") for x in split["train"]]
    print(f"Loading {len(train_names)} training videos", flush=True)
    train = [load_video(stem, args, args.train_evidence) for stem in train_names]
    print("Attaching frozen V2 raw scores", flush=True)
    attach_v2_raw_scores(train, args)
    groups = np.asarray([video.stem for video in train])
    names = {c["name"] for c in CONFIGS}
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    if only:
        missing = set(only) - names
        if missing:
            raise SystemExit(f"unknown configs: {sorted(missing)}")
        configs = [c for c in CONFIGS if c["name"] in only]
    else:
        configs = CONFIGS
    board = []
    board_path = args.output / "sweep_board.json"
    for cfg in configs:
        args.topk_pairs = int(cfg["topk"])
        args.pair_pos_cap = float(cfg["pair_cap"])
        args.source_pos_cap = float(cfg["source_cap"])
        args.use_v2_source_scores = bool(cfg["v2"])
        args.pair_backend = cfg["pair"]
        args.source_backend = cfg["source"]
        args.device = cfg.get("device", args.device)
        print(f"\n=== {cfg['name']} {cfg} ===", flush=True)
        try:
            result = train_family(
                train, [], [], args, groups,
                use_c=bool(cfg["use_c"]),
                label=cfg["name"],
                backend=cfg["source"],
            )
            oof = result["oof_frozen_threshold"]
            row = {
                "name": cfg["name"],
                "config": {k: v for k, v in cfg.items() if k != "name"},
                "mean_jaccard": oof["mean_jaccard"],
                "std_jaccard": oof["std_jaccard"],
                "threshold": oof["threshold"],
                "per_fold": oof["per_fold"],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"{cfg['name']} FAILED {exc}", flush=True)
            row = {
                "name": cfg["name"],
                "config": {k: v for k, v in cfg.items() if k != "name"},
                "mean_jaccard": None,
                "error": str(exc),
            }
        board.append(row)
        ranked = sorted(
            [r for r in board if r.get("mean_jaccard") is not None],
            key=lambda r: r["mean_jaccard"],
            reverse=True,
        )
        payload = {"baseline_hgb_v2c": 0.3868548830006573, "ranked": ranked, "all": board}
        board_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(row, indent=2), flush=True)
    print(json.dumps({"ranked": board_path.as_posix()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
