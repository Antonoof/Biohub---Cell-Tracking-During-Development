"""Fair P1/P2-only source-cardinality comparison.

The production cardinality experiment trained V2+Model-C and
V2+Model-C+P1/P2 heads.  Its later "without C" audit masked Model-C features
after fitting, which cannot answer whether native P1/P2 evidence can learn a
strong division head on its own.

This research-only script holds every other variable fixed and trains three
families from scratch:

* V2 geometry + Model-C evidence (control)
* V2 geometry + native P1/P2 evidence (requested treatment)
* V2 geometry + Model-C + native P1/P2 evidence (production-family control)

Thresholds are selected only from grouped-video OOF predictions and then
frozen for the untouched held-20 evaluation.  It does not alter production
artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import train_public934_source_cardinality_head as base


MODES = ("c_v2", "p1p2_v2", "combined")


def parse_args() -> argparse.Namespace:
    parser = base.parse_args()
    parser.output = Path("artifacts/unigraft_p1p2_cardinality_head_v1")
    return parser


def prepare(videos: list[object], mode: str) -> list[base.PreparedVideo]:
    if mode not in MODES:
        raise ValueError(f"Unknown evidence mode: {mode}")
    result: list[base.PreparedVideo] = []
    for video in videos:
        blocks = [video.pair_x]
        if mode in ("c_v2", "combined"):
            blocks.append(video.c_pair_x)
        if mode in ("p1p2_v2", "combined"):
            blocks.append(video.public_pair_x)
        pair_features = np.nan_to_num(
            np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        result.append(
            base.PreparedVideo(
                video=video,
                pair_features=pair_features,
                pair_rows=base.rows_by_source(video.pair_owner, len(video.source_x)),
            )
        )
    return result


def train_family(
    label: str,
    raw_train: list[object],
    raw_held: list[object],
    args: argparse.Namespace,
    trainer,
    mode: str,
) -> dict:
    train = prepare(raw_train, mode)
    held = prepare(raw_held, mode)
    groups = np.asarray([video.stem for video in raw_train])
    splitter = GroupKFold(n_splits=args.folds)
    oof = []
    dummy = np.zeros(len(train))
    fold_details = []
    for fold, (fit_index, val_index) in enumerate(splitter.split(dummy, groups=groups)):
        fit = [train[int(index)] for index in fit_index]
        val = [train[int(index)] for index in val_index]
        print(f"{label} fold {fold}: fit={len(fit)} val={len(val)}", flush=True)
        model, normalizer, details = base.fit_model(fit, args, args.seed + fold)
        oof.extend(base.score_videos(model, normalizer, val, args))
        fold_details.append(details)
    frozen = trainer.threshold_sweep(oof)
    print(f"{label} GROUPED OOF {json.dumps(frozen)}", flush=True)
    final_model, final_normalizer, final_details = base.fit_model(
        train, args, args.seed + 100
    )
    held_scored = base.score_videos(final_model, final_normalizer, held, args)
    held_frozen = trainer.metric(held_scored, frozen["threshold"])
    held_diagnostic = trainer.threshold_sweep(held_scored)
    by_embryo = {
        embryo: trainer.metric(
            [row for row in held_scored if row[0].embryo == embryo],
            frozen["threshold"],
        )
        for embryo in ("44b6", "6bba")
    }
    print(f"{label} HELD FROZEN {json.dumps(held_frozen)}", flush=True)
    return {
        "label": label,
        "mode": mode,
        "source_dim": int(final_normalizer.source_mean.shape[0]),
        "pair_dim": int(final_normalizer.pair_mean.shape[0]),
        "oof_frozen": frozen,
        "held_frozen": held_frozen,
        "held_diagnostic_best_do_not_deploy": held_diagnostic,
        "held_by_embryo": by_embryo,
        "fold_details": fold_details,
        "final_details": final_details,
        "model": final_model,
        "normalizer": final_normalizer,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = base.load_module(args.trainer, "p1p2_only_cardinality_decoder_trainer")
    tracker_data = base.load_module(
        args.trainer.parent / "tracker_e_data.py", "p1p2_only_cardinality_tracker_data"
    )
    split = json.loads(args.split.read_text())
    print("Loading exact-frame train-175", flush=True)
    train = base.load_partition(
        list(map(str, split["train"])), args, trainer, tracker_data.load_video
    )
    print("Loading untouched held-20", flush=True)
    held = base.load_partition(
        list(map(str, split["held"])), args, trainer, tracker_data.load_video
    )

    families = {
        "model_c_control": train_family(
            "CARDINALITY V2+C", train, held, args, trainer, "c_v2"
        ),
        "p1p2_only": train_family(
            "CARDINALITY V2+P1P2", train, held, args, trainer, "p1p2_v2"
        ),
        "combined_control": train_family(
            "CARDINALITY V2+C+P1P2", train, held, args, trainer, "combined"
        ),
    }

    requested = families["p1p2_only"]
    model_c = families["model_c_control"]
    combined = families["combined_control"]
    promotion = {
        "beats_model_c_oof": requested["oof_frozen"]["jaccard"]
        > model_c["oof_frozen"]["jaccard"],
        "beats_model_c_held": requested["held_frozen"]["jaccard"]
        > model_c["held_frozen"]["jaccard"],
        "beats_combined_oof": requested["oof_frozen"]["jaccard"]
        > combined["oof_frozen"]["jaccard"],
        "beats_combined_held": requested["held_frozen"]["jaccard"]
        > combined["held_frozen"]["jaccard"],
        "both_embryos_nonzero": all(
            requested["held_by_embryo"][embryo]["tp"] > 0
            for embryo in ("44b6", "6bba")
        ),
    }
    promotion["p1p2_replaces_model_c"] = bool(
        promotion["beats_model_c_oof"]
        and promotion["beats_model_c_held"]
        and promotion["both_embryos_nonzero"]
    )
    promotion["p1p2_replaces_combined"] = bool(
        promotion["beats_combined_oof"]
        and promotion["beats_combined_held"]
        and promotion["both_embryos_nonzero"]
    )

    # Preserve the treatment checkpoint for audit, never as production.
    treatment_config = {
        "source_dim": requested["source_dim"],
        "pair_dim": requested["pair_dim"],
        "hidden_source": args.hidden_source,
        "hidden_pair": args.hidden_pair,
        "max_pairs": args.max_pairs,
        "threshold": requested["oof_frozen"]["threshold"],
        "evidence_mode": "v2_geometry_plus_p1p2_without_model_c",
    }
    base.save_head(
        args.output / "p1p2_only_source_cardinality_head.pt",
        requested["model"],
        requested["normalizer"],
        treatment_config,
    )
    for family in families.values():
        family.pop("model")
        family.pop("normalizer")
    summary = {
        "version": "public934-p1p2-only-cardinality-head-v1",
        "purpose": "fair from-scratch P1/P2-only division-head comparison",
        "fixed_protocol": {
            "population": "same exact-frame 175/20 population as production cardinality v2",
            "labels": "exact division time; confirmed continuations; unknown sources excluded",
            "folds": args.folds,
            "max_pairs": args.max_pairs,
            "threshold": "selected on grouped train-video OOF and frozen for held-20",
            "null_class": False,
        },
        "families": families,
        "promotion": promotion,
        "treatment_config": treatment_config,
        "production_modified": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"DONE output={args.output} "
        f"p1p2_replaces_model_c={promotion['p1p2_replaces_model_c']} "
        f"p1p2_replaces_combined={promotion['p1p2_replaces_combined']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
