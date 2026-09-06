#!/usr/bin/env python3
"""Add the independent P1/P2 UG2 transactions after complete UG1 output."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from replay_model_c_full_population_gate import (
    apply_rows,
    candidate_rows,
    load_graph,
    save_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ug1-graphs",
        type=Path,
        default=Path(
            "/home/tweak/public944_cardinality_deployment_head_v1/"
            "held20_replay/graphs"
        ),
        help="Complete UG1 transaction graphs before finalization.",
    )
    parser.add_argument(
        "--ug2-population",
        type=Path,
        default=Path("/home/tweak/bio/unigraft_p1p2_ug2_held20_population_v1"),
    )
    parser.add_argument(
        "--ug2-head",
        type=Path,
        default=Path(
            "/home/tweak/bio/public934_p1p2_only_cardinality_head_v1/"
            "p1p2_only_source_cardinality_head.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/tweak/bio/unigraft_p1p2_late_ensemble_held20_v1"),
    )
    parser.add_argument("--expected-videos", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.ug2_head, map_location="cpu", weights_only=False)
    threshold = float(payload["config"]["threshold"])
    available = {
        path.name
        for path in args.ug2_population.iterdir()
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (path / "source_score.npy").is_file()
        and (path / "best_pair_nodes.npy").is_file()
    }
    if len(available) != args.expected_videos:
        raise RuntimeError(
            f"Expected {args.expected_videos} UG2 population videos, found {len(available)}"
        )
    graph_output = args.output / "graphs"
    if graph_output.exists():
        shutil.rmtree(graph_output)
    graph_output.mkdir(parents=True)

    totals: dict[str, int] = {}
    per_video: list[dict] = []
    for index, stem in enumerate(sorted(available), 1):
        root = args.ug2_population / stem
        source = np.load(root / "source_id.npy")
        tubes = np.load(root / "source_tube.npy")
        score = np.load(root / "source_score.npy")
        pairs = np.load(root / "best_pair_nodes.npy")
        rows = candidate_rows(
            score,
            tubes,
            pairs,
            threshold=threshold,
            rate_per_1000_tubes=None,
            floor=0.0,
        )
        graph_path = args.ug1_graphs / f"{stem}.geff"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing completed UG1 graph: {graph_path}")
        graph = load_graph(graph_path)
        counters = apply_rows(
            graph,
            source,
            tubes,
            pairs,
            score,
            rows,
            policy="unigraft-p1p2-independent-late-ensemble-v1",
        )
        save_graph(graph, graph_output / f"{stem}.geff")
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + int(value)
        per_video.append(
            {
                "dataset": stem,
                "ug2_threshold_passing": int(len(rows)),
                **{key: int(value) for key, value in counters.items()},
            }
        )
        print(
            f"[{index:02d}/{args.expected_videos}] {stem}: "
            f"UG2 passing={len(rows)} accepted={counters.get('selected_divisions', 0)}",
            flush=True,
        )

    summary = {
        "version": "unigraft-p1p2-independent-late-ensemble-held20-v1",
        "contract": {
            "UG1": "complete frozen production cardinality output",
            "UG2": "complete independent V2+P1/P2-only output",
            "merge_order": "UG1 complete first; UG2 accepted atomically only after both branches score independently",
            "UG2_threshold": threshold,
        },
        "graphs": len(per_video),
        "totals": totals,
        "per_video": per_video,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
