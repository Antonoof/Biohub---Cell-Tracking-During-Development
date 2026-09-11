#!/usr/bin/env python3
"""Atomically apply full-population cardinality-head decisions on held-20."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from replay_model_c_full_population_gate import (
    apply_rows,
    candidate_rows,
    load_graph,
    save_graph,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--population",
        type=Path,
        default=Path("data/public934_source_cardinality_held20_population_v2"),
    )
    p.add_argument(
        "--head-summary",
        type=Path,
        default=Path("data/public934_source_cardinality_head_v2/summary.json"),
    )
    p.add_argument(
        "--graphs",
        type=Path,
        default=Path(
            "data/public914_backbone_matched_v1/"
            "division_candidate_audit_v1/pre_safe_graphs"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/public934_source_cardinality_held20_replay_v2"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.head_summary.read_text())
    threshold = float(summary["saved_config"]["threshold"])
    expected = set()
    for worker in sorted(args.population.glob("worker_*.json")):
        expected.update(row["dataset"] for row in json.loads(worker.read_text()))
    available = {path.name for path in args.population.iterdir() if path.is_dir() and not path.name.startswith(".")}
    if len(available) != 20:
        raise RuntimeError(f"Expected 20 completed population videos, found {len(available)}")
    if expected and expected != available:
        raise RuntimeError(f"Worker manifest mismatch: {len(expected)} != {len(available)}")

    graph_output = args.output / "graphs"
    if graph_output.exists():
        shutil.rmtree(graph_output)
    graph_output.mkdir(parents=True)
    totals: dict[str, int] = {}
    per_video = []
    for index, stem in enumerate(sorted(available), 1):
        root = args.population / stem
        source = np.load(root / "source_id.npy")
        tubes = np.load(root / "source_tube.npy")
        score = np.load(root / "source_score.npy")
        pairs = np.load(root / "best_pair_nodes.npy")
        rows = candidate_rows(
            score, tubes, pairs,
            threshold=threshold, rate_per_1000_tubes=None, floor=0.0,
        )
        graph = load_graph(args.graphs / f"{stem}.geff")
        counters = apply_rows(
            graph, source, tubes, pairs, score, rows,
            policy="public934-source-cardinality-v1",
        )
        save_graph(graph, graph_output / f"{stem}.geff")
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + int(value)
        per_video.append({"dataset": stem, **counters})
        print(
            f"[{index:02d}/20] {stem}: passing={len(rows)} "
            f"accepted={counters.get('selected_divisions', 0)}",
            flush=True,
        )
    payload = {
        "version": "public934-source-cardinality-held20-replay-v2",
        "threshold": threshold,
        "threshold_source": "grouped train-175 OOF; held-20 untouched",
        "graphs": len(per_video),
        "totals": totals,
        "per_video": per_video,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
