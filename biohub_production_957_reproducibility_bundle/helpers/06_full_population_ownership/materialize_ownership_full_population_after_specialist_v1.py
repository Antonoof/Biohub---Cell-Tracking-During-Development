#!/usr/bin/env python3
"""Cheap exact falsification of the complete ownership serving population.

Apply every full-population OOF source winner above the frozen threshold to the
specialist-final graph.  Transactions are prevalidated and atomic; existing
forks and their daughters are protected.  Only videos carrying at least one
claim are written for exact host-patched evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
WINNERS = BIO / "ownership_full_population_oof_v1/selected_source_winners.parquet"
CONTROL = BIO / "live_v2_global_bundle_ug12_matched_v1/train175/final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply(graph, rows: pd.DataFrame) -> dict[str, int]:
    counters: Counter[str] = Counter()
    table = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).to_dicts()
    times = {int(item["node_id"]): int(item["t"]) for item in table}
    positions = {
        int(item["node_id"]): np.asarray([item["z"], item["y"], item["x"]], np.float32)
        for item in table
    }
    nodes = set(times)
    edge_keys = set(graph.edge_attr_keys())
    locked_sources = {node for node in nodes if graph.out_degree(node) >= 2}
    locked_targets = {
        int(child)
        for source in locked_sources
        for child in graph.successors(source)
    }
    initial_in = {node: int(graph.in_degree(node)) for node in nodes}
    initial_out = {node: int(graph.out_degree(node)) for node in nodes}

    for row in rows.sort_values("score", ascending=False).itertuples(index=False):
        source, a, b = int(row.source), int(row.a), int(row.b)
        if source in locked_sources or source not in nodes:
            counters["rejected_source"] += 1
            continue
        if a == b or a not in nodes or b not in nodes:
            counters["rejected_daughter"] += 1
            continue
        if a in locked_targets or b in locked_targets:
            counters["rejected_locked_target"] += 1
            continue
        if times[a] != times[source] + 1 or times[b] != times[source] + 1:
            counters["rejected_nonadjacent"] += 1
            continue
        protected = False
        for child in (a, b):
            for parent in map(int, graph.predecessors(child)):
                if parent != source and graph.out_degree(parent) >= 2:
                    protected = True
                    break
            if protected:
                break
        if protected:
            counters["rejected_protected_fork_target"] += 1
            continue

        pair = {a, b}
        for child in list(graph.successors(source)):
            child = int(child)
            if child not in pair:
                graph.remove_edge(source, child)
                counters["removed_source_continuation"] += 1
        for child in pair:
            for parent in list(graph.predecessors(child)):
                parent = int(parent)
                if parent != source:
                    graph.remove_edge(parent, child)
                    counters["removed_competing_claim"] += 1
        for child in pair:
            if graph.has_edge(source, child):
                continue
            attrs = {}
            if "edge_prob" in edge_keys:
                attrs["edge_prob"] = float(row.score)
            if "edge_dist" in edge_keys:
                attrs["edge_dist"] = float(np.linalg.norm(positions[source] - positions[child]))
            graph.add_edge(source, child, attrs)
            counters["added_edge"] += 1
        if graph.out_degree(source) != 2:
            raise RuntimeError(f"Atomic serving transaction failed for {source}")
        locked_sources.add(source)
        locked_targets.update(pair)
        counters["applied"] += 1

    if any(graph.in_degree(node) > max(1, initial_in[node]) for node in nodes):
        raise RuntimeError("Full serving replay worsened in-degree")
    if any(graph.out_degree(node) > max(2, initial_out[node]) for node in nodes):
        raise RuntimeError("Full serving replay worsened out-degree")
    return dict(counters)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True)
    sys.path.insert(0, str(WORKSPACE / "scripts"))
    import replay_model_c_full_population_gate as graph_io

    winners = pd.read_parquet(WINNERS)
    aggregate: Counter[str] = Counter()
    records: list[dict] = []
    for index, (stem, local) in enumerate(winners.groupby("dataset", sort=True), 1):
        graph = graph_io.load_graph(CONTROL / f"{stem}.geff")
        counters = apply(graph, local)
        graph_io.save_graph(graph, graph_dir / f"{stem}.geff")
        aggregate.update(counters)
        records.append({
            "dataset": stem, "proposed": int(len(local)),
            **{key: int(value) for key, value in counters.items()},
        })
        if index % 20 == 0 or index == winners.dataset.nunique():
            print(
                f"full-population atomic replay {index}/{winners.dataset.nunique()} "
                f"proposed={sum(record['proposed'] for record in records)} "
                f"applied={aggregate.get('applied', 0)}",
                flush=True,
            )
    pd.DataFrame(records).to_csv(args.output / "per_video.csv", index=False)
    payload = {
        "version": "ownership-full-population-after-specialist-v1",
        "contract": (
            "complete grouped-video OOF serving population applied after specialist-final; "
            "atomic replacement; existing forks and daughters protected"
        ),
        "videos": int(winners.dataset.nunique()),
        "proposed": int(len(winners)),
        "aggregate": dict(aggregate),
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
