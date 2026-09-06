#!/usr/bin/env python3
"""Replay strict and moderate native CandidateGRAFT/EndpointGRAFT OOF variants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
sys.path.insert(0, str(WORKSPACE / "scripts"))

import audit_edgegraft_component_oracle_v1 as edgebase
from smoke_edgegraft_v3_deploy_parity_v1 import load


VARIANTS = {
    "direct_strict": {"direct": 0.999, "leaf": None},
    "direct_moderate": {"direct": 0.990, "leaf": None},
    "leaf_strict": {"direct": None, "leaf": 0.995},
    "leaf_moderate": {"direct": None, "leaf": 0.975},
    "combined_strict": {"direct": 0.999, "leaf": 0.995},
    "combined_moderate": {"direct": 0.990, "leaf": 0.975},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path,
        default=BIO / "live_v2_bundle_boundary_ug12_edgegraft_v3_oof_v4/graphs",
    )
    parser.add_argument(
        "--raw", type=Path,
        default=BIO / "public914_backbone_matched_v1/repo/predictions/tweak/public914_train_raw/split_0",
    )
    parser.add_argument(
        "--scores", type=Path,
        default=BIO / "native_endpoint_candidate_graft_screen_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "native_endpoint_candidate_graft_replay_v1",
    )
    return parser.parse_args()


def selected(frame: pd.DataFrame, stem: str, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return frame.iloc[0:0]
    result = frame[(frame.dataset == stem) & (frame.oof_score >= threshold)].copy()
    return result.sort_values(["oof_score", "pmax"], ascending=False)


def add_direct(graph, rows: pd.DataFrame, counts: Counter) -> None:
    for row in rows.itertuples(index=False):
        source, target = int(row.source), int(row.target)
        counts["direct_selected"] += 1
        if not graph.has_node(source) or not graph.has_node(target):
            counts["direct_missing_node"] += 1
            continue
        if graph.out_degree(source) != 0 or graph.in_degree(target) != 0:
            counts["direct_conflict"] += 1
            continue
        graph.add_edge(source, target, {
            "edge_prob": float(row.pmax),
            "edge_dist": float(row.distance),
        })
        counts["direct_applied"] += 1


def add_leaf(graph, raw_nodes: dict, rows: pd.DataFrame, counts: Counter) -> None:
    for row in rows.itertuples(index=False):
        source, target = int(row.source), int(row.target)
        missing = int(row.missing)
        counts["leaf_selected"] += 1
        if missing not in raw_nodes:
            counts["leaf_missing_raw"] += 1
            continue
        source_exists, target_exists = graph.has_node(source), graph.has_node(target)
        if source_exists and graph.out_degree(source) != 0:
            counts["leaf_conflict"] += 1
            continue
        if target_exists and graph.in_degree(target) != 0:
            counts["leaf_conflict"] += 1
            continue
        if not source_exists and source != missing:
            counts["leaf_invalid"] += 1
            continue
        if not target_exists and target != missing:
            counts["leaf_invalid"] += 1
            continue
        if not graph.has_node(missing):
            raw = raw_nodes[missing]
            graph.add_node(
                {key: raw[key] for key in ("t", "z", "y", "x")},
                index=missing,
            )
            counts["leaf_nodes_added"] += 1
        if graph.out_degree(source) != 0 or graph.in_degree(target) != 0:
            counts["leaf_conflict_after_add"] += 1
            continue
        graph.add_edge(source, target, {
            "edge_prob": float(row.pmax),
            "edge_dist": float(row.distance),
        })
        counts["leaf_applied"] += 1


def validate(graph, protected: set[int]) -> None:
    for node in graph.node_ids():
        if graph.in_degree(node) > 1:
            raise RuntimeError(f"in-degree violation at {node}")
        if graph.out_degree(node) > 2:
            raise RuntimeError(f"out-degree violation at {node}")
    current = edgebase.protected_fork_nodes(graph)
    if not protected.issubset(current):
        raise RuntimeError("existing protected fork topology was changed")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    direct = pd.read_parquet(args.scores / "direct_edges_scored.parquet")
    leaf = pd.read_parquet(args.scores / "endpoint_leaves_scored.parquet")
    sys.path.insert(0, "/home/tweak/bio_track_repo/src")
    from biohub_tracking.io import save_graph

    totals = {name: Counter() for name in VARIANTS}
    records = []
    paths = sorted(args.baseline.glob("*.geff"))
    for index, path in enumerate(paths, 1):
        stem = path.stem
        raw_nodes, _ = load(args.raw / path.name)
        for name, thresholds in VARIANTS.items():
            graph = edgebase.load_graph(path)
            protected = edgebase.protected_fork_nodes(graph)
            counts = Counter()
            add_direct(graph, selected(direct, stem, thresholds["direct"]), counts)
            add_leaf(graph, raw_nodes, selected(leaf, stem, thresholds["leaf"]), counts)
            validate(graph, protected)
            destination = args.output / name / "graphs"
            destination.mkdir(parents=True, exist_ok=True)
            save_graph(graph, destination / path.name)
            totals[name].update(counts)
            records.append({"dataset": stem, "variant": name, **dict(counts)})
        if index % 10 == 0 or index == len(paths):
            print(f"[{index:03d}/{len(paths)}] {stem}", flush=True)

    pd.DataFrame(records).fillna(0).to_csv(args.output / "per_video.csv", index=False)
    summary = {
        "version": "native-endpoint-candidate-graft-replay-v1",
        "baseline": str(args.baseline),
        "variants": VARIANTS,
        "totals": {
            name: {key: int(value) for key, value in counts.items()}
            for name, counts in totals.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
