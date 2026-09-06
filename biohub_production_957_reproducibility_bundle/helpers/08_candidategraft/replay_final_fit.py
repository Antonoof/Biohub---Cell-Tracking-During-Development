#!/usr/bin/env python3
"""Replay the final all-known CandidateGRAFT fit on the exact 175 OOF graphs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
sys.path.insert(0, str(WORKSPACE / "scripts"))

import audit_edgegraft_component_oracle_v1 as edgebase
from replay_native_endpoint_candidate_graft_v1 import add_direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path,
        default=BIO / "live_v2_bundle_boundary_ug12_edgegraft_v3_oof_v4/graphs",
    )
    parser.add_argument(
        "--population", type=Path,
        default=BIO / "native_endpoint_candidate_graft_v2/direct_edges.parquet",
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=BIO / "candidategraft_direct_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "candidategraft_direct_final_fit_replay_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True)
    report = json.loads((args.artifact / "report.json").read_text())
    features = list(report["features"])
    threshold = float(report["threshold"])
    model = joblib.load(args.artifact / "candidategraft_direct.joblib")
    frame = pd.read_parquet(args.population)
    matrix = frame[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).to_numpy(np.float32)
    frame["oof_score"] = model.predict_proba(matrix)[:, 1]

    sys.path.insert(0, "/home/tweak/bio_track_repo/src")
    from biohub_tracking.io import save_graph

    total = Counter()
    per_video = []
    paths = sorted(args.baseline.glob("*.geff"))
    for index, path in enumerate(paths, 1):
        graph = edgebase.load_graph(path)
        protected = edgebase.protected_fork_nodes(graph)
        selected = frame[
            (frame.dataset == path.stem) & (frame.oof_score >= threshold)
        ].sort_values(["oof_score", "pmax"], ascending=False)
        counts = Counter()
        add_direct(graph, selected, counts)
        for node in graph.node_ids():
            if graph.in_degree(node) > 1 or graph.out_degree(node) > 2:
                raise RuntimeError(f"{path.stem}: graph-degree violation at {node}")
        if not protected.issubset(edgebase.protected_fork_nodes(graph)):
            raise RuntimeError(f"{path.stem}: protected fork topology changed")
        save_graph(graph, graph_dir / path.name)
        total.update(counts)
        per_video.append({"dataset": path.stem, **dict(counts)})
        if index % 10 == 0 or index == len(paths):
            print(f"[{index:03d}/{len(paths)}] {path.stem}: {dict(counts)}", flush=True)
    pd.DataFrame(per_video).fillna(0).to_csv(args.output / "per_video.csv", index=False)
    summary = {
        "version": "candidategraft-direct-final-fit-replay-v1",
        "threshold": threshold,
        "videos": len(paths),
        "totals": {key: int(value) for key, value in total.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
