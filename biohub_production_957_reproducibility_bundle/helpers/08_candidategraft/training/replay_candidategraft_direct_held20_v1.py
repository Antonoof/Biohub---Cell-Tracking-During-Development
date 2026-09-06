#!/usr/bin/env python3
"""Replay the frozen CandidateGRAFT direct gate on the untouched held-20.

The input is the saved, finalized UG1/UG2 + EdgeGRAFT V3 held-20 graph bank.
CandidateGRAFT is applied after destructive cleanup, matching deployment order.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))

from candidategraft_direct_runtime_v1 import CandidateGraftDirectRuntime  # noqa: E402
from replay_model_c_full_population_gate import save_graph  # noqa: E402
from screen_model_c_atomic_triplet_rescue import load_graph  # noqa: E402
from smoke_edgegraft_v3_deploy_parity_v1 import load  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path,
        default=BIO / "edgegraft_v3_ug12_prefinal_held20_v1/finalized",
    )
    parser.add_argument(
        "--raw", type=Path,
        default=BIO / "public914_backbone_matched_v1/repo/predictions/tweak/public914_train_raw/split_0",
    )
    parser.add_argument(
        "--root", type=Path,
        default=BIO / "public914_backbone_matched_v1",
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=BIO / "candidategraft_direct_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "candidategraft_direct_held20_v1",
    )
    return parser.parse_args()


def staged_evidence(root: Path, stem: str, temporary: Path) -> tuple[Path, Path]:
    """Add the current fused-node mapping to the frozen all-199 evidence."""
    with np.load(
        root / "public914_proposals" / f"{stem}.npz", allow_pickle=False,
    ) as proposal:
        graph_node_id = proposal["graph_node_id"].astype(np.int64, copy=False)
    staged = []
    for folder in (
        "public_primary_native_evidence_all199_v1",
        "public_secondary_native_evidence_all199_v1",
    ):
        source_path = root / folder / f"{stem}.npz"
        destination = temporary / f"{folder}.npz"
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        arrays["fused_graph_node_id"] = graph_node_id
        np.savez_compressed(destination, **arrays)
        staged.append(destination)
    return staged[0], staged[1]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite {args.output}")
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True)
    runtime = CandidateGraftDirectRuntime(args.artifact, SCRIPTS)
    totals = Counter()
    reports = []
    stems = sorted(path.stem for path in args.baseline.glob("*.geff"))
    if len(stems) != 20:
        raise RuntimeError(f"Expected exactly 20 held videos, found {len(stems)}")

    for index, stem in enumerate(stems, 1):
        baseline_path = args.baseline / f"{stem}.geff"
        raw_nodes, raw_edges = load(args.raw / f"{stem}.geff")
        final_nodes, final_edges = load(baseline_path)
        with tempfile.TemporaryDirectory(prefix=f"candidategraft_{stem}_") as temp:
            p1_path, p2_path = staged_evidence(args.root, stem, Path(temp))
            updated_edges, counters = runtime.apply(
                stem, raw_nodes, raw_edges, final_nodes, final_edges,
                p1_path, p2_path,
            )

        graph = load_graph(baseline_path)
        for source, target in list(graph.edge_list()):
            graph.remove_edge(int(source), int(target))
        positions = {
            int(node): np.asarray([row["z"], row["y"], row["x"]], np.float64)
            for node, row in final_nodes.items()
        }
        spacing = np.asarray((1.625, 0.40625, 0.40625), np.float64)
        edge_keys = set(graph.edge_attr_keys())
        for row in updated_edges:
            source, target = int(row["source_id"]), int(row["target_id"])
            attrs = {}
            if "edge_prob" in edge_keys:
                probability = row.get("edge_prob")
                attrs["edge_prob"] = 1.0 if probability is None else float(probability)
            if "edge_dist" in edge_keys:
                attrs["edge_dist"] = float(
                    np.linalg.norm((positions[target] - positions[source]) * spacing)
                )
            graph.add_edge(source, target, attrs)
        save_graph(graph, graph_dir / f"{stem}.geff")

        for key, value in counters.items():
            if isinstance(value, (int, np.integer)):
                totals[key] += int(value)
        reports.append({"stem": stem, **counters})
        print(
            f"[{index:02d}/20] {stem}: edges={len(final_edges):,}->{len(updated_edges):,} "
            f"population={counters.get('population', 0):,} "
            f"selected={counters.get('selected', 0):,} "
            f"applied={counters.get('applied', 0):,}",
            flush=True,
        )

    payload = {
        "version": "candidategraft-direct-held20-v1",
        "contract": "frozen final-fit model and threshold; no held-20 fitting or tuning",
        "baseline": str(args.baseline),
        "artifact": str(args.artifact),
        "totals": dict(totals),
        "per_video": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
