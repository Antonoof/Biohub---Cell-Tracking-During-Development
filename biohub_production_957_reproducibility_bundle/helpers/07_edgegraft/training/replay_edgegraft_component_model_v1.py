#!/usr/bin/env python3
"""Replay the frozen component model on the current EdgeGRAFT substrate.

This is the zero-training deployment-parity baseline for EdgeGRAFT.  It fixes
the two decisive defects in the historical replay:

* raw P1/P2 node IDs are mapped into the current final graph by physical
  identity when direct IDs no longer survive finalization;
* components are evaluated against the *current* continuation assignment, not
  the obsolete public-.944 raw-parent flag.

Existing forks are immutable and every accepted component is replaced
atomically.  No node is added or removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import audit_edgegraft_component_oracle_v1 as edgebase
from replay_public_p1p2_component_gate_hard20_v1 import safe_apply
from train_public_p1p2_component_gate_v1 import ComponentEdgeNet, EmbeddingStore, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("/home/tweak/bio/public944_p1p2_full_conflicts_all199_v1"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/tweak/bio/public944_p1p2_component_gate_tabular_v1/"
            "component_gate_best.pt"
        ),
    )
    parser.add_argument("--root", type=Path, default=edgebase.DEFAULT_ROOT)
    parser.add_argument("--baseline", type=Path, default=edgebase.DEFAULT_FINAL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/tweak/bio/edgegraft_component_model_v1"),
    )
    parser.add_argument("--threshold", type=float, default=float("nan"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def map_rows(frame: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    source = frame.source.map(mapping)
    target = frame.target.map(mapping)
    keep = source.notna() & target.notna()
    result = frame.loc[keep].copy()
    result["source"] = source.loc[keep].astype(np.int64)
    result["target"] = target.loc[keep].astype(np.int64)
    result = result.drop_duplicates(["component", "source", "target"], keep="first")
    return result


def proposed_assignment(group: pd.DataFrame, graph, protected: set[int]):
    """Return a model/base transaction on current assigned targets only."""
    current_parent: dict[int, int] = {}
    for target in map(int, group.target.unique()):
        parents = list(map(int, graph.predecessors(target)))
        if len(parents) != 1:
            continue
        parent = parents[0]
        if parent in protected or target in protected or graph.out_degree(parent) > 1:
            continue
        current_parent[target] = parent
    if not current_parent:
        return None

    represented = set(map(tuple, group[["source", "target"]].to_numpy(np.int64)))
    current_parent = {
        target: source
        for target, source in current_parent.items()
        if (source, target) in represented
    }
    if not current_parent:
        return None

    eligible_targets = set(current_parent)
    work = group[group.target.isin(eligible_targets)].copy()
    sources = np.sort(work.source.unique())
    targets = np.sort(work.target.unique())
    if not len(sources) or len(sources) < len(targets):
        return None

    source_index = {value: index for index, value in enumerate(sources)}
    target_index = {value: index for index, value in enumerate(targets)}
    matrix = np.full((len(targets), len(sources)), -1e6, np.float64)
    for row in work.itertuples():
        i = target_index[int(row.target)]
        j = source_index[int(row.source)]
        matrix[i, j] = max(matrix[i, j], float(row.score))

    base = {(int(source), int(target)) for target, source in current_parent.items()}
    if len(base) != len(targets):
        return None
    base_values = []
    for source, target in base:
        value = matrix[target_index[target], source_index.get(source, -1)]
        if value < -1e5:
            return None
        base_values.append(value)

    rows, columns = linear_sum_assignment(matrix, maximize=True)
    if len(rows) != len(targets) or np.any(matrix[rows, columns] < -1e5):
        return None
    model = {
        (int(sources[column]), int(targets[row]))
        for row, column in zip(rows, columns)
    }
    advantage = float(
        (matrix[rows, columns].sum() - np.sum(base_values)) / max(len(targets), 1)
    )
    return model, base, advantage


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True)

    sys.path.insert(0, "/home/tweak/bio_track_repo/src")
    from biohub_tracking.io import save_graph

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ComponentEdgeNet(len(checkpoint["features"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    store = EmbeddingStore("/dev/null", disabled=True)
    threshold = (
        float(checkpoint["threshold"])
        if not np.isfinite(args.threshold)
        else float(args.threshold)
    )

    raw_dir = args.root / "repo/predictions/tweak/public914_train_raw/split_0"
    stems = sorted(path.stem for path in args.baseline.glob("*.geff"))
    records = []
    aggregate: Counter = Counter()
    for index, stem in enumerate(stems, 1):
        graph = edgebase.load_graph(args.baseline / f"{stem}.geff")
        raw = edgebase.load_graph(raw_dir / f"{stem}.geff")
        mapping = edgebase.raw_to_final_map(raw, graph)
        frame = pd.read_parquet(args.rows / f"{stem}.parquet")
        frame = map_rows(frame, mapping)
        current = edgebase.edge_set(graph)
        frame["is_raw_parent"] = np.fromiter(
            (
                (int(source), int(target)) in current
                for source, target in frame[["source", "target"]].itertuples(index=False)
            ),
            np.float32,
            len(frame),
        )
        predicted = predict(
            model,
            frame,
            store,
            checkpoint["features"],
            checkpoint["mean"],
            checkpoint["std"],
            device,
            8,
        )
        protected = edgebase.protected_fork_nodes(graph)
        counts: Counter = Counter()
        decision_rows = []
        for component, group in predicted.groupby("component", sort=False):
            counts["components"] += 1
            value = proposed_assignment(group, graph, protected)
            if value is None:
                counts["ineligible"] += 1
                continue
            model_edges, base_edges, advantage = value
            counts["eligible"] += 1
            selected = model_edges != base_edges and advantage >= threshold
            applied = False
            reason = "below_threshold"
            removed = added = 0
            if selected:
                counts["selected"] += 1
                applied, removed, added, reason = safe_apply(
                    graph, model_edges, base_edges
                )
                if applied:
                    counts["applied"] += 1
                    counts["removed"] += removed
                    counts["added"] += added
            decision_rows.append(
                {
                    "dataset": stem,
                    "component": int(component),
                    "targets": len({target for _, target in base_edges}),
                    "changed": model_edges != base_edges,
                    "advantage": advantage,
                    "selected": selected,
                    "applied": applied,
                    "reason": reason,
                }
            )
        records.append({"dataset": stem, **dict(counts)})
        aggregate.update(counts)
        pd.DataFrame(decision_rows).to_parquet(
            args.output / f"{stem}_decisions.parquet", index=False
        )
        save_graph(graph, graph_dir / f"{stem}.geff")
        print(
            f"[{index:03d}/{len(stems)}] {stem}: rows={len(frame):,} "
            f"eligible={counts['eligible']} selected={counts['selected']} "
            f"applied={counts['applied']} -{counts['removed']} +{counts['added']}",
            flush=True,
        )

    payload = {
        "version": "edgegraft-component-model-v1",
        "checkpoint": str(args.checkpoint),
        "threshold": threshold,
        "baseline": str(args.baseline),
        "videos": records,
        "totals": {key: int(value) for key, value in aggregate.items()},
    }
    (args.output / "replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
