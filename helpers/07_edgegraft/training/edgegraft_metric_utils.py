"""Graph utilities required by the EdgeGRAFT transaction-gate trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tracksdata as td


def load_graph(path: Path):
    value = td.graph.IndexedRXGraph.from_geff(path)
    return value[0] if isinstance(value, tuple) else value


def graph_nodes(graph) -> dict[int, dict[str, object]]:
    return {
        int(row["node_id"]): {
            "t": int(row["t"]),
            "zyx": np.asarray([row["z"], row["y"], row["x"]], np.float32),
        }
        for row in graph.node_attrs(
            attr_keys=["node_id", "t", "z", "y", "x"]
        ).iter_rows(named=True)
    }


def matched_truth(final, gt, scale, max_distance):
    """Return matched, ordinary one-child truth edges in prediction ID space."""
    from tracking_cellmot.division_metrics import _match_full

    final_nodes = graph_nodes(final)
    matched = _match_full(final, gt, scale, max_distance)
    gt_to_pred = {
        int(row["match_node_id"]): int(row["node_id"])
        for row in matched.node_attrs().to_dicts()
        if int(row["match_node_id"]) >= 0
    }
    truth: set[tuple[int, int]] = set()
    for gt_source, gt_target in gt.edge_list():
        gt_source, gt_target = int(gt_source), int(gt_target)
        if gt.out_degree(gt_source) != 1:
            continue
        if gt_source not in gt_to_pred or gt_target not in gt_to_pred:
            continue
        source, target = gt_to_pred[gt_source], gt_to_pred[gt_target]
        if int(final_nodes[target]["t"]) == int(final_nodes[source]["t"]) + 1:
            truth.add((source, target))
    return truth
