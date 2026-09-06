#!/usr/bin/env python3
"""Kaggle-safe upstream P1/P2 graph reconstruction and endpoint recovery.

This runtime executes before motion repair and division.  It preserves the
fused detector population, rebuilds one-to-one continuations from independent
P1/P2 native candidates, and then adds only mutually supported missing nodes
that bridge two otherwise open track endpoints.  It never reads ground truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree


DEFAULT_SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)


def _load_native(path: Path) -> dict[str, np.ndarray]:
    names = (
        "source_id", "target_id", "probability",
        "alternative_parent_probability", "is_target_winner", "distance_um",
        "source_target_rank", "native_node_coords", "mapped_ab_node",
        "map_distance_um", "ab_node_coords", "spacing_um",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in names if name not in data]
        if missing:
            raise RuntimeError(f"Incomplete native evidence {path}: {missing}")
        return {name: np.asarray(data[name]).copy() for name in names}


def _coord_key(row) -> tuple[int, int, int, int]:
    return tuple(int(round(float(value))) for value in row)


def _proposal_to_graph(cache: dict, nodes: dict[int, dict]) -> np.ndarray:
    by_coord: dict[tuple[int, int, int, int], int] = {}
    duplicates: set[tuple[int, int, int, int]] = set()
    for node_id, node in nodes.items():
        key = _coord_key((node["t"], node["z"], node["y"], node["x"]))
        if key in by_coord:
            duplicates.add(key)
        else:
            by_coord[key] = int(node_id)
    output = np.full(len(cache["ab_node_coords"]), -1, np.int64)
    for index, row in enumerate(cache["ab_node_coords"]):
        key = _coord_key(row)
        if key not in duplicates:
            output[index] = int(by_coord.get(key, -1))
    return output


def _native_edges(
    cache: dict,
    proposal_to_graph: np.ndarray,
    nodes: dict[int, dict],
    threshold: float,
    rank_cap: int,
    max_distance: float,
) -> dict[tuple[int, int], dict[str, float]]:
    source = cache["source_id"].astype(np.int64, copy=False)
    target = cache["target_id"].astype(np.int64, copy=False)
    probability = cache["probability"].astype(np.float32, copy=False)
    alternative = cache["alternative_parent_probability"].astype(np.float32, copy=False)
    winner = cache["is_target_winner"].astype(bool, copy=False)
    distance = cache["distance_um"].astype(np.float32, copy=False)
    rank = cache["source_target_rank"].astype(np.int16, copy=False)
    mapped = cache["mapped_ab_node"].astype(np.int64, copy=False)
    map_distance = cache["map_distance_um"].astype(np.float32, copy=False)
    source_row, target_row = mapped[source], mapped[target]
    valid = (
        (probability >= threshold)
        & (rank <= rank_cap)
        & (distance <= max_distance)
        & (source_row >= 0)
        & (target_row >= 0)
        & (source_row < len(proposal_to_graph))
        & (target_row < len(proposal_to_graph))
    )
    output: dict[tuple[int, int], dict[str, float]] = {}
    for index in np.flatnonzero(valid):
        a = int(proposal_to_graph[int(source_row[index])])
        b = int(proposal_to_graph[int(target_row[index])])
        if a not in nodes or b not in nodes:
            continue
        if int(nodes[b]["t"]) != int(nodes[a]["t"]) + 1:
            continue
        row = {
            "prob": float(probability[index]),
            "margin": float(probability[index] - alternative[index]),
            "winner": float(winner[index]),
            "rank": float(rank[index]),
            "distance": float(distance[index]),
            "source_map": float(map_distance[int(source[index])]),
            "target_map": float(map_distance[int(target[index])]),
        }
        old = output.get((a, b))
        if old is None or row["prob"] > old["prob"]:
            output[(a, b)] = row
    return output


def _solve_transition(
    candidates: dict[tuple[int, int], tuple[float, float]],
) -> set[tuple[int, int]]:
    if not candidates:
        return set()
    sources = sorted({source for source, _ in candidates})
    targets = sorted({target for _, target in candidates})
    source_row = {node: index for index, node in enumerate(sources)}
    target_col = {node: index for index, node in enumerate(targets)}
    rows: list[int] = []
    cols: list[int] = []
    costs: list[float] = []
    for (source, target), (score, _distance) in candidates.items():
        rows.append(source_row[source])
        cols.append(target_col[target])
        costs.append(-float(score))
    for index in range(len(sources)):
        rows.append(index)
        cols.append(len(targets) + index)
        costs.append(1e-12)
    matrix = coo_matrix(
        (np.asarray(costs, np.float64), (rows, cols)),
        shape=(len(sources), len(targets) + len(sources)),
    ).tocsr()
    row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    return {
        (int(sources[int(row)]), int(targets[int(col)]))
        for row, col in zip(row_ind, col_ind)
        if col < len(targets)
    }


def _rebuild_continuations(
    nodes: dict[int, dict],
    raw_edges: list[dict],
    p1: dict,
    p2: dict,
    edge_threshold: float,
    native_threshold: float,
    candidate_rank: int,
    max_distance: float,
) -> tuple[list[dict], Counter]:
    p1_map = _proposal_to_graph(p1, nodes)
    p2_map = _proposal_to_graph(p2, nodes)
    one = _native_edges(
        p1, p1_map, nodes, native_threshold, candidate_rank, max_distance,
    )
    two = _native_edges(
        p2, p2_map, nodes, native_threshold, candidate_rank, max_distance,
    )
    old = {
        (int(edge["source_id"]), int(edge["target_id"])): dict(edge)
        for edge in raw_edges
    }
    by_t: dict[int, dict[tuple[int, int], tuple[float, float]]] = defaultdict(dict)
    for pair in set(one) | set(two):
        left, right = one.get(pair), two.get(pair)
        score = 0.5 * (
            (float(left["prob"]) if left is not None else 0.0)
            + (float(right["prob"]) if right is not None else 0.0)
        )
        if score < edge_threshold:
            continue
        distance = min(
            float(row["distance"]) for row in (left, right) if row is not None
        )
        by_t[int(nodes[pair[0]]["t"])][pair] = (score, distance)
    for pair, edge in old.items():
        transition = by_t[int(nodes[pair[0]]["t"])]
        if pair not in transition:
            probability = float(edge.get("edge_prob", edge_threshold) or edge_threshold)
            distance = float(edge.get("edge_dist", edge.get("distance_um", 0.0)) or 0.0)
            transition[pair] = (max(probability, edge_threshold), distance)
    selected: set[tuple[int, int]] = set()
    metadata: dict[tuple[int, int], tuple[float, float]] = {}
    for transition in by_t.values():
        take = _solve_transition(transition)
        selected.update(take)
        metadata.update({pair: transition[pair] for pair in take})
    edges = [
        {
            "source_id": int(source),
            "target_id": int(target),
            "edge_prob": float(metadata[(source, target)][0]),
            "edge_dist": float(metadata[(source, target)][1]),
        }
        for source, target in sorted(selected)
    ]
    counts = Counter({
        "fusion_p1_candidates": len(one),
        "fusion_p2_candidates": len(two),
        "fusion_old_edges": len(old),
        "fusion_edges": len(edges),
        "fusion_added": len(selected - set(old)),
        "fusion_removed": len(set(old) - selected),
    })
    return edges, counts


def _best_neighbors(cache: dict):
    count = len(cache["native_node_coords"])
    incoming_node = np.full(count, -1, np.int64)
    outgoing_node = np.full(count, -1, np.int64)
    incoming_prob = np.zeros(count, np.float32)
    outgoing_prob = np.zeros(count, np.float32)
    source = cache["source_id"].astype(np.int64, copy=False)
    target = cache["target_id"].astype(np.int64, copy=False)
    probability = cache["probability"].astype(np.float32, copy=False)
    valid = (source >= 0) & (source < count) & (target >= 0) & (target < count)
    order = np.argsort(probability[valid], kind="stable")[::-1]
    for s, t, probability_value in zip(
        source[valid][order], target[valid][order], probability[valid][order], strict=True,
    ):
        if outgoing_node[int(s)] < 0:
            outgoing_node[int(s)] = int(t)
            outgoing_prob[int(s)] = float(probability_value)
        if incoming_node[int(t)] < 0:
            incoming_node[int(t)] = int(s)
            incoming_prob[int(t)] = float(probability_value)
    return incoming_node, incoming_prob, outgoing_node, outgoing_prob


def _native_to_graph(cache: dict, proposal_to_graph: np.ndarray) -> np.ndarray:
    mapped = cache["mapped_ab_node"].astype(np.int64, copy=False)
    output = np.full(len(mapped), -1, np.int64)
    valid = (mapped >= 0) & (mapped < len(proposal_to_graph))
    output[valid] = proposal_to_graph[mapped[valid]]
    return output


def _consensus_missing(
    p1: dict,
    p2: dict,
    nodes: dict[int, dict],
    proposal_radius: float,
    dedupe_radius: float,
    spacing: np.ndarray,
):
    coords1 = p1["native_node_coords"].astype(np.float32, copy=False)
    coords2 = p2["native_node_coords"].astype(np.float32, copy=False)
    by1: dict[int, list[int]] = defaultdict(list)
    by2: dict[int, list[int]] = defaultdict(list)
    final_by_t: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(coords1):
        by1[int(row[0])].append(index)
    for index, row in enumerate(coords2):
        by2[int(row[0])].append(index)
    for node_id, row in nodes.items():
        final_by_t[int(row["t"])].append(int(node_id))
    result = []
    for time in sorted(set(by1) & set(by2)):
        indices1 = np.asarray(by1[time], np.int64)
        indices2 = np.asarray(by2[time], np.int64)
        xyz1 = coords1[indices1, 1:4]
        xyz2 = coords2[indices2, 1:4]
        tree2 = cKDTree(xyz2 * spacing)
        distance12, neighbor12 = tree2.query(xyz1 * spacing, k=1)
        tree1 = cKDTree(xyz1 * spacing)
        _, neighbor21 = tree1.query(xyz2 * spacing, k=1)
        final_ids = final_by_t.get(time, [])
        final_xyz = np.asarray(
            [[nodes[node]["z"], nodes[node]["y"], nodes[node]["x"]] for node in final_ids],
            np.float32,
        ).reshape(-1, 3)
        final_tree = cKDTree(final_xyz * spacing) if len(final_xyz) else None
        for local1, (distance, local2) in enumerate(
            zip(distance12, neighbor12, strict=True)
        ):
            if float(distance) > proposal_radius:
                continue
            if int(neighbor21[int(local2)]) != local1:
                continue
            native1 = int(indices1[local1])
            native2 = int(indices2[int(local2)])
            position = 0.5 * (coords1[native1, 1:4] + coords2[native2, 1:4])
            if final_tree is not None:
                nearest = float(final_tree.query(position * spacing, k=1)[0])
                if nearest < dedupe_radius:
                    continue
            result.append((time, native1, native2, float(distance), position))
    return result


def _degrees(edges: list[dict]):
    incoming: Counter[int] = Counter()
    outgoing: Counter[int] = Counter()
    for edge in edges:
        outgoing[int(edge["source_id"])] += 1
        incoming[int(edge["target_id"])] += 1
    return incoming, outgoing


def _bridge_missing_endpoints(
    nodes: dict[int, dict],
    edges: list[dict],
    p1: dict,
    p2: dict,
    pmin: float,
    proposal_radius: float,
    dedupe_radius: float,
) -> Counter:
    spacing = np.asarray(p1.get("spacing_um", DEFAULT_SPACING), np.float64)
    p1_rows = _proposal_to_graph(p1, nodes)
    p2_rows = _proposal_to_graph(p2, nodes)
    p1_map = _native_to_graph(p1, p1_rows)
    p2_map = _native_to_graph(p2, p2_rows)
    p1_in, p1_in_prob, p1_out, p1_out_prob = _best_neighbors(p1)
    p2_in, p2_in_prob, p2_out, p2_out_prob = _best_neighbors(p2)
    incoming, outgoing = _degrees(edges)
    original_forks = {node for node, degree in outgoing.items() if degree >= 2}
    candidates = []
    for time, native1, native2, distance, position in _consensus_missing(
        p1, p2, nodes, proposal_radius, dedupe_radius, spacing,
    ):
        left1 = int(p1_map[int(p1_in[native1])]) if p1_in[native1] >= 0 else -1
        left2 = int(p2_map[int(p2_in[native2])]) if p2_in[native2] >= 0 else -1
        right1 = int(p1_map[int(p1_out[native1])]) if p1_out[native1] >= 0 else -1
        right2 = int(p2_map[int(p2_out[native2])]) if p2_out[native2] >= 0 else -1
        left = left1 if left1 >= 0 and left1 == left2 and outgoing[left1] == 0 else -1
        right = right1 if right1 >= 0 and right1 == right2 and incoming[right1] == 0 else -1
        if left < 0 or right < 0:
            continue
        left_pmin = min(float(p1_in_prob[native1]), float(p2_in_prob[native2]))
        right_pmin = min(float(p1_out_prob[native1]), float(p2_out_prob[native2]))
        support = min(left_pmin, right_pmin)
        if support >= pmin:
            candidates.append((support, -distance, time, position, left, right, left_pmin, right_pmin))
    candidates.sort(reverse=True, key=lambda row: (row[0], row[1]))
    used_anchors: set[int] = set()
    next_id = max(nodes, default=-1) + 1
    counts = Counter({"endpoint_candidates": len(candidates)})
    template = dict(next(iter(nodes.values()))) if nodes else {}
    for support, _negative_distance, time, position, left, right, left_prob, right_prob in candidates:
        if left in used_anchors or right in used_anchors:
            counts["endpoint_anchor_conflict"] += 1
            continue
        node_id = next_id
        next_id += 1
        node = {
            key: (0.0 if isinstance(value, (float, np.floating)) else 0)
            for key, value in template.items()
        }
        node.update({
            "node_id": node_id,
            "t": int(time),
            "z": float(position[0]),
            "y": float(position[1]),
            "x": float(position[2]),
        })
        nodes[node_id] = node
        edges.extend((
            {"source_id": left, "target_id": node_id, "edge_prob": left_prob},
            {"source_id": node_id, "target_id": right, "edge_prob": right_prob},
        ))
        used_anchors.update((left, right))
        counts["endpoint_nodes_added"] += 1
        counts["endpoint_edges_added"] += 2
        counts["endpoint_bridges_added"] += 1
    after_in, after_out = _degrees(edges)
    if any(degree > 1 for degree in after_in.values()):
        raise RuntimeError("BaseGraph endpoint recovery produced in-degree > 1")
    if any(degree > 1 for degree in after_out.values()):
        raise RuntimeError("BaseGraph continuation recovery produced out-degree > 1")
    if {node for node, degree in after_out.items() if degree >= 2} != original_forks:
        raise RuntimeError("BaseGraph endpoint recovery changed fork topology")
    return counts


class BaseGraphCandidateFusionRuntime:
    """Frozen upstream graph policy validated on the complete 199-video bank."""

    def __init__(
        self,
        edge_threshold: float = 0.40,
        native_threshold: float = 0.01,
        candidate_rank: int = 8,
        max_distance: float = 14.0,
        endpoint_pmin: float = 0.03,
        endpoint_consensus_um: float = 3.0,
        endpoint_dedupe_um: float = 3.0,
    ):
        self.edge_threshold = float(edge_threshold)
        self.native_threshold = float(native_threshold)
        self.candidate_rank = int(candidate_rank)
        self.max_distance = float(max_distance)
        self.endpoint_pmin = float(endpoint_pmin)
        self.endpoint_consensus_um = float(endpoint_consensus_um)
        self.endpoint_dedupe_um = float(endpoint_dedupe_um)

    def apply(
        self,
        nodes_by_id: dict[int, dict],
        raw_edges: list[dict],
        p1_path: str | Path,
        p2_path: str | Path,
    ) -> tuple[dict[int, dict], list[dict], dict[str, int | float]]:
        nodes = {int(node_id): dict(row) for node_id, row in nodes_by_id.items()}
        edges = [dict(edge) for edge in raw_edges]
        p1 = _load_native(Path(p1_path))
        p2 = _load_native(Path(p2_path))
        edges, counts = _rebuild_continuations(
            nodes,
            edges,
            p1,
            p2,
            self.edge_threshold,
            self.native_threshold,
            self.candidate_rank,
            self.max_distance,
        )
        counts.update(_bridge_missing_endpoints(
            nodes,
            edges,
            p1,
            p2,
            self.endpoint_pmin,
            self.endpoint_consensus_um,
            self.endpoint_dedupe_um,
        ))
        counts.update({
            "policy_mean": 1,
            "edge_threshold_million": int(round(self.edge_threshold * 1_000_000)),
            "endpoint_pmin_million": int(round(self.endpoint_pmin * 1_000_000)),
            "final_nodes": len(nodes),
            "final_edges": len(edges),
        })
        return nodes, edges, dict(counts)

