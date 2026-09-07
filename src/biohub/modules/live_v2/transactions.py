from collections import defaultdict
from typing import Any

import numpy as np

FIRST_SOURCE_FLOOR = 0.15
FIRST_MIN_AGREEMENT = 2
RUNNER_MIN_SCORE = 0.75
RUNNER_MAX_WINNER_MARGIN = 0.02
RUNNER_MIN_PAIR_VOTES = 3


def canonical_pair(values) -> tuple[int, int]:
    a, b = map(int, values)
    return (a, b) if a <= b else (b, a)


def edge_maps(edges: list[dict[str, Any]]):
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    probability: dict[tuple[int, int], float] = {}
    for edge in edges:
        source = int(edge['source_id'])
        target = int(edge['target_id'])
        outgoing[source].append(target)
        incoming[target].append(source)
        value = edge.get('edge_prob', 0.0)
        try:
            probability[(source, target)] = float(value or 0.0)
        except (TypeError, ValueError):
            probability[(source, target)] = 0.0
    return dict(outgoing), dict(incoming), probability


def lineage_depth(adjacency: dict[int, list[int]], node: int, cap: int = 16) -> int:
    result = 0
    current = int(node)
    for _ in range(cap):
        values = adjacency.get(current, [])
        if len(values) != 1:
            break
        current = int(values[0])
        result += 1
    return result


def native_best_pair_nodes(
    owner: np.ndarray,
    pair_nodes: np.ndarray,
    native_x: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    winner_count = 12
    minimum_probability = 6
    mean_probability = 7
    rank_a = 15
    rank_b = 16
    result = np.full((n_sources, 2), -1, np.int64)
    order = np.argsort(owner, kind='stable')
    if not len(order):
        return result
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows = order[left:right]
        block = native_x[rows]
        rank_sum = block[:, rank_a] + block[:, rank_b]
        local_order = np.lexsort(
            (
                rank_sum,
                -block[:, mean_probability],
                -block[:, minimum_probability],
                -block[:, winner_count],
            )
        )
        result[int(sorted_owner[left])] = pair_nodes[int(rows[int(local_order[0])])]
    return result


def pair_rows(owner: np.ndarray, n_sources: int) -> list[np.ndarray]:
    rows = [np.empty(0, np.int64) for _ in range(n_sources)]
    order = np.argsort(owner, kind='stable')
    if not len(order):
        return rows
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows[int(sorted_owner[left])] = order[left:right]
    return rows
