from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)
MAPPING_RADIUS_UM = 6.0


def node_position(row: dict[str, Any]) -> np.ndarray:
    return np.asarray((row['z'], row['y'], row['x']), np.float64)


def greedy_native_map(native_coords, native_offsets, nodes, spacing):
    mapped = np.full(len(native_coords), -1, np.int64)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, row in nodes.items():
        by_frame[int(row['t'])].append(int(node_id))
    for frame in range(len(native_offsets) - 1):
        start, end = int(native_offsets[frame]), int(native_offsets[frame + 1])
        graph_ids = sorted(by_frame.get(frame, ()))
        if end <= start or not graph_ids:
            continue
        native_um = native_coords[start:end, 1:].astype(np.float32) * spacing
        graph_um = (
            np.asarray(
                [
                    [
                        round(float(nodes[node]['z'])),
                        round(float(nodes[node]['y'])),
                        round(float(nodes[node]['x'])),
                    ]
                    for node in graph_ids
                ],
                np.float32,
            )
            * spacing
        )
        k = min(4, len(graph_um))
        distances, indexes = cKDTree(graph_um).query(native_um, k=k)
        if k == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        options = []
        for native_row in range(len(native_um)):
            for rank in range(k):
                value = float(distances[native_row, rank])
                if value <= MAPPING_RADIUS_UM:
                    options.append((value, native_row, int(indexes[native_row, rank])))
        used_native: set[int] = set()
        used_graph: set[int] = set()
        for _distance, native_row, graph_row in sorted(options):
            if native_row in used_native or graph_row in used_graph:
                continue
            used_native.add(native_row)
            used_graph.add(graph_row)
            mapped[start + native_row] = graph_ids[graph_row]
    return mapped


def conflict_pairs(evidence):
    source_to_target: dict[int, set[int]] = defaultdict(set)
    target_to_source: dict[int, set[int]] = defaultdict(set)
    for model in evidence.values():
        for source, target in model:
            source_to_target[int(source)].add(int(target))
            target_to_source[int(target)].add(int(source))
    seen: set[int] = set()
    keep: set[tuple[int, int]] = set()
    component_of: dict[tuple[int, int], int] = {}
    component = 0
    for initial in source_to_target:
        if initial in seen:
            continue
        stack = [initial]
        sources: set[int] = set()
        targets: set[int] = set()
        while stack:
            source = stack.pop()
            if source in seen:
                continue
            seen.add(source)
            sources.add(source)
            for target in source_to_target[source]:
                targets.add(target)
                for other in target_to_source[target]:
                    if other not in seen:
                        stack.append(other)
        if len(targets) > 1:
            for source in sources:
                for target in source_to_target[source]:
                    keep.add((source, target))
                    component_of[(source, target)] = component
            component += 1
    return keep, component_of


def density_maps(nodes):
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, row in nodes.items():
        by_frame[int(row['t'])].append(int(node_id))
    result = {}
    for ids in by_frame.values():
        points = np.asarray([node_position(nodes[node]) for node in ids]) * SPACING
        tree = cKDTree(points)
        near = tree.query_ball_point(points, 5.0, return_length=True) - 1
        broad = tree.query_ball_point(points, 10.0, return_length=True) - 1
        for node, one, two in zip(ids, near, broad):
            result[node] = (float(one), float(two))
    return result


def cosine(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-6 else 0.0


def speed_ratio(first, second):
    return float(
        np.clip(np.log((np.linalg.norm(second) + 1e-3) / (np.linalg.norm(first) + 1e-3)), -4, 4)
    )


def node_identity(row):
    return (
        int(row['t']),
        round(float(row['z']), 5),
        round(float(row['y']), 5),
        round(float(row['x']), 5),
    )


def raw_to_final_map(raw_nodes, final_nodes):
    result = {
        node: node
        for node in set(raw_nodes) & set(final_nodes)
        if int(raw_nodes[node]['t']) == int(final_nodes[node]['t'])
    }
    final_by_key = {}
    duplicates = set()
    for node, row in final_nodes.items():
        key = node_identity(row)
        if key in final_by_key:
            duplicates.add(key)
        else:
            final_by_key[key] = node
    for key in duplicates:
        final_by_key.pop(key, None)
    for node, row in raw_nodes.items():
        if node in result:
            continue
        target = final_by_key.get(node_identity(row))
        if target is not None:
            result[node] = int(target)
    return result


def protected_fork_nodes(edges):
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source, target = int(edge['source_id']), int(edge['target_id'])
        outgoing[source].add(target)
        incoming[target].add(source)
    protected: set[int] = set()
    for source, children in outgoing.items():
        if len(children) < 2:
            continue
        protected.add(source)
        protected.update(children)
        protected.update(incoming.get(source, ()))
        for child in children:
            protected.update(outgoing.get(child, ()))
    return protected


def factorize(values):
    lookup = {}
    result = np.empty(len(values), np.int64)
    for index, value in enumerate(values):
        if value not in lookup:
            lookup[value] = len(lookup)
        result[index] = lookup[value]
    return result
