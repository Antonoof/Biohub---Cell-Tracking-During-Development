from collections import defaultdict

import numpy as np

from biohub.modules.graph.geometry import edge_distance_um, edge_sort_key


def filter_short_track_components(upgrade, nodes_by_id, edges, stats):
    if not upgrade.filter_short_tracks or upgrade.min_track_len <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        if source_id in parent and target_id in parent:
            root_a, root_b = find(source_id), find(target_id)
            if root_a != root_b:
                parent[root_a] = root_b
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    t_min_global = t_max_global = None
    if upgrade.boundary_track_rescue and nodes_by_id:
        boundary_times = [int(node['t']) for node in nodes_by_id.values()]
        t_min_global = min(boundary_times)
        t_max_global = max(boundary_times)

    keep: set[int] = set()
    for members in components.values():
        divides = upgrade.keep_division_components and any(
            out_count.get(i, 0) >= 2 for i in members
        )
        is_long_enough = len(members) >= upgrade.min_track_len
        is_boundary_truncated = False
        if (
            upgrade.boundary_track_rescue
            and not is_long_enough
            and not divides
            and len(members) >= upgrade.boundary_track_min_len
        ):
            member_times = [int(nodes_by_id[node_id]['t']) for node_id in members]
            span = max(member_times) - min(member_times) + 1
            touches_start = min(member_times) == t_min_global
            touches_end = max(member_times) == t_max_global
            is_boundary_truncated = span < upgrade.min_track_len and (touches_start or touches_end)

        if is_long_enough or divides or is_boundary_truncated:
            keep.update(members)
            if is_boundary_truncated:
                stats['boundary_track_rescued_components'] += 1
                stats['boundary_track_rescued_nodes'] += len(members)

    if not keep or len(keep) == len(nodes_by_id):
        return nodes_by_id, edges

    kept_nodes = {i: node for i, node in nodes_by_id.items() if i in keep}
    kept_edges = [
        edge
        for edge in edges
        if int(edge['source_id']) in kept_nodes and int(edge['target_id']) in kept_nodes
    ]
    stats['short_track_components_removed'] = sum(
        1 for members in components.values() if not (set(members) & keep)
    )
    stats['short_track_nodes_removed'] = len(nodes_by_id) - len(kept_nodes)
    stats['short_track_edges_removed'] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def enforce_node_budget(upgrade, nodes_by_id, edges, stats, detected_nodes: int):
    if not upgrade.node_budget or detected_nodes <= 0 or not nodes_by_id:
        return nodes_by_id, edges

    dense = (
        upgrade.node_budget_dense_min_detected > 0
        and detected_nodes >= upgrade.node_budget_dense_min_detected
    )
    ratio = upgrade.node_budget_dense_ratio if dense else upgrade.node_budget_ratio
    max_drop_frac = (
        upgrade.node_budget_dense_max_drop_frac if dense else upgrade.node_budget_max_drop_frac
    )
    stats['node_budget_dense_applied'] = int(dense)
    stats['node_budget_ratio_used'] = float(ratio)
    stats['node_budget_detected_nodes'] = int(detected_nodes)

    budget = int(round(detected_nodes * ratio))
    excess = len(nodes_by_id) - budget
    stats['node_budget_excess'] = int(max(0, excess))
    if excess <= 0:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        if source_id in parent and target_id in parent:
            root_a, root_b = find(source_id), find(target_id)
            if root_a != root_b:
                parent[root_a] = root_b
        out_count[source_id] = out_count.get(source_id, 0) + 1

    members: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        members.setdefault(find(node_id), []).append(node_id)

    probability_sum: dict[int, float] = {}
    probability_count: dict[int, int] = {}
    for edge in edges:
        root = find(int(edge['source_id']))
        value = edge.get('edge_prob')
        try:
            value = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        probability_sum[root] = probability_sum.get(root, 0.0) + value
        probability_count[root] = probability_count.get(root, 0) + 1

    ranked = sorted(
        (
            (
                any(out_count.get(node_id, 0) >= 2 for node_id in group),
                len(group),
                probability_sum.get(root, 0.0) / max(probability_count.get(root, 0), 1),
                root,
            )
            for root, group in members.items()
        )
    )

    max_drop = int(round(len(nodes_by_id) * max_drop_frac))
    dropped: set[int] = set()
    for has_fork, size, _probability, root in ranked:
        if has_fork or len(dropped) >= excess or len(dropped) + size > max_drop:
            continue
        dropped.update(members[root])
    if not dropped:
        stats['node_budget_dropped_nodes'] = 0
        return nodes_by_id, edges

    kept_nodes = {i: node for i, node in nodes_by_id.items() if i not in dropped}
    kept_edges = [
        edge
        for edge in edges
        if int(edge['source_id']) in kept_nodes and int(edge['target_id']) in kept_nodes
    ]
    stats['node_budget_dropped_nodes'] = len(dropped)
    stats['node_budget_dropped_edges'] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(upgrade, nodes_by_id, edges, stats):
    if (
        not upgrade.linefit_smooth
        or upgrade.linefit_weight <= 0
        or upgrade.linefit_window <= 0
        or not edges
    ):
        return nodes_by_id

    predecessor: dict[int, int | None] = {}
    successor: dict[int, int | None] = {}
    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None or int(target['t']) != int(source['t']) + 1:
            continue
        successor[source_id] = None if source_id in successor else target_id
        predecessor[target_id] = None if target_id in predecessor else source_id

    ids = sorted(nodes_by_id)
    index_of = {node_id: index for index, node_id in enumerate(ids)}
    coords = np.empty((len(ids), 3), dtype=np.float64)
    for index, node_id in enumerate(ids):
        node = nodes_by_id[node_id]
        coords[index] = (node['z'], node['y'], node['x'])

    shapes: dict[tuple[int, int], list[list[int]]] = {}
    for index, node_id in enumerate(ids):
        row = [index]
        counts = []
        for links in (predecessor, successor):
            current = node_id
            steps = 0
            while steps < upgrade.linefit_window:
                nearby = links.get(current)
                if nearby is None or nearby not in index_of:
                    break
                current = nearby
                row.append(index_of[current])
                steps += 1
            counts.append(steps)
        if len(row) < 3:
            stats['linefit_skipped_nodes'] += 1
            continue
        shapes.setdefault((counts[0], counts[1]), []).append(row)

    weight = float(np.clip(upgrade.linefit_weight, 0.0, 1.0))
    smoothed = coords.copy()
    touched: list[np.ndarray] = []
    for (back, forward), rows in shapes.items():
        offsets = np.array([0, *range(-1, -back - 1, -1), *range(1, forward + 1)], dtype=np.float64)
        centered = offsets - offsets.mean()
        denominator = float(centered @ centered)
        if denominator <= 0.0:
            stats['linefit_skipped_nodes'] += len(rows)
            continue
        members = np.asarray(rows, dtype=np.int64)
        window = coords[members]
        mean = window.mean(axis=1)
        slope = np.einsum('k,mkc->mc', centered, window - mean[:, None, :]) / denominator
        fitted = mean - slope * offsets.mean()
        usable = np.isfinite(fitted).all(axis=1)
        stats['linefit_skipped_nodes'] += int((~usable).sum())
        targets = members[usable, 0]
        smoothed[targets] = (1.0 - weight) * coords[targets] + weight * fitted[usable]
        touched.append(targets)

    updated = np.concatenate(touched) if touched else np.empty(0, dtype=np.int64)
    for index in updated:
        node = nodes_by_id[ids[int(index)]]
        point = smoothed[int(index)]
        node['z'], node['y'], node['x'] = float(point[0]), float(point[1]), float(point[2])
    stats['linefit_smoothed_nodes'] = int(updated.size)
    return nodes_by_id


def cleanup_after_edgegraft(upgrade, nodes_by_id, edges, stats):
    source_degree = defaultdict(int)
    for edge in edges:
        source_degree[int(edge['source_id'])] += 1
    fork_sources = {source for source, degree in source_degree.items() if degree >= 2}

    by_pair: dict[tuple[int, int], dict] = {}
    for original in edges:
        edge = dict(original)
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            stats['post_edgegraft_dropped_dangling_edges'] += 1
            continue
        if upgrade.enforce_next_frame and int(target['t']) != int(source['t']) + 1:
            stats['post_edgegraft_dropped_nonconsecutive_edges'] += 1
            continue
        distance = edge_distance_um(upgrade, source, target)
        edge['distance_um'] = distance
        if upgrade.edge_max_um > 0 and distance > upgrade.edge_max_um:
            if source_id not in fork_sources:
                stats['post_edgegraft_dropped_long_continuations'] += 1
                continue
            stats['post_edgegraft_kept_long_division_edges'] += 1
        pair = (source_id, target_id)
        previous = by_pair.get(pair)
        if previous is None or edge_sort_key(upgrade, edge) > edge_sort_key(upgrade, previous):
            by_pair[pair] = edge
        else:
            stats['post_edgegraft_dropped_duplicate_edges'] += 1

    cleaned = list(by_pair.values())
    best_by_target: dict[int, dict] = {}
    for edge in cleaned:
        target_id = int(edge['target_id'])
        previous = best_by_target.get(target_id)
        if previous is None or edge_sort_key(upgrade, edge) > edge_sort_key(upgrade, previous):
            best_by_target[target_id] = edge
    if len(best_by_target) != len(cleaned):
        stats['post_edgegraft_dropped_multi_parent_edges'] += len(cleaned) - len(best_by_target)
    cleaned = list(best_by_target.values())

    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for edge in cleaned:
        incoming[int(edge['target_id'])] += 1
        outgoing[int(edge['source_id'])] += 1
    if any(value > 1 for value in incoming.values()):
        raise RuntimeError('Post-EdgeGRAFT cleanup left in-degree > 1')
    if any(value > 2 for value in outgoing.values()):
        raise RuntimeError('Post-EdgeGRAFT cleanup left out-degree > 2')
    stats['post_edgegraft_contract_edges'] = len(cleaned)
    return cleaned


def validate_final_graph_contract(upgrade, nodes_by_id, edges, stats):
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    final_long_edges = 0
    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            raise RuntimeError('Final graph contains a dangling edge')
        if upgrade.enforce_next_frame and int(target['t']) != int(source['t']) + 1:
            raise RuntimeError('Final graph contains a nonconsecutive edge')
        incoming[target_id] += 1
        outgoing[source_id] += 1
        final_long_edges += int(
            upgrade.edge_max_um > 0
            and edge_distance_um(upgrade, source, target) > upgrade.edge_max_um
        )
    if any(value > 1 for value in incoming.values()):
        raise RuntimeError('Final graph contains in-degree > 1')
    if any(value > 2 for value in outgoing.values()):
        raise RuntimeError('Final graph contains out-degree > 2')
    stats['final_edges_over_max_um_after_linefit'] = final_long_edges
