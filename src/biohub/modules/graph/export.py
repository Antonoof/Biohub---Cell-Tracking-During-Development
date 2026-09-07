import csv
from pathlib import Path

import tracksdata as td


def _columnar_nodes_edges(upgrade, graph):
    try:
        node_table = graph.node_attrs()
        edge_table = graph.edge_attrs()
        node_ids = [int(value) for value in node_table['node_id'].to_list()]
        frames = [int(value) for value in node_table['t'].to_list()]
        depths = [float(value) for value in node_table['z'].to_list()]
        rows = [float(value) for value in node_table['y'].to_list()]
        columns = [float(value) for value in node_table['x'].to_list()]
        sources = [int(value) for value in edge_table['source_id'].to_list()]
        targets = [int(value) for value in edge_table['target_id'].to_list()]
        if 'edge_prob' in list(edge_table.columns):
            probabilities = [
                None if value is None else float(value)
                for value in edge_table['edge_prob'].to_list()
            ]
        else:
            probabilities = [None] * len(sources)
    except Exception:
        return None

    nodes = {
        node_id: {'node_id': node_id, 't': t, 'z': z, 'y': y, 'x': x}
        for node_id, t, z, y, x in zip(node_ids, frames, depths, rows, columns)
    }
    edges = [
        {'source_id': source, 'target_id': target, 'edge_prob': probability}
        for source, target, probability in zip(sources, targets, probabilities)
    ]
    return nodes, edges


def graph_nodes_edges(upgrade, path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(graph, tuple):
        graph = graph[0]

    if upgrade.fast_geff_reader:
        result = _columnar_nodes_edges(upgrade, graph)
        if result is not None:
            return result

    nodes: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row['node_id'])
        nodes[node_id] = {
            'node_id': node_id,
            't': int(row['t']),
            'z': float(row['z']),
            'y': float(row['y']),
            'x': float(row['x']),
        }
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        prob = row.get('edge_prob')
        edges.append(
            {
                'source_id': int(row['source_id']),
                'target_id': int(row['target_id']),
                'edge_prob': None if prob is None else float(prob),
            }
        )
    return nodes, edges


def write_dataset_shard(upgrade, dataset: str, nodes_by_id: dict, edges: list[dict]):
    destination = upgrade.shard_dir / f'{dataset}.csv'
    temporary = destination.with_suffix('.csv.tmp')
    row_id = 0
    out_degree: dict[int, int] = {}

    if upgrade.fast_shard_writer and not any(mark in dataset for mark in (',', '"', '\r', '\n')):
        lines = [','.join(upgrade.csv_columns)]
        for node_id in sorted(nodes_by_id):
            node = nodes_by_id[node_id]
            z = max(0, int(round(float(node['z']))))
            y = max(0, int(round(float(node['y']))))
            x = max(0, int(round(float(node['x']))))
            lines.append(
                f'{row_id},{dataset},node,{int(node["node_id"])},{int(node["t"])},{z},{y},{x},-1,-1'
            )
            row_id += 1
        for edge in edges:
            source_id = int(edge['source_id'])
            target_id = int(edge['target_id'])
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise AssertionError(f'{dataset}: dangling edge after filtering')
            lines.append(f'{row_id},{dataset},edge,-1,-1,-1,-1,-1,{source_id},{target_id}')
            row_id += 1
            out_degree[source_id] = out_degree.get(source_id, 0) + 1
        with temporary.open('w', newline='') as handle:
            handle.write('\r\n'.join(lines))
            handle.write('\r\n')
        temporary.replace(destination)
        return len(nodes_by_id), len(edges), sum(v >= 2 for v in out_degree.values())

    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=upgrade.csv_columns)
        writer.writeheader()
        for node_id in sorted(nodes_by_id):
            node = nodes_by_id[node_id]
            writer.writerow(
                {
                    'id': row_id,
                    'dataset': dataset,
                    'row_type': 'node',
                    'node_id': int(node['node_id']),
                    't': int(node['t']),
                    'z': max(0, int(round(float(node['z'])))),
                    'y': max(0, int(round(float(node['y'])))),
                    'x': max(0, int(round(float(node['x'])))),
                    'source_id': -1,
                    'target_id': -1,
                }
            )
            row_id += 1
        for edge in edges:
            source_id = int(edge['source_id'])
            target_id = int(edge['target_id'])
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise AssertionError(f'{dataset}: dangling edge after filtering')
            writer.writerow(
                {
                    'id': row_id,
                    'dataset': dataset,
                    'row_type': 'edge',
                    'node_id': -1,
                    't': -1,
                    'z': -1,
                    'y': -1,
                    'x': -1,
                    'source_id': source_id,
                    'target_id': target_id,
                }
            )
            row_id += 1
            out_degree[source_id] = out_degree.get(source_id, 0) + 1
    temporary.replace(destination)
    return len(nodes_by_id), len(edges), sum(v >= 2 for v in out_degree.values())
