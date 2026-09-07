import hashlib
import json
from pathlib import Path

import numpy as np

from biohub.contracts import GraphState, TopologyIssue


def validate_topology(graph: GraphState, *, next_frame: bool = True) -> list[TopologyIssue]:
    issues: list[TopologyIssue] = []
    if graph.node_ids.size != np.unique(graph.node_ids).size:
        issues.append(TopologyIssue('duplicate_node_id', 'Node IDs are not unique'))
    id_to_index = {int(node_id): i for i, node_id in enumerate(graph.node_ids.tolist())}
    id_to_t = {int(node_id): int(graph.t[i]) for node_id, i in id_to_index.items()}

    seen_edges: set[tuple[int, int]] = set()
    incoming: dict[int, int] = {}
    outgoing: dict[int, list[int]] = {}
    for source, target in zip(graph.source_ids.tolist(), graph.target_ids.tolist(), strict=True):
        src, tgt = int(source), int(target)
        if src not in id_to_index or tgt not in id_to_index:
            issues.append(
                TopologyIssue('missing_endpoint', f'Edge {src}->{tgt} has a missing endpoint')
            )
            continue
        key = (src, tgt)
        if key in seen_edges:
            issues.append(TopologyIssue('duplicate_edge', f'Duplicate edge {src}->{tgt}'))
        seen_edges.add(key)
        if next_frame and id_to_t[tgt] != id_to_t[src] + 1:
            issues.append(
                TopologyIssue('nonconsecutive_frame', f'Edge {src}->{tgt} is not t -> t+1')
            )
        incoming[tgt] = incoming.get(tgt, 0) + 1
        outgoing.setdefault(src, []).append(tgt)

    for node_id, degree in incoming.items():
        if degree > 1:
            issues.append(TopologyIssue('in_degree', f'Node {node_id} has in-degree {degree}'))
    for node_id, children in outgoing.items():
        if len(children) > 2:
            issues.append(
                TopologyIssue('out_degree', f'Node {node_id} has out-degree {len(children)}')
            )
        if len(children) == 2 and children[0] == children[1]:
            issues.append(
                TopologyIssue('duplicate_daughter', f'Node {node_id} has identical daughters')
            )
    return issues


def fingerprint(graph: GraphState) -> str:
    payload = np.concatenate(
        [
            graph.node_ids.astype(np.float64),
            graph.t.astype(np.float64),
            graph.z,
            graph.y,
            graph.x,
            graph.source_ids.astype(np.float64),
            graph.target_ids.astype(np.float64),
        ]
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def save_graph_npz(graph: GraphState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        'node_ids': graph.node_ids,
        't': graph.t,
        'z': graph.z,
        'y': graph.y,
        'x': graph.x,
        'source_ids': graph.source_ids,
        'target_ids': graph.target_ids,
    }
    for key, value in graph.node_attrs.items():
        arrays[f'node_attr__{key}'] = value
    for key, value in graph.edge_attrs.items():
        arrays[f'edge_attr__{key}'] = value
    np.savez_compressed(path, allow_pickle=False, **arrays)
    meta = {
        'movie_id': graph.movie_id,
        'estimated_number_of_nodes': graph.estimated_number_of_nodes,
        'extra': graph.extra,
        'fingerprint': fingerprint(graph),
    }
    path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def load_graph_npz(path: Path) -> GraphState:
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        node_attrs = {
            key.removeprefix('node_attr__'): bundle[key]
            for key in bundle.files
            if key.startswith('node_attr__')
        }
        edge_attrs = {
            key.removeprefix('edge_attr__'): bundle[key]
            for key in bundle.files
            if key.startswith('edge_attr__')
        }
        graph = GraphState(
            movie_id='',
            node_ids=bundle['node_ids'],
            t=bundle['t'],
            z=bundle['z'],
            y=bundle['y'],
            x=bundle['x'],
            source_ids=bundle['source_ids'],
            target_ids=bundle['target_ids'],
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
        )
    meta_path = path.with_suffix('.meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        graph.movie_id = str(meta.get('movie_id', path.stem))
        graph.estimated_number_of_nodes = meta.get('estimated_number_of_nodes')
        graph.extra = dict(meta.get('extra') or {})
    else:
        graph.movie_id = path.stem
    return graph
