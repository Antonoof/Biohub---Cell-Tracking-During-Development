import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata

from biohub.contracts import GraphState, MovieRecord
from biohub.data.graph import load_graph_npz


def load_graph(path: Path, movie_id: str | None = None) -> GraphState:
    path = Path(path)
    if path.suffix == '.npz':
        graph = load_graph_npz(path)
        if movie_id:
            graph.movie_id = movie_id
        return graph
    if path.suffix == '.geff' or path.is_dir():
        return load_geff(path, movie_id=movie_id)
    raise ValueError(f'unsupported graph path: {path}')


def load_geff(path: Path, movie_id: str | None = None) -> GraphState:
    path = Path(path)
    result = td.graph.IndexedRXGraph.from_geff(path)
    tracks = result[0] if isinstance(result, tuple) else result
    node_attrs = tracks.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    edge_attrs = tracks.edge_attrs(attr_keys=['source_id', 'target_id'])
    extra_node = {
        key: node_attrs[key].to_numpy()
        for key in node_attrs.columns
        if key not in {'node_id', 't', 'z', 'y', 'x'}
    }
    extra_edge = {
        key: edge_attrs[key].to_numpy()
        for key in edge_attrs.columns
        if key not in {'source_id', 'target_id', 'edge_id'}
    }
    estimated = _estimated_nodes(path)
    return GraphState(
        movie_id=movie_id or path.name.removesuffix('.geff'),
        node_ids=node_attrs['node_id'].to_numpy().astype(np.int64),
        t=node_attrs['t'].to_numpy().astype(np.int64),
        z=node_attrs['z'].to_numpy().astype(np.float64),
        y=node_attrs['y'].to_numpy().astype(np.float64),
        x=node_attrs['x'].to_numpy().astype(np.float64),
        source_ids=(
            edge_attrs['source_id'].to_numpy().astype(np.int64)
            if len(edge_attrs)
            else np.zeros(0, np.int64)
        ),
        target_ids=(
            edge_attrs['target_id'].to_numpy().astype(np.int64)
            if len(edge_attrs)
            else np.zeros(0, np.int64)
        ),
        node_attrs=extra_node,
        edge_attrs=extra_edge,
        estimated_number_of_nodes=estimated,
    )


def graph_to_tracksdata(graph: GraphState):
    td_graph = td.graph.InMemoryGraph()
    for key in ('z', 'y', 'x'):
        td_graph.add_node_attr_key(key, cast(pl.DataType, pl.Float64), -999999.0)
    records = [
        {'t': int(t), 'z': float(z), 'y': float(y), 'x': float(x)}
        for t, z, y, x in zip(
            graph.t.tolist(),
            graph.z.tolist(),
            graph.y.tolist(),
            graph.x.tolist(),
            strict=True,
        )
    ]
    assigned = td_graph.bulk_add_nodes(records)
    original = graph.node_ids.tolist()
    remap = {int(old): int(new) for old, new in zip(original, assigned, strict=True)}
    if graph.n_edges:
        td_graph.bulk_add_edges(
            [
                {'source_id': remap[int(src)], 'target_id': remap[int(tgt)]}
                for src, tgt in zip(
                    graph.source_ids.tolist(), graph.target_ids.tolist(), strict=True
                )
                if int(src) in remap and int(tgt) in remap
            ]
        )
    return td_graph, remap


def _estimated_nodes(path: Path) -> float | None:
    try:
        meta = GeffMetadata.read(path)
        value = (meta.extra or {}).get('estimated_number_of_nodes')
        return float(value) if value is not None else None
    except Exception:
        meta_path = Path(path) / 'zarr.json'
        if not meta_path.is_file():
            return None
        payload = json.loads(meta_path.read_text())
        extra = ((payload.get('attributes') or {}).get('geff') or {}).get('extra') or {}
        value = extra.get('estimated_number_of_nodes')
        return float(value) if value is not None else None


def estimated_nodes_for_movie(movie: MovieRecord) -> float | None:
    if movie.estimated_number_of_nodes is not None:
        return movie.estimated_number_of_nodes
    if movie.geff_path is not None:
        return _estimated_nodes(movie.geff_path)
    return None
