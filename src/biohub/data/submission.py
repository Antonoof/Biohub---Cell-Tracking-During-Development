from pathlib import Path

import numpy as np
import polars as pl

from biohub.contracts import GraphState

SUBMISSION_COLUMNS = (
    'id',
    'dataset',
    'row_type',
    'node_id',
    't',
    'z',
    'y',
    'x',
    'source_id',
    'target_id',
)


def load_submission_graphs(path: Path) -> dict[str, GraphState]:
    frame = pl.read_csv(path)
    missing = [column for column in SUBMISSION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f'submission csv is missing columns {missing}: {path}')
    graphs: dict[str, GraphState] = {}
    datasets = frame.get_column('dataset').unique(maintain_order=True).to_list()
    for dataset in datasets:
        group = frame.filter(pl.col('dataset') == dataset)
        nodes = group.filter(pl.col('row_type') == 'node')
        edges = group.filter(pl.col('row_type') == 'edge')
        graphs[str(dataset)] = GraphState(
            movie_id=str(dataset),
            node_ids=nodes.get_column('node_id').to_numpy().astype(np.int64, copy=False),
            t=nodes.get_column('t').to_numpy().astype(np.int64, copy=False),
            z=nodes.get_column('z').to_numpy().astype(np.float64, copy=False),
            y=nodes.get_column('y').to_numpy().astype(np.float64, copy=False),
            x=nodes.get_column('x').to_numpy().astype(np.float64, copy=False),
            source_ids=edges.get_column('source_id').to_numpy().astype(np.int64, copy=False),
            target_ids=edges.get_column('target_id').to_numpy().astype(np.int64, copy=False),
        )
    return graphs
