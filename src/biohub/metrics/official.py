import warnings
from typing import Literal, NamedTuple

import polars as pl
import tracksdata as td
from scipy.sparse import SparseEfficiencyWarning
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options

from biohub.metrics.divisions import evaluate_divisions


class EvaluationResult(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int


class DatasetsResult(NamedTuple):
    edge_jaccard: float
    division_jaccard: float
    score: float


ADJUSTMENT_ALPHA: float = 0.1

SCORE_DIVISION_WEIGHT: float = 0.1

COUNT_COLUMNS: tuple[str, ...] = (
    'edge_tp',
    'edge_fp',
    'edge_fn',
    'division_tp',
    'division_fp',
    'division_fn',
    'num_pred_nodes',
)
METRIC_COLUMNS: tuple[str, ...] = COUNT_COLUMNS + (
    'node_recall',
    'total_node_ratio',
    'edge_jaccard',
    'adj_edge_jaccard',
)


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float('nan')


def _evaluate_matched_graph(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> pl.DataFrame:
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK])
    edge_attrs = edge_attrs.sort(
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
        descending=True,
    ).unique(
        subset=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET],
        keep='first',
    )
    node_attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )

    gt_node_ids = gt_graph.node_ids()
    gt_node_attrs = pl.DataFrame(
        {
            td.DEFAULT_ATTR_KEYS.NODE_ID: gt_node_ids,
            'out_degree': gt_graph.out_degree(gt_node_ids),
            'in_degree': gt_graph.in_degree(gt_node_ids),
        }
    ).with_columns(
        (pl.col('out_degree') > 0).alias('out_valid'),
        (pl.col('in_degree') > 0).alias('in_valid'),
    )

    node_attrs = node_attrs.join(
        gt_node_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how='left',
    ).with_columns(
        pl.col('out_valid').fill_null(False),
        pl.col('in_valid').fill_null(False),
    )

    edge_attrs = edge_attrs.join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, 'out_valid'),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how='left',
    ).join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, 'in_valid'),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how='left',
    )

    edge_attrs = edge_attrs.with_columns(
        (pl.col('out_valid') | pl.col('in_valid')).alias('pred_valid'),
    )

    assert edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)['pred_valid'].all()

    return edge_attrs


def _compute_score(
    edge_attrs: pl.DataFrame,
    gt_num_edges: int,
    metric: Literal['jaccard', 'dice'],
) -> float:
    intersection = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    n_valid_pred_edges = int(edge_attrs['pred_valid'].sum())

    if metric == 'jaccard':
        num = intersection
        denom = gt_num_edges + n_valid_pred_edges - intersection
    elif metric == 'dice':
        num = 2 * intersection
        denom = gt_num_edges + n_valid_pred_edges
    else:
        raise ValueError(f'Invalid metric: {metric}')

    return num / denom if denom > 0 else float('nan')


def _evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    metric: Literal['jaccard', 'dice'],
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> float:
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in graph.node_attr_keys():
        warnings.warn('Graph already matched, overwriting previous matching.')
        all_node_ids = graph.node_ids()
        graph.update_node_attrs(
            node_ids=all_node_ids,
            attrs={
                td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1,
                td.DEFAULT_ATTR_KEYS.MATCH_SCORE: 0.0,
            },
        )
        all_edge_ids = graph.edge_ids()
        if len(all_edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=all_edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    if graph.num_edges() == 0 or graph.num_nodes() == 0:
        warnings.warn('Predicted graph has no edges or no nodes, returning score 0.0.')
        return 0.0

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
            graph.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    return _compute_score(edge_attrs, gt_graph.num_edges(), metric)


def evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> EvaluationResult:
    _evaluate(graph, gt_graph, 'jaccard', scale, max_distance)

    if graph.num_edges() == 0:
        edge_tp = 0
        edge_fp = 0
        edge_fn = gt_graph.num_edges()
    else:
        edge_attrs = _evaluate_matched_graph(graph, gt_graph)
        edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        edge_valid_pred = int(edge_attrs['pred_valid'].sum())
        edge_fp = edge_valid_pred - edge_tp
        edge_fn = gt_graph.num_edges() - edge_tp

    div = evaluate_divisions(
        graph,
        gt_graph,
        scale=scale,
        max_distance=max_distance,
    )

    return EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div.tp,
        division_fp=div.fp,
        division_fn=div.fn,
        num_pred_nodes=graph.num_nodes(),
    )


def evaluate_datasets(
    graph_pairs: list[tuple[td.graph.BaseGraph, td.graph.BaseGraph]],
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DatasetsResult:
    edge_tp = edge_fp = edge_fn = 0
    div_tp = div_fp = div_fn = 0
    for pred, gt in graph_pairs:
        r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        edge_tp += r.edge_tp
        edge_fp += r.edge_fp
        edge_fn += r.edge_fn
        div_tp += r.division_tp
        div_fp += r.division_fp
        div_fn += r.division_fn

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    has_divisions = (div_tp + div_fp + div_fn) > 0
    division_jaccard = _jaccard(div_tp, div_fp, div_fn) if has_divisions else float('nan')
    score = (
        edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard if has_divisions else edge_jaccard
    )

    return DatasetsResult(
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        score=score,
    )


def _matched_node_ids(graph: td.graph.BaseGraph) -> pl.DataFrame:
    node_attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    return node_attrs


def node_recall(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> float:
    node_attrs = _matched_node_ids(graph)
    matched = node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )
    n_matched_gt = matched[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].n_unique()
    return n_matched_gt / gt_graph.num_nodes()


def per_sample_metrics(
    er: EvaluationResult,
    n_total: float,
    node_recall: float,
) -> dict:
    if n_total > 0:
        total_node_ratio = (er.num_pred_nodes - n_total) / n_total
    else:
        total_node_ratio = float('nan')

    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float('nan')
    if edge_jaccard == edge_jaccard and total_node_ratio == total_node_ratio:
        adj_edge_jaccard = max(
            0.0,
            edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio),
        )
    else:
        adj_edge_jaccard = float('nan')

    return {
        'edge_tp': er.edge_tp,
        'edge_fp': er.edge_fp,
        'edge_fn': er.edge_fn,
        'division_tp': er.division_tp,
        'division_fp': er.division_fp,
        'division_fn': er.division_fn,
        'num_pred_nodes': er.num_pred_nodes,
        'node_recall': node_recall,
        'total_node_ratio': total_node_ratio,
        'edge_jaccard': edge_jaccard,
        'adj_edge_jaccard': adj_edge_jaccard,
    }


def nan_metrics_row() -> dict:
    return {col: float('nan') for col in METRIC_COLUMNS}


def summarise(rows: list[dict]) -> dict:
    valid = [r for r in rows if r['edge_tp'] == r['edge_tp']]
    if not valid:
        return {
            'n': 0,
            'edge_jaccard': float('nan'),
            'division_jaccard': float('nan'),
            'division_tp': 0,
            'division_fp': 0,
            'division_fn': 0,
            'node_recall': float('nan'),
            'adj_edge_jaccard': float('nan'),
            'n_adj': 0,
            'score': float('nan'),
        }
    totals = {c: sum(r[c] for r in valid) for c in COUNT_COLUMNS}

    adj_rows = [r for r in valid if r['adj_edge_jaccard'] == r['adj_edge_jaccard']]
    weights = [r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in adj_rows]
    total_w = sum(weights)
    if total_w > 0:
        adj_edge_jaccard = (
            sum(w * r['adj_edge_jaccard'] for w, r in zip(weights, adj_rows)) / total_w
        )
    else:
        adj_edge_jaccard = float('nan')

    division_total = totals['division_tp'] + totals['division_fp'] + totals['division_fn']
    if division_total == 0:
        warnings.warn(
            'No divisions present across any sample in this split; '
            'dropping division term from the combined score.'
        )
        division_jaccard = float('nan')
        score = adj_edge_jaccard
    else:
        division_jaccard = _jaccard(
            totals['division_tp'],
            totals['division_fp'],
            totals['division_fn'],
        )
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    return {
        'n': len(valid),
        'edge_jaccard': _jaccard(
            totals['edge_tp'],
            totals['edge_fp'],
            totals['edge_fn'],
        ),
        'division_jaccard': division_jaccard,
        'division_tp': totals['division_tp'],
        'division_fp': totals['division_fp'],
        'division_fn': totals['division_fn'],
        'node_recall': sum(r['node_recall'] for r in valid) / len(valid),
        'adj_edge_jaccard': adj_edge_jaccard,
        'n_adj': len(adj_rows),
        'score': score,
    }
