import math

from biohub.constants import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT
from biohub.contracts import MetricRow

COUNT_COLUMNS = (
    'edge_tp',
    'edge_fp',
    'edge_fn',
    'division_tp',
    'division_fp',
    'division_fn',
    'num_pred_nodes',
)


def _jaccard(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float('nan')


def per_sample_metrics(
    *,
    edge_tp: float,
    edge_fp: float,
    edge_fn: float,
    division_tp: float,
    division_fp: float,
    division_fn: float,
    num_pred_nodes: float,
    n_total: float,
    node_recall: float,
    movie_id: str,
) -> MetricRow:
    if n_total > 0:
        total_node_ratio = (num_pred_nodes - n_total) / n_total
    else:
        total_node_ratio = float('nan')
    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    if math.isfinite(edge_jaccard) and math.isfinite(total_node_ratio):
        adj_edge_jaccard = max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
    else:
        adj_edge_jaccard = float('nan')
    return MetricRow(
        movie_id=movie_id,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=division_tp,
        division_fp=division_fp,
        division_fn=division_fn,
        num_pred_nodes=num_pred_nodes,
        node_recall=node_recall,
        total_node_ratio=total_node_ratio,
        edge_jaccard=edge_jaccard,
        adj_edge_jaccard=adj_edge_jaccard,
        estimated_number_of_nodes=n_total,
    )


def nan_row(movie_id: str, status: str, error: str | None = None) -> MetricRow:
    nan = float('nan')
    return MetricRow(
        movie_id=movie_id,
        edge_tp=nan,
        edge_fp=nan,
        edge_fn=nan,
        division_tp=nan,
        division_fp=nan,
        division_fn=nan,
        num_pred_nodes=nan,
        node_recall=nan,
        total_node_ratio=nan,
        edge_jaccard=nan,
        adj_edge_jaccard=nan,
        estimated_number_of_nodes=nan,
        status=status,
        error=error,
    )


def summarise(rows: list[MetricRow]) -> dict[str, float | int]:
    valid = [row for row in rows if row.status == 'ok' and row.edge_tp == row.edge_tp]
    if not valid:
        return {
            'n': 0,
            'n_expected': len(rows),
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
    totals = {column: sum(getattr(row, column) for row in valid) for column in COUNT_COLUMNS}
    adj_rows = [row for row in valid if row.adj_edge_jaccard == row.adj_edge_jaccard]
    weights = [row.edge_tp + row.edge_fp + row.edge_fn for row in adj_rows]
    total_w = sum(weights)
    if total_w > 0:
        adj_edge_jaccard = (
            sum(
                weight * row.adj_edge_jaccard for weight, row in zip(weights, adj_rows, strict=True)
            )
            / total_w
        )
    else:
        adj_edge_jaccard = float('nan')
    division_total = totals['division_tp'] + totals['division_fp'] + totals['division_fn']
    if division_total == 0:
        division_jaccard = float('nan')
        score = adj_edge_jaccard
    else:
        division_jaccard = _jaccard(
            totals['division_tp'], totals['division_fp'], totals['division_fn']
        )
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    return {
        'n': len(valid),
        'n_expected': len(rows),
        'edge_jaccard': _jaccard(totals['edge_tp'], totals['edge_fp'], totals['edge_fn']),
        'division_jaccard': division_jaccard,
        'division_tp': int(totals['division_tp']),
        'division_fp': int(totals['division_fp']),
        'division_fn': int(totals['division_fn']),
        'node_recall': sum(row.node_recall for row in valid) / len(valid),
        'adj_edge_jaccard': adj_edge_jaccard,
        'n_adj': len(adj_rows),
        'score': score,
    }
