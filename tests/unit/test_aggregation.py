import math

from biohub.contracts import EvaluationLevel
from biohub.metrics.aggregation import nan_row, per_sample_metrics, summarise
from biohub.metrics.scorer import IncompleteEvaluationError, completeness


def _row(
    movie_id: str, n_pred: float, n_total: float, tp: float = 10, fp: float = 0, fn: float = 0
):
    return per_sample_metrics(
        edge_tp=tp,
        edge_fp=fp,
        edge_fn=fn,
        division_tp=0,
        division_fp=0,
        division_fn=0,
        num_pred_nodes=n_pred,
        n_total=n_total,
        node_recall=1.0,
        movie_id=movie_id,
    )


def test_node_penalty_below_equal_above_estimate() -> None:
    below = _row('a', n_pred=90, n_total=100)
    equal = _row('b', n_pred=100, n_total=100)
    above = _row('c', n_pred=110, n_total=100)
    assert below.adj_edge_jaccard > equal.adj_edge_jaccard
    assert math.isclose(equal.adj_edge_jaccard, equal.edge_jaccard)
    assert above.adj_edge_jaccard < equal.adj_edge_jaccard
    assert above.adj_edge_jaccard == 1.0 * (1 - 0.1 * 0.10)


def test_no_divisions_drops_division_term() -> None:
    row = _row('a', n_pred=100, n_total=100)
    summary = summarise([row])
    assert math.isnan(summary['division_jaccard'])
    assert math.isclose(summary['score'], summary['adj_edge_jaccard'])


def test_sample_weighted_adjusted_aggregation() -> None:
    small = _row('small', n_pred=100, n_total=100, tp=1, fp=0, fn=0)
    large = _row('large', n_pred=200, n_total=100, tp=100, fp=0, fn=0)
    summary = summarise([small, large])
    expected = (1 * small.adj_edge_jaccard + 100 * large.adj_edge_jaccard) / 101
    assert math.isclose(summary['adj_edge_jaccard'], expected)


def test_missing_movie_fails_require_complete() -> None:
    rows = [_row('kept', n_pred=100, n_total=100)]
    completeness(
        rows, ['kept'], evaluation_level=EvaluationLevel.LEGACY_PARITY, require_complete=True
    )
    try:
        completeness(
            rows,
            ['kept', 'missing'],
            evaluation_level=EvaluationLevel.LEGACY_PARITY,
            require_complete=True,
        )
    except IncompleteEvaluationError:
        return
    raise AssertionError('missing movie must invalidate promotion')


def test_failed_movie_is_not_silently_skipped() -> None:
    rows = [_row('ok', n_pred=100, n_total=100), nan_row('bad', 'failed', 'boom')]
    summary = summarise(rows)
    assert summary['n'] == 1
    assert summary['n_expected'] == 2
    report = completeness(
        rows,
        ['ok', 'bad'],
        evaluation_level=EvaluationLevel.LEGACY_PARITY,
        require_complete=False,
    )
    assert report.failed_movies == ('bad',)
    assert not report.complete
