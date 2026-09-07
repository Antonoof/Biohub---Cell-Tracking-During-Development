import json
from pathlib import Path

from biohub.constants import MATCH_MAX_DISTANCE_UM, VOXEL_SCALE_ZYX
from biohub.contracts import CompletenessReport, EvaluationLevel, GraphState, MetricRow
from biohub.data.geff import graph_to_tracksdata
from biohub.metrics import official as metrics
from biohub.metrics.aggregation import per_sample_metrics, summarise


class IncompleteEvaluationError(RuntimeError):
    pass


def score_graph_pair(
    pred: GraphState,
    gt: GraphState,
    *,
    n_total: float | None = None,
    scale: tuple[float, float, float] = VOXEL_SCALE_ZYX,
    max_distance: float = MATCH_MAX_DISTANCE_UM,
) -> MetricRow:
    pred_td, _ = graph_to_tracksdata(pred)
    gt_td, _ = graph_to_tracksdata(gt)
    result = metrics.evaluate(pred_td, gt_td, scale=scale, max_distance=max_distance)
    recall = metrics.node_recall(pred_td, gt_td)
    estimate = n_total if n_total is not None else pred.estimated_number_of_nodes
    if estimate is None:
        estimate = float('nan')
    return per_sample_metrics(
        edge_tp=result.edge_tp,
        edge_fp=result.edge_fp,
        edge_fn=result.edge_fn,
        division_tp=result.division_tp,
        division_fp=result.division_fp,
        division_fn=result.division_fn,
        num_pred_nodes=result.num_pred_nodes,
        n_total=float(estimate),
        node_recall=recall,
        movie_id=pred.movie_id or gt.movie_id,
    )


def completeness(
    rows: list[MetricRow],
    expected_ids: list[str],
    *,
    evaluation_level: EvaluationLevel,
    require_complete: bool,
) -> CompletenessReport:
    by_id = {row.movie_id: row for row in rows}
    missing = tuple(movie_id for movie_id in expected_ids if movie_id not in by_id)
    failed = tuple(
        movie_id
        for movie_id in expected_ids
        if movie_id in by_id and by_id[movie_id].status != 'ok'
    )
    missing_estimates = tuple(
        movie_id
        for movie_id in expected_ids
        if movie_id in by_id
        and by_id[movie_id].status == 'ok'
        and by_id[movie_id].estimated_number_of_nodes != by_id[movie_id].estimated_number_of_nodes
    )
    scored = tuple(
        movie_id for movie_id in expected_ids if movie_id in by_id and movie_id not in failed
    )
    complete = not missing and not failed and not missing_estimates
    report = CompletenessReport(
        expected_movies=tuple(expected_ids),
        scored_movies=scored,
        missing_movies=missing,
        failed_movies=failed,
        missing_node_estimates=missing_estimates,
        complete=complete,
        evaluation_level=evaluation_level,
        require_complete=require_complete,
    )
    if require_complete and not complete:
        raise IncompleteEvaluationError(
            'Evaluation is incomplete: '
            f'missing={list(missing)} failed={list(failed)} '
            f'missing_node_estimates={list(missing_estimates)}'
        )
    return report


def score_rows(rows: list[MetricRow]) -> dict[str, float | int]:
    return summarise(rows)


def write_evaluation(path: Path, rows: list[MetricRow], summary: dict[str, float | int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / 'per_movie.json').write_text(
        json.dumps([row.as_dict() for row in rows], indent=2) + '\n'
    )
    (path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
