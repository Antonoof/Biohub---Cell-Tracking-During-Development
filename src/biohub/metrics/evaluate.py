import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from biohub.contracts import EvaluationLevel, MetricRow
from biohub.data.geff import estimated_nodes_for_movie, load_geff, load_graph
from biohub.data.graph import save_graph_npz
from biohub.data.movies import load_movie_catalog, movie_record_from_catalog, panel_movie_ids
from biohub.data.submission import load_submission_graphs
from biohub.log import setup_logging
from biohub.metrics.aggregation import nan_row, summarise
from biohub.metrics.scorer import completeness, score_graph_pair, write_evaluation
from biohub.paths import PROJECT_ROOT
from biohub.utils.runs import finish_run, init_run, new_run_id
from biohub.utils.yaml_config import load_yaml, resolve_path


def _find_prediction(pred_dir: Path, movie_id: str) -> Path:
    for candidate in (
        pred_dir / f'{movie_id}.npz',
        pred_dir / f'{movie_id}.geff',
        pred_dir / movie_id / 'final.geff',
        pred_dir / movie_id / 'final.npz',
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No prediction for {movie_id} under {pred_dir}')


def _evaluate_one(status: str, movie_id: str, catalog_row: dict | None, payload: dict) -> MetricRow:
    if status != 'ok' or catalog_row is None:
        return nan_row(movie_id, 'missing_catalog', 'movie is not in the catalog')
    train_dir = Path(payload['train_dir'])
    scale = tuple(payload['voxel_scale_um'])
    record = movie_record_from_catalog(catalog_row, train_dir, scale)
    try:
        if record.geff_path is None:
            raise FileNotFoundError(f'GT GEFF missing for {movie_id}')
        gt = load_geff(record.geff_path, movie_id=movie_id)
        if payload['pred_source'] == 'gt':
            pred = gt.copy()
        else:
            pred = load_graph(
                _find_prediction(Path(payload['pred_dir']), movie_id), movie_id=movie_id
            )
        n_total = estimated_nodes_for_movie(record)
        if n_total is None:
            n_total = gt.estimated_number_of_nodes
        return score_graph_pair(
            pred,
            gt,
            n_total=n_total,
            scale=scale,
            max_distance=payload['match_max_distance_um'],
        )
    except Exception as exc:
        return nan_row(movie_id, 'failed', str(exc))


def _evaluate_parallel(jobs: list, payload: dict, n_workers: int) -> list[MetricRow]:
    rows: list[MetricRow | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_evaluate_one, status, movie_id, catalog_row, payload): index
            for index, (status, movie_id, catalog_row) in enumerate(jobs)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    return [row for row in rows if row is not None]


def run_evaluate(
    *,
    config: Path,
    pred_dir: Path | None,
    pred_source: str,
    panel: str,
    require_complete: bool,
    n_workers: int | None,
    run_id: str | None,
) -> dict:
    raw = load_yaml(config)
    movie_ids = panel_movie_ids(panel)
    catalog_file = raw.get('movie_catalog')
    catalog = {
        item['movie_id']: item
        for item in load_movie_catalog(Path(catalog_file) if catalog_file else None)['movies']
    }
    resolved_run_id = run_id or new_run_id('eval')
    run_path = init_run(
        resolved_run_id,
        config=raw,
        extra={'panel': panel, 'pred_source': pred_source, 'command': 'evaluate'},
    )
    setup_logging(run_path)
    if pred_source == 'csv':
        if pred_dir is None:
            raise SystemExit('evaluate --pred-source csv requires --pred-dir')
        graphs = load_submission_graphs(pred_dir)
        pred_dir = run_path / 'pred_graphs'
        pred_dir.mkdir(parents=True, exist_ok=True)
        for movie_id, graph in graphs.items():
            save_graph_npz(graph, pred_dir / f'{movie_id}.npz')
    jobs = []
    for movie_id in movie_ids:
        catalog_row = catalog.get(movie_id)
        jobs.append(('ok' if catalog_row is not None else 'missing', movie_id, catalog_row))
    workers = n_workers if n_workers is not None else 0
    if workers <= 0:
        workers = min(32, os.cpu_count() or 1)
    payload = {
        'train_dir': str(resolve_path(raw['train_dir'])),
        'voxel_scale_um': list(raw['voxel_scale_um']),
        'match_max_distance_um': float(raw.get('match_max_distance_um', 7.0)),
        'pred_source': 'gt' if pred_source == 'gt' else 'dir',
        'pred_dir': str(pred_dir) if pred_dir else None,
    }
    if workers > 1 and len(jobs) > 1:
        rows = _evaluate_parallel(jobs, payload, workers)
    else:
        rows = [
            _evaluate_one(status, movie_id, catalog_row, payload)
            for status, movie_id, catalog_row in jobs
        ]
    level = EvaluationLevel('legacy_parity')
    report = completeness(
        rows, movie_ids, evaluation_level=level, require_complete=require_complete
    )
    completeness_payload = {
        'complete': report.complete,
        'missing_movies': list(report.missing_movies),
        'failed_movies': list(report.failed_movies),
        'missing_node_estimates': list(report.missing_node_estimates),
        'evaluation_level': report.evaluation_level.value,
    }
    summary = summarise(rows)
    write_evaluation(run_path / 'evaluation', rows, summary)
    (run_path / 'evaluation' / 'completeness.json').write_text(
        json.dumps(completeness_payload, indent=2) + '\n'
    )
    finish_run(run_path, 'ok' if report.complete else 'incomplete', extra={'summary': summary})
    return {'run': str(run_path), 'summary': summary, 'completeness': completeness_payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Official Biohub evaluation')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'infer.yaml')
    parser.add_argument('--panel', default='smoke')
    parser.add_argument('--pred-dir', type=Path, default=None)
    parser.add_argument('--pred-source', choices=('dir', 'gt', 'csv'), default='dir')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--n-workers', type=int, default=None)
    args = parser.parse_args(argv)
    if args.pred_source in {'dir', 'csv'} and args.pred_dir is None:
        raise SystemExit('evaluate --pred-source dir/csv requires --pred-dir')
    result = run_evaluate(
        config=args.config,
        pred_dir=args.pred_dir,
        pred_source=args.pred_source,
        panel=args.panel,
        require_complete=args.require_complete,
        n_workers=args.n_workers,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result['completeness']['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
