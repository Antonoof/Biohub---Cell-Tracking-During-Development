import json
from pathlib import Path
from typing import Any


def load_summary(run_path: Path) -> dict[str, Any]:
    summary = json.loads((run_path / 'evaluation' / 'summary.json').read_text())
    per_movie = json.loads((run_path / 'evaluation' / 'per_movie.json').read_text())
    manifest = (
        json.loads((run_path / 'manifest.json').read_text())
        if (run_path / 'manifest.json').is_file()
        else {}
    )
    completeness_path = run_path / 'evaluation' / 'completeness.json'
    completeness = json.loads(completeness_path.read_text()) if completeness_path.is_file() else {}
    return {
        'summary': summary,
        'per_movie': per_movie,
        'manifest': manifest,
        'completeness': completeness,
    }


def evaluation_level_of(payload: dict[str, Any]) -> str | None:
    level = payload.get('manifest', {}).get('evaluation_level')
    if level:
        return str(level)
    completeness_level = payload.get('completeness', {}).get('evaluation_level')
    if completeness_level:
        return str(completeness_level)
    return None


def compare_runs(baseline: Path, candidate: Path) -> dict[str, Any]:
    left = load_summary(baseline)
    right = load_summary(candidate)
    left_level = evaluation_level_of(left)
    right_level = evaluation_level_of(right)
    if not left_level or not right_level:
        raise ValueError('Refusing to compare runs without evaluation_level')
    if left_level != right_level:
        raise ValueError(
            f'Refusing to compare different evaluation levels: {left_level} vs {right_level}'
        )
    left_rows = {row['movie_id']: row for row in left['per_movie']}
    right_rows = {row['movie_id']: row for row in right['per_movie']}
    if set(left_rows) != set(right_rows):
        raise ValueError(
            f'Refusing to compare different movie sets: {sorted(left_rows)} vs {sorted(right_rows)}'
        )
    movie_ids = sorted(left_rows)
    deltas = []
    incomplete = []
    for movie_id in movie_ids:
        a = left_rows[movie_id]
        b = right_rows[movie_id]
        if a.get('status') != 'ok' or b.get('status') != 'ok':
            incomplete.append(movie_id)
            deltas.append({'movie_id': movie_id, 'status': 'incomplete'})
            continue
        deltas.append(
            {
                'movie_id': movie_id,
                'status': 'ok',
                'adj_edge_jaccard_delta': b['adj_edge_jaccard'] - a['adj_edge_jaccard'],
                'division_tp_delta': b['division_tp'] - a['division_tp'],
                'division_fp_delta': b['division_fp'] - a['division_fp'],
                'division_fn_delta': b['division_fn'] - a['division_fn'],
                'num_pred_nodes_delta': b['num_pred_nodes'] - a['num_pred_nodes'],
            }
        )
    if incomplete:
        raise ValueError(f'Refusing to compare incomplete movies: {incomplete}')
    return {
        'baseline': str(baseline),
        'candidate': str(candidate),
        'evaluation_level': left_level,
        'baseline_score': left['summary'].get('score'),
        'candidate_score': right['summary'].get('score'),
        'score_delta': (right['summary'].get('score') or 0) - (left['summary'].get('score') or 0),
        'per_movie': deltas,
    }
