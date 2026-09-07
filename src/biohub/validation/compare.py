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
    return {'summary': summary, 'per_movie': per_movie, 'manifest': manifest}


def compare_runs(baseline: Path, candidate: Path) -> dict[str, Any]:
    left = load_summary(baseline)
    right = load_summary(candidate)
    left_level = left['manifest'].get('evaluation_level')
    right_level = right['manifest'].get('evaluation_level')
    if left_level and right_level and left_level != right_level:
        raise ValueError(
            f'Refusing to compare different evaluation levels: {left_level} vs {right_level}'
        )
    left_rows = {row['movie_id']: row for row in left['per_movie']}
    right_rows = {row['movie_id']: row for row in right['per_movie']}
    movie_ids = sorted(set(left_rows) | set(right_rows))
    deltas = []
    for movie_id in movie_ids:
        a = left_rows.get(movie_id)
        b = right_rows.get(movie_id)
        if not a or not b or a.get('status') != 'ok' or b.get('status') != 'ok':
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
    return {
        'baseline': str(baseline),
        'candidate': str(candidate),
        'evaluation_level': left_level or right_level,
        'baseline_score': left['summary'].get('score'),
        'candidate_score': right['summary'].get('score'),
        'score_delta': (right['summary'].get('score') or 0) - (left['summary'].get('score') or 0),
        'per_movie': deltas,
    }
