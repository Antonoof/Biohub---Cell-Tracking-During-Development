from pathlib import Path

import pytest

from biohub.validation.compare import compare_runs


def _write_run(path: Path, *, level: str, score: float, movie_id: str = 'a') -> None:
    (path / 'evaluation').mkdir(parents=True)
    (path / 'evaluation' / 'summary.json').write_text(f'{{"score": {score}}}\n')
    (path / 'evaluation' / 'per_movie.json').write_text(
        f'[{{"movie_id": "{movie_id}", "status": "ok", "adj_edge_jaccard": {score},'
        f' "division_tp": 0, "division_fp": 0, "division_fn": 0, "num_pred_nodes": 1}}]\n'
    )
    (path / 'manifest.json').write_text(f'{{"evaluation_level": "{level}"}}\n')


def test_compare_refuses_mixed_evaluation_levels(tmp_path: Path) -> None:
    baseline = tmp_path / 'base'
    candidate = tmp_path / 'cand'
    _write_run(baseline, level='legacy_parity', score=0.9)
    _write_run(candidate, level='strict_nested', score=0.91)
    with pytest.raises(ValueError, match='evaluation levels'):
        compare_runs(baseline, candidate)


def test_compare_reports_per_movie_delta(tmp_path: Path) -> None:
    baseline = tmp_path / 'base'
    candidate = tmp_path / 'cand'
    _write_run(baseline, level='legacy_parity', score=0.90)
    _write_run(candidate, level='legacy_parity', score=0.95)
    report = compare_runs(baseline, candidate)
    assert report['score_delta'] == pytest.approx(0.05)
    assert report['per_movie'][0]['adj_edge_jaccard_delta'] == pytest.approx(0.05)
