import json
from pathlib import Path

import yaml

from biohub.data.geff import graph_to_tracksdata
from biohub.data.synthetic import continuation_pair
from biohub.data.volume import save_graph
from biohub.metrics.evaluate import run_evaluate
from biohub.validation.compare import compare_runs, evaluation_level_of, load_summary


def _eval_config(tmp_path: Path, train_dir: Path, catalog: Path) -> Path:
    payload = {
        'train_dir': str(train_dir),
        'voxel_scale_um': [1.625, 0.40625, 0.40625],
        'match_max_distance_um': 7.0,
        'movie_catalog': str(catalog),
    }
    path = tmp_path / 'eval.yaml'
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_gt(train_dir: Path) -> str:
    _pred, gt = continuation_pair()
    movie_id = gt.movie_id
    tracks, _remap = graph_to_tracksdata(gt)
    save_graph(tracks, train_dir / f'{movie_id}.geff')
    return movie_id


def test_evaluate_writes_level_and_compare_runs(tmp_path: Path) -> None:
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    movie_id = _write_gt(train_dir)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(
        json.dumps(
            {
                'movies': [
                    {
                        'movie_id': movie_id,
                        'estimated_number_of_nodes': 3,
                    }
                ]
            }
        )
        + '\n'
    )
    config = _eval_config(tmp_path, train_dir, catalog)
    left = run_evaluate(
        config=config,
        pred_dir=None,
        pred_source='gt',
        panel='smoke',
        require_complete=True,
        n_workers=1,
        run_id='eval_left',
        runs_root=tmp_path / 'runs',
        movie_ids=[movie_id],
    )
    right = run_evaluate(
        config=config,
        pred_dir=None,
        pred_source='gt',
        panel='smoke',
        require_complete=True,
        n_workers=1,
        run_id='eval_right',
        runs_root=tmp_path / 'runs',
        movie_ids=[movie_id],
    )
    left_path = Path(left['run'])
    right_path = Path(right['run'])
    left_summary = load_summary(left_path)
    assert left_summary['manifest']['evaluation_level'] == 'legacy_parity'
    assert left_summary['completeness']['evaluation_level'] == 'legacy_parity'
    assert evaluation_level_of(left_summary) == 'legacy_parity'
    report = compare_runs(left_path, right_path)
    assert report['evaluation_level'] == 'legacy_parity'
    assert report['score_delta'] == 0


def test_compare_reads_completeness_when_manifest_omits_level(tmp_path: Path) -> None:
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    movie_id = _write_gt(train_dir)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(
        json.dumps({'movies': [{'movie_id': movie_id, 'estimated_number_of_nodes': 3}]})
        + '\n'
    )
    config = _eval_config(tmp_path, train_dir, catalog)
    payload = run_evaluate(
        config=config,
        pred_dir=None,
        pred_source='gt',
        panel='smoke',
        require_complete=True,
        n_workers=1,
        run_id='eval_legacy_manifest',
        runs_root=tmp_path / 'runs',
        movie_ids=[movie_id],
    )
    run_path = Path(payload['run'])
    manifest = json.loads((run_path / 'manifest.json').read_text())
    del manifest['evaluation_level']
    (run_path / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = compare_runs(run_path, run_path)
    assert report['evaluation_level'] == 'legacy_parity'
