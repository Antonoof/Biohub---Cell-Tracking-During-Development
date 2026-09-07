import argparse
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from biohub.config import FrozenModel
from biohub.infer.config import RuntimeConfig, load_tracking_config
from biohub.paths import PROJECT_ROOT


def test_infer_yaml_thresholds() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / 'configs/infer.yaml').read_text())
    assert payload['graph']['relink_max_match_cost'] == 7.5
    assert payload['division']['model_c_threshold'] == 0.96
    assert payload['graph']['gap_close_um'] == 5.8
    assert payload['detection']['threshold'] == 0.96875
    assert payload['association']['secondary_detection_weight'] == 0.475
    assert payload['ilp']['disappearance'] == 1.5
    assert payload['seed'] == 2026


def test_tracking_config_bundle_paths(tmp_path: Path) -> None:
    bundle = tmp_path / 'bundle'
    for name in (
        'model_c/weights',
        'deepcenter',
        'motion_corrector',
        'source_cardinality',
        'unigraft_p1p2',
        'live_v2_bundle',
        'ownership',
        'edgegraft',
        'candidategraft',
    ):
        (bundle / name).mkdir(parents=True)
    tracking = load_tracking_config(
        bundle_dir=bundle,
        test_dir=tmp_path / 'movies',
        work_dir=tmp_path / 'work',
    )
    assert tracking.bundle.model_c_checkpoint.name == 'edge_predictor_best.pth'
    assert tracking.bundle.deepcenter_checkpoint.name == 'best.pt'
    assert tracking.bundle.motion_checkpoint.name == 'motion_corrector_best.pt'
    assert tracking.detection.threshold == 0.96875


def test_unknown_config_key_is_rejected() -> None:
    class Sample(FrozenModel):
        name: str = 'local'

    with pytest.raises(ValidationError):
        Sample.model_validate({'name': 'local', 'unexpected': True})


def test_stage_train_yaml_has_seed() -> None:
    for root in (PROJECT_ROOT / 'configs', PROJECT_ROOT / 'configs' / 'smoke'):
        for path in sorted(root.glob('*.yaml')):
            payload = yaml.safe_load(path.read_text())
            assert 'seed' in payload, str(path.relative_to(PROJECT_ROOT))


def test_smoke_configs_live_under_smoke_dir() -> None:
    smoke = PROJECT_ROOT / 'configs' / 'smoke'
    names = {path.name for path in smoke.glob('*.yaml')}
    assert names == {
        '01_p1.yaml',
        '01_p2.yaml',
        '02_model_c.yaml',
        '02_division.yaml',
        '02_division_decoder.yaml',
        '03_cardinality.yaml',
        '04_unigraft.yaml',
        '06_ownership.yaml',
        '07_edgegraft_ranker.yaml',
        '07_edgegraft_gate.yaml',
        '07_edgegraft_oracle.yaml',
        '08_candidategraft.yaml',
        '08_candidategraft_screen.yaml',
        '09_motion.yaml',
        '09_motion_cache.yaml',
        '10_deepcenter.yaml',
    }
    leftover = list((PROJECT_ROOT / 'configs').glob('*_smoke.yaml'))
    assert leftover == []


def test_numbered_train_scripts_exist() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub' / 'train'
    for name in (
        '01_p1.py',
        '01_p2.py',
        '02_model_c.py',
        '02_division.py',
        '02_division_decoder.py',
        '03_cardinality.py',
        '04_unigraft.py',
        '06_ownership.py',
        '07_edgegraft_ranker.py',
        '07_edgegraft_gate.py',
        '07_edgegraft_oracle.py',
        '08_candidategraft.py',
        '09_motion.py',
        '09_motion_cache.py',
        '10_deepcenter.py',
    ):
        assert (root / name).is_file()


def test_numbered_infer_scripts_exist() -> None:
    root = PROJECT_ROOT / 'src' / 'biohub' / 'infer'
    for name in (
        '01_detect.py',
        '04_unigraft.py',
        '05_live_v2.py',
        '06_ownership.py',
        '07_edgegraft.py',
        '08_candidategraft.py',
        '09_motion.py',
        '10_deepcenter.py',
        'run.py',
    ):
        assert (root / name).is_file()


def test_infer_notebook_uses_package_apis() -> None:
    payload = json.loads((PROJECT_ROOT / 'notebooks' / 'infer.ipynb').read_text())
    source = '\n'.join(''.join(cell.get('source', [])) for cell in payload['cells'])
    assert 'run_inference(' not in source
    assert 'python -m biohub.infer.run' not in source
    assert 'biohub_pipeline' not in source
    assert 'detect_job' in source
    assert 'predict_from_job' in source
    assert 'GraphUpgrade' in source
    assert 'process_graph' in source
    assert 'assemble_submission' in source
    assert "configs' / ('infer_kaggle.yaml' if ON_KAGGLE else 'infer.yaml')" in source
    assert 'upgrade.csv_columns' in source
    assert '/kaggle/input' in source
    assert '/kaggle/working' in source
    assert 'BIOHUB_ROOT' in source
    assert 'sys.path' in source
    assert 'submission.csv' in source
    assert 'DATASETS_DIR' in source
    assert 'ensure_dependencies' in source
    assert '--find-links' in source
    assert '--no-index' in source
    assert 'pilkwang/biohub-tracking-support-pack-50ep-v1' in source
    assert 'tweakai/biohub-edgegraft-v3-full-population-v1' in source
    assert "artifact(MODELS['model_c_dir'], 'deploy_spec.json')" in source
    assert 'Float16' in source
    assert 'weights.zip' in source
    assert "uv', 'pip', 'install'" not in source


def _argparse_dests(fn) -> set[str]:
    captured: list[set[str]] = []
    original = argparse.ArgumentParser.parse_args

    def fake(self, args=None, namespace=None):
        dests = {action.dest for action in self._actions if action.dest != 'help'}
        captured.append(dests)
        ns = argparse.Namespace()
        for action in self._actions:
            if action.dest != 'help':
                setattr(ns, action.dest, action.default)
        return ns

    argparse.ArgumentParser.parse_args = fake
    try:
        fn()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured[0]


def test_stage_yaml_covers_argparse_dests() -> None:
    from biohub.train import (
        candidategraft,
        cardinality,
        decoder,
        deepcenter,
        detector,
        division,
        edgegraft_gate,
        edgegraft_oracle,
        edgegraft_ranker,
        motion,
        motion_cache,
        ownership,
    )

    mapping = {
        '01_p1.yaml': detector.parse_args,
        '01_p2.yaml': detector.parse_args,
        '02_model_c.yaml': detector.parse_args,
        '02_division.yaml': division.argspec,
        '02_division_decoder.yaml': decoder.parse_args,
        '03_cardinality.yaml': cardinality.parse_args,
        '04_unigraft.yaml': cardinality.parse_args,
        '06_ownership.yaml': ownership.parse_args,
        '07_edgegraft_ranker.yaml': edgegraft_ranker.parse_args,
        '07_edgegraft_gate.yaml': edgegraft_gate.parse_args,
        '07_edgegraft_oracle.yaml': edgegraft_oracle.parse_args,
        '08_candidategraft.yaml': candidategraft.fit_parse_args,
        '08_candidategraft_screen.yaml': candidategraft.parse_args,
        '09_motion.yaml': motion.argspec,
        '09_motion_cache.yaml': motion_cache.parse_args,
        '10_deepcenter.yaml': deepcenter.parse_args,
    }
    for name, parse_fn in mapping.items():
        dests = _argparse_dests(parse_fn)
        for root in (PROJECT_ROOT / 'configs', PROJECT_ROOT / 'configs' / 'smoke'):
            path = root / name
            payload = yaml.safe_load(path.read_text())
            keys = set(payload)
            assert keys == dests, (
                f'{path.relative_to(PROJECT_ROOT)} extra={sorted(keys - dests)} '
                f'missing={sorted(dests - keys)}'
            )


def test_infer_yaml_has_every_tracking_field() -> None:
    from biohub.infer.config import (
        AssociationConfig,
        DetectionConfig,
        DivisionConfig,
        GraphConfig,
        IlpConfig,
        ModelBundleConfig,
        RuntimeConfig,
        SpeedConfig,
        TrackingConfig,
    )

    payload = yaml.safe_load((PROJECT_ROOT / 'configs/infer.yaml').read_text())
    sections = {
        'detection': DetectionConfig,
        'association': AssociationConfig,
        'ilp': IlpConfig,
        'graph': GraphConfig,
        'division': DivisionConfig,
        'models': ModelBundleConfig,
        'speed': SpeedConfig,
        'runtime': RuntimeConfig,
    }
    for key in ('experiment_tag', 'voxel_scale_um', *sections):
        assert key in payload
        assert key in TrackingConfig.model_fields
    for name, model in sections.items():
        assert set(payload[name]) == set(model.model_fields), name
    speed = payload['speed']
    assert 'vectorized_candidates' not in speed
    assert 'function_profile' not in speed
    assert 'runtime_accel' not in speed
    assert payload['runtime']['cpu_workers'] == 64
    assert payload['detection']['unet_batch_size'] == 32
    assert payload['detection']['det_tta'] is True
    assert payload['graph']['deepcenter_device'] == 'cuda'


def test_kaggle_infer_uses_smaller_resource_pools() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / 'configs/infer_kaggle.yaml').read_text())
    assert set(payload['runtime']) == set(RuntimeConfig.model_fields)
    assert payload['runtime']['cpu_workers'] == 2
    assert payload['runtime']['deepcenter_gpu_workers'] == 1
    assert payload['graph']['deepcenter_score_cache_max_frames'] == 8
    assert payload['graph']['frame_cache_max_frames'] == 8


def test_run_starts_detect_before_graph_upgrade() -> None:
    source = (PROJECT_ROOT / 'src/biohub/infer/run.py').read_text()
    start = source.index('\ndef run(')
    nxt = source.find('\ndef ', start + 1)
    body = source[start:] if nxt < 0 else source[start:nxt]
    detect_at = body.index('run_detect_jobs(')
    upgrade_at = body.index('run_upgrade(upgrade)')
    graph_at = body.index('upgrade = GraphUpgrade(tracking)')
    assert detect_at < graph_at < upgrade_at
