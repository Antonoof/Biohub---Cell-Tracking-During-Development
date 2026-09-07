from pathlib import Path

import yaml

from biohub.paths import PROJECT_ROOT
from tests.parity.notebook import notebook_cell_source

PATH_KEYS = {'root', 'dir', 'path', 'paths', 'gt_dir', 'report_path'}
THROUGHPUT_KEYS = {
    ('detection', 'unet_batch_size'),
    ('graph', 'deepcenter_device'),
    ('graph', 'deepcenter_score_cache_max_frames'),
    ('graph', 'frame_cache_max_frames'),
}


def _without_paths(value):
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items() if key not in PATH_KEYS}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _exec_notebook_cfg(tmp_path: Path):
    source = notebook_cell_source(1)
    work = tmp_path / 'work'
    work.mkdir()
    source = source.replace('Path("/kaggle/input")', 'Path("/tmp/input")')
    source = source.replace('Path("/kaggle/working")', f'Path({str(work)!r})')
    namespace: dict = {}
    exec(source, namespace)
    return namespace['CFG']


def test_tracking_yaml_matches_notebook_cfg_numbers(tmp_path: Path) -> None:
    notebook_cfg = _without_paths(_exec_notebook_cfg(tmp_path))
    ours = yaml.safe_load((PROJECT_ROOT / 'configs/infer.yaml').read_text())
    ours = _without_paths(ours)
    assert ours['voxel_scale_um'] == notebook_cfg['voxel_scale_um']
    for section in ('detection', 'association', 'ilp', 'graph', 'division'):
        for key, value in notebook_cfg[section].items():
            if (section, key) in THROUGHPUT_KEYS:
                continue
            assert ours[section][key] == value, f'{section}.{key}'
    for key, value in notebook_cfg['models'].items():
        if isinstance(value, dict) and 'enabled' in value:
            assert ours['models'][f'{key}_enabled'] == value['enabled'], key
    assert ours['runtime']['gpu_workers'] == notebook_cfg['runtime']['gpu_workers']
    assert ours['runtime']['hard_limit_seconds'] == notebook_cfg['runtime']['hard_limit_seconds']
    assert ours['runtime']['emergency_shard'] is True
    assert ours['speed']['longest_first'] is True
    assert ours['speed']['uncompressed_evidence'] is True
