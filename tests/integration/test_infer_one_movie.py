import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from biohub.paths import PROJECT_ROOT

MOVIE = '44b6_0113de3b'
GOLDEN = PROJECT_ROOT / 'tests' / 'fixtures' / '44b6_0113de3b_package_submission.csv'
TRAIN_DIR = (
    PROJECT_ROOT
    / 'kaggle'
    / 'input'
    / 'competitions'
    / 'biohub-cell-tracking-during-development'
    / 'train'
)
BUNDLE = PROJECT_ROOT / 'kaggle' / 'input' / 'datasets' / 'antonoof' / 'all_files'


@pytest.mark.slow
@pytest.mark.skipif(not GOLDEN.is_file(), reason='package golden CSV is missing')
@pytest.mark.skipif(
    not (TRAIN_DIR / f'{MOVIE}.zarr').exists() or not BUNDLE.exists(),
    reason='train movie or frozen weights are missing',
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU is missing')
def test_infer_one_movie_matches_package_golden(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / 'configs' / 'infer.yaml').read_text())
    raw['graph']['deepcenter_device'] = 'cpu'
    raw['runtime']['cpu_workers'] = 1
    raw['speed']['thread_pool_total'] = 8
    config_path = tmp_path / 'infer.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    runs_root = tmp_path / 'runs'
    env = os.environ.copy()
    src = str(PROJECT_ROOT / 'src')
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = src if not pythonpath else src + os.pathsep + pythonpath
    env['PYTHONUNBUFFERED'] = '1'
    for key in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        env[key] = '8'
    subprocess.check_call(
        [
            sys.executable,
            '-m',
            'biohub.infer.run',
            '--config',
            str(config_path),
            '--movie',
            MOVIE,
            '--movies-dir',
            str(TRAIN_DIR),
            '--gpu-workers',
            '1',
            '--run-id',
            'package_golden',
            '--runs-root',
            str(runs_root),
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )
    produced = runs_root / 'package_golden' / 'workdir' / 'submission.csv'
    assert produced.read_bytes() == GOLDEN.read_bytes()
