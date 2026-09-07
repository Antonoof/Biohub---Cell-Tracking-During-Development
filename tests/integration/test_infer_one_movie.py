import json
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
def test_infer_one_movie_matches_package_golden(tmp_path: Path) -> None:
    movie = TRAIN_DIR / f'{MOVIE}.zarr'
    if not GOLDEN.is_file():
        pytest.skip('package golden CSV is missing')
    if not movie.exists() or not BUNDLE.exists():
        pytest.skip('train movie or frozen weights are missing')
    if not torch.cuda.is_available():
        pytest.skip('GPU is missing')
    raw = yaml.safe_load((PROJECT_ROOT / 'configs' / 'infer.yaml').read_text())
    raw['graph']['deepcenter_device'] = 'cpu'
    raw['runtime']['cpu_workers'] = 1
    raw['speed']['thread_pool_total'] = 8
    config_path = tmp_path / 'infer.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    runs_root = tmp_path / 'runs'
    payload_path = tmp_path / 'payload.json'
    script = tmp_path / 'run_one.py'
    script.write_text(
        'import json\n'
        'from pathlib import Path\n'
        'from biohub.infer.run import run_inference\n'
        'payload = run_inference(\n'
        f'    config=Path({str(config_path)!r}),\n'
        f'    movie={MOVIE!r},\n'
        f'    movies_dir=Path({str(TRAIN_DIR)!r}),\n'
        '    gpu_workers=1,\n'
        "    run_id='package_golden',\n"
        f'    runs_root=Path({str(runs_root)!r}),\n'
        ')\n'
        f'Path({str(payload_path)!r}).write_text(json.dumps(payload) + chr(10))\n'
    )
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
    subprocess.check_call([sys.executable, str(script)], cwd=PROJECT_ROOT, env=env)
    payload = json.loads(payload_path.read_text())
    produced = Path(payload['submission'])
    assert produced.read_bytes() == GOLDEN.read_bytes()
