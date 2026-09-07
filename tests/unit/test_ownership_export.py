from pathlib import Path

import numpy as np
import pandas as pd

from biohub.modules.ownership import FEATURES, FullPopulationOwnershipRuntime
from biohub.train.ownership import train_from_config


def _bank(path: Path) -> Path:
    rows = []
    rng = np.random.default_rng(0)
    for video_index in range(5):
        dataset = f'vid_{video_index}'
        for source in range(6):
            row = {name: float(rng.normal()) for name in FEATURES}
            row.update(
                {
                    'dataset': dataset,
                    'panel': 'train175',
                    'source': source,
                    'a': source,
                    'b': source + 1,
                    'current_child': source + 2,
                    'source_label': 1 if source % 2 == 0 else 0,
                    'best_pair_label': 1 if source % 3 == 0 else 0,
                    'source_score': float(rng.random()),
                    'v2_source_score_raw': float(rng.random()),
                    'source_time': source,
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return path


def test_ownership_trainer_exports_runtime_artifact(tmp_path: Path) -> None:
    bank = _bank(tmp_path / 'bank.parquet')
    output = tmp_path / 'ownership'
    train_from_config(
        {
            'bank': str(bank),
            'output': str(output),
            'seed': 7,
            'folds': 5,
            'n_estimators': 8,
            'max_depth': 2,
            'min_samples_leaf': 1,
            'max_features': 0.75,
            'n_jobs': 1,
        }
    )
    spec = (output / 'deploy_spec.json').read_text()
    assert 'full-population-ownership-v1' in spec
    assert (output / 'models' / 'ownership_fold_0.joblib').is_file()
    runtime = FullPopulationOwnershipRuntime(None, output)
    assert len(runtime.models) == 5
    assert list(runtime.models[0].classes_) is not None
    scores = runtime.models[0].predict_proba(np.zeros((2, len(FEATURES)), np.float32))[:, 1]
    assert scores.shape == (2,)
