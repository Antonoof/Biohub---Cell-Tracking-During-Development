#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

METADATA = {
    'dataset',
    'partition',
    'fold',
    'video_fold',
    'component',
    'group',
    'source',
    'target',
    'y',
}

RELATIVE_MAX = [
    'p1_probability',
    'p2_probability',
    'model_probability_max',
    'model_probability_mean',
    'raw_edge_probability',
    'ctx_p1_path_min',
    'ctx_p1_path_mean',
    'ctx_p1_path_geomean',
    'ctx_p2_path_min',
    'ctx_p2_path_mean',
    'ctx_p2_path_geomean',
]
RELATIVE_MIN = [
    'distance_um',
    'velocity_residual_um',
    'geom_distance_um_exact',
    'geom_incoming_residual_um',
    'geom_outgoing_residual_um',
]


def add_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby(['dataset', 'target'], sort=False)
    for column in RELATIVE_MAX:
        if column not in result:
            continue
        maximum = groups[column].transform('max')
        result[f'rankrel_{column}_to_max'] = result[column] - maximum
        result[f'rankrel_{column}_rank'] = groups[column].rank(method='average', ascending=False)
    for column in RELATIVE_MIN:
        if column not in result:
            continue
        minimum = groups[column].transform('min')
        result[f'rankrel_{column}_from_min'] = result[column] - minimum
        result[f'rankrel_{column}_rank'] = groups[column].rank(method='average', ascending=True)
    return result


def load_labels(root: Path) -> pd.DataFrame:
    frames = []
    paths = sorted(root.glob('*.parquet'))
    if not paths:
        raise RuntimeError(f'No label files found: {root}')
    for index, path in enumerate(paths, 1):
        frame = pd.read_parquet(path)
        if len(frame):
            frames.append(frame)
        if index % 25 == 0 or index == len(paths):
            print(f'loaded {index}/{len(paths)}', flush=True)
    result = pd.concat(frames, ignore_index=True)
    result = add_relative_features(result)
    return result


def feature_names(frame: pd.DataFrame) -> list[str]:
    output = []
    for column in frame.columns:
        if column in METADATA:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            output.append(column)
    return output


def target_weights(frame: pd.DataFrame) -> np.ndarray:
    negative_count = frame.groupby(['dataset', 'target'], sort=False).y.transform(
        lambda x: max(int((x == 0).sum()), 1)
    )
    return np.where(frame.y.to_numpy() == 1, 0.5, 0.5 / negative_count.to_numpy()).astype(
        np.float32
    )


def target_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, int | float]:
    work = frame[['dataset', 'target', 'source', 'y', 'is_current_parent']].copy()
    work['score'] = score
    chosen = work.loc[work.groupby(['dataset', 'target'], sort=False).score.idxmax()].copy()
    targets = len(chosen)
    model_correct = int(chosen.y.sum())
    base_correct = int(
        work.loc[(work.y == 1) & (work.is_current_parent == 1), ['dataset', 'target']]
        .drop_duplicates()
        .shape[0]
    )
    chosen['base_correct'] = chosen.set_index(['dataset', 'target']).index.map(
        work.loc[(work.y == 1) & (work.is_current_parent == 1)]
        .set_index(['dataset', 'target'])
        .index.unique()
        .__contains__
    )
    recovered = int(((chosen.y == 1) & ~chosen.base_correct).sum())
    lost = int(((chosen.y == 0) & chosen.base_correct).sum())
    return {
        'targets': targets,
        'baseline_correct': base_correct,
        'baseline_accuracy': base_correct / max(targets, 1),
        'model_correct': model_correct,
        'model_accuracy': model_correct / max(targets, 1),
        'net_correct': model_correct - base_correct,
        'recovered': recovered,
        'lost': lost,
    }


FEATURES = [
    'top_score',
    'runner_up_score',
    'top_margin',
    'current_present',
    'current_score_filled',
    'advantage_filled',
    'remove_count',
    'topology_2to1',
    'embryo_6bba',
]
