from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

EMBRYOS = ('44b6', '6bba')
MOVIE_GROUP_FOLDS = 5


def movie_group_kfold(
    groups: Any,
    n_splits: int = MOVIE_GROUP_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros(len(groups))
    splits = min(int(n_splits), len(set(map(str, groups))))
    if splits < 2:
        index = np.arange(len(groups))
        return [(index, index[:0])]
    return list(GroupKFold(n_splits=splits).split(dummy, groups=groups))


def payload_movie_names(payload: dict | list) -> list[str]:
    names: list[str] = []

    def _extend(values: Any) -> None:
        if isinstance(values, list):
            names.extend(str(item) for item in values)

    if isinstance(payload, list):
        for fold_data in payload:
            if isinstance(fold_data, dict):
                _extend(fold_data.get('train'))
                _extend(fold_data.get('test', fold_data.get('val', fold_data.get('held', []))))
        return list(dict.fromkeys(names))
    for key in ('train', 'test', 'val', 'held', 'practice'):
        _extend(payload.get(key))
    for value in payload.values():
        if isinstance(value, dict) and 'train' in value:
            _extend(value.get('train'))
            _extend(value.get('test', value.get('val', value.get('held', []))))
    return list(dict.fromkeys(names))


def movie_group_fold_names(
    movies: Sequence[str],
    fold: int,
    n_splits: int = MOVIE_GROUP_FOLDS,
) -> tuple[list[str], list[str]]:
    unique = list(dict.fromkeys(str(name) for name in movies))
    if len(unique) < 2:
        return unique, []
    splits = movie_group_kfold(np.asarray(unique), n_splits)
    index = int(fold) % len(splits)
    fit, held = splits[index]
    return [unique[int(i)] for i in fit], [unique[int(i)] for i in held]


def embryo_two_fold(
    embryos: Sequence[str] = EMBRYOS,
) -> list[tuple[str, str]]:
    values = tuple(embryos)
    if len(values) < 2:
        raise RuntimeError(f'Need two embryos for cross-validation; got {values}')
    return [(values[0], values[1]), (values[1], values[0])]
