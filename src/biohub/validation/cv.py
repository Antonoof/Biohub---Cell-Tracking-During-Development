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
    return list(GroupKFold(n_splits=n_splits).split(dummy, groups=groups))


def embryo_two_fold(
    embryos: Sequence[str] = EMBRYOS,
) -> list[tuple[str, str]]:
    values = tuple(embryos)
    if len(values) < 2:
        raise RuntimeError(f'Need two embryos for cross-validation; got {values}')
    return [(values[0], values[1]), (values[1], values[0])]
