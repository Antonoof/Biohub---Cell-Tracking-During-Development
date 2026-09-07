import numpy as np
from sklearn.model_selection import GroupKFold

from biohub.validation.cv import EMBRYOS, embryo_two_fold, movie_group_kfold
from biohub.validation.splits import detector_validation_role


def test_movie_group_kfold_is_five_fold_by_movie() -> None:
    groups = np.asarray([f'movie_{index % 10}' for index in range(50)])
    splits = movie_group_kfold(groups, n_splits=5)
    assert len(splits) == 5
    expected = list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))
    for (fit, held), (exp_fit, exp_held) in zip(splits, expected):
        np.testing.assert_array_equal(fit, exp_fit)
        np.testing.assert_array_equal(held, exp_held)
        assert set(groups[held]).isdisjoint(set(groups[fit]))


def test_embryo_two_fold_swaps_44b6_and_6bba() -> None:
    assert embryo_two_fold() == [
        (EMBRYOS[0], EMBRYOS[1]),
        (EMBRYOS[1], EMBRYOS[0]),
    ]
    assert embryo_two_fold() == [('44b6', '6bba'), ('6bba', '44b6')]


def test_detector_validation_role_flags_train_test_overlap() -> None:
    assert detector_validation_role(['a', 'b'], ['b']) == 'in_sample_production_fit'
    assert detector_validation_role(['a', 'b'], ['c']) == 'held_split'
