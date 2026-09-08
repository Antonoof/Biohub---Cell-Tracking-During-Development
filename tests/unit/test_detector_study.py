import sys

import optuna
import pytest

from biohub.train.detector_study import TrialExecutionError, optimize_trials, run_command


def test_bad_child_trial_does_not_abort_study(tmp_path):
    study = optuna.create_study(direction='maximize')

    def objective(trial):
        command = 'raise SystemExit(7)' if trial.number == 0 else 'print("training complete")'
        run_command([sys.executable, '-c', command], tmp_path / f'{trial.number}.log', timeout=5)
        return 0.75

    optimize_trials(study, objective, n_trials=2)
    assert [t.state for t in study.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert study.best_value == 0.75


def test_child_timeout_is_trial_failure_and_next_child_runs(tmp_path):
    with pytest.raises(TrialExecutionError, match='timed out'):
        run_command(
            [sys.executable, '-c', 'import time; time.sleep(10)'],
            tmp_path / 'timeout.log',
            timeout=0.1,
        )
    run_command([sys.executable, '-c', 'print("ok")'], tmp_path / 'next.log', timeout=5)
    assert (tmp_path / 'next.log').read_text().strip() == 'ok'


def test_search_driver_bug_is_not_silently_swallowed():
    def objective(trial):
        raise KeyError('driver bug')

    with pytest.raises(KeyError, match='driver bug'):
        optimize_trials(optuna.create_study(), objective, n_trials=2)
