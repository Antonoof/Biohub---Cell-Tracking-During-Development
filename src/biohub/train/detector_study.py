"""Process-isolated detector Optuna worker. Run one worker per visible GPU."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import optuna
import yaml

from biohub.paths import PROJECT_ROOT
from biohub.train.detector import validate_config
from biohub.train.detector_search import (
    BASE_CONFIG,
    pooled_oof_score,
    sample_search_params,
    trial_config,
)


class TrialExecutionError(RuntimeError):
    """One trial failed; its log remains available and the study can continue."""


def run_command(command: list[str], log_path: Path, timeout: float | None = None) -> None:
    if os.name != 'posix':
        raise RuntimeError('Isolated detector worker currently requires Linux/macOS')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be positive or null')
    with log_path.open('x') as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise TrialExecutionError(f'Training exited with code {code}; see {log_path}')
        except subprocess.TimeoutExpired as exc:
            raise TrialExecutionError(f'Training timed out; see {log_path}') from exc
        finally:
            # Includes DataLoader grandchildren, also on timeout, Ctrl-C or child crash.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_isolated_training(cfg: dict, directory: Path, *, timeout: float | None = None) -> None:
    checked = validate_config(cfg)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'config.yaml'
    path.write_text(yaml.safe_dump(checked, sort_keys=False))
    run_command(
        [sys.executable, '-m', 'biohub.train.01_p1', '--config', str(path)],
        directory / 'train.log',
        timeout,
    )


def optimize_trials(study, objective, *, n_trials: int) -> None:
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    # No catch-all: bugs in the search driver still surface, training failures are FAIL.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(TrialExecutionError,))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=40, help='Trials for THIS worker')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--validation-folds', type=int, default=5, choices=range(1, 6))
    parser.add_argument('--timeout', type=float, default=None, help='Seconds per fold')
    parser.add_argument(
        '--sampler-seed',
        type=int,
        default=None,
        help='Use different seeds for simultaneous workers',
    )
    args = parser.parse_args(argv)
    validate_config({'device': args.device})
    if args.trials < 1:
        parser.error('--trials must be positive')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(args.journal))
    )
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, constant_liar=True),
    )
    # Different evaluation panels must never be silently combined in one study.
    code_hash = hashlib.sha256()
    for path in [BASE_CONFIG, *sorted((PROJECT_ROOT / 'src' / 'biohub').rglob('*.py'))]:
        code_hash.update(str(path.relative_to(PROJECT_ROOT)).encode())
        code_hash.update(path.read_bytes())
    contract = {
        'validation_folds': args.validation_folds,
        'batch_size': 16,
        'objective': 'pooled_window_proxy_v2',
        'output': str(args.output.resolve()),
        'source_sha256': code_hash.hexdigest(),
    }
    previous = study.user_attrs.get('detector_contract')
    if previous is not None and previous != contract:
        raise ValueError('Study contract differs; use another study name/output')
    study.set_user_attr('detector_contract', contract)

    def objective(trial):
        params = sample_search_params(trial)
        directory = args.output / f'trial_{trial.number:05d}'
        directory.mkdir(exist_ok=False)
        trial.set_user_attr('directory', str(directory.resolve()))
        try:
            for fold in range(args.validation_folds):
                cfg = trial_config(params, fold=fold, weights_dir=directory)
                cfg['device'] = args.device
                run_isolated_training(cfg, directory / f'launcher_{fold}', timeout=args.timeout)
                partial_score, _ = pooled_oof_score(directory, n_folds=fold + 1)
                trial.report(partial_score, step=fold)
                if fold + 1 < args.validation_folds and trial.should_prune():
                    raise optuna.TrialPruned(f'Pruned after {fold + 1} validation folds')
            score, metrics = pooled_oof_score(directory, n_folds=args.validation_folds)
            if not math.isfinite(score):
                raise TrialExecutionError('Non-finite objective')
            (directory / 'objective.json').write_text(json.dumps(metrics, indent=2) + '\n')
            return score
        except TrialExecutionError as exc:
            trial.set_user_attr('failure', str(exc))
            raise

    optimize_trials(study, objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
