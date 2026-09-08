import json
from pathlib import Path

from biohub.train.detector_search import (
    SEARCH_PARAM_NAMES,
    apply_search_params,
    load_base_config,
    params_from_config,
    pooled_oof_score,
    sample_search_params,
    trial_config,
)


class _FixedTrial:
    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        return float(low)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(low)


def test_sample_params_cover_search_names() -> None:
    params = sample_search_params(_FixedTrial())
    assert set(params) == set(SEARCH_PARAM_NAMES)


def test_p1_yaml_enqueues_into_search_space() -> None:
    params = params_from_config(load_base_config())
    assert set(params) == set(SEARCH_PARAM_NAMES)
    assert params['optimizer'] == 'adamw'
    assert params['n_heads'] in {4, 8}
    assert params['hidden_dim'] % params['n_heads'] == 0
    assert params['noise_aug_proba'] == 0.0
    assert params['rot90_aug'] is False


def test_apply_search_params_derives_conditionals() -> None:
    params = sample_search_params(_FixedTrial())
    params['scheduler'] = 'none'
    params['use_ema'] = False
    params['use_layer_scale'] = False
    params['match_assign'] = 'greedy'
    params['match_soft'] = True
    params['use_peak_topk'] = False
    params['use_edge_gate'] = False
    params['drop_path'] = 0.1
    overlay = apply_search_params(params)
    assert overlay['warmup_epochs'] == 0
    assert overlay['min_lr'] == 0.0
    assert overlay['ema_decay'] == 0.0
    assert overlay['layer_scale_init'] == 0.0
    assert overlay['match_soft'] is False
    assert overlay['train_peak_topk'] == 0
    assert overlay['edge_gate_distance'] == 0.0
    assert overlay['drop_path_decay'] is True
    assert overlay['batch_size'] == 32
    assert overlay['amp'] == 'bf16'
    assert overlay['seed'] == 42
    assert overlay['unet_layers'] == [32, 64, 128]


def test_sinkhorn_keeps_soft_match() -> None:
    params = sample_search_params(_FixedTrial())
    params['match_assign'] = 'sinkhorn'
    params['match_soft'] = True
    overlay = apply_search_params(params)
    assert overlay['match_soft'] is True


def test_trial_config_sets_fold_and_weights(tmp_path: Path) -> None:
    params = sample_search_params(_FixedTrial())
    cfg = trial_config(params, fold=3, weights_dir=tmp_path)
    assert cfg['split'] == '3'
    assert cfg['weights_dir'] == str(tmp_path)
    assert cfg['data_parallel'] is False
    assert cfg['hidden_dim'] % cfg['n_heads'] == 0


def test_pooled_oof_sums_fold_counts(tmp_path: Path) -> None:
    for fold in range(5):
        path = tmp_path / 'unet_transformer' / f'split_{fold}' / 'metrics.json'
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    'edge_tp': 10,
                    'edge_fp': 1,
                    'edge_fn': 1,
                    'division_tp': 2,
                    'division_fp': 0,
                    'division_fn': 0,
                    'num_pred_nodes': 20,
                    'gt_total': 20,
                }
            )
        )
    score, bundled = pooled_oof_score(tmp_path)
    assert bundled['edge_tp'] == 50.0
    assert bundled['gt_total'] == 100.0
    assert score > 0.0


def test_optuna_enqueues_p1_params() -> None:
    import optuna
    from optuna.samplers import TPESampler

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))
    study.enqueue_trial(params_from_config(load_base_config()))

    def objective(trial: optuna.Trial) -> float:
        params = sample_search_params(trial)
        overlay = apply_search_params(params)
        assert overlay['hidden_dim'] % overlay['n_heads'] == 0
        return 0.0

    study.optimize(objective, n_trials=1)
    assert study.trials[0].params['optimizer'] == 'adamw'
