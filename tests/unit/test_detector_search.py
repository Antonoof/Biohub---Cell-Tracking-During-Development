import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from biohub.data.windows import FrameWindowData, pad_window
from biohub.features.position import POS_EMBED_DIM
from biohub.losses.detection import gaussian_heatmap_target
from biohub.models.detector import UNetNodeTransformer
from biohub.models.temporal_unet import TemporalUNet3D, unet_in_channels
from biohub.train.detector import (
    SCORE_THRESHOLDS,
    _augmentations_from_cfg,
    _recipe_kwargs,
    evaluate,
    score_threshold_key,
    select_best_threshold_metrics,
    train_det_threshold,
    train_epoch,
)
from biohub.train.detector_search import (
    MSNT_TRAIN_DET_PROB,
    SEARCH_PARAM_NAMES,
    apply_search_params,
    load_base_config,
    params_from_config,
    pooled_oof_score,
    sample_search_params,
    seed_trial_params,
    trial_config,
)
from biohub.train.schedule import build_optimizer


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
    assert overlay['batch_size'] == 16
    assert overlay['frame_cache_mb'] == 256.0
    assert overlay['batch_padding'] is True
    assert overlay['pair_chunk_size'] == 512
    assert overlay['checkpoint_metric'] == 'acc_times_recall'
    assert overlay['det_threshold'] == pytest.approx(MSNT_TRAIN_DET_PROB)
    assert overlay['edge_threshold'] == 0.5
    assert 0.5 in SCORE_THRESHOLDS
    assert overlay['amp'] == 'bf16'
    assert overlay['seed'] == 42
    assert overlay['unet_layers'] == [32, 64, 128]
    assert overlay['poisson_aug'] is False
    assert overlay['poisson_aug_proba'] == 0.0
    assert 'poisson_scale' not in overlay
    assert 'use_ema' not in overlay
    assert 'use_peak_topk' not in overlay
    assert train_det_threshold(0.15) == 0.15
    assert train_det_threshold(0.5) == 0.5
    assert train_det_threshold(0.97) == 0.97


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
    assert set(cfg) <= set(load_base_config())


def test_divisible_heads_coerces_incompatible_counts() -> None:
    from biohub.models.attention import divisible_heads

    assert divisible_heads(128, 8) == 8
    assert divisible_heads(192, 8) == 8
    assert divisible_heads(128, 3) == 2
    assert divisible_heads(32, 4) == 4


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
                    'gt_matched': 8,
                    'gt_total': 20,
                    'pair_correct': 10,
                    'pair_total': 20,
                }
            )
        )
    score, bundled = pooled_oof_score(tmp_path)
    assert bundled['edge_tp'] == 50.0
    assert bundled['gt_total'] == 100.0
    assert bundled['gt_matched'] == 40.0
    assert bundled['pair_correct'] == 50.0
    assert bundled['pair_total'] == 100.0
    assert bundled['acc_times_recall'] == 0.2
    assert bundled['acc_times_recall'] == score
    assert bundled['competition_metric'] > 0.0
    assert bundled['edge_jaccard'] == pytest.approx(50.0 / 60.0)
    assert score == 0.2


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


def test_seed_trial_params_matches_yaml() -> None:
    yaml_params = params_from_config(load_base_config())
    seed = seed_trial_params(load_base_config())
    assert yaml_params['det_loss'] == 'weighted_bce'
    assert seed['det_loss'] == 'gaussian_heatmap'
    assert seed['det_heatmap_sigma'] == pytest.approx(1.0)
    assert seed['det_neg_weight'] == pytest.approx(0.01)
    assert seed['det_loss_weight'] == pytest.approx(1.0)
    assert set(seed) == set(SEARCH_PARAM_NAMES)


def test_optuna_enqueues_yaml_seed() -> None:
    import optuna
    from optuna.samplers import TPESampler

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))
    study.enqueue_trial(seed_trial_params(load_base_config()))

    def objective(trial: optuna.Trial) -> float:
        params = sample_search_params(trial)
        assert params['det_loss'] == 'gaussian_heatmap'
        assert params['det_heatmap_sigma'] == pytest.approx(1.0)
        assert params['det_neg_weight'] == pytest.approx(0.01)
        assert params['det_loss_weight'] == pytest.approx(1.0)
        return 0.0

    study.optimize(objective, n_trials=1)


def _tiny_detector(**knobs) -> UNetNodeTransformer:
    in_channels = unet_in_channels(
        coord_kind=str(knobs.get('coord_kind', 'none')),
        fourier_bands=int(knobs.get('fourier_bands', 4)),
        flow_input=str(knobs.get('flow_input', 'none')),
        extra_encoder=str(knobs.get('extra_encoder', 'none')),
        extra_encoder_channels=int(knobs.get('extra_encoder_channels', 8)),
    )
    unet = TemporalUNet3D(
        in_channels,
        int(knobs.get('unet_out_channels', 8)),
        layers=tuple(knobs.get('layers', (8, 16))),
        skip_fullres_temporal=True,
        temporal_n_heads=4,
        se_ratio=float(knobs.get('se_ratio', 0.0)),
        unet_block=str(knobs.get('unet_block', 'plain')),
        unet_norm=str(knobs.get('unet_norm', 'batchnorm')),
        unet_gn_groups=8,
        unet_deform=bool(knobs.get('unet_deform', False)),
        temporal_mix=str(knobs.get('temporal_mix', 'attn')),
    )
    out_ch = int(knobs.get('unet_out_channels', 8))
    return UNetNodeTransformer(
        unet,
        out_ch,
        4 * POS_EMBED_DIM,
        hidden_dim=int(knobs.get('hidden_dim', 32)),
        n_heads=int(knobs.get('n_heads', 4)),
        n_blocks=int(knobs.get('n_blocks', 1)),
        dropout=0,
        use_self_attn=bool(knobs.get('use_self_attn', False)),
        norm=str(knobs.get('norm', 'layernorm')),
        ffn_act=str(knobs.get('ffn_act', 'gelu')),
        pair_head=str(knobs.get('pair_head', 'mlp')),
        pair_geom=str(knobs.get('pair_geom', 'rel')),
        feature_sample=str(knobs.get('feature_sample', 'nearest')),
        coord_kind=str(knobs.get('coord_kind', 'none')),
        fourier_bands=int(knobs.get('fourier_bands', 4)),
        flow_input=str(knobs.get('flow_input', 'none')),
        extra_encoder=str(knobs.get('extra_encoder', 'none')),
        extra_encoder_channels=int(knobs.get('extra_encoder_channels', 8)),
        extra_encoder_freeze=False,
        layer_scale_init=float(knobs.get('layer_scale_init', 0.0)),
        drop_path=float(knobs.get('drop_path', 0.0)),
        drop_path_decay=bool(knobs.get('drop_path_decay', False)),
    )


@pytest.mark.parametrize(
    'knobs',
    [
        {'temporal_mix': 'attn', 'unet_block': 'plain', 'unet_norm': 'batchnorm'},
        {
            'temporal_mix': 'both',
            'unet_block': 'convnext',
            'unet_norm': 'groupnorm',
            'unet_deform': True,
            'se_ratio': 0.25,
            'coord_kind': 'fourier',
            'fourier_bands': 4,
            'flow_input': 'frame_diff_grad',
            'extra_encoder': 'conv',
            'extra_encoder_channels': 4,
            'feature_sample': 'trilinear',
            'use_self_attn': True,
            'norm': 'rmsnorm',
            'ffn_act': 'swiglu',
            'pair_head': 'bilinear',
            'pair_geom': 'rel_dist',
            'layer_scale_init': 1e-5,
            'drop_path': 0.1,
            'drop_path_decay': True,
        },
        {
            'temporal_mix': 'conv',
            'unet_block': 'residual',
            'coord_kind': 'coord',
            'flow_input': 'spatial_grad',
            'feature_sample': 'nearest',
            'ffn_act': 'silu',
            'pair_head': 'mlp',
            'pair_geom': 'dist',
            'use_self_attn': True,
        },
    ],
)
def test_search_architecture_combos_backward(knobs: dict[str, object]) -> None:
    torch.manual_seed(0)
    model = _tiny_detector(**knobs)
    imgs = torch.rand(2, 2, 4, 8, 8)
    unet_out, det_logits = model.encode(imgs)
    coords = torch.rand(2, 3, 3) * 3
    mask = torch.ones(2, 3, dtype=torch.bool)
    pos = torch.randn(2, 3, 4 * POS_EMBED_DIM)
    feat = model.index_features(unet_out[:, 0], coords, mask)
    edges = model.predict_edges(feat, feat, coords, coords, pos, pos, mask, mask)
    loss = sum(logit.float().pow(2).mean() for logit in det_logits) + edges.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).all()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_encode_stacked_matches_encode_list() -> None:
    torch.manual_seed(1)
    model = _tiny_detector()
    model.eval()
    imgs = torch.rand(2, 3, 4, 8, 8)
    with torch.no_grad():
        unet_out, det_list = model.encode(imgs)
        stacked_out, stacked = model.encode_stacked(imgs)
    torch.testing.assert_close(unet_out, stacked_out)
    torch.testing.assert_close(stacked, torch.stack(det_list, dim=1))


class _Once(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def _combo_sample(n_frames: int = 2, spatial: tuple[int, int, int] = (4, 8, 8)) -> dict:
    window = FrameWindowData(
        0,
        n_frames,
        [torch.zeros(2, 32)] * n_frames,
        [torch.ones(2, 3)] * n_frames,
        [2] * n_frames,
        [torch.eye(2)] * max(n_frames - 1, 0),
    )
    return {
        **pad_window(window, 5),
        'imgs': torch.rand(n_frames, *spatial),
        'image_shape': torch.tensor((n_frames, *spatial)),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }


def test_evaluate_logs_competition_thresholds() -> None:
    torch.manual_seed(0)
    metrics = evaluate(
        _tiny_detector(),
        DataLoader(_Once(_combo_sample()), batch_size=1),
        torch.device('cpu'),
        det_threshold=0.97,
        edge_threshold=0.5,
        threshold_metric='acc_times_recall',
    )
    scores = [metrics[score_threshold_key(threshold)] for threshold in SCORE_THRESHOLDS]
    assert metrics['acc_times_recall'] == max(scores)
    assert metrics['acc_times_recall'] == metrics[score_threshold_key(metrics['score_threshold'])]
    assert 'edge_jaccard' in metrics
    assert metrics['det_threshold'] == 0.97
    assert metrics['edge_threshold'] == 0.5


def test_select_best_threshold_metrics_uses_peak_then_higher_threshold() -> None:
    def item(score: float) -> dict[str, float]:
        return {
            'acc_times_recall': score,
            'competition_metric': score,
            'acc': score,
            'num_pred_nodes': 1.0,
            'recall': 0.8,
            'node_ratio': 1.0,
        }

    metrics = select_best_threshold_metrics(
        [item(0.0), item(0.2), item(1.1), item(1.1), item(0.5)],
        (0.1, 0.5, 0.6, 0.7, 0.97),
        metric='acc_times_recall',
    )
    assert metrics['acc_times_recall'] == 1.1
    assert metrics['score_threshold'] == 0.7
    assert metrics[score_threshold_key(0.5)] == 0.2
    assert metrics[score_threshold_key(0.6)] == 1.1
    assert metrics[score_threshold_key(0.7)] == 1.1
    assert metrics['acc'] == 1.1


def test_select_best_threshold_rejects_collapsed_recall() -> None:
    rows = [
        {
            'competition_metric': 1.1,
            'acc_times_recall': 0.01,
            'recall': 0.01,
            'node_ratio': 0.01,
            'acc': 1.0,
            'num_pred_nodes': 1.0,
        },
        {
            'competition_metric': 0.4,
            'acc_times_recall': 0.18,
            'recall': 0.2,
            'node_ratio': 1.1,
            'acc': 0.9,
            'num_pred_nodes': 10.0,
        },
    ]
    metrics = select_best_threshold_metrics(rows, (0.5, 0.9), metric='acc_times_recall')
    assert metrics['acc_times_recall'] == 0.18
    assert metrics['score_threshold'] == 0.9
    assert metrics['recall'] == 0.2


@pytest.mark.parametrize(
    'knobs',
    [
        {'temporal_mix': 'none'},
        {'hidden_dim': 192, 'n_heads': 8, 'n_blocks': 2},
        {'hidden_dim': 256, 'n_heads': 8},
        {'flow_input': 'frame_diff'},
        {'flow_input': 'frame_diff_grad', 'coord_kind': 'fourier'},
        {'pair_geom': 'rel', 'norm': 'layernorm', 'ffn_act': 'gelu'},
    ],
)
def test_search_remaining_architecture_values_backward(knobs: dict[str, object]) -> None:
    torch.manual_seed(2)
    model = _tiny_detector(**knobs)
    imgs = torch.rand(2, 2, 4, 8, 8)
    unet_out, det_logits = model.encode(imgs)
    coords = torch.rand(2, 3, 3) * 3
    mask = torch.ones(2, 3, dtype=torch.bool)
    pos = torch.randn(2, 3, 4 * POS_EMBED_DIM)
    feat = model.index_features(unet_out[:, 0], coords, mask)
    loss = model.predict_edges(feat, feat, coords, coords, pos, pos, mask, mask).float().mean()
    loss = loss + det_logits[0].float().mean()
    loss.backward()
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    'loop',
    [
        {'det_loss_kind': 'weighted_bce', 'match_assign': 'greedy'},
        {'det_loss_kind': 'focal', 'match_assign': 'hungarian'},
        {'det_loss_kind': 'gaussian_heatmap', 'match_assign': 'sinkhorn', 'match_soft': True},
        {'det_loss_kind': 'pu_bce', 'match_assign': 'greedy'},
        {'det_loss_kind': 'pu_heatmap', 'match_assign': 'greedy'},
        {'edge_loss': 'ce_softmax', 'target_mode': 'gt_nodes'},
        {'edge_loss': 'asl_softmax', 'target_mode': 'mixed', 'target_gt_frac': 0.5},
        {'train_peak_topk': 4, 'edge_gate_distance': 12.0},
        {
            'aux_division_weight': 0.1,
            'aux_contrastive_weight': 0.1,
            'aux_offset_weight': 0.1,
            'offset_target': 'parabolic',
        },
        {'use_self_attn': True, 'pair_head': 'bilinear', 'feature_sample': 'trilinear'},
    ],
)
def test_search_train_loop_knobs_one_step(loop: dict[str, Any]) -> None:
    torch.manual_seed(3)
    params = dict(loop)
    model = _tiny_detector(
        use_self_attn=bool(params.pop('use_self_attn', False)),
        pair_head=str(params.pop('pair_head', 'mlp')),
        feature_sample=str(params.pop('feature_sample', 'nearest')),
    )
    sample = _combo_sample()
    if params.get('det_loss_kind') == 'gaussian_heatmap':
        sample['heatmap_target'] = gaussian_heatmap_target(
            sample['coords'], sample['masks'], (4, 8, 8)
        )
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(_Once(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        **params,
    )
    assert all(np.isfinite(losses))


class _OverrideTrial(_FixedTrial):
    def __init__(self, overrides: dict[str, object]) -> None:
        self.overrides = overrides

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        if name in self.overrides:
            value = self.overrides[name]
            assert value in choices, name
            return value
        return choices[0]


class _CaptureTrial(_FixedTrial):
    def __init__(self) -> None:
        self.categoricals: dict[str, list[object]] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.categoricals[name] = list(choices)
        return choices[0]


_ARCH_OVERLAY_KEYS = (
    'temporal_mix',
    'unet_block',
    'unet_norm',
    'unet_deform',
    'se_ratio',
    'coord_kind',
    'fourier_bands',
    'flow_input',
    'extra_encoder',
    'extra_encoder_channels',
    'feature_sample',
    'use_self_attn',
    'norm',
    'ffn_act',
    'pair_head',
    'pair_geom',
    'layer_scale_init',
    'drop_path',
    'drop_path_decay',
)


def _tiny_from_overlay(overlay: dict[str, Any]) -> UNetNodeTransformer:
    knobs = {key: overlay[key] for key in _ARCH_OVERLAY_KEYS}
    stages = len(overlay['unet_layers'])
    knobs['layers'] = tuple(8 * (i + 1) for i in range(stages))
    knobs['unet_out_channels'] = 8
    knobs['hidden_dim'] = 32
    knobs['n_heads'] = 4
    knobs['n_blocks'] = 1
    return _tiny_detector(**knobs)


def _encode_shape(overlay: dict[str, Any]) -> tuple[int, int, int]:
    stages = len(overlay['unet_layers'])
    depth = 2**stages
    return depth, depth * 2, depth * 2


def test_search_four_stage_and_wide_head_backward() -> None:
    torch.manual_seed(4)
    model = _tiny_detector(layers=(8, 12, 16, 24), unet_out_channels=48)
    imgs = torch.rand(1, 2, 8, 16, 16)
    unet_out, det_logits = model.encode(imgs)
    coords = torch.rand(1, 3, 3) * 3
    mask = torch.ones(1, 3, dtype=torch.bool)
    pos = torch.randn(1, 3, 4 * POS_EMBED_DIM)
    feat = model.index_features(unet_out[:, 0], coords, mask)
    loss = model.predict_edges(feat, feat, coords, coords, pos, pos, mask, mask).float().mean()
    loss = loss + det_logits[0].float().mean()
    loss.backward()
    assert torch.isfinite(loss).all()
    assert unet_out.shape[2] == 48


@pytest.mark.parametrize('name', ['adamw', 'adan', 'adamp', 'muonwithauxadam', 'sgd'])
def test_search_optimizers_step_tiny_detector(name: str) -> None:
    torch.manual_seed(5)
    model = _tiny_detector()
    opt = build_optimizer(model, name=name, lr=0.01, weight_decay=0.01)
    imgs = torch.rand(2, 2, 4, 8, 8)
    loss = model.encode(imgs)[1][0].float().pow(2).mean()
    loss.backward()
    before = [param.detach().clone() for param in model.parameters() if param.grad is not None]
    opt.step()
    after = [param.detach() for param in model.parameters() if param.grad is not None]
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_search_trial_config_recipe_and_augmentations(tmp_path: Path) -> None:
    params = sample_search_params(_FixedTrial())
    params['rot90_aug'] = True
    params['scheduler'] = 'cosine_warmup'
    params['use_ema'] = True
    cfg = trial_config(params, fold=1, weights_dir=tmp_path)
    recipe = _recipe_kwargs(cfg)
    augs = _augmentations_from_cfg(cfg)
    assert recipe['optimizer_name'] == 'adamw'
    assert recipe['amp'] == 'bf16'
    assert recipe['batch_padding'] is True
    assert recipe['match_soft'] is False
    assert cfg['batch_size'] == 16
    assert cfg['poisson_aug'] is False
    assert cfg['poisson_aug_proba'] == 0.0
    names = [getattr(aug, 'func', aug).__name__ for aug in augs]
    assert 'poisson_augment' not in names
    assert len(augs) >= 2


def test_every_search_categorical_builds_and_steps(tmp_path: Path) -> None:
    capture = _CaptureTrial()
    sample_search_params(capture)
    assert capture.categoricals
    for name, choices in capture.categoricals.items():
        for choice in choices:
            params = sample_search_params(_OverrideTrial({name: choice}))
            overlay = apply_search_params(params)
            cfg = trial_config(params, fold=0, weights_dir=tmp_path)
            recipe = _recipe_kwargs(cfg)
            _augmentations_from_cfg(cfg)
            build_optimizer(
                _tiny_detector(), name=str(recipe['optimizer_name']), lr=0.01, weight_decay=0.0
            )
            model = _tiny_from_overlay(overlay)
            depth, height, width = _encode_shape(overlay)
            imgs = torch.rand(1, 2, depth, height, width)
            unet_out, det_logits = model.encode(imgs)
            n_nodes = 2
            coords = torch.rand(1, n_nodes, 3)
            coords[..., 0] *= depth - 1
            coords[..., 1] *= height - 1
            coords[..., 2] *= width - 1
            mask = torch.ones(1, n_nodes, dtype=torch.bool)
            pos = torch.randn(1, n_nodes, 4 * POS_EMBED_DIM)
            feat = model.index_features(unet_out[:, 0], coords, mask)
            loss = (
                model.predict_edges(feat, feat, coords, coords, pos, pos, mask, mask).float().mean()
                + det_logits[0].float().mean()
            )
            loss.backward()
            assert torch.isfinite(loss).all()


class _LastTrial(_FixedTrial):
    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        return choices[-1]

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        return float(high)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(high)


def test_search_window_and_epoch_bounds() -> None:
    low = sample_search_params(_FixedTrial())
    high = sample_search_params(_LastTrial())
    assert low['window_size'] == 2
    assert high['window_size'] == 4
    assert low['epochs'] == 5
    assert high['epochs'] == 30
    p1 = params_from_config(load_base_config())
    assert p1['window_size'] == 2
    assert p1['epochs'] == 30
    overlay = apply_search_params(high)
    assert overlay['window_size'] == 4
    assert overlay['epochs'] == 30


def test_search_window_size_five_train_step() -> None:
    torch.manual_seed(6)
    model = _tiny_detector()
    sample = _combo_sample(n_frames=5)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(_Once(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        match_assign='greedy',
    )
    assert all(np.isfinite(losses))


def test_search_last_choice_overlay_train_step() -> None:
    torch.manual_seed(7)
    params = sample_search_params(_LastTrial())
    overlay = apply_search_params(params)
    model = _tiny_from_overlay(overlay)
    spatial = _encode_shape(overlay)
    sample = _combo_sample(n_frames=int(overlay['window_size']), spatial=spatial)
    if overlay['det_loss'] == 'gaussian_heatmap':
        sample['heatmap_target'] = gaussian_heatmap_target(
            sample['coords'], sample['masks'], spatial
        )
    opt = build_optimizer(model, name='adamw', lr=0.001, weight_decay=0.0)
    losses = train_epoch(
        model,
        DataLoader(_Once(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        det_loss_kind=str(overlay['det_loss']),
        match_assign=str(overlay['match_assign']),
        match_soft=bool(overlay['match_soft']),
        edge_loss=str(overlay['edge_loss']),
        train_peak_topk=int(overlay['train_peak_topk']),
        edge_gate_distance=float(overlay['edge_gate_distance']),
        aux_division_weight=float(overlay['aux_division_weight']),
        aux_contrastive_weight=float(overlay['aux_contrastive_weight']),
        aux_offset_weight=float(overlay['aux_offset_weight']),
        offset_target=str(overlay['offset_target']),
    )
    assert all(np.isfinite(losses))


def test_search_disables_poisson_even_for_last_choice() -> None:
    params = sample_search_params(_LastTrial())
    overlay = apply_search_params(params)
    assert overlay['poisson_aug'] is False
    assert overlay['poisson_aug_proba'] == 0.0
    names = [getattr(aug, 'func', aug).__name__ for aug in _augmentations_from_cfg(overlay)]
    assert 'poisson_augment' not in names


def test_zero_proba_augs_are_omitted() -> None:
    augs = _augmentations_from_cfg(
        {
            'brightness_aug': True,
            'brightness_aug_proba': 0.0,
            'flip_aug': True,
            'flip_aug_proba': 1.0,
            'noise_aug': True,
            'noise_aug_proba': 0.0,
            'contrast_aug': True,
            'contrast_aug_proba': 0.0,
            'gamma_aug': True,
            'gamma_aug_proba': 0.0,
            'rot90_aug': True,
            'rot90_aug_proba': 0.0,
            'translate_aug': True,
            'translate_aug_proba': 0.0,
            'cutout_aug': True,
            'cutout_aug_proba': 0.0,
            'blur_aug': True,
            'blur_aug_proba': 0.0,
            'bleach_aug': True,
            'bleach_aug_proba': 0.0,
            'poisson_aug': True,
            'poisson_aug_proba': 0.0,
            'haze_aug': True,
            'haze_aug_proba': 0.0,
        }
    )
    names = [getattr(aug, 'func', aug).__name__ for aug in augs]
    assert names == ['flip_augment']
