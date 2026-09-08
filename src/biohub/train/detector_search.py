import json
from pathlib import Path
from typing import Any, Protocol

import yaml

from biohub.metrics.aggregation import competition_score
from biohub.models.attention import divisible_heads
from biohub.paths import PROJECT_ROOT

BASE_CONFIG = PROJECT_ROOT / 'configs' / '01_p1.yaml'
UNET_LAYER_CHOICES = ('32-64-128', '32-64-128-256')
UNET_LAYERS = {
    '32-64-128': [32, 64, 128],
    '32-64-128-256': [32, 64, 128, 256],
}
OOF_COUNT_KEYS = (
    'edge_tp',
    'edge_fp',
    'edge_fn',
    'division_tp',
    'division_fp',
    'division_fn',
    'num_pred_nodes',
    'gt_matched',
    'gt_total',
    'pair_correct',
    'pair_total',
)
SEARCH_PARAM_NAMES = (
    'epochs',
    'optimizer',
    'lr',
    'weight_decay',
    'scheduler',
    'warmup_epochs',
    'use_ema',
    'ema_decay',
    'unet_out_channels',
    'unet_layers',
    'hidden_dim',
    'n_heads',
    'n_blocks',
    'mlp_ratio',
    'dropout',
    'attn_dropout',
    'drop_path',
    'use_self_attn',
    'norm',
    'ffn_act',
    'use_layer_scale',
    'layer_scale_init',
    'pair_head',
    'pair_geom',
    'rel_coord_scale',
    'unet_block',
    'unet_norm',
    'unet_deform',
    'temporal_mix',
    'se_ratio',
    'coord_kind',
    'fourier_bands',
    'flow_input',
    'extra_encoder',
    'extra_encoder_channels',
    'feature_sample',
    'window_size',
    'pool_kernel_um',
    'max_match_distance',
    'det_loss_weight',
    'det_neg_weight',
    'det_loss',
    'det_heatmap_sigma',
    'edge_loss',
    'edge_focal_gamma',
    'edge_div_weight',
    'match_assign',
    'match_soft',
    'sinkhorn_tau',
    'sinkhorn_iters',
    'use_peak_topk',
    'train_peak_topk',
    'use_edge_gate',
    'edge_gate_distance',
    'aux_division_weight',
    'aux_contrastive_weight',
    'aux_contrastive_temp',
    'aux_offset_weight',
    'offset_target',
    'brightness_aug_proba',
    'brightness_shift',
    'flip_aug_proba',
    'noise_aug_std',
    'noise_aug_proba',
    'contrast_aug_range',
    'contrast_aug_proba',
    'gamma_aug_range',
    'gamma_aug_proba',
    'rot90_aug',
    'rot90_aug_proba',
    'translate_aug_px',
    'translate_aug_proba',
    'cutout_holes',
    'cutout_size',
    'cutout_aug_proba',
    'blur_sigma',
    'blur_aug_proba',
    'bleach_strength',
    'bleach_aug_proba',
    'haze_amount',
    'haze_aug_proba',
)
INT_SEARCH_PARAMS = {
    'epochs',
    'warmup_epochs',
    'unet_out_channels',
    'hidden_dim',
    'n_heads',
    'n_blocks',
    'fourier_bands',
    'extra_encoder_channels',
    'window_size',
    'sinkhorn_iters',
    'train_peak_topk',
    'translate_aug_px',
    'cutout_holes',
    'cutout_size',
}
BOOL_SEARCH_PARAMS = {
    'use_ema',
    'use_self_attn',
    'use_layer_scale',
    'unet_deform',
    'match_soft',
    'use_peak_topk',
    'use_edge_gate',
    'rot90_aug',
}
FIXED_TRAIN_KEYS = {
    'cv_mode': 'group_kfold',
    'n_folds': 5,
    'seed': 42,
    'patience': 5,
    'checkpoint_metric': 'competition_metric',
    'batch_size': 16,
    'accum_steps': 1,
    'grad_clip_norm': 2.0,
    'amp': 'bf16',
    'skip_fullres_temporal': True,
    'unet_n_heads': 4,
    'unet_gn_groups': 8,
    'extra_encoder_freeze': False,
    'extra_encoder_weights': None,
    'pair_chunk_size': 512,
    'gradient_checkpointing': False,
    'downsample': [1, 4, 4],
    'det_threshold': 0.97,
    'edge_threshold': 0.97,
    'target_mode': 'matched_det',
    'target_gt_frac': 0.0,
    'brightness_aug': True,
    'flip_aug': True,
    'noise_aug': True,
    'contrast_aug': True,
    'gamma_aug': True,
    'translate_aug': True,
    'cutout_aug': True,
    'blur_aug': True,
    'bleach_aug': True,
    'poisson_aug': False,
    'poisson_aug_proba': 0.0,
    'haze_aug': True,
    'scale_aug': False,
    'time_stretch_aug': False,
    'time_stretch_aug_proba': 0.0,
    'time_warp_aug': False,
    'time_warp_aug_proba': 0.0,
    'data_parallel': False,
    'deterministic': False,
    'overwrite': True,
    'num_workers': 4,
    'device': 'cuda',
    'max_iters': None,
    'debug_video': None,
    'max_frames': None,
    'unet_weights': None,
    'batch_padding': True,
    'frame_cache_mb': 256.0,
}
SEARCH_META_KEYS = ('use_ema', 'use_layer_scale', 'use_peak_topk', 'use_edge_gate')


class TrialLike(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float: ...

    def suggest_int(self, name: str, low: int, high: int) -> int: ...


def load_base_config() -> dict[str, Any]:
    payload = yaml.safe_load(BASE_CONFIG.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f'Config {BASE_CONFIG} must be a mapping')
    return payload


def _unet_layers_key(value: Any) -> str:
    layers = [int(item) for item in value]
    for key, expected in UNET_LAYERS.items():
        if layers == expected:
            return key
    return '32-64-128'


def _suggest_lr(trial: TrialLike, optimizer: str) -> float:
    if optimizer in {'adamw', 'adamp'}:
        return trial.suggest_float('lr', 3e-5, 3e-4, log=True)
    if optimizer == 'adan':
        return trial.suggest_float('lr', 5e-5, 5e-4, log=True)
    if optimizer == 'muonwithauxadam':
        return trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    return trial.suggest_float('lr', 1e-3, 1e-2, log=True)


def _suggest_weight_decay(trial: TrialLike, optimizer: str) -> float:
    if optimizer == 'muonwithauxadam':
        return trial.suggest_float('weight_decay', 0.0, 0.01)
    return trial.suggest_float('weight_decay', 0.01, 0.05)


def sample_search_params(trial: TrialLike) -> dict[str, Any]:
    optimizer = str(
        trial.suggest_categorical('optimizer', ['adamw', 'adan', 'adamp', 'muonwithauxadam', 'sgd'])
    )
    scheduler = str(trial.suggest_categorical('scheduler', ['none', 'cosine_warmup']))
    use_ema = bool(trial.suggest_categorical('use_ema', [False, True]))
    use_layer_scale = bool(trial.suggest_categorical('use_layer_scale', [False, True]))
    match_assign = str(
        trial.suggest_categorical('match_assign', ['greedy', 'hungarian', 'sinkhorn'])
    )
    use_peak_topk = bool(trial.suggest_categorical('use_peak_topk', [False, True]))
    use_edge_gate = bool(trial.suggest_categorical('use_edge_gate', [False, True]))
    drop_path = trial.suggest_float('drop_path', 0.0, 0.2)
    warmup_epochs = trial.suggest_int('warmup_epochs', 2, 5)
    params: dict[str, Any] = {
        'epochs': trial.suggest_int('epochs', 10, 50),
        'optimizer': optimizer,
        'lr': _suggest_lr(trial, optimizer),
        'weight_decay': _suggest_weight_decay(trial, optimizer),
        'scheduler': scheduler,
        'warmup_epochs': 0 if scheduler == 'none' else warmup_epochs,
        'use_ema': use_ema,
        'ema_decay': trial.suggest_float('ema_decay', 0.998, 0.9999),
        'unet_out_channels': int(trial.suggest_categorical('unet_out_channels', [32, 48])),
        'unet_layers': str(trial.suggest_categorical('unet_layers', list(UNET_LAYER_CHOICES))),
        'hidden_dim': int(trial.suggest_categorical('hidden_dim', [128, 192, 256])),
        'n_heads': int(trial.suggest_categorical('n_heads', [4, 8])),
        'n_blocks': trial.suggest_int('n_blocks', 4, 8),
        'mlp_ratio': trial.suggest_float('mlp_ratio', 2.0, 4.0),
        'dropout': trial.suggest_float('dropout', 0.05, 0.4),
        'attn_dropout': trial.suggest_float('attn_dropout', 0.0, 0.3),
        'drop_path': drop_path,
        'use_self_attn': bool(trial.suggest_categorical('use_self_attn', [False, True])),
        'norm': str(trial.suggest_categorical('norm', ['layernorm', 'rmsnorm'])),
        'ffn_act': str(trial.suggest_categorical('ffn_act', ['gelu', 'silu', 'swiglu'])),
        'use_layer_scale': use_layer_scale,
        'layer_scale_init': trial.suggest_float('layer_scale_init', 1e-6, 1e-4, log=True),
        'pair_head': str(trial.suggest_categorical('pair_head', ['mlp', 'bilinear'])),
        'pair_geom': str(trial.suggest_categorical('pair_geom', ['rel', 'dist', 'rel_dist'])),
        'rel_coord_scale': trial.suggest_float('rel_coord_scale', 50.0, 200.0),
        'unet_block': str(
            trial.suggest_categorical('unet_block', ['plain', 'residual', 'convnext'])
        ),
        'unet_norm': str(trial.suggest_categorical('unet_norm', ['batchnorm', 'groupnorm'])),
        'unet_deform': bool(trial.suggest_categorical('unet_deform', [False, True])),
        'temporal_mix': str(trial.suggest_categorical('temporal_mix', ['attn', 'conv', 'both'])),
        'se_ratio': float(trial.suggest_categorical('se_ratio', [0.0, 0.25])),
        'coord_kind': str(trial.suggest_categorical('coord_kind', ['none', 'coord', 'fourier'])),
        'fourier_bands': trial.suggest_int('fourier_bands', 4, 8),
        'flow_input': str(
            trial.suggest_categorical(
                'flow_input', ['none', 'frame_diff', 'spatial_grad', 'frame_diff_grad']
            )
        ),
        'extra_encoder': str(trial.suggest_categorical('extra_encoder', ['none', 'conv'])),
        'extra_encoder_channels': trial.suggest_int('extra_encoder_channels', 4, 16),
        'feature_sample': str(
            trial.suggest_categorical('feature_sample', ['nearest', 'trilinear'])
        ),
        'window_size': trial.suggest_int('window_size', 2, 4),
        'pool_kernel_um': trial.suggest_float('pool_kernel_um', 3.0, 7.0),
        'max_match_distance': trial.suggest_float('max_match_distance', 3.0, 8.0),
        'det_loss_weight': trial.suggest_float('det_loss_weight', 0.5, 2.0),
        'det_neg_weight': trial.suggest_float('det_neg_weight', 0.003, 0.03, log=True),
        'det_loss': str(
            trial.suggest_categorical('det_loss', ['weighted_bce', 'focal', 'gaussian_heatmap'])
        ),
        'det_heatmap_sigma': trial.suggest_float('det_heatmap_sigma', 0.8, 2.0),
        'edge_loss': str(
            trial.suggest_categorical('edge_loss', ['focal_softmax', 'ce_softmax', 'asl_softmax'])
        ),
        'edge_focal_gamma': trial.suggest_float('edge_focal_gamma', 1.5, 3.0),
        'edge_div_weight': trial.suggest_float('edge_div_weight', 1.0, 3.0),
        'match_assign': match_assign,
        'match_soft': bool(trial.suggest_categorical('match_soft', [False, True])),
        'sinkhorn_tau': trial.suggest_float('sinkhorn_tau', 0.05, 0.3),
        'sinkhorn_iters': trial.suggest_int('sinkhorn_iters', 10, 30),
        'use_peak_topk': use_peak_topk,
        'train_peak_topk': trial.suggest_int('train_peak_topk', 64, 256),
        'use_edge_gate': use_edge_gate,
        'edge_gate_distance': trial.suggest_float('edge_gate_distance', 15.0, 40.0),
        'aux_division_weight': trial.suggest_float('aux_division_weight', 0.0, 0.5),
        'aux_contrastive_weight': trial.suggest_float('aux_contrastive_weight', 0.0, 0.3),
        'aux_contrastive_temp': trial.suggest_float('aux_contrastive_temp', 0.05, 0.2),
        'aux_offset_weight': trial.suggest_float('aux_offset_weight', 0.0, 0.2),
        'offset_target': str(trial.suggest_categorical('offset_target', ['frac', 'parabolic'])),
        'brightness_aug_proba': trial.suggest_float('brightness_aug_proba', 0.5, 1.0),
        'brightness_shift': trial.suggest_float('brightness_shift', 0.05, 0.2),
        'flip_aug_proba': trial.suggest_float('flip_aug_proba', 0.5, 1.0),
        'noise_aug_std': trial.suggest_float('noise_aug_std', 0.02, 0.1),
        'noise_aug_proba': trial.suggest_float('noise_aug_proba', 0.0, 0.7),
        'contrast_aug_range': trial.suggest_float('contrast_aug_range', 0.1, 0.3),
        'contrast_aug_proba': trial.suggest_float('contrast_aug_proba', 0.0, 0.5),
        'gamma_aug_range': trial.suggest_float('gamma_aug_range', 0.1, 0.3),
        'gamma_aug_proba': trial.suggest_float('gamma_aug_proba', 0.0, 0.5),
        'rot90_aug': bool(trial.suggest_categorical('rot90_aug', [False, True])),
        'rot90_aug_proba': trial.suggest_float('rot90_aug_proba', 0.0, 0.5),
        'translate_aug_px': trial.suggest_int('translate_aug_px', 2, 8),
        'translate_aug_proba': trial.suggest_float('translate_aug_proba', 0.0, 0.5),
        'cutout_holes': trial.suggest_int('cutout_holes', 1, 3),
        'cutout_size': trial.suggest_int('cutout_size', 4, 8),
        'cutout_aug_proba': trial.suggest_float('cutout_aug_proba', 0.0, 0.3),
        'blur_sigma': trial.suggest_float('blur_sigma', 0.4, 1.5),
        'blur_aug_proba': trial.suggest_float('blur_aug_proba', 0.0, 0.5),
        'bleach_strength': trial.suggest_float('bleach_strength', 0.2, 0.5),
        'bleach_aug_proba': trial.suggest_float('bleach_aug_proba', 0.0, 0.3),
        'haze_amount': trial.suggest_float('haze_amount', 0.05, 0.2),
        'haze_aug_proba': trial.suggest_float('haze_aug_proba', 0.0, 0.3),
    }
    return params


def apply_search_params(params: dict[str, Any]) -> dict[str, Any]:
    overlay = dict(params)
    drop_path = float(overlay['drop_path'])
    overlay['drop_path_decay'] = drop_path > 0.0
    overlay['ema_decay'] = float(overlay['ema_decay']) if overlay['use_ema'] else 0.0
    overlay['layer_scale_init'] = (
        float(overlay['layer_scale_init']) if overlay['use_layer_scale'] else 0.0
    )
    overlay['match_soft'] = bool(overlay['match_soft']) and overlay['match_assign'] == 'sinkhorn'
    overlay['train_peak_topk'] = int(overlay['train_peak_topk']) if overlay['use_peak_topk'] else 0
    overlay['edge_gate_distance'] = (
        float(overlay['edge_gate_distance']) if overlay['use_edge_gate'] else 0.0
    )
    overlay['min_lr'] = 0.0 if overlay['scheduler'] == 'none' else float(overlay['lr']) / 100.0
    overlay['warmup_epochs'] = (
        0 if overlay['scheduler'] == 'none' else int(overlay['warmup_epochs'])
    )
    overlay['offset_target'] = (
        str(overlay['offset_target']) if float(overlay['aux_offset_weight']) > 0.0 else 'frac'
    )
    overlay['unet_layers'] = list(UNET_LAYERS[str(overlay['unet_layers'])])
    overlay.update(FIXED_TRAIN_KEYS)
    overlay['n_heads'] = divisible_heads(int(overlay['hidden_dim']), int(overlay['n_heads']))
    overlay['unet_n_heads'] = divisible_heads(
        min(int(width) for width in overlay['unet_layers']),
        int(overlay['unet_n_heads']),
    )
    for key in SEARCH_META_KEYS:
        overlay.pop(key, None)
    return overlay


def params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        'epochs': min(50, max(10, int(cfg.get('epochs', 50)))),
        'optimizer': str(cfg.get('optimizer', 'adamw')),
        'lr': float(cfg.get('lr', 1e-4)),
        'weight_decay': float(cfg.get('weight_decay', 0.01)),
        'scheduler': str(cfg.get('scheduler', 'none')),
        'warmup_epochs': max(2, int(cfg.get('warmup_epochs', 2))),
        'use_ema': float(cfg.get('ema_decay', 0.0)) > 0.0,
        'ema_decay': float(cfg.get('ema_decay', 0.0)) or 0.999,
        'unet_out_channels': int(cfg.get('unet_out_channels', 32)),
        'unet_layers': _unet_layers_key(cfg.get('unet_layers', [32, 64, 128])),
        'hidden_dim': int(cfg.get('hidden_dim', 128)),
        'n_heads': int(cfg.get('n_heads', 4)),
        'n_blocks': int(cfg.get('n_blocks', 4)),
        'mlp_ratio': float(cfg.get('mlp_ratio', 2.0)),
        'dropout': float(cfg.get('dropout', 0.3)),
        'attn_dropout': float(cfg.get('attn_dropout', 0.3)),
        'drop_path': float(cfg.get('drop_path', 0.0)),
        'use_self_attn': bool(cfg.get('use_self_attn', False)),
        'norm': str(cfg.get('norm', 'layernorm')),
        'ffn_act': str(cfg.get('ffn_act', 'gelu')),
        'use_layer_scale': float(cfg.get('layer_scale_init', 0.0)) > 0.0,
        'layer_scale_init': float(cfg.get('layer_scale_init', 0.0)) or 1e-6,
        'pair_head': str(cfg.get('pair_head', 'mlp')),
        'pair_geom': str(cfg.get('pair_geom', 'rel')),
        'rel_coord_scale': float(cfg.get('rel_coord_scale', 100.0)),
        'unet_block': str(cfg.get('unet_block', 'plain')),
        'unet_norm': str(cfg.get('unet_norm', 'batchnorm')),
        'unet_deform': bool(cfg.get('unet_deform', False)),
        'temporal_mix': str(cfg.get('temporal_mix', 'attn')),
        'se_ratio': float(cfg.get('se_ratio', 0.0)),
        'coord_kind': str(cfg.get('coord_kind', 'none')),
        'fourier_bands': int(cfg.get('fourier_bands', 4)),
        'flow_input': str(cfg.get('flow_input', 'none')),
        'extra_encoder': str(cfg.get('extra_encoder', 'none')),
        'extra_encoder_channels': int(cfg.get('extra_encoder_channels', 8)),
        'feature_sample': str(cfg.get('feature_sample', 'nearest')),
        'window_size': min(4, max(2, int(cfg.get('window_size', 2)))),
        'pool_kernel_um': float(cfg.get('pool_kernel_um', 5.0)),
        'max_match_distance': float(cfg.get('max_match_distance', 5.0)),
        'det_loss_weight': float(cfg.get('det_loss_weight', 1.0)),
        'det_neg_weight': float(cfg.get('det_neg_weight', 0.01)),
        'det_loss': str(cfg.get('det_loss', 'weighted_bce')),
        'det_heatmap_sigma': float(cfg.get('det_heatmap_sigma', 1.0)),
        'edge_loss': str(cfg.get('edge_loss', 'focal_softmax')),
        'edge_focal_gamma': float(cfg.get('edge_focal_gamma', 2.0)),
        'edge_div_weight': float(cfg.get('edge_div_weight', 1.0)),
        'match_assign': str(cfg.get('match_assign', 'greedy')),
        'match_soft': bool(cfg.get('match_soft', False)),
        'sinkhorn_tau': float(cfg.get('sinkhorn_tau', 0.1)),
        'sinkhorn_iters': int(cfg.get('sinkhorn_iters', 20)),
        'use_peak_topk': int(cfg.get('train_peak_topk', 0)) > 0,
        'train_peak_topk': max(64, int(cfg.get('train_peak_topk', 0))),
        'use_edge_gate': float(cfg.get('edge_gate_distance', 0.0)) > 0.0,
        'edge_gate_distance': float(cfg.get('edge_gate_distance', 0.0)) or 15.0,
        'aux_division_weight': float(cfg.get('aux_division_weight', 0.0)),
        'aux_contrastive_weight': float(cfg.get('aux_contrastive_weight', 0.0)),
        'aux_contrastive_temp': float(cfg.get('aux_contrastive_temp', 0.1)),
        'aux_offset_weight': float(cfg.get('aux_offset_weight', 0.0)),
        'offset_target': str(cfg.get('offset_target', 'frac')),
        'brightness_aug_proba': float(cfg.get('brightness_aug_proba', 1.0)),
        'brightness_shift': float(cfg.get('brightness_shift', 0.1)),
        'flip_aug_proba': float(cfg.get('flip_aug_proba', 1.0)),
        'noise_aug_std': float(cfg.get('noise_aug_std', 0.05)),
        'noise_aug_proba': (
            float(cfg.get('noise_aug_proba', 0.0)) if cfg.get('noise_aug', False) else 0.0
        ),
        'contrast_aug_range': float(cfg.get('contrast_aug_range', 0.2)),
        'contrast_aug_proba': (
            float(cfg.get('contrast_aug_proba', 0.0)) if cfg.get('contrast_aug', False) else 0.0
        ),
        'gamma_aug_range': float(cfg.get('gamma_aug_range', 0.2)),
        'gamma_aug_proba': (
            float(cfg.get('gamma_aug_proba', 0.0)) if cfg.get('gamma_aug', False) else 0.0
        ),
        'rot90_aug': bool(cfg.get('rot90_aug', False)),
        'rot90_aug_proba': float(cfg.get('rot90_aug_proba', 0.5)),
        'translate_aug_px': int(cfg.get('translate_aug_px', 4)),
        'translate_aug_proba': (
            float(cfg.get('translate_aug_proba', 0.0)) if cfg.get('translate_aug', False) else 0.0
        ),
        'cutout_holes': int(cfg.get('cutout_holes', 1)),
        'cutout_size': int(cfg.get('cutout_size', 4)),
        'cutout_aug_proba': (
            float(cfg.get('cutout_aug_proba', 0.0)) if cfg.get('cutout_aug', False) else 0.0
        ),
        'blur_sigma': float(cfg.get('blur_sigma', 0.8)),
        'blur_aug_proba': (
            float(cfg.get('blur_aug_proba', 0.0)) if cfg.get('blur_aug', False) else 0.0
        ),
        'bleach_strength': float(cfg.get('bleach_strength', 0.4)),
        'bleach_aug_proba': (
            float(cfg.get('bleach_aug_proba', 0.0)) if cfg.get('bleach_aug', False) else 0.0
        ),
        'haze_amount': float(cfg.get('haze_amount', 0.1)),
        'haze_aug_proba': (
            float(cfg.get('haze_aug_proba', 0.0)) if cfg.get('haze_aug', False) else 0.0
        ),
    }
    if params['unet_out_channels'] not in {32, 48}:
        params['unet_out_channels'] = 32
    if params['hidden_dim'] not in {128, 192, 256}:
        params['hidden_dim'] = 128
    if params['n_heads'] not in {4, 8}:
        params['n_heads'] = 4
    if params['temporal_mix'] not in {'attn', 'conv', 'both'}:
        params['temporal_mix'] = 'attn'
    if params['extra_encoder'] not in {'none', 'conv'}:
        params['extra_encoder'] = 'none'
    for name in INT_SEARCH_PARAMS:
        params[name] = int(params[name])
    for name in BOOL_SEARCH_PARAMS:
        params[name] = bool(params[name])
    return params


def seed_trial_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = params_from_config(cfg)
    params['det_loss'] = 'gaussian_heatmap'
    return params


def trial_config(
    params: dict[str, Any],
    *,
    fold: int,
    weights_dir: Path,
) -> dict[str, Any]:
    cfg = load_base_config()
    cfg.update(apply_search_params(params))
    cfg['split'] = str(fold)
    cfg['weights_dir'] = str(weights_dir)
    return cfg


def dump_yaml(payload: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def fold_metrics_path(trial_dir: Path, fold: int) -> Path:
    return trial_dir / 'unet_transformer' / f'split_{fold}' / 'metrics.json'


def pooled_oof_score(trial_dir: Path, n_folds: int = 5) -> tuple[float, dict[str, float]]:
    totals = {key: 0.0 for key in OOF_COUNT_KEYS}
    folds: list[dict[str, float]] = []
    for fold in range(n_folds):
        path = fold_metrics_path(trial_dir, fold)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f'Invalid metrics payload at {path}')
        row = {key: float(payload[key]) for key in OOF_COUNT_KEYS}
        folds.append(row)
        for key, value in row.items():
            totals[key] += value
    competition = competition_score(
        edge_tp=totals['edge_tp'],
        edge_fp=totals['edge_fp'],
        edge_fn=totals['edge_fn'],
        division_tp=totals['division_tp'],
        division_fp=totals['division_fp'],
        division_fn=totals['division_fn'],
        num_pred_nodes=totals['num_pred_nodes'],
        n_total=totals['gt_total'],
    )
    acc = totals['pair_correct'] / max(totals['pair_total'], 1.0)
    recall = totals['gt_matched'] / max(totals['gt_total'], 1.0)
    bundled = {
        'oof': competition,
        'competition_metric': competition,
        'acc_times_recall': acc * recall,
        **totals,
        **{f'fold{fold}_{key}': row[key] for fold, row in enumerate(folds) for key in row},
    }
    return float(competition), bundled
