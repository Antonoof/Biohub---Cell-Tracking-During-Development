"""Fail-fast detector configuration contract, shared by CLI, Python and search."""

import math
import re
from numbers import Integral, Real

from biohub.train.schedule import normalize_amp

ENUMS = {
    'checkpoint_metric': None,
    'edge_loss': ('focal_softmax', 'ce_softmax', 'asl_softmax'),
    'det_loss': ('weighted_bce', 'focal', 'gaussian_heatmap'),
    'target_mode': ('matched_det', 'gt_nodes', 'mixed'),
    'optimizer': ('adamw', 'adam', 'sgd', 'adan', 'adamp', 'muonwithauxadam'),
    'scheduler': ('none', 'cosine', 'cosine_warmup'),
    'amp': ('off', 'fp16', 'bf16'),
    'norm': ('layernorm', 'rmsnorm'),
    'pair_head': ('mlp', 'bilinear'),
    'cv_mode': ('group_kfold', 'file'),
    'ffn_act': ('gelu', 'silu', 'swiglu'),
    'pair_geom': ('rel', 'dist', 'rel_dist'),
    'unet_block': ('plain', 'residual', 'convnext'),
    'unet_norm': ('batchnorm', 'groupnorm'),
    'temporal_mix': ('attn', 'conv', 'both', 'none'),
    'coord_kind': ('none', 'coord', 'fourier'),
    'flow_input': ('none', 'frame_diff', 'spatial_grad', 'frame_diff_grad'),
    'extra_encoder': ('none', 'conv', 'cellpose', 'sam'),
    'feature_sample': ('nearest', 'trilinear'),
    'match_assign': ('greedy', 'hungarian', 'sinkhorn'),
    'offset_target': ('frac', 'parabolic'),
}


def validate_detector_config(cfg: dict, defaults: dict, checkpoint_metrics) -> dict:
    provided = dict(cfg)
    if 'n_epochs' in provided:
        if 'epochs' in provided and provided['epochs'] != provided['n_epochs']:
            raise ValueError('epochs and n_epochs disagree')
        provided['epochs'] = provided.pop('n_epochs')
    unknown = set(provided) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown detector configuration keys: {sorted(unknown)}')
    values = defaults | provided
    values['amp'] = normalize_amp(values['amp'])
    for name, default in defaults.items():
        value = values[name]
        if name == 'pair_chunk_size' and value is None:
            continue
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f'{name} must be a boolean, not {value!r}')
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite and numeric, got {value!r}')
            if isinstance(default, int) and not isinstance(value, Integral):
                raise ValueError(f'{name} must be an integer')
    for name, choices in ENUMS.items():
        choices = checkpoint_metrics if choices is None else choices
        if values[name] not in choices:
            raise ValueError(f'{name}={values[name]!r}; expected one of {choices}')
    positive = (
        'epochs',
        'lr',
        'batch_size',
        'unet_out_channels',
        'window_size',
        'hidden_dim',
        'n_heads',
        'mlp_ratio',
        'unet_n_heads',
        'det_heatmap_sigma',
        'aux_contrastive_temp',
        'accum_steps',
        'rel_coord_scale',
        'unet_gn_groups',
        'fourier_bands',
        'extra_encoder_channels',
        'sinkhorn_tau',
        'sinkhorn_iters',
        'poisson_scale',
    )
    for name in positive:
        if values[name] <= 0:
            raise ValueError(f'{name} must be > 0')
    nonnegative = (
        'num_workers',
        'n_blocks',
        'weight_decay',
        'patience',
        'grad_clip_norm',
        'pool_kernel_um',
        'max_match_distance',
        'min_lr',
        'warmup_epochs',
        'se_ratio',
        'train_peak_topk',
        'edge_gate_distance',
        'frame_cache_mb',
        'layer_scale_init',
        'edge_focal_gamma',
        'edge_div_weight',
        'det_loss_weight',
        'det_neg_weight',
        'aux_division_weight',
        'aux_contrastive_weight',
        'aux_offset_weight',
        'brightness_shift',
        'noise_aug_std',
        'translate_aug_px',
        'cutout_holes',
        'cutout_size',
        'blur_sigma',
        'bleach_strength',
    )
    for name in nonnegative:
        if values[name] < 0:
            raise ValueError(f'{name} must be >= 0')
    for name in (
        'dropout',
        'attn_dropout',
        'target_gt_frac',
        'det_threshold',
        'edge_threshold',
        'haze_amount',
        *[k for k in values if k.endswith('_proba')],
    ):
        if not 0 <= values[name] <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    for name in (
        'drop_path',
        'ema_decay',
        'gamma_aug_range',
        'contrast_aug_range',
        'scale_aug_range',
        'time_stretch_scale',
        'time_warp_magnitude',
    ):
        if not 0 <= values[name] < 1:
            raise ValueError(f'{name} must be in [0, 1)')
    for name in ('seed', 'max_iters', 'max_frames', 'pair_chunk_size'):
        value = values[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not math.isfinite(value)
            or int(value) != value
        ):
            raise ValueError(f'{name} must be an integer or null')
    for name in ('max_iters', 'max_frames'):
        if values[name] is not None and values[name] <= 0:
            raise ValueError(f'{name} must be positive or null')
    if values['seed'] is not None and not 0 <= values['seed'] < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    if values['hidden_dim'] < 2 or values['hidden_dim'] % values['n_heads']:
        raise ValueError('hidden_dim must be >= 2 and divisible by n_heads')
    if int(values['hidden_dim'] * values['mlp_ratio']) < 1:
        raise ValueError('mlp_ratio produces an empty FFN')
    for name, default in (('unet_layers', (32, 64, 128)), ('downsample', (1, 4, 4))):
        raw = values[name].split(',') if isinstance(values[name], str) else values[name]
        if raw is not None and any(float(v) != int(v) or int(v) <= 0 for v in raw):
            raise ValueError(f'{name} must contain positive integers')
        values[name] = tuple(int(v) for v in raw) if raw is not None else default
    if len(values['unet_layers']) < 2 or len(values['downsample']) != 3:
        raise ValueError('unet_layers needs >= 2 stages; downsample needs exactly 3 axes')
    if values['temporal_mix'] in ('attn', 'both'):
        widths = values['unet_layers'][int(values['skip_fullres_temporal']) :]
        if any(w % values['unet_n_heads'] for w in widths):
            raise ValueError('Every active temporal stage width must divide by unet_n_heads')
    if values['min_lr'] > values['lr']:
        raise ValueError('min_lr cannot exceed lr')
    if values['max_frames'] is not None and values['max_frames'] < values['window_size']:
        raise ValueError('max_frames cannot be smaller than window_size')
    if values['match_soft'] and values['match_assign'] != 'sinkhorn':
        raise ValueError('match_soft requires match_assign=sinkhorn')
    for aug in ('time_stretch_aug', 'time_warp_aug'):
        if values[aug] and values[f'{aug}_proba'] > 0:
            raise ValueError(f'{aug} cannot resample images without corresponding GT tracks')
    if values['extra_encoder'] in ('sam', 'cellpose'):
        raise ValueError('SAM/Cellpose training adapters are not validated; use none/conv')
    if values['extra_encoder_weights']:
        raise ValueError('extra_encoder_weights is not used by the none/conv adapters')
    if values['cv_mode'] == 'group_kfold' and values['n_folds'] < 2:
        raise ValueError('group_kfold requires n_folds >= 2')
    if str(values['split']) != 'all':
        if isinstance(values['split'], (bool, float)):
            raise ValueError('split must be a nonnegative integer or all')
        try:
            fold = int(values['split'])
        except (TypeError, ValueError) as exc:
            raise ValueError('split must be a nonnegative integer or all') from exc
        if fold < 0:
            raise ValueError('split must be nonnegative')
        if values['cv_mode'] == 'group_kfold' and fold >= values['n_folds']:
            raise ValueError('split must be smaller than n_folds')
    if re.fullmatch(r'auto|cpu|cuda(?::[0-9]+)?', str(values['device'])) is None:
        raise ValueError('device must be auto, cpu, cuda or cuda:N (logical ordinal)')
    return values
