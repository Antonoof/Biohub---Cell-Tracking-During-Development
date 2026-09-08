import argparse
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from biohub.augmentations import (
    bleach_augment,
    blur_augment,
    brightness_augment,
    contrast_augment,
    cutout_augment,
    flip_augment,
    gamma_augment,
    haze_augment,
    noise_augment,
    poisson_augment,
    rot90_augment,
    scale_augment,
    time_stretch_augment,
    time_warp_augment,
    translate_augment,
)
from biohub.data.windows import (
    FrameWindowData,
    FrameWindowDataset,
    VideoGroupedSampler,
    VideoMeta,
    collate_windows,
    load_dataset_windows,
)
from biohub.features.position import POS_EMBED_DIM, pos_embed_torch
from biohub.losses.association import (
    EDGE_LOSSES,
    compute_batch_loss,
    evaluate_pairs_batched,
)
from biohub.losses.aux import contrastive_aux_loss, division_aux_loss, offset_aux_loss
from biohub.losses.detection import DET_LOSSES, detection_loss
from biohub.metrics.aggregation import competition_score
from biohub.models import TemporalUNet3D, UNetNodeTransformer
from biohub.models.temporal_unet import unet_in_channels
from biohub.train.assign import greedy_assign_batched, hard_coupling, match_peaks
from biohub.train.detector_config import validate_detector_config
from biohub.train.schedule import (
    ModelEma,
    amp_dtype,
    build_optimizer,
    build_scheduler,
    normalize_amp,
)
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.parallel import ordered_thread_map
from biohub.utils.seed import (
    configure_determinism,
    dataloader_generator,
    seed_everything,
    seed_worker,
)
from biohub.validation.cv import movie_group_fold_names, payload_movie_names
from biohub.validation.splits import detector_fold_names, detector_validation_role

DEFAULT_METHOD = 'unet_transformer'
DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]

VALIDATION_MATCH_DISTANCE = 5.0
VALIDATION_MATCH_ASSIGN = 'greedy'
CHECKPOINT_METRICS = (
    'acc_times_recall',
    'competition_metric',
    'acc',
    'recall',
    'neg_loss',
    'edge_precision',
    'edge_recall',
    'edge_f1',
    'edge_jaccard',
    'division_precision',
    'division_recall',
    'division_f1',
    'division_jaccard',
    'det_precision',
    'node_ratio',
)


def _rates(tp: float, fp: float, fn: float) -> tuple[float, float, float, float]:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
    jac = tp / max(tp + fp + fn, 1)
    return prec, rec, f1, jac


def require_finite(tensor: torch.Tensor, what: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'{what} is not finite')


def require_finite_grads(model: nn.Module) -> None:
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]

    if grads and not torch.stack([torch.isfinite(g).all() for _, g in grads]).all():
        for name, grad in grads:
            require_finite(grad, f'Detector gradient {name}')


def detector_optimizer_step(
    model, optimizer, scaler, grad_clip_norm, ema=None, *, check_finite: bool = False
) -> None:
    updated = True
    if scaler is not None:
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= old_scale
    else:
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm, error_if_nonfinite=True
            )
        else:
            require_finite_grads(model)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None and updated:
        ema.update(model)


def prepare_detector_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RuntimeError(f'Refusing to overwrite: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_train_device(spec: str) -> torch.device:
    text = str(spec).strip()
    lowered = text.lower()
    if lowered in {'auto', ''}:
        return torch.device(
            f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        )
    if lowered == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda but CUDA is not available')
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    if lowered == 'cpu':
        return torch.device('cpu')
    if lowered.startswith('cuda:'):
        resolved = torch.device(lowered)
        if not torch.cuda.is_available():
            raise RuntimeError(f'device={lowered} but CUDA is not available')
        if resolved.index is None or resolved.index >= torch.cuda.device_count():
            raise ValueError(f'{lowered} is outside the visible logical CUDA devices')
        return resolved
    raise ValueError(f'Unknown device {spec!r}')


def configure_cuda_backends() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = not torch.are_deterministic_algorithms_enabled()


def optional_positive_int(value) -> int | None:
    if value is None:
        return None
    number = int(value)
    return None if number <= 0 else number


def checkpoint_score(metric: str, values: dict[str, float]) -> float:
    if metric not in CHECKPOINT_METRICS:
        raise ValueError(
            f'Unknown checkpoint_metric {metric!r}; expected one of {CHECKPOINT_METRICS}'
        )
    return float(values[metric])


def _topk_coords_padded(
    scores: torch.Tensor,
    k: int,
    min_score: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = scores.shape[0]
    spatial = scores.shape[1:]
    k = min(max(int(k), 0), int(scores[0].numel()))
    if k <= 0:
        coords = scores.new_zeros(batch, 0, 3)
        keep = torch.zeros(batch, 0, dtype=torch.bool, device=scores.device)
        return coords, keep
    vals, idx = torch.topk(scores.reshape(batch, -1), k, dim=1)
    keep = vals > min_score
    coords = torch.stack(torch.unravel_index(idx, spatial), dim=-1).to(dtype=torch.float32)
    return coords.masked_fill(~keep.unsqueeze(-1), 0), keep


def _pack_valid_nodes(
    coords: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = coords.shape[0]
    if coords.shape[1] == 0:
        packed = coords.new_zeros(batch, 1, 3)
        mask = torch.zeros(batch, 1, dtype=torch.bool, device=coords.device)
        return packed, mask, torch.zeros(batch, dtype=torch.long, device=coords.device)
    order = keep.to(dtype=torch.int64).argsort(dim=1, descending=True, stable=True)
    packed = coords.gather(1, order.unsqueeze(-1).expand_as(coords))
    packed_keep = keep.gather(1, order)
    n_keep = packed_keep.sum(dim=1)
    width = max(int(n_keep.max().item()), 1)
    return packed[:, :width], packed_keep[:, :width], n_keep


def detect_and_match(
    det_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, ...],
    det_threshold: float = 0.5,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 5.0,
    voxel_size: tuple[float, ...] | None = None,
    frame_index: int | torch.Tensor = 0,
    window_size: int | None = None,
    match_assign: str = 'greedy',
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 20,
    train_peak_topk: int = 0,
    gt_counts: list[int] | None = None,
    return_couplings: bool = True,
    packed_matches: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor] | torch.Tensor,
    list[torch.Tensor],
]:
    B = det_logits.shape[0]
    device = det_logits.device
    vs = (
        torch.tensor(voxel_size, dtype=torch.float32, device=device)
        if voxel_size is not None
        else None
    )

    if voxel_size is not None:
        pool_kernel = tuple(
            max(1, k if k % 2 == 1 else k + 1)
            for k in (max(1, round(pool_kernel_um / s)) for s in voxel_size)
        )
    else:
        k = max(1, round(pool_kernel_um))
        pool_kernel = (k if k % 2 == 1 else k + 1,) * 3

    pad = tuple(k // 2 for k in pool_kernel)
    gt_n = (
        torch.as_tensor(gt_counts, device=device, dtype=torch.long)
        if gt_counts is not None
        else mask.sum(dim=1).long()
    )
    max_gt_n = int(gt_coords.shape[1])

    with torch.no_grad():
        pooled = F.max_pool3d(det_logits, pool_kernel, stride=1, padding=pad)
        probs = torch.sigmoid(det_logits[:, 0])
        is_peak = (det_logits[:, 0] == pooled[:, 0]) & (probs > det_threshold)
        # Prediction budget must not depend on GT count/padding or batch composition.
        cap = max(512, int(train_peak_topk))
        peak_coords, peak_keep = _topk_coords_padded(probs.masked_fill(~is_peak, -1.0), cap, 0.0)
        if train_peak_topk > 0:
            extra_c, extra_k = _topk_coords_padded(
                probs.masked_fill(is_peak, -1.0), int(train_peak_topk), 0.0
            )
            peak_coords = torch.cat([peak_coords, extra_c], dim=1)
            peak_keep = torch.cat([peak_keep, extra_k], dim=1)
        padded_coords, padded_mask, det_counts_t = _pack_valid_nodes(peak_coords, peak_keep)

    max_det = padded_coords.shape[1]
    max_gt = max(max_gt_n, 1)
    if gt_coords.shape[1] < max_gt:
        gt_coords = F.pad(gt_coords, (0, 0, 0, max_gt - gt_coords.shape[1]))
    gt_valid = torch.arange(max_gt, device=device).unsqueeze(0) < gt_n.unsqueeze(1)
    det_valid = padded_mask
    need_counts = match_assign != 'greedy' or return_couplings or not packed_matches
    det_counts = [int(v) for v in det_counts_t.tolist()] if need_counts else None
    nt_per_sample = [int(v) for v in gt_n.tolist()] if need_counts else None

    sample_couplings: list[torch.Tensor] = []
    matched_pad = torch.full((B, max_det), -1, device=device, dtype=torch.long)
    if match_assign == 'greedy':
        left = padded_coords if vs is None else padded_coords * vs
        right = gt_coords[:, :max_gt] if vs is None else gt_coords[:, :max_gt] * vs
        dists = torch.cdist(left, right)
        far = (~det_valid).unsqueeze(-1) | (~gt_valid).unsqueeze(1)
        dists = dists.masked_fill(far, float('inf'))
        matched_pad = greedy_assign_batched(dists, max_match_distance)
        matched_pad = matched_pad.masked_fill(~det_valid, -1)
        if return_couplings:
            assert nt_per_sample is not None and det_counts is not None
            for b in range(B):
                n = det_counts[b]
                nt = int(nt_per_sample[b])
                sample_couplings.append(hard_coupling(matched_pad[b, :n], nt).to(dtype=dists.dtype))
    else:
        assert nt_per_sample is not None and det_counts is not None
        for b in range(B):
            n_det = det_counts[b]
            nt = int(nt_per_sample[b])
            det_b = padded_coords[b, :n_det]
            gt_b = gt_coords[b, :nt]
            if n_det > 0 and nt > 0:
                if vs is not None:
                    dists = torch.cdist(det_b * vs, gt_b * vs)
                else:
                    dists = torch.cdist(det_b, gt_b)
                matched, coupling = match_peaks(
                    dists,
                    max_match_distance,
                    kind=match_assign,
                    tau=sinkhorn_tau,
                    iters=sinkhorn_iters,
                )
            else:
                matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
                coupling = torch.zeros(n_det, nt, device=device)
            if n_det:
                matched_pad[b, :n_det] = matched
            if return_couplings:
                sample_couplings.append(coupling)

    if torch.is_tensor(frame_index):
        t_col = frame_index.to(device=device, dtype=torch.float32).reshape(B).view(B, 1, 1)
        t_col = t_col.expand(B, max_det, 1)
    else:
        t_col = torch.full((B, max_det, 1), float(frame_index), device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)
    pos_shape = (window_size,) + image_shape[1:] if window_size is not None else image_shape
    padded_pos = pos_embed_torch(full_coords, pos_shape)
    matches_out: list[torch.Tensor] | torch.Tensor
    if packed_matches:
        matches_out = matched_pad
    else:
        assert det_counts is not None
        matches_out = [matched_pad[b, : det_counts[b]] for b in range(B)]
    return padded_coords, padded_pos, padded_mask, matches_out, sample_couplings


def _as_coupling(matched: torch.Tensor, n_gt: int) -> torch.Tensor:
    if matched.ndim == 2:
        return matched
    coupling = torch.zeros(matched.shape[0], n_gt, device=matched.device, dtype=torch.float32)
    if n_gt == 0:
        return coupling
    valid = matched >= 0
    coupling[
        torch.arange(matched.shape[0], device=matched.device),
        matched.clamp(min=0),
    ] = valid.to(dtype=torch.float32)
    return coupling


def _stack_match_indices(
    rows: list[torch.Tensor] | torch.Tensor,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(rows):
        if rows.shape[1] == width and rows.device == device:
            return rows
        out = torch.full((rows.shape[0], width), -1, device=device, dtype=torch.long)
        n = min(int(rows.shape[1]), width)
        if n:
            out[:, :n] = rows[:, :n].to(device=device)
        return out
    out = torch.full((len(rows), width), -1, device=device, dtype=torch.long)
    for b, row in enumerate(rows):
        n = min(int(row.shape[0]), width)
        if n:
            out[b, :n] = row[:n]
    return out


def build_matched_edge_targets(
    match_t: list[torch.Tensor] | torch.Tensor,
    match_t1: list[torch.Tensor] | torch.Tensor,
    gt_target: torch.Tensor,
    max_det_t: int,
    max_det_t1: int,
    *,
    match_soft: bool = False,
    couplings_t: list[torch.Tensor] | None = None,
    couplings_t1: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    B = gt_target.shape[0]
    device = gt_target.device
    if match_soft:
        target = torch.zeros(B, max_det_t, max_det_t1, device=device)
        for b in range(B):
            gt_trans = gt_target[b]
            n_gt_t, n_gt_t1 = gt_trans.shape
            left = couplings_t[b] if couplings_t is not None else match_t[b]
            right = couplings_t1[b] if couplings_t1 is not None else match_t1[b]
            c0 = _as_coupling(left, n_gt_t)
            c1 = _as_coupling(right, n_gt_t1)
            n_t, n_t1 = c0.shape[0], c1.shape[0]
            if n_t == 0 or n_t1 == 0:
                continue
            with torch.autocast(device.type, enabled=False):
                target[b, :n_t, :n_t1] = (
                    c0.float() @ gt_trans[: c0.shape[1], : c1.shape[1]].float() @ c1.float().T
                )
        return target
    if max_det_t == 0 or max_det_t1 == 0:
        return torch.zeros(B, max_det_t, max_det_t1, device=device)
    n_gt_t, n_gt_t1 = gt_target.shape[-2], gt_target.shape[-1]
    if n_gt_t == 0 or n_gt_t1 == 0:
        return torch.zeros(B, max_det_t, max_det_t1, device=device)
    left = _stack_match_indices(match_t, max_det_t, device)
    right = _stack_match_indices(match_t1, max_det_t1, device)
    gathered = gt_target[
        torch.arange(B, device=device).view(B, 1, 1),
        left.clamp(min=0, max=n_gt_t - 1).unsqueeze(-1),
        right.clamp(min=0, max=n_gt_t1 - 1).unsqueeze(1),
    ]
    valid = (left >= 0).unsqueeze(-1) & (right >= 0).unsqueeze(1)
    return gathered.masked_fill(~valid, 0)


def _pair_edge_targets(
    *,
    use_gt: bool,
    match_soft: bool,
    matches_w: torch.Tensor | None,
    targets: torch.Tensor,
    nodes: int,
    couplings_w: list[list[torch.Tensor]] | None,
) -> torch.Tensor:
    pair_batch = int(targets.shape[0] * targets.shape[1])
    if use_gt:
        return targets.reshape(pair_batch, targets.shape[2], targets.shape[3])
    assert matches_w is not None
    if not match_soft:
        match_nodes = matches_w.shape[2]
        return build_matched_edge_targets(
            matches_w[:, :-1].reshape(pair_batch, match_nodes),
            matches_w[:, 1:].reshape(pair_batch, match_nodes),
            targets.reshape(pair_batch, targets.shape[2], targets.shape[3]),
            nodes,
            nodes,
        )
    pairs = int(targets.shape[1])
    stacked = [
        build_matched_edge_targets(
            matches_w[:, i],
            matches_w[:, i + 1],
            targets[:, i],
            nodes,
            nodes,
            match_soft=True,
            couplings_t=None if couplings_w is None else couplings_w[i],
            couplings_t1=None if couplings_w is None else couplings_w[i + 1],
        )
        for i in range(pairs)
    ]
    return torch.stack(stacked, dim=1).reshape(pair_batch, nodes, nodes)


def _split_per_frame(
    values: list[torch.Tensor], batch: int, frames: int
) -> list[list[torch.Tensor]]:
    return [[values[sample * frames + frame] for sample in range(batch)] for frame in range(frames)]


def _window_detections(
    model: UNetNodeTransformer,
    unet_out: torch.Tensor,
    det_logits: list[torch.Tensor] | torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    image_shape: tuple[int, ...],
    voxel_size: tuple[float, ...],
    *,
    use_gt: bool,
    det_threshold: float,
    pool_kernel_um: float,
    max_match_distance: float,
    match_assign: str,
    sinkhorn_tau: float,
    sinkhorn_iters: int,
    train_peak_topk: int,
    match_soft: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    list[list[torch.Tensor]] | None,
]:
    batch, frames = unet_out.shape[:2]
    device = unet_out.device
    flat = batch * frames
    if use_gt:
        nodes = coords.shape[2]
        frame_ids = torch.arange(frames, device=device, dtype=coords.dtype).view(1, frames, 1, 1)
        t_col = frame_ids.expand(batch, frames, nodes, 1)
        pos = pos_embed_torch(
            torch.cat([t_col, coords], dim=-1).reshape(flat, nodes, 4),
            (frames,) + image_shape[1:],
        ).view(batch, frames, nodes, -1)
        feat = model.index_features(
            unet_out.reshape(flat, *unet_out.shape[2:]),
            coords.reshape(flat, nodes, 3),
            masks.reshape(flat, nodes),
        ).view(batch, frames, nodes, -1)
        return coords, pos, masks, None, feat, None

    det_stack = det_logits if torch.is_tensor(det_logits) else torch.stack(det_logits, dim=1)
    frame_index = torch.arange(frames, device=device).unsqueeze(0).expand(batch, frames).reshape(-1)
    det_c, det_p, det_m, matches, couplings = detect_and_match(
        det_stack.reshape(flat, *det_stack.shape[2:]),
        coords.reshape(flat, *coords.shape[2:]),
        masks.reshape(flat, *masks.shape[2:]),
        image_shape,
        det_threshold=det_threshold,
        voxel_size=voxel_size,
        pool_kernel_um=pool_kernel_um,
        max_match_distance=max_match_distance,
        frame_index=frame_index,
        window_size=frames,
        match_assign=match_assign,
        sinkhorn_tau=sinkhorn_tau,
        sinkhorn_iters=sinkhorn_iters,
        train_peak_topk=train_peak_topk,
        return_couplings=match_soft,
        packed_matches=True,
    )
    nodes = det_c.shape[1]
    det_c = det_c.view(batch, frames, nodes, 3)
    det_p = det_p.view(batch, frames, nodes, -1)
    det_m = det_m.view(batch, frames, nodes)
    feat = model.index_features(
        unet_out.reshape(flat, *unet_out.shape[2:]),
        det_c.reshape(flat, nodes, 3),
        det_m.reshape(flat, nodes),
    ).view(batch, frames, nodes, -1)
    assert torch.is_tensor(matches)
    matches_w = matches.view(batch, frames, nodes)
    couplings_w = _split_per_frame(couplings, batch, frames) if couplings else None
    return det_c, det_p, det_m, matches_w, feat, couplings_w


def train_epoch(
    model: UNetNodeTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    det_loss_weight: float = 0.1,
    det_neg_weight: float = 0.1,
    max_iters: int | None = None,
    pool_kernel_um: float = 5.0,
    det_threshold: float = 0.5,
    max_match_distance: float = 5.0,
    grad_clip_norm: float = 1.0,
    edge_loss: str = 'focal_softmax',
    edge_focal_gamma: float = 2.0,
    edge_div_weight: float = 1.0,
    det_loss_kind: str = 'weighted_bce',
    det_heatmap_sigma: float = 1.0,
    target_mode: str = 'matched_det',
    target_gt_frac: float = 0.5,
    aux_division_weight: float = 0.0,
    aux_contrastive_weight: float = 0.0,
    aux_contrastive_temp: float = 0.1,
    aux_offset_weight: float = 0.0,
    accum_steps: int = 1,
    amp_kind: str = 'off',
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEma[UNetNodeTransformer] | None = None,
    match_assign: str = 'greedy',
    match_soft: bool = False,
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 20,
    train_peak_topk: int = 0,
    edge_gate_distance: float = 0.0,
    offset_target: str = 'frac',
) -> tuple[float, float]:
    model.train()
    total_edge_loss = torch.zeros((), device=device)
    total_det_loss = torch.zeros((), device=device)
    n_samples = 0
    accum = max(int(accum_steps), 1)
    dtype = amp_dtype(amp_kind)
    autocast_on = dtype is not None and device.type == 'cuda'

    n_steps = int(max_iters) if max_iters is not None else len(loader)
    if n_steps <= 0 or len(loader) == 0:
        raise ValueError('Detector training requires a nonempty loader and positive steps')
    batch_iter = iter(loader)
    pbar = tqdm(range(n_steps), desc='  batches', leave=False, disable=False)

    t_data, t_forward, t_backward = 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step_i, _ in enumerate(pbar):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch['coords'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        heatmap_batch = (
            batch['heatmap_target'].to(device, non_blocking=True)
            if 'heatmap_target' in batch
            else None
        )
        image_shape = tuple(batch['image_shape'][0].tolist())
        voxel_size = tuple(batch['voxel_size'][0].tolist())
        ds_scale = batch['downsample'][0].to(device)

        t1 = time.perf_counter()
        t_data += t1 - t0

        B, W = imgs.shape[:2]
        if target_mode == 'gt_nodes':
            use_gt = True
        elif target_mode == 'mixed':
            use_gt = float(torch.rand((), device='cpu').item()) < target_gt_frac
        elif target_mode == 'matched_det':
            use_gt = False
        else:
            raise ValueError(
                f'Unknown target_mode {target_mode!r}; expected gt_nodes, mixed, or matched_det'
            )

        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=autocast_on):
            unet_out, det_stack = model.encode_stacked(imgs)

            flat = B * W
            det_loss = detection_loss(
                det_loss_kind,
                det_stack.reshape(flat, *det_stack.shape[2:]),
                coords.reshape(flat, *coords.shape[2:]),
                masks.reshape(flat, *masks.shape[2:]),
                neg_weight=det_neg_weight,
                heatmap_sigma=det_heatmap_sigma,
                focal_gamma=edge_focal_gamma,
                heatmap_target=(
                    None
                    if heatmap_batch is None
                    else heatmap_batch.reshape(flat, *heatmap_batch.shape[2:])
                ),
            )

            det_c, det_p, det_m, matches_w, feat, couplings_w = _window_detections(
                model,
                unet_out,
                det_stack,
                coords,
                masks,
                image_shape,
                voxel_size,
                use_gt=use_gt,
                det_threshold=det_threshold,
                pool_kernel_um=pool_kernel_um,
                max_match_distance=max_match_distance,
                match_assign=match_assign,
                sinkhorn_tau=sinkhorn_tau,
                sinkhorn_iters=sinkhorn_iters,
                train_peak_topk=train_peak_topk,
                match_soft=match_soft,
            )

            aux_div = []
            aux_con = []
            pairs = W - 1
            query: torch.Tensor | None = None
            key: torch.Tensor | None = None
            edge_logits: torch.Tensor | None = None
            pair_target: torch.Tensor | None = None
            if pairs > 0:
                pair_batch = B * pairs
                nodes = feat.shape[2]
                src_feat = feat[:, :-1].reshape(pair_batch, nodes, feat.shape[3])
                tgt_feat = feat[:, 1:].reshape(pair_batch, nodes, feat.shape[3])
                src_c = (det_c[:, :-1] * ds_scale).reshape(pair_batch, nodes, 3)
                tgt_c = (det_c[:, 1:] * ds_scale).reshape(pair_batch, nodes, 3)
                src_p = det_p[:, :-1].reshape(pair_batch, nodes, det_p.shape[-1])
                tgt_p = det_p[:, 1:].reshape(pair_batch, nodes, det_p.shape[-1])
                src_m = det_m[:, :-1].reshape(pair_batch, nodes)
                tgt_m = det_m[:, 1:].reshape(pair_batch, nodes)
                if aux_contrastive_weight > 0:
                    edge_logits, query, key = model.predict_edges_embeddings(
                        src_feat, tgt_feat, src_c, tgt_c, src_p, tgt_p, src_m, tgt_m
                    )
                else:
                    edge_logits = model.predict_edges(
                        src_feat, tgt_feat, src_c, tgt_c, src_p, tgt_p, src_m, tgt_m
                    )
                pair_target = _pair_edge_targets(
                    use_gt=use_gt,
                    match_soft=match_soft,
                    matches_w=matches_w,
                    targets=targets,
                    nodes=nodes,
                    couplings_w=couplings_w,
                )
                assert edge_logits is not None
                edge_loss_val = compute_batch_loss(
                    edge_logits,
                    pair_target,
                    src_m,
                    tgt_m,
                    kind=edge_loss,
                    focal_gamma=edge_focal_gamma,
                    div_weight=edge_div_weight,
                    coords_src=src_c,
                    coords_tgt=tgt_c,
                    gate_distance=edge_gate_distance,
                )
                if aux_division_weight > 0:
                    aux_div.append(division_aux_loss(edge_logits, pair_target, src_m, tgt_m))
                if aux_contrastive_weight > 0 and query is not None and key is not None:
                    aux_con.append(
                        contrastive_aux_loss(
                            query, key, pair_target, src_m, tgt_m, temperature=aux_contrastive_temp
                        )
                    )
            else:
                edge_loss_val = torch.zeros((), device=device)
            loss = edge_loss_val + det_loss_weight * det_loss
            if aux_div:
                loss = loss + aux_division_weight * (sum(aux_div) / len(aux_div))
            if aux_con:
                loss = loss + aux_contrastive_weight * (sum(aux_con) / len(aux_con))
            if aux_offset_weight > 0:
                offset_pred = model.offset_head(unet_out.reshape(flat, *unet_out.shape[2:]))
                offset_det = (
                    None
                    if offset_target != 'parabolic'
                    else det_stack.reshape(flat, *det_stack.shape[2:])
                )
                loss = loss + aux_offset_weight * offset_aux_loss(
                    offset_pred,
                    coords.reshape(flat, *coords.shape[2:]),
                    masks.reshape(flat, *masks.shape[2:]),
                    target=offset_target,
                    det_logits=offset_det,
                )

            group_size = min(accum, n_steps - (step_i // accum) * accum)
            loss = loss / group_size

        require_finite(loss, 'Detector training loss')

        t2 = time.perf_counter()
        t_forward += t2 - t1

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step_i + 1) % accum == 0 or step_i + 1 == n_steps:
            detector_optimizer_step(
                model,
                optimizer,
                scaler,
                grad_clip_norm,
                ema,
                check_finite=scaler is None,
            )

        t3 = time.perf_counter()
        t_backward += t3 - t2

        total_edge_loss = total_edge_loss + edge_loss_val.detach() * B
        total_det_loss = total_det_loss + det_loss.detach() * B
        n_samples += B

        t0 = time.perf_counter()

    t_total = t_data + t_forward + t_backward
    if t_total > 0:
        print(
            f'  [timing] data: {t_data:.1f}s ({100 * t_data / t_total:.0f}%) | '
            f'forward: {t_forward:.1f}s ({100 * t_forward / t_total:.0f}%) | '
            f'backward: {t_backward:.1f}s ({100 * t_backward / t_total:.0f}%) | '
            f'total: {t_total:.1f}s'
        )

    return (
        float(total_edge_loss / max(n_samples, 1)),
        float(total_det_loss / max(n_samples, 1)),
    )


@torch.inference_mode()
def evaluate(
    model: UNetNodeTransformer,
    loader: DataLoader,
    device: torch.device,
    pool_kernel_um: float = 5.0,
    det_threshold: float = 0.5,
    max_match_distance: float = 5.0,
    edge_threshold: float = 0.5,
    match_assign: str = 'greedy',
    match_soft: bool = False,
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 20,
    train_peak_topk: int = 0,
    amp_kind: str = 'off',
) -> dict[str, float]:
    model.eval()
    n_pairs = 0
    pair_stats = torch.zeros(9, device=device, dtype=torch.float64)
    gt_total_t = torch.zeros((), device=device, dtype=torch.long)
    gt_matched_t = torch.zeros((), device=device, dtype=torch.long)
    num_pred_t = torch.zeros((), device=device, dtype=torch.long)
    edge_tp = edge_fp = edge_fn = 0
    division_tp = division_fp = division_fn = 0
    dtype = amp_dtype(amp_kind)
    autocast_on = dtype is not None and device.type == 'cuda'

    for batch in loader:
        imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch['coords'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        image_shape = tuple(batch['image_shape'][0].tolist())
        voxel_size = tuple(batch['voxel_size'][0].tolist())
        ds_scale = batch['downsample'][0].to(device)

        B, W = imgs.shape[:2]
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=autocast_on):
            unet_out, det_stack = model.encode_stacked(imgs)
        det_c, det_p, det_m, matches_w, feat, couplings_w = _window_detections(
            model,
            unet_out,
            det_stack,
            coords,
            masks,
            image_shape,
            voxel_size,
            use_gt=False,
            det_threshold=det_threshold,
            pool_kernel_um=pool_kernel_um,
            max_match_distance=max_match_distance,
            match_assign=match_assign,
            sinkhorn_tau=sinkhorn_tau,
            sinkhorn_iters=sinkhorn_iters,
            train_peak_topk=train_peak_topk,
            match_soft=match_soft,
        )
        assert matches_w is not None
        gt_total_t += masks.sum().long()
        gt_matched_t += (matches_w >= 0).sum()
        num_pred_t += det_m.sum().long()

        pairs = W - 1
        if pairs > 0:
            pair_batch = B * pairs
            nodes = feat.shape[2]
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=autocast_on):
                pair_logits_all = model.predict_edges(
                    feat[:, :-1].reshape(pair_batch, nodes, feat.shape[3]),
                    feat[:, 1:].reshape(pair_batch, nodes, feat.shape[3]),
                    (det_c[:, :-1] * ds_scale).reshape(pair_batch, nodes, 3),
                    (det_c[:, 1:] * ds_scale).reshape(pair_batch, nodes, 3),
                    det_p[:, :-1].reshape(pair_batch, nodes, det_p.shape[-1]),
                    det_p[:, 1:].reshape(pair_batch, nodes, det_p.shape[-1]),
                    det_m[:, :-1].reshape(pair_batch, nodes),
                    det_m[:, 1:].reshape(pair_batch, nodes),
                )
            pair_target = _pair_edge_targets(
                use_gt=False,
                match_soft=match_soft,
                matches_w=matches_w,
                targets=targets,
                nodes=nodes,
                couplings_w=couplings_w,
            )
            pair_stats += evaluate_pairs_batched(
                pair_logits_all,
                pair_target,
                det_m[:, :-1].reshape(pair_batch, nodes),
                det_m[:, 1:].reshape(pair_batch, nodes),
                edge_threshold,
            )
            n_pairs += pair_batch

    (
        total_loss,
        correct,
        total,
        edge_tp,
        edge_fp,
        edge_fn,
        division_tp,
        division_fp,
        division_fn,
    ) = pair_stats.tolist()
    gt_total = int(gt_total_t.item())
    gt_matched = int(gt_matched_t.item())
    num_pred_nodes = int(num_pred_t.item())
    node_recall = gt_matched / max(gt_total, 1)
    det_precision = gt_matched / max(num_pred_nodes, 1)
    acc = correct / max(total, 1)
    loss = total_loss / max(n_pairs, 1)
    metric = competition_score(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=division_tp,
        division_fp=division_fp,
        division_fn=division_fn,
        num_pred_nodes=num_pred_nodes,
        n_total=gt_total,
    )
    edge_p, edge_r, edge_f1, edge_j = _rates(edge_tp, edge_fp, edge_fn)
    div_p, div_r, div_f1, div_j = _rates(division_tp, division_fp, division_fn)
    return {
        'loss': loss,
        'acc': acc,
        'recall': node_recall,
        'acc_times_recall': acc * node_recall,
        'competition_metric': metric,
        'neg_loss': -loss,
        'edge_precision': edge_p,
        'edge_recall': edge_r,
        'edge_f1': edge_f1,
        'edge_jaccard': edge_j,
        'division_precision': div_p,
        'division_recall': div_r,
        'division_f1': div_f1,
        'division_jaccard': div_j,
        'det_precision': det_precision,
        'node_ratio': num_pred_nodes / max(gt_total, 1),
        'edge_tp': float(edge_tp),
        'edge_fp': float(edge_fp),
        'edge_fn': float(edge_fn),
        'division_tp': float(division_tp),
        'division_fp': float(division_fp),
        'division_fn': float(division_fn),
        'num_pred_nodes': float(num_pred_nodes),
        'gt_matched': float(gt_matched),
        'gt_total': float(gt_total),
        'pair_correct': float(correct),
        'pair_total': float(total),
    }


def train(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_dir: Path,
    method: str = DEFAULT_METHOD,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    num_workers: int = 4,
    unet_out_channels: int = 32,
    unet_layers: list[int] | None = None,
    unet_weights: Path | None = None,
    downsample: tuple[int, ...] = (1, 4, 4),
    det_loss_weight: float = 1e1,
    det_neg_weight: float = 1e-2,
    max_iters: int | None = None,
    debug_video: Path | None = None,
    seed: int | None = None,
    max_frames: int | None = None,
    window_size: int = 2,
    augmentations: list | None = DEFAULT_AUGMENTATIONS,
    pool_kernel_um: float = 5.0,
    data_parallel: bool = True,
    hidden_dim: int = 128,
    n_heads: int = 4,
    n_blocks: int = 4,
    dropout: float = 0.3,
    weight_decay: float = 0.01,
    overwrite: bool = False,
    checkpoint_metric: str = 'competition_metric',
    patience: int = 0,
    det_threshold: float = 0.5,
    max_match_distance: float = 5.0,
    grad_clip_norm: float = 1.0,
    mlp_ratio: float = 2.0,
    pair_chunk_size: int | None = 32,
    gradient_checkpointing: bool = False,
    skip_fullres_temporal: bool = True,
    unet_n_heads: int = 4,
    device: str = 'auto',
    edge_threshold: float = 0.5,
    edge_loss: str = 'focal_softmax',
    edge_focal_gamma: float = 2.0,
    edge_div_weight: float = 1.0,
    det_loss: str = 'weighted_bce',
    det_heatmap_sigma: float = 1.0,
    target_mode: str = 'matched_det',
    target_gt_frac: float = 0.5,
    aux_division_weight: float = 0.0,
    aux_contrastive_weight: float = 0.0,
    aux_contrastive_temp: float = 0.1,
    aux_offset_weight: float = 0.0,
    optimizer_name: str = 'adamw',
    scheduler: str = 'none',
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    ema_decay: float = 0.0,
    accum_steps: int = 1,
    amp: str = 'off',
    drop_path: float = 0.0,
    use_self_attn: bool = False,
    norm: str = 'layernorm',
    rel_coord_scale: float = 100.0,
    pair_head: str = 'mlp',
    layer_scale_init: float = 0.0,
    cv_mode: str = 'group_kfold',
    n_folds: int = 5,
    ffn_act: str = 'gelu',
    attn_dropout: float = 0.3,
    drop_path_decay: bool = False,
    pair_geom: str = 'rel',
    se_ratio: float = 0.0,
    unet_block: str = 'plain',
    unet_norm: str = 'batchnorm',
    unet_gn_groups: int = 8,
    unet_deform: bool = False,
    temporal_mix: str = 'attn',
    coord_kind: str = 'none',
    fourier_bands: int = 4,
    flow_input: str = 'none',
    extra_encoder: str = 'none',
    extra_encoder_channels: int = 8,
    extra_encoder_freeze: bool = True,
    extra_encoder_weights: str | None = None,
    feature_sample: str = 'nearest',
    match_assign: str = 'greedy',
    match_soft: bool = False,
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 20,
    train_peak_topk: int = 0,
    edge_gate_distance: float = 0.0,
    offset_target: str = 'frac',
    batch_padding: bool = True,
    frame_cache_mb: float = 0.0,
    epoch_callback: Callable[[int, dict[str, float]], bool | None] | None = None,
) -> UNetNodeTransformer:
    # Validate direct Python calls as well as YAML/CLI before touching data or output.
    call_config = dict(locals())
    for old, new in (
        ('fold', 'split'),
        ('splits_file', 'splits'),
        ('n_epochs', 'epochs'),
        ('optimizer_name', 'optimizer'),
    ):
        call_config[new] = call_config.pop(old)
    call_config.pop('augmentations')
    call_config.pop('epoch_callback')
    validate_config(call_config)
    train_device = resolve_train_device(device)
    if train_device.type == 'cuda':
        torch.cuda.set_device(train_device)
        if normalize_amp(amp) == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError(f'amp=bf16 is unsupported on {train_device}')
    if seed is not None:
        seed_everything(int(seed), deterministic=torch.are_deterministic_algorithms_enabled())
    if unet_layers is None:
        unet_layers = [32, 64, 128]
    if int(window_size) < 1:
        raise ValueError('window_size must be >= 1')
    checkpoint_score(checkpoint_metric, {name: 0.0 for name in CHECKPOINT_METRICS})

    if debug_video is not None:
        train_names = test_names = [debug_video.name]
        train_files = test_files = [debug_video]
        print(f'Debug mode: using single video {debug_video.name}', flush=True)
    else:
        if splits_file.exists():
            payload = json.loads(splits_file.read_text())
        elif cv_mode == 'group_kfold':
            payload = {}
        else:
            raise FileNotFoundError(splits_file)
        if cv_mode == 'group_kfold':
            movies = payload_movie_names(payload)
            if not movies:
                movies = [path.name for path in sorted(data_dir.glob('*.zarr'))]
            if n_folds > len(movies):
                raise ValueError('n_folds exceeds available movies; refusing repeated/empty folds')
            train_names, test_names = movie_group_fold_names(movies, fold, n_folds)
        else:
            train_names, test_names = detector_fold_names(payload, fold)
        train_files = [data_dir / name for name in train_names]
        test_files = [data_dir / name for name in test_names]
        print(f'Fold {fold}: {len(train_files)} train, {len(test_files)} val', flush=True)
    validation_role = detector_validation_role(
        [str(name) for name in train_names],
        [str(name) for name in test_names],
    )
    if validation_role == 'in_sample_production_fit':
        print(
            f'Validation role {validation_role}: overlapping train/val movies; '
            f'{checkpoint_metric} is diagnostic, not a holdout',
            flush=True,
        )

    output_dir = weights_dir / method / f'split_{fold}'

    def _load(
        files: list[Path],
        desc: str,
    ) -> list[tuple[VideoMeta, list[FrameWindowData]]]:
        print(f'Loading {desc} ({len(files)} datasets)...', flush=True)

        def _load_one(path: Path) -> tuple[VideoMeta, list[FrameWindowData]]:
            return load_dataset_windows(
                path,
                window_size=window_size,
                max_frames=max_frames,
                downsample=downsample,
            )

        data = ordered_thread_map(_load_one, files)
        n_windows = sum(len(w) for _, w in data)
        print(f'  {desc} done: {n_windows} windows total', flush=True)
        return data

    train_video_data = _load(train_files, 'train')
    test_video_data = _load(test_files, 'test')
    for name, videos in (('train', train_video_data), ('validation', test_video_data)):
        if not any(windows for _, windows in videos):
            raise ValueError(f'No supervised {name} windows for window_size={window_size}')
        for vm, windows in videos:
            if windows and min(vm.image_shape[1:]) < 2 ** (len(unet_layers) - 1):
                raise ValueError(
                    f'{vm.zarr_path.name}: downsample/UNet depth leaves an empty spatial axis'
                )
    prepare_detector_output(output_dir, overwrite=overwrite)

    all_windows = [w for _, ws in train_video_data + test_video_data for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    print(f'max_nodes={max_nodes}', flush=True)

    pos_feat_dim = 4 * POS_EMBED_DIM

    writer = open_writer(output_dir)
    train_loader = test_loader = None
    try:
        model_config = {
            'unet_out_channels': unet_out_channels,
            'unet_layers': unet_layers,
            'downsample': list(downsample),
            'window_size': window_size,
            'pool_kernel_um': pool_kernel_um,
            'hidden_dim': hidden_dim,
            'n_heads': n_heads,
            'n_blocks': n_blocks,
            'dropout': dropout,
            'pos_feat_dim': pos_feat_dim,
            'mlp_ratio': mlp_ratio,
            'pair_chunk_size': pair_chunk_size,
            'skip_fullres_temporal': skip_fullres_temporal,
            'unet_n_heads': unet_n_heads,
            'checkpoint_metric': checkpoint_metric,
            'checkpoint_selection': checkpoint_metric,
            'promotion_requires': 'official_evaluate',
            'validation_role': validation_role,
            'det_threshold': det_threshold,
            'max_match_distance': max_match_distance,
            'edge_threshold': edge_threshold,
            'grad_clip_norm': grad_clip_norm,
            'patience': patience,
            'drop_path': drop_path,
            'use_self_attn': use_self_attn,
            'norm': norm,
            'rel_coord_scale': rel_coord_scale,
            'pair_head': pair_head,
            'layer_scale_init': layer_scale_init,
            'ffn_act': ffn_act,
            'attn_dropout': attn_dropout,
            'drop_path_decay': drop_path_decay,
            'pair_geom': pair_geom,
            'se_ratio': se_ratio,
            'unet_block': unet_block,
            'unet_norm': unet_norm,
            'unet_gn_groups': unet_gn_groups,
            'unet_deform': unet_deform,
            'temporal_mix': temporal_mix,
            'coord_kind': coord_kind,
            'fourier_bands': fourier_bands,
            'flow_input': flow_input,
            'extra_encoder': extra_encoder,
            'extra_encoder_channels': extra_encoder_channels,
            'extra_encoder_freeze': extra_encoder_freeze,
            'extra_encoder_weights': extra_encoder_weights,
            'feature_sample': feature_sample,
            'match_assign': match_assign,
            'match_soft': match_soft,
            'sinkhorn_tau': sinkhorn_tau,
            'sinkhorn_iters': sinkhorn_iters,
            'train_peak_topk': train_peak_topk,
            'edge_gate_distance': edge_gate_distance,
            'offset_target': offset_target,
            'cv_mode': cv_mode,
            'n_folds': n_folds,
            'edge_loss': edge_loss,
            'det_loss': det_loss,
            'target_mode': target_mode,
            'validation_match_distance': VALIDATION_MATCH_DISTANCE,
            'validation_match_assign': VALIDATION_MATCH_ASSIGN,
            'validation_match_soft': False,
            'validation_peak_topk': 0,
            'batch_padding': batch_padding,
            'frame_cache_mb': frame_cache_mb,
        }
        (output_dir / 'config.json').write_text(json.dumps(model_config, indent=2) + '\n')

        dataset_seed = int(seed) if seed is not None else 0
        train_ds = FrameWindowDataset(
            train_video_data,
            max_nodes=max_nodes,
            augmentations=augmentations,
            seed=dataset_seed,
            batch_padding=batch_padding,
            frame_cache_mb=frame_cache_mb,
        )
        test_ds = FrameWindowDataset(
            test_video_data,
            max_nodes=max_nodes,
            seed=dataset_seed,
            batch_padding=batch_padding,
            frame_cache_mb=frame_cache_mb,
        )
        g = dataloader_generator(seed)
        worker_init_fn = seed_worker if num_workers > 0 else None

        if train_device.type == 'cuda':
            configure_cuda_backends()
        n_visible = torch.cuda.device_count() if train_device.type == 'cuda' else 0
        print(f'Using device: {train_device} | visible CUDA GPUs: {n_visible}', flush=True)
        prefetch = 4 if num_workers > 0 else None
        pin_memory = train_device.type == 'cuda'

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=VideoGroupedSampler(train_ds, generator=g, num_workers=num_workers),
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
            generator=g,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_windows if batch_padding else None,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
            generator=g,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_windows if batch_padding else None,
        )

        unet = TemporalUNet3D(
            in_channels=unet_in_channels(
                coord_kind=coord_kind,
                fourier_bands=fourier_bands,
                flow_input=flow_input,
                extra_encoder=extra_encoder,
                extra_encoder_channels=extra_encoder_channels,
            ),
            out_channels=unet_out_channels,
            layers=unet_layers,
            gradient_checkpointing=gradient_checkpointing,
            skip_fullres_temporal=skip_fullres_temporal,
            temporal_n_heads=unet_n_heads,
            se_ratio=se_ratio,
            unet_block=unet_block,
            unet_norm=unet_norm,
            unet_gn_groups=unet_gn_groups,
            unet_deform=unet_deform,
            temporal_mix=temporal_mix,
        )
        if unet_weights is not None:
            state = torch.load(unet_weights, map_location='cpu', weights_only=True)
            missing, unexpected = unet.load_state_dict(state, strict=False)
            print(
                f'  UNet weights: {len(missing)} missing, {len(unexpected)} unexpected', flush=True
            )

        model = UNetNodeTransformer(
            unet=unet,
            unet_out_channels=unet_out_channels,
            pos_feat_dim=pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            pair_chunk_size=pair_chunk_size,
            drop_path=drop_path,
            use_self_attn=use_self_attn,
            norm=norm,
            rel_coord_scale=rel_coord_scale,
            pair_head=pair_head,
            layer_scale_init=layer_scale_init,
            gradient_checkpointing=gradient_checkpointing,
            ffn_act=ffn_act,
            attn_dropout=attn_dropout,
            drop_path_decay=drop_path_decay,
            pair_geom=pair_geom,
            feature_sample=feature_sample,
            coord_kind=coord_kind,
            fourier_bands=fourier_bands,
            flow_input=flow_input,
            extra_encoder=extra_encoder,
            extra_encoder_channels=extra_encoder_channels,
            extra_encoder_freeze=extra_encoder_freeze,
            extra_encoder_weights=extra_encoder_weights,
        ).to(train_device)

        if data_parallel and train_device.type == 'cuda' and n_visible > 1:
            primary = (
                train_device.index
                if train_device.index is not None
                else torch.cuda.current_device()
            )
            device_ids = [primary, *[i for i in range(n_visible) if i != primary]]
            model.unet = nn.DataParallel(model.unet, device_ids=device_ids, output_device=primary)
            print(
                f'DataParallel: UNet split across {n_visible} GPUs '
                f'(effective per-GPU batch {max(1, batch_size // n_visible)})',
                flush=True,
            )
        elif train_device.type == 'cuda':
            reason = '--single-gpu set' if not data_parallel else f'only {n_visible} GPU visible'
            print(
                f'Single-GPU training ({reason}). '
                'For 2 GPUs set the Kaggle accelerator to GPU T4 x2.',
                flush=True,
            )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model parameters: {n_params:,}', flush=True)

        optimizer = build_optimizer(model, name=optimizer_name, lr=lr, weight_decay=weight_decay)
        lr_sched = build_scheduler(
            optimizer,
            name=scheduler,
            n_epochs=n_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            base_lr=lr,
        )
        ema: ModelEma[UNetNodeTransformer] | None = (
            ModelEma(model, ema_decay) if ema_decay > 0 else None
        )
        scaler = (
            torch.amp.GradScaler('cuda', enabled=True)
            if amp == 'fp16' and train_device.type == 'cuda'
            else None
        )
        print(f'Starting training for {n_epochs} epochs (batch_size={batch_size})...', flush=True)

        best_score = float('-inf')
        best_metrics: dict[str, float] | None = None
        stale = 0
        save_path = output_dir / 'edge_predictor_best.pth'
        pbar = tqdm(range(n_epochs), desc='Training', disable=False)
        print(f'Detection loss: weight={det_loss_weight}, neg_weight={det_neg_weight}', flush=True)
        print(
            f'Checkpoint metric: {checkpoint_metric}; patience={patience}; '
            'promotion requires official evaluate',
            flush=True,
        )

        for epoch in pbar:
            train_ds.set_epoch(epoch)
            t0 = time.monotonic()
            train_edge_loss, train_det_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                train_device,
                det_loss_weight,
                det_neg_weight,
                max_iters=max_iters,
                pool_kernel_um=pool_kernel_um,
                det_threshold=det_threshold,
                max_match_distance=max_match_distance,
                grad_clip_norm=grad_clip_norm,
                edge_loss=edge_loss,
                edge_focal_gamma=edge_focal_gamma,
                edge_div_weight=edge_div_weight,
                det_loss_kind=det_loss,
                det_heatmap_sigma=det_heatmap_sigma,
                target_mode=target_mode,
                target_gt_frac=target_gt_frac,
                aux_division_weight=aux_division_weight,
                aux_contrastive_weight=aux_contrastive_weight,
                aux_contrastive_temp=aux_contrastive_temp,
                aux_offset_weight=aux_offset_weight,
                accum_steps=accum_steps,
                amp_kind=amp,
                scaler=scaler,
                ema=ema,
                match_assign=match_assign,
                match_soft=match_soft,
                sinkhorn_tau=sinkhorn_tau,
                sinkhorn_iters=sinkhorn_iters,
                train_peak_topk=train_peak_topk,
                edge_gate_distance=edge_gate_distance,
                offset_target=offset_target,
            )
            train_time = time.monotonic() - t0

            eval_model = ema.shadow if ema is not None else model
            t0 = time.monotonic()
            metrics = evaluate(
                eval_model,
                test_loader,
                train_device,
                pool_kernel_um=pool_kernel_um,
                det_threshold=det_threshold,
                max_match_distance=VALIDATION_MATCH_DISTANCE,
                edge_threshold=edge_threshold,
                match_assign=VALIDATION_MATCH_ASSIGN,
                match_soft=False,
                sinkhorn_tau=sinkhorn_tau,
                sinkhorn_iters=sinkhorn_iters,
                train_peak_topk=0,
                amp_kind=amp,
            )
            test_time = time.monotonic() - t0

            score = checkpoint_score(checkpoint_metric, metrics)
            is_best = score >= best_score

            if is_best:
                best_score = score
                best_metrics = {key: float(value) for key, value in metrics.items()}
                stale = 0
                (output_dir / 'metrics.json').write_text(json.dumps(best_metrics, indent=2) + '\n')
                torch.save(
                    {
                        k.replace('unet.module.', 'unet.', 1): v
                        for k, v in (
                            ema.state_dict() if ema is not None else model.state_dict()
                        ).items()
                    },
                    save_path,
                )
            else:
                stale += 1

            marker = '*' if is_best else ' '
            tb_values = {
                'train/edge_loss': train_edge_loss,
                'train/det_loss': train_det_loss,
                'train/lr': optimizer.param_groups[0]['lr'],
                'val/score': score,
                'val/best_score': best_score,
            }
            for key, value in metrics.items():
                tb_values[f'val/{key}'] = value
            log_scalars(writer, epoch, tb_values)
            pbar.set_postfix(
                edge=f'{train_edge_loss:.4f}',
                det=f'{train_det_loss:.4f}',
                acc=f'{metrics["acc"]:.4f}',
                comp=f'{metrics["competition_metric"]:.4f}',
            )
            print(
                f'  Epoch {epoch:3d}/{n_epochs} | edge={train_edge_loss:.4f} | '
                f'det={train_det_loss:.4f} | '
                f'test_loss={metrics["loss"]:.4f} | acc={metrics["acc"]:.4f} | '
                f'recall={metrics["recall"]:.4f} | '
                f'competition={metrics["competition_metric"]:.4f} | '
                f'{checkpoint_metric}={score:.4f} | best={best_score:.4f} {marker} | '
                f'train={train_time:.1f}s test={test_time:.1f}s',
                flush=True,
            )
            if lr_sched is not None:
                lr_sched.step()
            if epoch_callback is not None:
                keep_training = epoch_callback(epoch, dict(metrics))
                if keep_training is False:
                    print(f'Stopped by epoch callback at epoch {epoch}', flush=True)
                    break
            if patience > 0 and stale >= patience:
                print(
                    f'Early stopping at epoch {epoch} ({patience} epochs without '
                    f'{checkpoint_metric} improvement)',
                    flush=True,
                )
                break

        if best_metrics is not None:
            (output_dir / 'metrics.json').write_text(json.dumps(best_metrics, indent=2) + '\n')
        print(
            f'\nBest {checkpoint_metric}: {best_score:.4f}, saved to {save_path}. '
            'Window competition_metric is a proxy; promotion requires official evaluate '
            'on a disjoint panel.',
            flush=True,
        )
        writer.close()
        if save_path.exists():
            state = torch.load(save_path, map_location=train_device, weights_only=True)
            if isinstance(model.unet, nn.DataParallel):
                state = {
                    (k.replace('unet.', 'unet.module.', 1) if k.startswith('unet.') else k): v
                    for k, v in state.items()
                }
            model.load_state_dict(state)
        return model
    finally:
        for loader in (train_loader, test_loader):
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
        writer.close()


def _as_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(','))
    return tuple(int(item) for item in value)


def _augmentations_from_cfg(cfg: dict) -> list:
    if any(
        cfg.get(key, False) and float(cfg.get(f'{key}_proba', 0.5)) > 0
        for key in ('time_stretch_aug', 'time_warp_aug')
    ):
        raise ValueError(
            'Temporal resampling changes images without resampling GT tracks; '
            'disable time_stretch_aug and time_warp_aug for detector training'
        )
    augs = []
    brightness_p = float(cfg.get('brightness_aug_proba', 1.0))
    if cfg.get('brightness_aug', True) and brightness_p > 0.0:
        augs.append(
            partial(
                brightness_augment,
                shift_range=float(cfg.get('brightness_shift', 0.1)),
                proba=brightness_p,
            )
        )
    flip_p = float(cfg.get('flip_aug_proba', 1.0))
    if cfg.get('flip_aug', True) and flip_p > 0.0:
        augs.append(partial(flip_augment, proba=flip_p))
    noise_p = float(cfg.get('noise_aug_proba', 0.5))
    if cfg.get('noise_aug', False) and noise_p > 0.0:
        augs.append(
            partial(
                noise_augment,
                std=float(cfg.get('noise_aug_std', 0.05)),
                proba=noise_p,
            )
        )
    contrast_p = float(cfg.get('contrast_aug_proba', 0.5))
    if cfg.get('contrast_aug', False) and contrast_p > 0.0:
        augs.append(
            partial(
                contrast_augment,
                contrast_range=float(cfg.get('contrast_aug_range', 0.2)),
                proba=contrast_p,
            )
        )
    gamma_p = float(cfg.get('gamma_aug_proba', 0.5))
    if cfg.get('gamma_aug', False) and gamma_p > 0.0:
        augs.append(
            partial(
                gamma_augment,
                gamma_range=float(cfg.get('gamma_aug_range', 0.2)),
                proba=gamma_p,
            )
        )
    rot90_p = float(cfg.get('rot90_aug_proba', 0.5))
    if cfg.get('rot90_aug', False) and rot90_p > 0.0:
        augs.append(partial(rot90_augment, proba=rot90_p))
    translate_p = float(cfg.get('translate_aug_proba', 0.5))
    if cfg.get('translate_aug', False) and translate_p > 0.0:
        augs.append(
            partial(
                translate_augment,
                px=int(cfg.get('translate_aug_px', 4)),
                proba=translate_p,
            )
        )
    cutout_p = float(cfg.get('cutout_aug_proba', 0.5))
    if cfg.get('cutout_aug', False) and cutout_p > 0.0:
        augs.append(
            partial(
                cutout_augment,
                holes=int(cfg.get('cutout_holes', 1)),
                size=int(cfg.get('cutout_size', 4)),
                proba=cutout_p,
            )
        )
    time_stretch_p = float(cfg.get('time_stretch_aug_proba', 0.5))
    if cfg.get('time_stretch_aug', False) and time_stretch_p > 0.0:
        augs.append(
            partial(
                time_stretch_augment,
                scale=float(cfg.get('time_stretch_scale', 0.25)),
                proba=time_stretch_p,
            )
        )
    time_warp_p = float(cfg.get('time_warp_aug_proba', 0.5))
    if cfg.get('time_warp_aug', False) and time_warp_p > 0.0:
        augs.append(
            partial(
                time_warp_augment,
                magnitude=float(cfg.get('time_warp_magnitude', 0.2)),
                proba=time_warp_p,
            )
        )
    blur_p = float(cfg.get('blur_aug_proba', 0.5))
    if cfg.get('blur_aug', False) and blur_p > 0.0:
        augs.append(
            partial(
                blur_augment,
                sigma=float(cfg.get('blur_sigma', 0.8)),
                proba=blur_p,
            )
        )
    scale_p = float(cfg.get('scale_aug_proba', 0.5))
    if cfg.get('scale_aug', False) and scale_p > 0.0:
        augs.append(
            partial(
                scale_augment,
                scale_range=float(cfg.get('scale_aug_range', 0.15)),
                proba=scale_p,
            )
        )
    bleach_p = float(cfg.get('bleach_aug_proba', 0.5))
    if cfg.get('bleach_aug', False) and bleach_p > 0.0:
        augs.append(
            partial(
                bleach_augment,
                strength=float(cfg.get('bleach_strength', 0.4)),
                proba=bleach_p,
            )
        )
    poisson_p = float(cfg.get('poisson_aug_proba', 0.5))
    if cfg.get('poisson_aug', False) and poisson_p > 0.0:
        augs.append(
            partial(
                poisson_augment,
                scale=float(cfg.get('poisson_scale', 30.0)),
                proba=poisson_p,
            )
        )
    haze_p = float(cfg.get('haze_aug_proba', 0.5))
    if cfg.get('haze_aug', False) and haze_p > 0.0:
        augs.append(
            partial(
                haze_augment,
                amount=float(cfg.get('haze_amount', 0.1)),
                proba=haze_p,
            )
        )
    return augs


def _recipe_kwargs(cfg: dict) -> dict:
    return {
        'edge_loss': str(cfg.get('edge_loss', 'focal_softmax')),
        'edge_focal_gamma': float(cfg.get('edge_focal_gamma', 2.0)),
        'edge_div_weight': float(cfg.get('edge_div_weight', 1.0)),
        'det_loss': str(cfg.get('det_loss', 'weighted_bce')),
        'det_heatmap_sigma': float(cfg.get('det_heatmap_sigma', 1.0)),
        'target_mode': str(cfg.get('target_mode', 'matched_det')),
        'target_gt_frac': float(cfg.get('target_gt_frac', 0.5)),
        'aux_division_weight': float(cfg.get('aux_division_weight', 0.0)),
        'aux_contrastive_weight': float(cfg.get('aux_contrastive_weight', 0.0)),
        'aux_contrastive_temp': float(cfg.get('aux_contrastive_temp', 0.1)),
        'aux_offset_weight': float(cfg.get('aux_offset_weight', 0.0)),
        'optimizer_name': str(cfg.get('optimizer', 'adamw')),
        'scheduler': str(cfg.get('scheduler', 'none')),
        'warmup_epochs': int(cfg.get('warmup_epochs', 0)),
        'min_lr': float(cfg.get('min_lr', 0.0)),
        'ema_decay': float(cfg.get('ema_decay', 0.0)),
        'accum_steps': int(cfg.get('accum_steps', 1)),
        'amp': normalize_amp(cfg.get('amp', 'off')),
        'drop_path': float(cfg.get('drop_path', 0.0)),
        'use_self_attn': bool(cfg.get('use_self_attn', False)),
        'norm': str(cfg.get('norm', 'layernorm')),
        'rel_coord_scale': float(cfg.get('rel_coord_scale', 100.0)),
        'pair_head': str(cfg.get('pair_head', 'mlp')),
        'layer_scale_init': float(cfg.get('layer_scale_init', 0.0)),
        'cv_mode': str(cfg.get('cv_mode', 'group_kfold')),
        'n_folds': int(cfg.get('n_folds', 5)),
        'ffn_act': str(cfg.get('ffn_act', 'gelu')),
        'attn_dropout': float(cfg.get('attn_dropout', 0.3)),
        'drop_path_decay': bool(cfg.get('drop_path_decay', False)),
        'pair_geom': str(cfg.get('pair_geom', 'rel')),
        'se_ratio': float(cfg.get('se_ratio', 0.0)),
        'unet_block': str(cfg.get('unet_block', 'plain')),
        'unet_norm': str(cfg.get('unet_norm', 'batchnorm')),
        'unet_gn_groups': int(cfg.get('unet_gn_groups', 8)),
        'unet_deform': bool(cfg.get('unet_deform', False)),
        'temporal_mix': str(cfg.get('temporal_mix', 'attn')),
        'coord_kind': str(cfg.get('coord_kind', 'none')),
        'fourier_bands': int(cfg.get('fourier_bands', 4)),
        'flow_input': str(cfg.get('flow_input', 'none')),
        'extra_encoder': str(cfg.get('extra_encoder', 'none')),
        'extra_encoder_channels': int(cfg.get('extra_encoder_channels', 8)),
        'extra_encoder_freeze': bool(cfg.get('extra_encoder_freeze', True)),
        'extra_encoder_weights': (
            None if not cfg.get('extra_encoder_weights') else str(cfg.get('extra_encoder_weights'))
        ),
        'feature_sample': str(cfg.get('feature_sample', 'nearest')),
        'match_assign': str(cfg.get('match_assign', 'greedy')),
        'match_soft': bool(cfg.get('match_soft', False)),
        'sinkhorn_tau': float(cfg.get('sinkhorn_tau', 0.1)),
        'sinkhorn_iters': int(cfg.get('sinkhorn_iters', 20)),
        'train_peak_topk': int(cfg.get('train_peak_topk', 0)),
        'edge_gate_distance': float(cfg.get('edge_gate_distance', 0.0)),
        'offset_target': str(cfg.get('offset_target', 'frac')),
        'batch_padding': bool(cfg.get('batch_padding', True)),
        'frame_cache_mb': float(cfg.get('frame_cache_mb', 0.0)),
    }


def train_from_config(
    cfg: dict,
    *,
    epoch_callback: Callable[[int, dict[str, float]], bool | None] | None = None,
) -> None:
    cfg = validate_config(cfg)
    configure_determinism(cfg['deterministic'])
    if cfg.get('seed') is not None:
        seed_everything(int(cfg['seed']), deterministic=bool(cfg.get('deterministic', False)))
    data_dir = Path(cfg['data_dir'])
    splits_file = Path(cfg['splits']) if cfg.get('splits') else data_dir / 'dataset_splits.json'
    weights_dir = Path(cfg['weights_dir'])
    unet_layers = list(_as_int_tuple(cfg.get('unet_layers'), (32, 64, 128)))
    downsample = _as_int_tuple(cfg.get('downsample'), (1, 4, 4))
    unet_weights = Path(cfg['unet_weights']) if cfg.get('unet_weights') else None
    debug_video = Path(cfg['debug_video']) if cfg.get('debug_video') else None
    split = cfg.get('split', 0)
    n_folds = int(cfg.get('n_folds', 5))
    folds = (
        [0]
        if debug_video is not None
        else (range(n_folds) if str(split) == 'all' else [int(split)])
    )
    for fold in folds:
        train(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_dir=weights_dir,
            method=str(cfg.get('method', DEFAULT_METHOD)),
            n_epochs=int(cfg.get('epochs', cfg.get('n_epochs', 50))),
            lr=float(cfg.get('lr', 1e-4)),
            batch_size=int(cfg.get('batch_size', 16)),
            num_workers=int(cfg.get('num_workers', 8)),
            unet_out_channels=int(cfg.get('unet_out_channels', 32)),
            unet_layers=unet_layers,
            unet_weights=unet_weights,
            downsample=downsample,
            det_loss_weight=float(cfg.get('det_loss_weight', 1.0)),
            det_neg_weight=float(cfg.get('det_neg_weight', 0.01)),
            max_iters=cfg.get('max_iters'),
            debug_video=debug_video,
            seed=cfg.get('seed'),
            max_frames=cfg.get('max_frames'),
            window_size=int(cfg.get('window_size', 2)),
            pool_kernel_um=float(cfg.get('pool_kernel_um', 5.0)),
            data_parallel=bool(cfg.get('data_parallel', True)),
            hidden_dim=int(cfg.get('hidden_dim', 128)),
            n_heads=int(cfg.get('n_heads', 4)),
            n_blocks=int(cfg.get('n_blocks', 4)),
            dropout=float(cfg.get('dropout', 0.3)),
            weight_decay=float(cfg.get('weight_decay', 0.01)),
            augmentations=_augmentations_from_cfg(cfg),
            overwrite=bool(cfg.get('overwrite', False)),
            checkpoint_metric=str(cfg.get('checkpoint_metric', 'competition_metric')),
            patience=int(cfg.get('patience', 0)),
            det_threshold=float(cfg.get('det_threshold', 0.5)),
            max_match_distance=float(cfg.get('max_match_distance', 5.0)),
            grad_clip_norm=float(cfg.get('grad_clip_norm', 1.0)),
            mlp_ratio=float(cfg.get('mlp_ratio', 2.0)),
            pair_chunk_size=optional_positive_int(cfg.get('pair_chunk_size', 32)),
            gradient_checkpointing=bool(cfg.get('gradient_checkpointing', False)),
            skip_fullres_temporal=bool(cfg.get('skip_fullres_temporal', True)),
            unet_n_heads=int(cfg.get('unet_n_heads', 4)),
            device=str(cfg.get('device', 'auto')),
            edge_threshold=float(cfg.get('edge_threshold', 0.5)),
            epoch_callback=epoch_callback,
            **_recipe_kwargs(cfg),
        )


def _resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    raise SystemExit('--data-dir is required')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train UNet + transformer edge predictor end-to-end.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--method', type=str, default=DEFAULT_METHOD)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--weights-dir', type=str, required=True)
    parser.add_argument('--splits', type=str, default=None)
    parser.add_argument('--split', type=str, default='0', help="Split index (0-4) or 'all'.")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--frame-cache-mb',
        type=float,
        default=0.0,
        help='Normalized FP32 frame LRU budget per dataset per worker.',
    )
    parser.add_argument('--no-batch-padding', dest='batch_padding', action='store_false')
    parser.add_argument('--unet-out-channels', type=int, default=32)
    parser.add_argument('--unet-layers', type=str, default='32,64,128')
    parser.add_argument('--unet-weights', type=str, default=None)
    parser.add_argument('--downsample', type=str, default='1,4,4')
    parser.add_argument('--det-loss-weight', type=float, default=1e0)
    parser.add_argument('--det-neg-weight', type=float, default=1e-2)
    parser.add_argument('--max-iters', type=int, default=None)
    parser.add_argument('--debug-video', type=str, default=None)
    parser.add_argument('--window-size', type=int, default=2)
    parser.add_argument('--pool-kernel-um', type=float, default=5.0)
    parser.add_argument('--data-parallel', dest='data_parallel', action='store_true', default=True)
    parser.add_argument('--single-gpu', dest='data_parallel', action='store_false')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--brightness-aug', action='store_true', default=True)
    parser.add_argument('--no-brightness-aug', dest='brightness_aug', action='store_false')
    parser.add_argument('--flip-aug', action='store_true', default=True)
    parser.add_argument('--no-flip-aug', dest='flip_aug', action='store_false')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--checkpoint-metric',
        choices=list(CHECKPOINT_METRICS),
        default='competition_metric',
    )
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--det-threshold', type=float, default=0.5)
    parser.add_argument('--max-match-distance', type=float, default=5.0)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--mlp-ratio', type=float, default=2.0)
    parser.add_argument('--pair-chunk-size', type=int, default=32)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument(
        '--no-gradient-checkpointing', dest='gradient_checkpointing', action='store_false'
    )
    parser.add_argument('--skip-fullres-temporal', action='store_true', default=True)
    parser.add_argument(
        '--no-skip-fullres-temporal', dest='skip_fullres_temporal', action='store_false'
    )
    parser.add_argument('--unet-n-heads', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--edge-threshold', type=float, default=0.5)
    parser.add_argument('--brightness-aug-proba', type=float, default=1.0)
    parser.add_argument('--brightness-shift', type=float, default=0.1)
    parser.add_argument('--flip-aug-proba', type=float, default=1.0)
    parser.add_argument('--noise-aug', action='store_true')
    parser.add_argument('--noise-aug-proba', type=float, default=0.5)
    parser.add_argument('--noise-aug-std', type=float, default=0.05)
    parser.add_argument('--contrast-aug', action='store_true')
    parser.add_argument('--contrast-aug-proba', type=float, default=0.5)
    parser.add_argument('--contrast-aug-range', type=float, default=0.2)
    parser.add_argument('--gamma-aug', action='store_true')
    parser.add_argument('--gamma-aug-proba', type=float, default=0.5)
    parser.add_argument('--gamma-aug-range', type=float, default=0.2)
    parser.add_argument('--rot90-aug', action='store_true')
    parser.add_argument('--rot90-aug-proba', type=float, default=0.5)
    parser.add_argument('--translate-aug', action='store_true')
    parser.add_argument('--translate-aug-proba', type=float, default=0.5)
    parser.add_argument('--translate-aug-px', type=int, default=4)
    parser.add_argument('--cutout-aug', action='store_true')
    parser.add_argument('--cutout-aug-proba', type=float, default=0.5)
    parser.add_argument('--cutout-holes', type=int, default=1)
    parser.add_argument('--cutout-size', type=int, default=4)
    parser.add_argument('--time-stretch-aug', action='store_true')
    parser.add_argument('--time-stretch-aug-proba', type=float, default=0.5)
    parser.add_argument('--time-stretch-scale', type=float, default=0.25)
    parser.add_argument('--time-warp-aug', action='store_true')
    parser.add_argument('--time-warp-aug-proba', type=float, default=0.5)
    parser.add_argument('--time-warp-magnitude', type=float, default=0.2)
    parser.add_argument('--blur-aug', action='store_true')
    parser.add_argument('--blur-aug-proba', type=float, default=0.5)
    parser.add_argument('--blur-sigma', type=float, default=0.8)
    parser.add_argument('--scale-aug', action='store_true')
    parser.add_argument('--scale-aug-proba', type=float, default=0.5)
    parser.add_argument('--scale-aug-range', type=float, default=0.15)
    parser.add_argument('--bleach-aug', action='store_true')
    parser.add_argument('--bleach-aug-proba', type=float, default=0.5)
    parser.add_argument('--bleach-strength', type=float, default=0.4)
    parser.add_argument('--poisson-aug', action='store_true')
    parser.add_argument('--poisson-aug-proba', type=float, default=0.5)
    parser.add_argument('--poisson-scale', type=float, default=30.0)
    parser.add_argument('--haze-aug', action='store_true')
    parser.add_argument('--haze-aug-proba', type=float, default=0.5)
    parser.add_argument('--haze-amount', type=float, default=0.1)
    parser.add_argument('--edge-loss', choices=list(EDGE_LOSSES), default='focal_softmax')
    parser.add_argument('--edge-focal-gamma', type=float, default=2.0)
    parser.add_argument('--edge-div-weight', type=float, default=1.0)
    parser.add_argument('--det-loss', choices=list(DET_LOSSES), default='weighted_bce')
    parser.add_argument('--det-heatmap-sigma', type=float, default=1.0)
    parser.add_argument(
        '--target-mode', choices=('matched_det', 'gt_nodes', 'mixed'), default='matched_det'
    )
    parser.add_argument('--target-gt-frac', type=float, default=0.5)
    parser.add_argument('--aux-division-weight', type=float, default=0.0)
    parser.add_argument('--aux-contrastive-weight', type=float, default=0.0)
    parser.add_argument('--aux-contrastive-temp', type=float, default=0.1)
    parser.add_argument('--aux-offset-weight', type=float, default=0.0)
    parser.add_argument(
        '--optimizer',
        choices=('adamw', 'adam', 'sgd', 'adan', 'adamp', 'muonwithauxadam'),
        default='adamw',
    )
    parser.add_argument('--scheduler', choices=('none', 'cosine', 'cosine_warmup'), default='none')
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--min-lr', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--amp', choices=('off', 'fp16', 'bf16'), default='off')
    parser.add_argument('--drop-path', type=float, default=0.0)
    parser.add_argument('--use-self-attn', action='store_true')
    parser.add_argument('--norm', choices=('layernorm', 'rmsnorm'), default='layernorm')
    parser.add_argument('--rel-coord-scale', type=float, default=100.0)
    parser.add_argument('--pair-head', choices=('mlp', 'bilinear'), default='mlp')
    parser.add_argument('--layer-scale-init', type=float, default=0.0)
    parser.add_argument('--cv-mode', choices=('group_kfold', 'file'), default='group_kfold')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--ffn-act', choices=('gelu', 'silu', 'swiglu'), default='gelu')
    parser.add_argument('--attn-dropout', type=float, default=0.3)
    parser.add_argument('--drop-path-decay', action='store_true')
    parser.add_argument('--pair-geom', choices=('rel', 'dist', 'rel_dist'), default='rel')
    parser.add_argument('--se-ratio', type=float, default=0.0)
    parser.add_argument('--unet-block', choices=('plain', 'residual', 'convnext'), default='plain')
    parser.add_argument('--unet-norm', choices=('batchnorm', 'groupnorm'), default='batchnorm')
    parser.add_argument('--unet-gn-groups', type=int, default=8)
    parser.add_argument('--unet-deform', action='store_true')
    parser.add_argument('--temporal-mix', choices=('attn', 'conv', 'both', 'none'), default='attn')
    parser.add_argument('--coord-kind', choices=('none', 'coord', 'fourier'), default='none')
    parser.add_argument('--fourier-bands', type=int, default=4)
    parser.add_argument(
        '--flow-input',
        choices=('none', 'frame_diff', 'spatial_grad', 'frame_diff_grad'),
        default='none',
    )
    parser.add_argument(
        '--extra-encoder', choices=('none', 'conv', 'cellpose', 'sam'), default='none'
    )
    parser.add_argument('--extra-encoder-channels', type=int, default=8)
    parser.add_argument('--extra-encoder-freeze', action='store_true', default=True)
    parser.add_argument(
        '--no-extra-encoder-freeze', dest='extra_encoder_freeze', action='store_false'
    )
    parser.add_argument('--extra-encoder-weights', type=str, default=None)
    parser.add_argument('--feature-sample', choices=('nearest', 'trilinear'), default='nearest')
    parser.add_argument(
        '--match-assign', choices=('greedy', 'hungarian', 'sinkhorn'), default='greedy'
    )
    parser.add_argument('--match-soft', action='store_true')
    parser.add_argument('--sinkhorn-tau', type=float, default=0.1)
    parser.add_argument('--sinkhorn-iters', type=int, default=20)
    parser.add_argument('--train-peak-topk', type=int, default=0)
    parser.add_argument('--edge-gate-distance', type=float, default=0.0)
    parser.add_argument('--offset-target', choices=('frac', 'parabolic'), default='frac')
    return parser.parse_args(argv)


def validate_config(cfg: dict) -> dict:
    return validate_detector_config(
        cfg, vars(parse_args(['--weights-dir', 'unused'])), CHECKPOINT_METRICS
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_config(vars(args))
    configure_determinism(args.deterministic)
    if args.seed is not None:
        seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    data_dir = _resolve_data_dir(args.data_dir)
    splits_file = Path(args.splits) if args.splits else data_dir / 'dataset_splits.json'
    weights_dir = Path(args.weights_dir)
    unet_layers = [int(x) for x in args.unet_layers.split(',')]
    unet_weights = Path(args.unet_weights) if args.unet_weights else None
    debug_video = Path(args.debug_video) if args.debug_video else None
    downsample = tuple(int(x) for x in args.downsample.split(','))

    n_folds = int(args.n_folds)
    folds = (
        [0]
        if debug_video is not None
        else (range(n_folds) if args.split == 'all' else [int(args.split)])
    )
    for fold in folds:
        train(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_dir=weights_dir,
            method=args.method,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            unet_out_channels=args.unet_out_channels,
            unet_layers=unet_layers,
            unet_weights=unet_weights,
            downsample=downsample,
            det_loss_weight=args.det_loss_weight,
            det_neg_weight=args.det_neg_weight,
            max_iters=args.max_iters,
            debug_video=debug_video,
            seed=args.seed,
            max_frames=args.max_frames,
            window_size=args.window_size,
            pool_kernel_um=args.pool_kernel_um,
            data_parallel=args.data_parallel,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            augmentations=_augmentations_from_cfg(vars(args)),
            overwrite=bool(args.overwrite),
            checkpoint_metric=str(args.checkpoint_metric),
            patience=int(args.patience),
            det_threshold=float(args.det_threshold),
            max_match_distance=float(args.max_match_distance),
            grad_clip_norm=float(args.grad_clip_norm),
            mlp_ratio=float(args.mlp_ratio),
            pair_chunk_size=optional_positive_int(args.pair_chunk_size),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            skip_fullres_temporal=bool(args.skip_fullres_temporal),
            unet_n_heads=int(args.unet_n_heads),
            device=str(args.device),
            edge_threshold=float(args.edge_threshold),
            **_recipe_kwargs(vars(args)),
        )


if __name__ == '__main__':
    main()
