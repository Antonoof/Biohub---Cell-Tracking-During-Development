import argparse
import json
import time
from functools import partial
from itertools import cycle
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
    VideoMeta,
    load_dataset_windows,
)
from biohub.features.position import POS_EMBED_DIM, pos_embed_torch
from biohub.losses.association import (
    EDGE_LOSSES,
    compute_batch_loss,
    evaluate_pair,
    pair_event_counts,
)
from biohub.losses.aux import contrastive_aux_loss, division_aux_loss, offset_aux_loss
from biohub.losses.detection import DET_LOSSES, detection_loss
from biohub.metrics.aggregation import competition_score
from biohub.models import TemporalUNet3D, UNetNodeTransformer
from biohub.models.temporal_unet import unet_in_channels
from biohub.train.assign import match_peaks
from biohub.train.schedule import (
    ModelEma,
    amp_dtype,
    build_optimizer,
    build_scheduler,
    normalize_amp,
)
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.parallel import ordered_thread_map
from biohub.utils.seed import dataloader_generator, seed_everything, seed_worker
from biohub.validation.cv import movie_group_fold_names, payload_movie_names
from biohub.validation.splits import detector_fold_names, detector_validation_role

DEFAULT_METHOD = 'unet_transformer'
DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]
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
    for name, param in model.named_parameters():
        if param.grad is not None:
            require_finite(param.grad, f'Detector gradient {name}')


def prepare_detector_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RuntimeError(f'Refusing to overwrite: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_train_device(spec: str) -> torch.device:
    text = str(spec).strip()
    lowered = text.lower()
    if lowered in {'auto', ''}:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if lowered == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda but CUDA is not available')
        return torch.device('cuda')
    if lowered == 'cpu':
        return torch.device('cpu')
    if lowered.startswith('cuda:'):
        return torch.device(text)
    raise ValueError(f'Unknown device {spec!r}')


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


def detect_and_match(
    det_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, ...],
    det_threshold: float = 0.5,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 5.0,
    voxel_size: tuple[float, ...] | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
    match_assign: str = 'greedy',
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 20,
    train_peak_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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

    with torch.no_grad():
        pooled = F.max_pool3d(det_logits, pool_kernel, stride=1, padding=pad)
        is_peak = (det_logits == pooled) & (torch.sigmoid(det_logits) > det_threshold)
        peak_idx = torch.nonzero(is_peak[:, 0])

    batch_ids = peak_idx[:, 0]
    peak_coords = peak_idx[:, 1:].float()
    if train_peak_topk > 0:
        scores = torch.sigmoid(det_logits[:, 0]).masked_fill(is_peak[:, 0], -1.0)
        extra_b: list[torch.Tensor] = []
        extra_c: list[torch.Tensor] = []
        for b in range(B):
            flat = scores[b].reshape(-1)
            k = min(int(train_peak_topk), int(flat.numel()))
            if k <= 0:
                continue
            vals, idx = torch.topk(flat, k)
            idx = idx[vals > 0]
            if idx.numel() == 0:
                continue
            zyx = torch.stack(torch.unravel_index(idx, scores[b].shape), dim=-1).float()
            extra_b.append(torch.full((zyx.shape[0],), b, device=device, dtype=batch_ids.dtype))
            extra_c.append(zyx)
        if extra_c:
            batch_ids = torch.cat([batch_ids, *extra_b])
            peak_coords = torch.cat([peak_coords, *extra_c])

    nt_per_sample = mask.sum(dim=1).long()

    sample_matches: list[torch.Tensor] = []
    sample_couplings: list[torch.Tensor] = []
    sample_coords: list[torch.Tensor] = []
    max_det = 0

    for b in range(B):
        sel = batch_ids == b
        det_b = peak_coords[sel]
        n_det = det_b.shape[0]
        nt = int(nt_per_sample[b].item())
        gt_b = gt_coords[b, :nt]
        n_gt = gt_b.shape[0]

        if n_det > 0 and n_gt > 0:
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
            coupling = torch.zeros(n_det, n_gt, device=device)

        sample_matches.append(matched)
        sample_couplings.append(coupling)
        sample_coords.append(det_b)
        if n_det > max_det:
            max_det = n_det

    max_det = max(max_det, 1)

    padded_coords = torch.zeros(B, max_det, 3, device=device)
    padded_mask = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    for b in range(B):
        n = sample_coords[b].shape[0]
        if n == 0:
            continue
        padded_coords[b, :n] = sample_coords[b]
        padded_mask[b, :n] = True

    t_col = torch.full((B, max_det, 1), frame_index, device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)
    pos_shape = (window_size,) + image_shape[1:] if window_size is not None else image_shape
    padded_pos = pos_embed_torch(full_coords, pos_shape)

    return padded_coords, padded_pos, padded_mask, sample_matches, sample_couplings


def _as_coupling(matched: torch.Tensor, n_gt: int) -> torch.Tensor:
    if matched.ndim == 2:
        return matched
    coupling = torch.zeros(matched.shape[0], n_gt, device=matched.device, dtype=torch.float32)
    valid = matched >= 0
    if valid.any():
        coupling[torch.arange(matched.shape[0], device=matched.device)[valid], matched[valid]] = 1.0
    return coupling


def build_matched_edge_targets(
    match_t: list[torch.Tensor],
    match_t1: list[torch.Tensor],
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
    target = torch.zeros(B, max_det_t, max_det_t1, device=device)

    for b in range(B):
        gt_trans = gt_target[b]
        n_gt_t, n_gt_t1 = gt_trans.shape
        left = couplings_t[b] if match_soft and couplings_t is not None else match_t[b]
        right = couplings_t1[b] if match_soft and couplings_t1 is not None else match_t1[b]
        c0 = _as_coupling(left, n_gt_t)
        c1 = _as_coupling(right, n_gt_t1)
        n_t, n_t1 = c0.shape[0], c1.shape[0]
        if n_t == 0 or n_t1 == 0:
            continue
        target[b, :n_t, :n_t1] = c0 @ gt_trans @ c1.T

    return target


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
    total_edge_loss = 0.0
    total_det_loss = 0.0
    n_samples = 0
    accum = max(int(accum_steps), 1)
    dtype = amp_dtype(amp_kind)
    autocast_on = dtype is not None and device.type == 'cuda'

    if max_iters is not None:
        batch_iter = cycle(loader)
        pbar = tqdm(range(max_iters), desc='  iters', leave=False, disable=False)
    else:
        batch_iter = iter(loader)
        pbar = tqdm(range(len(loader)), desc='  batches', leave=False, disable=False)

    t_data, t_forward, t_backward = 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    optimizer.zero_grad()

    for step_i, _ in enumerate(pbar):
        batch = next(batch_iter)

        imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch['coords'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        image_shape = tuple(batch['image_shape'][0].tolist())
        voxel_size = tuple(batch['voxel_size'][0].tolist())
        ds_scale = batch['downsample'][0].to(device)

        t1 = time.perf_counter()
        t_data += t1 - t0

        B, W = imgs.shape[:2]
        if target_mode == 'gt_nodes':
            use_gt = True
        elif target_mode == 'mixed':
            use_gt = float(torch.rand((), device=device).item()) < target_gt_frac
        else:
            use_gt = False

        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=autocast_on):
            unet_out, det_logits = model.encode(imgs)

            det_losses = [
                detection_loss(
                    det_loss_kind,
                    det_logits[i],
                    coords[:, i],
                    masks[:, i],
                    neg_weight=det_neg_weight,
                    heatmap_sigma=det_heatmap_sigma,
                    focal_gamma=edge_focal_gamma,
                )
                for i in range(W)
            ]
            det_loss = sum(det_losses) / W

            frame_det: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor] | None,
                    torch.Tensor,
                    list[torch.Tensor] | None,
                ]
            ] = []
            for i in range(W):
                if use_gt:
                    det_c = coords[:, i]
                    det_m = masks[:, i]
                    t_col = torch.full((B, det_c.shape[1], 1), i, device=device, dtype=det_c.dtype)
                    pos_shape = (W,) + image_shape[1:]
                    det_p = pos_embed_torch(torch.cat([t_col, det_c], dim=-1), pos_shape)
                    matches: list[torch.Tensor] | None = None
                    couplings: list[torch.Tensor] | None = None
                else:
                    det_c, det_p, det_m, matches, couplings = detect_and_match(
                        det_logits[i],
                        coords[:, i],
                        masks[:, i],
                        image_shape,
                        det_threshold=det_threshold,
                        voxel_size=voxel_size,
                        pool_kernel_um=pool_kernel_um,
                        max_match_distance=max_match_distance,
                        frame_index=i,
                        window_size=W,
                        match_assign=match_assign,
                        sinkhorn_tau=sinkhorn_tau,
                        sinkhorn_iters=sinkhorn_iters,
                        train_peak_topk=train_peak_topk,
                    )
                unet_feat = model.index_features(unet_out[:, i], det_c, det_m)
                frame_det.append((det_c, det_p, det_m, matches, unet_feat, couplings))

            block_losses = []
            aux_div = []
            aux_con = []
            for i in range(W - 1):
                ns = frame_det[i][0].shape[1]
                nt = frame_det[i + 1][0].shape[1]
                if use_gt:
                    pair_target = targets[:, i]
                else:
                    matches_t = frame_det[i][3]
                    matches_t1 = frame_det[i + 1][3]
                    assert matches_t is not None and matches_t1 is not None
                    pair_target = build_matched_edge_targets(
                        matches_t,
                        matches_t1,
                        targets[:, i],
                        ns,
                        nt,
                        match_soft=match_soft,
                        couplings_t=frame_det[i][5],
                        couplings_t1=frame_det[i + 1][5],
                    )
                query: torch.Tensor | None = None
                key: torch.Tensor | None = None
                if aux_contrastive_weight > 0:
                    edge_logits, query, key = model.predict_edges_embeddings(
                        frame_det[i][4],
                        frame_det[i + 1][4],
                        frame_det[i][0] * ds_scale,
                        frame_det[i + 1][0] * ds_scale,
                        frame_det[i][1],
                        frame_det[i + 1][1],
                        frame_det[i][2],
                        frame_det[i + 1][2],
                    )
                else:
                    edge_logits = model.predict_edges(
                        frame_det[i][4],
                        frame_det[i + 1][4],
                        frame_det[i][0] * ds_scale,
                        frame_det[i + 1][0] * ds_scale,
                        frame_det[i][1],
                        frame_det[i + 1][1],
                        frame_det[i][2],
                        frame_det[i + 1][2],
                    )
                block_losses.append(
                    compute_batch_loss(
                        edge_logits,
                        pair_target,
                        frame_det[i][2],
                        frame_det[i + 1][2],
                        kind=edge_loss,
                        focal_gamma=edge_focal_gamma,
                        div_weight=edge_div_weight,
                        coords_src=frame_det[i][0] * ds_scale,
                        coords_tgt=frame_det[i + 1][0] * ds_scale,
                        gate_distance=edge_gate_distance,
                    )
                )
                if aux_division_weight > 0:
                    for b in range(B):
                        ns_b = int(frame_det[i][2][b].sum().item())
                        nt_b = int(frame_det[i + 1][2][b].sum().item())
                        aux_div.append(
                            division_aux_loss(
                                edge_logits[b, :ns_b, :nt_b],
                                pair_target[b, :ns_b, :nt_b],
                            )
                        )
                if aux_contrastive_weight > 0 and query is not None and key is not None:
                    for b in range(B):
                        ns_b = int(frame_det[i][2][b].sum().item())
                        nt_b = int(frame_det[i + 1][2][b].sum().item())
                        aux_con.append(
                            contrastive_aux_loss(
                                query[b, :ns_b],
                                key[b, :nt_b],
                                pair_target[b, :ns_b, :nt_b],
                                temperature=aux_contrastive_temp,
                            )
                        )
            edge_loss_val = sum(block_losses) / len(block_losses)
            loss = edge_loss_val + det_loss_weight * det_loss
            if aux_div:
                loss = loss + aux_division_weight * (sum(aux_div) / len(aux_div))
            if aux_con:
                loss = loss + aux_contrastive_weight * (sum(aux_con) / len(aux_con))
            if aux_offset_weight > 0:
                offset_terms = [
                    offset_aux_loss(
                        model.offset_head(unet_out[:, i]),
                        coords[:, i],
                        masks[:, i],
                        target=offset_target,
                        det_logits=det_logits[i],
                    )
                    for i in range(W)
                ]
                loss = loss + aux_offset_weight * (sum(offset_terms) / W)
            loss = loss / accum

        require_finite(loss, 'Detector training loss')

        t2 = time.perf_counter()
        t_forward += t2 - t1

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step_i + 1) % accum == 0:
            require_finite_grads(model)
            if scaler is not None:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        t3 = time.perf_counter()
        t_backward += t3 - t2

        total_edge_loss += float(edge_loss_val) * B
        total_det_loss += float(det_loss) * B
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
        total_edge_loss / max(n_samples, 1),
        total_det_loss / max(n_samples, 1),
    )


@torch.no_grad()
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
) -> dict[str, float]:
    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0
    num_pred_nodes = 0
    edge_tp = edge_fp = edge_fn = 0
    division_tp = division_fp = division_fn = 0

    for batch in loader:
        imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch['coords'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        image_shape = tuple(batch['image_shape'][0].tolist())
        voxel_size = tuple(batch['voxel_size'][0].tolist())
        ds_scale = batch['downsample'][0].to(device)

        B, W = imgs.shape[:2]
        unet_out, det_logits = model.encode(imgs)
        frame_det: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                list[torch.Tensor],
                torch.Tensor,
                list[torch.Tensor],
            ]
        ] = []
        for i in range(W):
            det_c, det_p, det_m, matches, couplings = detect_and_match(
                det_logits[i],
                coords[:, i],
                masks[:, i],
                image_shape,
                det_threshold=det_threshold,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                max_match_distance=max_match_distance,
                frame_index=i,
                window_size=W,
                match_assign=match_assign,
                sinkhorn_tau=sinkhorn_tau,
                sinkhorn_iters=sinkhorn_iters,
                train_peak_topk=train_peak_topk,
            )
            unet_feat = model.index_features(
                unet_out[:, i],
                det_c,
                det_m,
            )
            frame_det.append((det_c, det_p, det_m, matches, unet_feat, couplings))

            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = (matches[b] >= 0).sum().item()
                gt_total += n_gt
                gt_matched += n_matched
                num_pred_nodes += int(det_m[b].sum().item())

        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3],
                frame_det[i + 1][3],
                targets[:, i],
                ns,
                nt,
                match_soft=match_soft,
                couplings_t=frame_det[i][5],
                couplings_t1=frame_det[i + 1][5],
            )
            pair_logits = model.predict_edges(
                frame_det[i][4],
                frame_det[i + 1][4],
                frame_det[i][0] * ds_scale,
                frame_det[i + 1][0] * ds_scale,
                frame_det[i][1],
                frame_det[i + 1][1],
                frame_det[i][2],
                frame_det[i + 1][2],
            )

            for b in range(B):
                ns_b = int(frame_det[i][2][b].sum().item())
                nt_b = int(frame_det[i + 1][2][b].sum().item())
                logits_b = pair_logits[b, :ns_b, :nt_b]
                target_b = pair_target[b, :ns_b, :nt_b]
                pair_loss, pair_correct, pair_total = evaluate_pair(
                    logits_b,
                    target_b,
                    edge_threshold=edge_threshold,
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                n_pairs += 1
                counts = pair_event_counts(logits_b, target_b, edge_threshold=edge_threshold)
                edge_tp += counts[0]
                edge_fp += counts[1]
                edge_fn += counts[2]
                division_tp += counts[3]
                division_fp += counts[4]
                division_fn += counts[5]

    node_recall = gt_matched / max(gt_total, 1)
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
        'det_precision': gt_matched / max(num_pred_nodes, 1),
        'node_ratio': num_pred_nodes / max(gt_total, 1),
        'edge_tp': float(edge_tp),
        'edge_fp': float(edge_fp),
        'edge_fn': float(edge_fn),
        'division_tp': float(division_tp),
        'division_fp': float(division_fp),
        'division_fn': float(division_fn),
        'num_pred_nodes': float(num_pred_nodes),
        'gt_total': float(gt_total),
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
) -> UNetNodeTransformer:
    if unet_layers is None:
        unet_layers = [32, 64, 128]
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
    prepare_detector_output(output_dir, overwrite=overwrite)
    writer = open_writer(output_dir)

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

    all_windows = [w for _, ws in train_video_data + test_video_data for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    print(f'max_nodes={max_nodes}', flush=True)

    pos_feat_dim = 4 * POS_EMBED_DIM

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
    }
    (output_dir / 'config.json').write_text(json.dumps(model_config, indent=2) + '\n')

    dataset_seed = int(seed) if seed is not None else 0
    train_ds = FrameWindowDataset(
        train_video_data,
        max_nodes=max_nodes,
        augmentations=augmentations,
        seed=dataset_seed,
    )
    test_ds = FrameWindowDataset(test_video_data, max_nodes=max_nodes, seed=dataset_seed)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = dataloader_generator(seed)
        worker_init_fn = seed_worker

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )

    train_device = resolve_train_device(device)
    n_visible = torch.cuda.device_count() if train_device.type == 'cuda' else 0
    print(f'Using device: {train_device} | visible CUDA GPUs: {n_visible}', flush=True)

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
        print(f'  UNet weights: {len(missing)} missing, {len(unexpected)} unexpected', flush=True)

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
        model.unet = nn.DataParallel(model.unet)
        print(
            f'DataParallel: UNet split across {n_visible} GPUs '
            f'(effective per-GPU batch {max(1, batch_size // n_visible)})',
            flush=True,
        )
    elif train_device.type == 'cuda':
        reason = '--single-gpu set' if not data_parallel else f'only {n_visible} GPU visible'
        print(
            f'Single-GPU training ({reason}). For 2 GPUs set the Kaggle accelerator to GPU T4 x2.',
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
            max_match_distance=max_match_distance,
            edge_threshold=edge_threshold,
            match_assign=match_assign,
            match_soft=match_soft,
            sinkhorn_tau=sinkhorn_tau,
            sinkhorn_iters=sinkhorn_iters,
            train_peak_topk=train_peak_topk,
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


def _as_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(','))
    return tuple(int(item) for item in value)


def _augmentations_from_cfg(cfg: dict) -> list:
    augs = []
    if cfg.get('brightness_aug', True):
        augs.append(
            partial(
                brightness_augment,
                shift_range=float(cfg.get('brightness_shift', 0.1)),
                proba=float(cfg.get('brightness_aug_proba', 1.0)),
            )
        )
    if cfg.get('flip_aug', True):
        augs.append(partial(flip_augment, proba=float(cfg.get('flip_aug_proba', 1.0))))
    if cfg.get('noise_aug', False):
        augs.append(
            partial(
                noise_augment,
                std=float(cfg.get('noise_aug_std', 0.05)),
                proba=float(cfg.get('noise_aug_proba', 0.5)),
            )
        )
    if cfg.get('contrast_aug', False):
        augs.append(
            partial(
                contrast_augment,
                contrast_range=float(cfg.get('contrast_aug_range', 0.2)),
                proba=float(cfg.get('contrast_aug_proba', 0.5)),
            )
        )
    if cfg.get('gamma_aug', False):
        augs.append(
            partial(
                gamma_augment,
                gamma_range=float(cfg.get('gamma_aug_range', 0.2)),
                proba=float(cfg.get('gamma_aug_proba', 0.5)),
            )
        )
    if cfg.get('rot90_aug', False):
        augs.append(partial(rot90_augment, proba=float(cfg.get('rot90_aug_proba', 0.5))))
    if cfg.get('translate_aug', False):
        augs.append(
            partial(
                translate_augment,
                px=int(cfg.get('translate_aug_px', 4)),
                proba=float(cfg.get('translate_aug_proba', 0.5)),
            )
        )
    if cfg.get('cutout_aug', False):
        augs.append(
            partial(
                cutout_augment,
                holes=int(cfg.get('cutout_holes', 1)),
                size=int(cfg.get('cutout_size', 4)),
                proba=float(cfg.get('cutout_aug_proba', 0.5)),
            )
        )
    if cfg.get('time_stretch_aug', False):
        augs.append(
            partial(
                time_stretch_augment,
                scale=float(cfg.get('time_stretch_scale', 0.25)),
                proba=float(cfg.get('time_stretch_aug_proba', 0.5)),
            )
        )
    if cfg.get('time_warp_aug', False):
        augs.append(
            partial(
                time_warp_augment,
                magnitude=float(cfg.get('time_warp_magnitude', 0.2)),
                proba=float(cfg.get('time_warp_aug_proba', 0.5)),
            )
        )
    if cfg.get('blur_aug', False):
        augs.append(
            partial(
                blur_augment,
                sigma=float(cfg.get('blur_sigma', 0.8)),
                proba=float(cfg.get('blur_aug_proba', 0.5)),
            )
        )
    if cfg.get('scale_aug', False):
        augs.append(
            partial(
                scale_augment,
                scale_range=float(cfg.get('scale_aug_range', 0.15)),
                proba=float(cfg.get('scale_aug_proba', 0.5)),
            )
        )
    if cfg.get('bleach_aug', False):
        augs.append(
            partial(
                bleach_augment,
                strength=float(cfg.get('bleach_strength', 0.4)),
                proba=float(cfg.get('bleach_aug_proba', 0.5)),
            )
        )
    if cfg.get('poisson_aug', False):
        augs.append(
            partial(
                poisson_augment,
                scale=float(cfg.get('poisson_scale', 30.0)),
                proba=float(cfg.get('poisson_aug_proba', 0.5)),
            )
        )
    if cfg.get('haze_aug', False):
        augs.append(
            partial(
                haze_augment,
                amount=float(cfg.get('haze_amount', 0.1)),
                proba=float(cfg.get('haze_aug_proba', 0.5)),
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
    }


def train_from_config(cfg: dict) -> None:
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
