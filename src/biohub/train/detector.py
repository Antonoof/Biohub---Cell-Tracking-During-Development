import argparse
import json
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from biohub.augmentations import brightness_augment, flip_augment
from biohub.data.windows import (
    FrameWindowData,
    FrameWindowDataset,
    VideoMeta,
    load_dataset_windows,
)
from biohub.features.position import POS_EMBED_DIM, pos_embed_torch
from biohub.losses.association import (
    compute_batch_loss,
    evaluate_pair,
)
from biohub.losses.detection import compute_detection_loss
from biohub.models import TemporalUNet3D, UNetNodeTransformer
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.seed import seed_everything

DEFAULT_METHOD = 'unet_transformer'
DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def detect_and_match(
    det_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, ...],
    det_threshold: float = 0.3,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 5.0,
    voxel_size: tuple[float, ...] | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
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
        is_peak = (det_logits == pooled) & (det_logits > det_threshold)
        peak_idx = torch.nonzero(is_peak[:, 0])

    batch_ids = peak_idx[:, 0]
    peak_coords = peak_idx[:, 1:].float()

    nt_per_sample = mask.sum(dim=1).long()

    sample_matches: list[torch.Tensor] = []
    sample_coords: list[torch.Tensor] = []
    max_det = 0

    for b in range(B):
        sel = batch_ids == b
        det_b = peak_coords[sel]
        n_det = det_b.shape[0]
        nt = int(nt_per_sample[b].item())
        gt_b = gt_coords[b, :nt]
        n_gt = gt_b.shape[0]

        matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
        if n_det > 0 and n_gt > 0:
            if vs is not None:
                dists = torch.cdist(det_b * vs, gt_b * vs)
            else:
                dists = torch.cdist(det_b, gt_b)
            min_d, min_i = dists.min(dim=1)
            order = min_d.argsort()
            gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True

        sample_matches.append(matched)
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

    return padded_coords, padded_pos, padded_mask, sample_matches


def build_matched_edge_targets(
    match_t: list[torch.Tensor],
    match_t1: list[torch.Tensor],
    gt_target: torch.Tensor,
    max_det_t: int,
    max_det_t1: int,
) -> torch.Tensor:
    B = gt_target.shape[0]
    device = gt_target.device
    target = torch.zeros(B, max_det_t, max_det_t1, device=device)

    for b in range(B):
        mt = match_t[b]
        mt1 = match_t1[b]
        gt_trans = gt_target[b]
        n_t, n_t1 = mt.shape[0], mt1.shape[0]
        if n_t == 0 or n_t1 == 0:
            continue

        valid_t = mt >= 0
        valid_t1 = mt1 >= 0
        valid_mask = valid_t.unsqueeze(1) & valid_t1.unsqueeze(0)
        safe_t = mt.clamp(min=0)
        safe_t1 = mt1.clamp(min=0)
        block = gt_trans[safe_t][:, safe_t1] * valid_mask.float()
        target[b, :n_t, :n_t1] = block

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
) -> tuple[float, float]:
    model.train()
    total_edge_loss = 0.0
    total_det_loss = 0.0
    n_samples = 0

    if max_iters is not None:
        batch_iter = cycle(loader)
        pbar = tqdm(range(max_iters), desc='  iters', leave=False, disable=False)
    else:
        batch_iter = iter(loader)
        pbar = tqdm(range(len(loader)), desc='  batches', leave=False, disable=False)

    t_data, t_forward, t_backward = 0.0, 0.0, 0.0
    t0 = time.perf_counter()

    for _ in pbar:
        batch = next(batch_iter)

        imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch['coords'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        image_shape = tuple(batch['image_shape'][0].tolist())
        voxel_size = tuple(batch['voxel_size'][0].tolist())
        ds_scale = batch['downsample'][0].to(device)

        _cuda_sync()
        t1 = time.perf_counter()
        t_data += t1 - t0

        B, W = imgs.shape[:2]

        unet_out, det_logits = model.encode(imgs)

        det_losses = [
            compute_detection_loss(
                det_logits[i],
                coords[:, i],
                masks[:, i],
                det_neg_weight,
            )
            for i in range(W)
        ]
        det_loss = sum(det_losses) / W

        frame_det: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]
        ] = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i],
                coords[:, i],
                masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i,
                window_size=W,
            )
            unet_feat = model.index_features(
                unet_out[:, i],
                det_c,
                det_m,
            )
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

        block_losses = []
        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3],
                frame_det[i + 1][3],
                targets[:, i],
                ns,
                nt,
            )
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
                )
            )
        edge_loss = sum(block_losses) / len(block_losses)

        loss = edge_loss + det_loss_weight * det_loss

        _cuda_sync()
        t2 = time.perf_counter()
        t_forward += t2 - t1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _cuda_sync()
        t3 = time.perf_counter()
        t_backward += t3 - t2

        total_edge_loss += float(edge_loss) * B
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
) -> tuple[float, float, float]:
    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0

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
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]
        ] = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i],
                coords[:, i],
                masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i,
                window_size=W,
            )
            unet_feat = model.index_features(
                unet_out[:, i],
                det_c,
                det_m,
            )
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = (matches[b] >= 0).sum().item()
                gt_total += n_gt
                gt_matched += n_matched

        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3],
                frame_det[i + 1][3],
                targets[:, i],
                ns,
                nt,
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
                pair_loss, pair_correct, pair_total = evaluate_pair(
                    pair_logits[b, :ns_b, :nt_b],
                    pair_target[b, :ns_b, :nt_b],
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                n_pairs += 1

    node_recall = gt_matched / max(gt_total, 1)
    return total_loss / max(n_pairs, 1), correct / max(total, 1), node_recall


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
) -> UNetNodeTransformer:
    if unet_layers is None:
        unet_layers = [32, 64, 128]

    if debug_video is not None:
        train_files = test_files = [debug_video]
        print(f'Debug mode: using single video {debug_video.name}', flush=True)
    else:
        folds = json.loads(splits_file.read_text())
        fold_data = folds[fold]
        train_files = [data_dir / name for name in fold_data['train']]
        test_files = [data_dir / name for name in fold_data['test']]
        print(f'Fold {fold}: {len(train_files)} train, {len(test_files)} test', flush=True)

    output_dir = weights_dir / method / f'split_{fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = open_writer(output_dir)

    model_config = {
        'unet_out_channels': unet_out_channels,
        'unet_layers': unet_layers,
        'downsample': list(downsample),
        'window_size': window_size,
        'pool_kernel_um': pool_kernel_um,
    }
    (output_dir / 'config.json').write_text(json.dumps(model_config, indent=2))

    def _load(
        files: list[Path],
        desc: str,
    ) -> list[tuple[VideoMeta, list[FrameWindowData]]]:
        print(f'Loading {desc} ({len(files)} datasets)...', flush=True)
        data: list[tuple[VideoMeta, list[FrameWindowData]]] = []
        for f in tqdm(files, desc=desc, disable=False):
            video_meta, windows = load_dataset_windows(
                f,
                window_size=window_size,
                max_frames=max_frames,
                downsample=downsample,
            )
            data.append((video_meta, windows))
        n_windows = sum(len(w) for _, w in data)
        print(f'  {desc} done: {n_windows} windows total', flush=True)
        return data

    train_video_data = _load(train_files, 'train')
    test_video_data = _load(test_files, 'test')

    all_windows = [w for _, ws in train_video_data + test_video_data for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    print(f'max_nodes={max_nodes}', flush=True)

    pos_feat_dim = 4 * POS_EMBED_DIM

    train_ds = FrameWindowDataset(
        train_video_data, max_nodes=max_nodes, augmentations=augmentations
    )
    test_ds = FrameWindowDataset(test_video_data, max_nodes=max_nodes)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def worker_init_fn(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
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
        pin_memory=False,
        generator=g,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_visible = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f'Using device: {device} | visible CUDA GPUs: {n_visible}', flush=True)

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=unet_out_channels,
        layers=unet_layers,
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
    ).to(device)

    if data_parallel and device.type == 'cuda' and n_visible > 1:
        model.unet = nn.DataParallel(model.unet)
        print(
            f'DataParallel: UNet split across {n_visible} GPUs '
            f'(effective per-GPU batch {max(1, batch_size // n_visible)})',
            flush=True,
        )
    elif device.type == 'cuda':
        reason = '--single-gpu set' if not data_parallel else f'only {n_visible} GPU visible'
        print(
            f'Single-GPU training ({reason}). For 2 GPUs set the Kaggle accelerator to GPU T4 x2.',
            flush=True,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f'Starting training for {n_epochs} epochs (batch_size={batch_size})...', flush=True)

    best_score = 0.0
    save_path = output_dir / 'edge_predictor_best.pth'
    pbar = tqdm(range(n_epochs), desc='Training', disable=False)
    print(f'Detection loss: weight={det_loss_weight}, neg_weight={det_neg_weight}', flush=True)

    for epoch in pbar:
        t0 = time.monotonic()
        edge_loss, det_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            det_loss_weight,
            det_neg_weight,
            max_iters=max_iters,
            pool_kernel_um=pool_kernel_um,
        )
        train_time = time.monotonic() - t0

        t0 = time.monotonic()
        test_loss, test_acc, test_recall = evaluate(
            model, test_loader, device, pool_kernel_um=pool_kernel_um
        )
        test_time = time.monotonic() - t0

        score = test_acc * test_recall
        is_best = score >= best_score

        if is_best:
            best_score = score
            torch.save(
                {k.replace('unet.module.', 'unet.', 1): v for k, v in model.state_dict().items()},
                save_path,
            )

        marker = '*' if is_best else ' '
        log_scalars(
            writer,
            epoch,
            {
                'train/edge_loss': edge_loss,
                'train/det_loss': det_loss,
                'val/loss': test_loss,
                'val/acc': test_acc,
                'val/recall': test_recall,
                'val/score': score,
                'val/best_score': best_score,
            },
        )
        pbar.set_postfix(edge=f'{edge_loss:.4f}', det=f'{det_loss:.4f}', acc=f'{test_acc:.4f}')
        print(
            f'  Epoch {epoch:3d}/{n_epochs} | edge={edge_loss:.4f} | det={det_loss:.4f} | '
            f'test_loss={test_loss:.4f} | acc={test_acc:.4f} | '
            f'recall={test_recall:.4f} | best={best_score:.4f} {marker} | '
            f'train={train_time:.1f}s test={test_time:.1f}s',
            flush=True,
        )

    print(f'\nBest score (acc*recall): {best_score:.4f}, saved to {save_path}', flush=True)
    writer.close()
    if save_path.exists():
        state = torch.load(save_path, map_location=device, weights_only=True)
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
        augs.append(brightness_augment)
    if cfg.get('flip_aug', True):
        augs.append(flip_augment)
    return augs


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
    folds = [0] if debug_video is not None else (range(5) if str(split) == 'all' else [int(split)])
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

    folds = (
        [0] if debug_video is not None else (range(5) if args.split == 'all' else [int(args.split)])
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
            augmentations=_augmentations_from_cfg(
                {'brightness_aug': args.brightness_aug, 'flip_aug': args.flip_aug}
            ),
        )


if __name__ == '__main__':
    main()
