#!/usr/bin/env python

import argparse
import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tracksdata as td
import zarr
from torch.utils.data import DataLoader

from biohub.data.deepcenter import (
    FullFrameDataset,
    FullFrameTrainingConfig,
    read_zarr_meta,
)
from biohub.losses.deepcenter import weighted_bce_loss
from biohub.models.deepcenter import DeepCenterUNet3D
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.utils.seed import dataloader_generator, seed_everything, seed_worker
from biohub.validation.cv import movie_group_fold_names

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def config_from_args(args: argparse.Namespace) -> FullFrameTrainingConfig:
    return FullFrameTrainingConfig(
        seed=args.seed,
        pool_factor=args.pool_factor,
        base_channels=args.base_channels,
        gauss_sigma=args.gauss_sigma,
        pos_thresh=args.pos_thresh,
        bg_quantile=args.bg_quantile,
        w_pos=args.w_pos,
        w_bg=args.w_bg,
        w_ignore=args.w_ignore,
        norm_lo_pct=args.norm_lo_pct,
        norm_hi_pct=args.norm_hi_pct,
        norm_clip_lo=args.norm_clip_lo,
        norm_clip_hi=args.norm_clip_hi,
        batch_size=args.batch_size,
        epochs=args.epochs,
        frames_per_movie=args.frames_per_movie,
        movie_limit=args.movie_limit,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        random_flip=not args.no_random_flip,
        brightness_jitter=args.brightness_jitter,
    )


def config_from_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> FullFrameTrainingConfig:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = dict(checkpoint.get('config', {}))
    valid_keys = {field.name for field in fields(FullFrameTrainingConfig)}
    defaults = asdict(FullFrameTrainingConfig())
    merged = {**defaults, **{k: v for k, v in saved.items() if k in valid_keys}}

    merged['epochs'] = args.epochs
    merged['batch_size'] = args.batch_size
    merged['num_workers'] = args.num_workers
    return FullFrameTrainingConfig(**merged)


def read_geff_nodes(geff_path: Path) -> dict[int, np.ndarray]:
    try:
        graph = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = graph[0] if isinstance(graph, tuple) else graph
        out: dict[int, list[list[float]]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            out.setdefault(int(row['t']), []).append(
                [float(row['z']), float(row['y']), float(row['x'])]
            )
        return {t: np.asarray(coords, dtype=np.float32) for t, coords in out.items() if len(coords)}
    except Exception:
        pass

    root = zarr.open_group(str(geff_path), mode='r')
    t_values = np.asarray(root['nodes/props/t/values'])
    z_values = np.asarray(root['nodes/props/z/values'])
    y_values = np.asarray(root['nodes/props/y/values'])
    x_values = np.asarray(root['nodes/props/x/values'])
    out_list: dict[int, list[list[float]]] = {}
    for t, z, y, x in zip(t_values, z_values, y_values, x_values):
        out_list.setdefault(int(t), []).append([float(z), float(y), float(x)])
    return {
        t: np.asarray(coords, dtype=np.float32) for t, coords in out_list.items() if len(coords)
    }


def discover_samples(data_dir: Path, cfg: FullFrameTrainingConfig) -> list[dict[str, Any]]:
    zarrs = sorted(data_dir.glob('*.zarr'))
    if cfg.movie_limit is not None:
        zarrs = zarrs[: int(cfg.movie_limit)]
    samples: list[dict[str, Any]] = []
    for zarr_path in zarrs:
        geff_path = data_dir / f'{zarr_path.stem}.geff'
        if not geff_path.exists():
            continue
        shape, dtype = read_zarr_meta(zarr_path)
        centers_by_t = read_geff_nodes(geff_path)
        if not centers_by_t:
            continue
        samples.append(
            {
                'name': zarr_path.stem,
                'zarr': zarr_path,
                'geff': geff_path,
                'shape': shape,
                'dtype': dtype,
                'centers_by_t': centers_by_t,
            }
        )
    if not samples:
        raise FileNotFoundError(f'No paired .zarr/.geff samples with labels found in {data_dir}')
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
    *,
    cv_mode: str = 'group_kfold',
    fold: int = 0,
    n_folds: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = [str(sample['name']) for sample in samples]
    if cv_mode == 'group_kfold' and len(set(names)) >= 2:
        train_names, val_names = movie_group_fold_names(names, fold, n_folds)
        train_set = set(train_names)
        val_set = set(val_names)
        train = [sample for sample in samples if str(sample['name']) in train_set]
        val = [sample for sample in samples if str(sample['name']) in val_set]
        if train:
            return train, val
    rng = random.Random(seed)
    by_embryo: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        embryo = str(sample['name']).split('_', 1)[0]
        by_embryo.setdefault(embryo, []).append(sample)
    embryos = sorted(by_embryo)
    rng.shuffle(embryos)
    n_val = max(1, int(round(len(embryos) * val_fraction))) if len(embryos) > 1 else 0
    val_embryos = set(embryos[:n_val])
    train = [
        sample for sample in samples if str(sample['name']).split('_', 1)[0] not in val_embryos
    ]
    val = [sample for sample in samples if str(sample['name']).split('_', 1)[0] in val_embryos]
    if not train and val:
        train, val = val, []
    return train, val


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: FullFrameTrainingConfig,
    epoch: int,
    best_score: float,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'config': asdict(cfg),
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': int(epoch),
            'best_score': float(best_score),
            'history': history,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    if 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return (
        int(checkpoint.get('epoch', 0)),
        float(checkpoint.get('best_score', float('-inf'))),
        list(checkpoint.get('history', [])),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, (image, target, weights) in enumerate(loader, start=1):
        image = image.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)
        logits = model(image)
        losses.append(float(weighted_bce_loss(logits, target, weights).detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return float(np.mean(losses)) if losses else float('nan')


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError('Expected at least one comma-separated threshold value.')
    return sorted(set(values))


def peak_distance_um(
    peak_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    pool_factor: int,
) -> np.ndarray:
    if gt_zyx.size == 0:
        return np.empty((0,), dtype=np.float32)
    xy_offset = (pool_factor - 1) / 2.0 if pool_factor > 1 else 0.0
    peak_orig = np.asarray(
        [
            peak_zyx[0],
            peak_zyx[1] * pool_factor + xy_offset,
            peak_zyx[2] * pool_factor + xy_offset,
        ],
        dtype=np.float32,
    )
    delta = (gt_zyx.astype(np.float32) - peak_orig[None, :]) * np.asarray(
        VOXEL_SCALE_UM, dtype=np.float32
    )
    return np.sqrt(np.sum(delta * delta, axis=1))


def find_heatmap_peaks(
    heatmap: np.ndarray,
    threshold: float,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(0, int(min_distance))
    tensor = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    if radius > 0:
        kernel = 2 * radius + 1
        local = F.max_pool3d(tensor, kernel_size=kernel, stride=1, padding=radius)
    else:
        local = tensor
    mask = (tensor == local) & (tensor >= float(threshold))
    coords = torch.nonzero(mask[0, 0], as_tuple=False).cpu().numpy()
    if coords.size == 0:
        return coords.reshape(0, 3).astype(np.float32), np.empty((0,), dtype=np.float32)
    scores = heatmap[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    order = np.argsort(-scores)
    return coords[order].astype(np.float32), scores[order].astype(np.float32)


def greedy_match_peaks(
    peaks: np.ndarray,
    scores: np.ndarray,
    gt_zyx: np.ndarray,
    pool_factor: int,
    match_radius_um: float,
) -> tuple[int, list[float], list[bool]]:
    if len(peaks) == 0:
        return 0, [], []
    used_gt: set[int] = set()
    nearest_distances: list[float] = []
    matched_flags: list[bool] = []
    for peak in peaks:
        distances = peak_distance_um(peak, gt_zyx, pool_factor)
        if distances.size == 0:
            nearest_distances.append(float('nan'))
            matched_flags.append(False)
            continue
        nearest_distances.append(float(np.min(distances)))
        order = np.argsort(distances)
        chosen = None
        for gt_idx in order:
            if int(gt_idx) in used_gt:
                continue
            if float(distances[gt_idx]) <= match_radius_um:
                chosen = int(gt_idx)
            break
        if chosen is None:
            matched_flags.append(False)
        else:
            used_gt.add(chosen)
            matched_flags.append(True)
    return len(used_gt), nearest_distances, matched_flags


@torch.no_grad()
def evaluate_gate_metrics(
    model: nn.Module,
    dataset: FullFrameDataset,
    cfg: FullFrameTrainingConfig,
    device: torch.device,
    output_dir: Path,
    thresholds: list[float],
    max_frames: int,
    peak_min_distance: int,
    match_radius_um: float,
    peak_sample_limit: int,
) -> dict[str, Any]:
    if max_frames <= 0 or len(dataset) == 0:
        return {}

    model.eval()
    n_eval = min(int(max_frames), len(dataset))
    if n_eval == len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = np.linspace(0, len(dataset) - 1, n_eval).round().astype(int).tolist()

    min_threshold = min(thresholds)
    xy_offset = (cfg.pool_factor - 1) / 2.0 if cfg.pool_factor > 1 else 0.0
    accum = {threshold: {'frames': 0, 'gt': 0, 'pred': 0, 'matched': 0} for threshold in thresholds}
    frame_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []

    batch_size = max(1, int(cfg.batch_size))
    eval_idx = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        images = []
        metas = []
        for dataset_idx in chunk:
            sample_idx, t = dataset.items[int(dataset_idx)]
            sample = dataset.samples[sample_idx]
            image, _, _ = dataset[int(dataset_idx)]
            images.append(image)
            metas.append((sample, int(t)))
        logits = model(torch.stack(images).to(device=device, dtype=torch.float32))
        heatmaps = torch.sigmoid(logits[:, 0]).detach().cpu().numpy().astype(np.float32)
        for heatmap, (sample, t) in zip(heatmaps, metas):
            eval_idx += 1
            gt_zyx = sample['centers_by_t'].get(t, np.empty((0, 3), dtype=np.float32))
            base_peaks, base_scores = find_heatmap_peaks(heatmap, min_threshold, peak_min_distance)
            base_match_count, base_nearest, base_matched = greedy_match_peaks(
                base_peaks,
                base_scores,
                gt_zyx,
                cfg.pool_factor,
                match_radius_um,
            )
            if len(peak_rows) < peak_sample_limit:
                remaining = peak_sample_limit - len(peak_rows)
                for peak, score, nearest, matched in zip(
                    base_peaks[:remaining],
                    base_scores[:remaining],
                    base_nearest[:remaining],
                    base_matched[:remaining],
                ):
                    peak_rows.append(
                        {
                            'dataset': sample['name'],
                            't': int(t),
                            'z': float(peak[0]),
                            'y': float(peak[1]),
                            'x': float(peak[2]),
                            'z_orig': float(peak[0]),
                            'y_orig': float(peak[1] * cfg.pool_factor + xy_offset),
                            'x_orig': float(peak[2] * cfg.pool_factor + xy_offset),
                            'score': float(score),
                            'nearest_label_um': nearest,
                            'matched_within_radius': int(matched),
                        }
                    )

            for threshold in thresholds:
                keep = base_scores >= float(threshold)
                peaks = base_peaks[keep]
                scores = base_scores[keep]
                matched_count, nearest_distances, matched_flags = greedy_match_peaks(
                    peaks,
                    scores,
                    gt_zyx,
                    cfg.pool_factor,
                    match_radius_um,
                )
                n_gt = int(len(gt_zyx))
                n_pred = int(len(peaks))
                precision = matched_count / n_pred if n_pred else float('nan')
                recall = matched_count / n_gt if n_gt else float('nan')
                nearest_arr = np.asarray(
                    [v for v in nearest_distances if np.isfinite(v)],
                    dtype=np.float32,
                )
                frame_rows.append(
                    {
                        'dataset': sample['name'],
                        't': int(t),
                        'threshold': float(threshold),
                        'n_gt_sparse': n_gt,
                        'n_pred': n_pred,
                        'n_matched': int(matched_count),
                        'precision_sparse': precision,
                        'recall_sparse': recall,
                        'score_mean': float(np.mean(scores)) if len(scores) else float('nan'),
                        'score_p90': float(np.percentile(scores, 90))
                        if len(scores)
                        else float('nan'),
                        'nearest_um_median': float(np.median(nearest_arr))
                        if nearest_arr.size
                        else float('nan'),
                    }
                )
                accum[threshold]['frames'] += 1
                accum[threshold]['gt'] += n_gt
                accum[threshold]['pred'] += n_pred
                accum[threshold]['matched'] += int(matched_count)

            if eval_idx == 1 or eval_idx % 25 == 0 or eval_idx == n_eval:
                print(f'gate-eval frame={eval_idx}/{n_eval}', flush=True)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = accum[threshold]
        n_pred = int(row['pred'])
        n_gt = int(row['gt'])
        matched = int(row['matched'])
        precision = matched / n_pred if n_pred else float('nan')
        recall = matched / n_gt if n_gt else float('nan')
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else float('nan')
        )
        threshold_rows.append(
            {
                'threshold': float(threshold),
                'frames': int(row['frames']),
                'n_gt_sparse': n_gt,
                'n_pred': n_pred,
                'n_matched': matched,
                'precision_sparse': precision,
                'recall_sparse': recall,
                'f1_sparse': f1,
                'pred_per_frame': n_pred / max(int(row['frames']), 1),
                'gt_per_frame': n_gt / max(int(row['frames']), 1),
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / 'gate_threshold_metrics.csv', threshold_rows)
    write_csv(output_dir / 'gate_frame_metrics.csv', frame_rows)
    write_csv(output_dir / 'gate_peak_samples.csv', peak_rows)

    summary = {
        'thresholds': thresholds,
        'max_frames': int(max_frames),
        'evaluated_frames': int(n_eval),
        'peak_min_distance': int(peak_min_distance),
        'match_radius_um': float(match_radius_um),
        'peak_sample_limit': int(peak_sample_limit),
        'threshold_metrics': threshold_rows,
        'notes': [
            'Sparse-label precision/recall are calibration features, not complete-cell metrics.',
            'Use high precision thresholds as conservative node-rescue gates.',
        ],
    }
    (output_dir / 'gate_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    return summary


def append_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'score', 'minutes']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        'path': dst.name,
        'bytes': dst.stat().st_size,
    }


def evaluate_best_checkpoint_gate(
    checkpoint_path: Path,
    cfg: FullFrameTrainingConfig,
    val_ds: FullFrameDataset,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    max_frames: int,
) -> dict[str, Any]:
    if max_frames <= 0:
        return {}
    if not checkpoint_path.exists():
        print(f'Gate evaluation skipped: {checkpoint_path} was not found.')
        return {}

    eval_model = DeepCenterUNet3D(base_channels=cfg.base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(checkpoint['model_state'])
    return evaluate_gate_metrics(
        model=eval_model,
        dataset=val_ds,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        thresholds=parse_float_list(args.gate_thresholds),
        max_frames=max_frames,
        peak_min_distance=args.gate_peak_min_distance,
        match_radius_um=args.gate_match_radius_um,
        peak_sample_limit=args.gate_peak_sample_limit,
    )


def write_epoch_snapshot(
    snapshot_root: Path,
    snapshot_prefix: str,
    output_dir: Path,
    epoch: int,
    cfg: FullFrameTrainingConfig,
    best_score: float,
    val_ds: FullFrameDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> Path:
    snapshot_dir = snapshot_root / f'{snapshot_prefix}{epoch:04d}'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Any] = {}
    for name in [
        'best.pt',
        'checkpoint_last.pt',
        'config.json',
        'history.csv',
        'split_manifest.json',
    ]:
        info = copy_if_exists(output_dir / name, snapshot_dir / name)
        if info is not None:
            copied[name] = info

    gate_frames = (
        args.snapshot_gate_eval_frames
        if args.snapshot_gate_eval_frames is not None
        else args.gate_eval_frames
    )
    gate_summary = evaluate_best_checkpoint_gate(
        checkpoint_path=output_dir / 'best.pt',
        cfg=cfg,
        val_ds=val_ds,
        device=device,
        output_dir=snapshot_dir,
        args=args,
        max_frames=int(gate_frames),
    )
    for name in [
        'gate_summary.json',
        'gate_threshold_metrics.csv',
        'gate_frame_metrics.csv',
        'gate_peak_samples.csv',
    ]:
        path = snapshot_dir / name
        if path.exists():
            copied[name] = {
                'path': name,
                'bytes': path.stat().st_size,
            }

    snapshot_manifest = {
        'snapshot_epoch': int(epoch),
        'snapshot_prefix': snapshot_prefix,
        'source_output_dir': str(output_dir),
        'best_score_so_far': float(best_score),
        'config': asdict(cfg),
        'gate_eval_frames': int(gate_frames),
        'gate_summary_written': bool(gate_summary),
        'files': copied,
        'notes': [
            'best.pt is the best validation-loss checkpoint observed up to this snapshot epoch.',
            'checkpoint_last.pt is the exact training state at this snapshot epoch.',
            'Gate diagnostics are computed from best.pt on the fixed validation split.',
        ],
    }
    (snapshot_dir / 'SNAPSHOT_MANIFEST.json').write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True) + '\n'
    )
    print(f'snapshot written: {snapshot_dir}', flush=True)
    return snapshot_dir


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    last_path = output_dir / 'checkpoint_last.pt'
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f'--resume requires {last_path}')
        cfg = config_from_checkpoint(last_path, args)
    else:
        cfg = config_from_args(args)

    seed_everything(int(cfg.seed), deterministic=bool(getattr(args, 'deterministic', False)))

    data_dir = args.data_dir.expanduser().resolve()
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f'{output_dir} is not empty. Use --resume or --overwrite intentionally.'
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.json').write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True) + '\n'
    )

    print('== Full-frame center detector training ==')
    print('data_dir:', data_dir)
    print('output_dir:', output_dir)
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    samples = discover_samples(data_dir, cfg)
    train_samples, val_samples = split_samples(
        samples,
        cfg.val_fraction,
        cfg.seed,
        cv_mode=str(args.cv_mode),
        fold=int(args.fold),
        n_folds=int(args.n_folds),
    )
    print(f'samples: total={len(samples)} train={len(train_samples)} val={len(val_samples)}')
    print('train sample examples:', [s['name'] for s in train_samples[:5]])
    print('val sample examples:', [s['name'] for s in val_samples[:5]])
    split_manifest = {
        'seed': int(cfg.seed),
        'val_fraction': float(cfg.val_fraction),
        'train': [str(sample['name']) for sample in train_samples],
        'val': [str(sample['name']) for sample in val_samples],
        'all': [str(sample['name']) for sample in samples],
    }
    (output_dir / 'split_manifest.json').write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + '\n'
    )

    train_ds = FullFrameDataset(train_samples, cfg, training=True)
    val_ds = FullFrameDataset(val_samples or train_samples[:1], cfg, training=False)
    print(f'frames: train={len(train_ds)} val={len(val_ds)}')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
        generator=dataloader_generator(cfg.seed),
        worker_init_fn=seed_worker if cfg.num_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, min(cfg.num_workers, 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=max(0, min(cfg.num_workers, 2)) > 0,
        drop_last=False,
        generator=dataloader_generator(cfg.seed + 1),
        worker_init_fn=seed_worker if max(0, min(cfg.num_workers, 2)) else None,
    )

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cuda':
        print('device:', torch.cuda.get_device_name(0))
    else:
        print('device:', device)

    model = DeepCenterUNet3D(base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_path = output_dir / 'best.pt'
    history_path = output_dir / 'history.csv'
    snapshot_root = (
        args.snapshot_dir.expanduser().resolve()
        if args.snapshot_dir is not None
        else output_dir.parent / f'{output_dir.name}_snapshots'
    )

    start_epoch = 0
    best_score = float('-inf')
    history: list[dict[str, float]] = []
    if args.resume:
        start_epoch, best_score, history = load_checkpoint(last_path, model, optimizer, device)
        print(f'Resumed from epoch {start_epoch}; best_score={best_score:.6f}')

    total_batches = len(train_loader)
    last_completed_epoch = start_epoch
    snapshotted_epochs: set[int] = set()
    writer = open_writer(output_dir)
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        seen_batches = 0
        for batch_idx, (image, target, weights) in enumerate(train_loader, start=1):
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            weights = weights.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = weighted_bce_loss(logits, target, weights)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            running += float(loss.detach().cpu())
            seen_batches += 1
            if (
                batch_idx == 1
                or batch_idx % args.progress_interval == 0
                or batch_idx == total_batches
            ):
                avg = running / max(seen_batches, 1)
                pct = 100.0 * batch_idx / max(total_batches, 1)
                print(
                    f'epoch={epoch}/{cfg.epochs} batch={batch_idx}/{total_batches} '
                    f'({pct:.1f}%) train_loss={avg:.6f}',
                    flush=True,
                )

        train_loss = running / max(seen_batches, 1)
        val_loss = evaluate(model, val_loader, device, args.val_batches)
        score = -val_loss if np.isfinite(val_loss) else -train_loss
        minutes = (time.time() - epoch_start) / 60.0
        row = {
            'epoch': float(epoch),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'score': float(score),
            'minutes': float(minutes),
        }
        history.append(row)
        append_history_csv(history_path, history)
        log_scalars(
            writer,
            epoch,
            {
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/score': score,
                'val/best_score': best_score,
            },
        )

        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, cfg, epoch, best_score, history)
            print(f'new best epoch={epoch} score={best_score:.6f} val_loss={val_loss:.6f}')

        save_checkpoint(last_path, model, optimizer, cfg, epoch, best_score, history)
        last_completed_epoch = epoch
        print(
            f'epoch={epoch} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} '
            f'best_score={best_score:.6f} minutes={minutes:.2f}',
            flush=True,
        )
        if args.snapshot_interval > 0 and epoch % args.snapshot_interval == 0:
            write_epoch_snapshot(
                snapshot_root=snapshot_root,
                snapshot_prefix=args.snapshot_prefix,
                output_dir=output_dir,
                epoch=epoch,
                cfg=cfg,
                best_score=best_score,
                val_ds=val_ds,
                device=device,
                args=args,
            )
            snapshotted_epochs.add(epoch)

    print('Training complete.')
    writer.close()
    print('best:', best_path)
    print('last:', last_path)

    if args.gate_eval_frames > 0:
        if not best_path.exists():
            print('Gate evaluation skipped: best.pt was not found.')
        else:
            print('Loading best checkpoint for gate evaluation:', best_path)
            gate_summary = evaluate_best_checkpoint_gate(
                checkpoint_path=best_path,
                cfg=cfg,
                val_ds=val_ds,
                device=device,
                output_dir=output_dir,
                args=args,
                max_frames=args.gate_eval_frames,
            )
            if gate_summary:
                print('Gate evaluation complete:')
                print(json.dumps(gate_summary['threshold_metrics'], indent=2))

    if (
        args.snapshot_final
        and last_completed_epoch > 0
        and last_completed_epoch not in snapshotted_epochs
    ):
        write_epoch_snapshot(
            snapshot_root=snapshot_root,
            snapshot_prefix=args.snapshot_prefix,
            output_dir=output_dir,
            epoch=last_completed_epoch,
            cfg=cfg,
            best_score=best_score,
            val_ds=val_ds,
            device=device,
            args=args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--pool-factor', type=int, default=4)
    parser.add_argument('--base-channels', type=int, default=24)
    parser.add_argument('--gauss-sigma', type=float, default=1.0)
    parser.add_argument('--pos-thresh', type=float, default=0.05)
    parser.add_argument('--bg-quantile', type=float, default=0.40)
    parser.add_argument('--w-pos', type=float, default=12.0)
    parser.add_argument('--w-bg', type=float, default=1.0)
    parser.add_argument('--w-ignore', type=float, default=0.05)
    parser.add_argument('--norm-lo-pct', type=float, default=50.0)
    parser.add_argument('--norm-hi-pct', type=float, default=99.5)
    parser.add_argument('--norm-clip-lo', type=float, default=-0.5)
    parser.add_argument('--norm-clip-hi', type=float, default=6.0)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames-per-movie', type=int, default=0)
    parser.add_argument('--movie-limit', type=int, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.10)
    parser.add_argument('--cv-mode', choices=('group_kfold', 'embryo'), default='group_kfold')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1.0e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--grad-clip-norm', type=float, default=None)
    parser.add_argument('--no-random-flip', action='store_true')
    parser.add_argument('--brightness-jitter', type=float, default=0.0)
    parser.add_argument('--progress-interval', type=int, default=50)
    parser.add_argument('--val-batches', type=int, default=24)
    parser.add_argument(
        '--gate-eval-frames',
        type=int,
        default=240,
        help='Validation frames to use for post-training gate calibration; set 0 to disable.',
    )
    parser.add_argument(
        '--gate-thresholds',
        default='0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80',
        help='Comma-separated heatmap thresholds for sparse GT calibration.',
    )
    parser.add_argument('--gate-peak-min-distance', type=int, default=1)
    parser.add_argument('--gate-match-radius-um', type=float, default=7.0)
    parser.add_argument('--gate-peak-sample-limit', type=int, default=50000)
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=0,
        help='Write best/last/config/history/split/gate diagnostics every N epochs; 0 disables.',
    )
    parser.add_argument(
        '--snapshot-dir',
        type=Path,
        default=None,
    )
    parser.add_argument('--snapshot-prefix', default='ep')
    parser.add_argument(
        '--snapshot-final',
        action='store_true',
    )
    parser.add_argument(
        '--snapshot-gate-eval-frames',
        type=int,
        default=None,
        help='Validation frames for snapshot gate diagnostics. Defaults to --gate-eval-frames.',
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)
