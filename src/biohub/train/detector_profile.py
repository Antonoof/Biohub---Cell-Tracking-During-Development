"""Reproducible CPU/CUDA detector microprofile; never writes production weights."""

import argparse
import inspect
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from biohub.data.windows import FrameWindowDataset, collate_windows, load_dataset_windows
from biohub.losses.detection import gaussian_heatmap_target
from biohub.models import TemporalUNet3D, UNetNodeTransformer
from biohub.models.temporal_unet import unet_in_channels
from biohub.train import detector
from biohub.utils.seed import seed_everything


def measure(fn, device, repeats=5):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    fn()
    values = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        values.append(1000 * (time.perf_counter() - start))
    return statistics.median(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--movie', type=Path, required=True)
    parser.add_argument(
        '--config', type=Path, help='Profile this training recipe; CLI shape overrides it'
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--unet-layers', default='8,16')
    args = parser.parse_args(argv)
    cfg = detector.validate_config(yaml.safe_load(args.config.read_text())) if args.config else None
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (cfg['batch_size'] if cfg else 1)
    )
    args.window_size = (
        args.window_size if args.window_size is not None else (cfg['window_size'] if cfg else 2)
    )
    if min(args.steps, args.threads, args.batch_size, args.window_size) < 1:
        parser.error('steps, threads, batch-size and window-size must be positive')
    torch.set_num_threads(args.threads)
    seed_everything(42)
    device = detector.resolve_train_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        detector.configure_cuda_backends()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vm, windows = load_dataset_windows(
        args.movie, window_size=args.window_size, downsample=cfg['downsample'] if cfg else (1, 4, 4)
    )
    metadata_ms = 1000 * (time.perf_counter() - start)
    # Use the densest supervised windows (some movies only annotate one lineage).
    selected = sorted(windows, key=lambda w: sum(w.node_counts), reverse=True)[
        : max(args.steps * args.batch_size, args.batch_size)
    ]
    if len(selected) < args.steps * args.batch_size:
        parser.error('Movie has too few supervised windows for steps * batch-size')
    padding = cfg['batch_padding'] if cfg else True
    raw_ds = FrameWindowDataset([(vm, selected)], batch_padding=padding)
    cached_ds = FrameWindowDataset(
        [(vm, selected)],
        batch_padding=padding,
        frame_cache_mb=cfg['frame_cache_mb'] if cfg else 256,
        augmentations=detector._augmentations_from_cfg(cfg) if cfg else None,
    )
    sample = raw_ds[0]
    cpu = torch.device('cpu')
    result = {
        'device': str(device),
        'torch': torch.__version__,
        'threads': args.threads,
        'batch_size': args.batch_size,
        'window_size': args.window_size,
        'spatial': list(sample['imgs'].shape[1:]),
        'nodes': sample['node_counts'].tolist(),
        'metadata_ms': metadata_ms,
        'read_process_sample_ms': measure(lambda: raw_ds[0], cpu),
        'cached_process_sample_ms': measure(lambda: cached_ds[0], cpu),
    }
    batch = collate_windows([sample] * args.batch_size)
    if device.type == 'cuda':
        batch = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in batch.items()}
    result['collate_ms'] = measure(lambda: collate_windows([sample] * args.batch_size), cpu)
    result['transfer_ms'] = measure(
        lambda: [
            batch[k].to(device, non_blocking=True) for k in ('imgs', 'coords', 'masks', 'targets')
        ],
        device,
    )
    aug_cfg = {
        name: True
        for name in (
            'noise_aug',
            'gamma_aug',
            'contrast_aug',
            'rot90_aug',
            'translate_aug',
            'cutout_aug',
            'blur_aug',
            'bleach_aug',
            'poisson_aug',
            'haze_aug',
        )
    }
    result['augmentation_ms'] = {}
    for aug in detector._augmentations_from_cfg(aug_cfg):
        name = aug.func.__name__

        def run_aug(aug=aug):
            return aug(
                sample['imgs'],
                sample['coords'],
                sample['masks'],
                proba=1.0,
                rng=np.random.default_rng(123),
            )

        result['augmentation_ms'][name] = measure(run_aug, cpu)

    dense_coords = torch.rand(2, 128, 3) * 63
    dense_mask = torch.ones(2, 128, dtype=torch.bool)
    result['gaussian_2x128_ms'] = measure(
        lambda: gaussian_heatmap_target(dense_coords, dense_mask, (64, 64, 64)), cpu, repeats=3
    )
    q, k = torch.randn(16, 64, 32), torch.randn(16, 64, 32)
    edge = torch.randn(16, 64, 64)
    target = torch.eye(64).expand(16, -1, -1)
    result['aux_16x64_loop_ms'] = measure(
        lambda: [
            detector.division_aux_loss(edge[i], target[i])
            + detector.contrastive_aux_loss(q[i], k[i], target[i])
            for i in range(16)
        ],
        cpu,
    )
    result['aux_16x64_batch_ms'] = measure(
        lambda: detector.division_aux_loss(edge, target)
        + detector.contrastive_aux_loss(q, k, target),
        cpu,
    )

    model = UNetNodeTransformer(
        TemporalUNet3D(1, 8, layers=tuple(map(int, args.unet_layers.split(',')))),
        8,
        32,
        hidden_dim=args.hidden_dim,
        n_heads=4,
        n_blocks=2,
        dropout=0,
        pair_chunk_size=64,
    ).to(device)
    if cfg:
        unet_keys = inspect.signature(TemporalUNet3D).parameters
        node_keys = inspect.signature(UNetNodeTransformer).parameters
        in_keys = inspect.signature(unet_in_channels).parameters
        unet_kw = {k: v for k, v in cfg.items() if k in unet_keys}
        unet_kw.update(
            in_channels=unet_in_channels(**{k: cfg[k] for k in in_keys}),
            out_channels=cfg['unet_out_channels'],
            layers=cfg['unet_layers'],
            temporal_n_heads=cfg['unet_n_heads'],
        )
        model = UNetNodeTransformer(
            TemporalUNet3D(**unet_kw),
            pos_feat_dim=32,
            **{k: v for k, v in cfg.items() if k in node_keys},
        ).to(device)
    opt = detector.build_optimizer(
        model,
        name=cfg['optimizer'] if cfg else 'adamw',
        lr=cfg['lr'] if cfg else 1e-4,
        weight_decay=cfg['weight_decay'] if cfg else 0.01,
    )
    loader = DataLoader(
        cached_ds,
        batch_size=args.batch_size,
        collate_fn=collate_windows,
        pin_memory=device.type == 'cuda',
    )
    amp = (cfg['amp'] if cfg else 'bf16') if device.type == 'cuda' else 'off'
    kw: dict[str, Any] = dict(
        max_iters=args.steps,
        amp_kind=amp,
        train_peak_topk=4,
        aux_division_weight=0.1,
        aux_contrastive_weight=0.1,
        aux_offset_weight=0.1,
    )
    if cfg:
        kw.update(
            {
                k: v
                for k, v in cfg.items()
                if k in inspect.signature(detector.train_epoch).parameters
                and k not in ('device', 'optimizer', 'model', 'loader')
            }
        )
        kw.update(
            det_loss_kind=cfg['det_loss'],
            amp_kind=amp,
            max_iters=args.steps,
            scaler=torch.amp.GradScaler('cuda') if amp == 'fp16' else None,
            ema=detector.ModelEma(model, cfg['ema_decay']) if cfg['ema_decay'] > 0 else None,
        )
    result['config'] = str(args.config) if cfg else 'tiny_microbenchmark'
    result['loader_workers'] = (
        0  # Instrument workers separately: profiler scopes live in this process.
    )
    val_loader = DataLoader(
        raw_ds,
        batch_size=args.batch_size,
        collate_fn=collate_windows,
        pin_memory=device.type == 'cuda',
    )
    result['validation_ms_per_batch'] = measure(
        lambda: detector.evaluate(model, val_loader, device, amp_kind=amp), device, repeats=2
    ) / len(val_loader)
    if cfg:
        warmup_kw = dict(kw)
        warmup_kw['max_iters'] = 1
        detector.train_epoch(model, loader, opt, device, **warmup_kw)
    else:
        detector.train_epoch(model, loader, opt, device, max_iters=1, amp_kind=amp)

    def tagged(fn, name):
        def wrapped(*a, **k):
            with record_function(name):
                return fn(*a, **k)

        return wrapped

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with ExitStack() as stack:
        for obj, name, label in [
            (FrameWindowDataset, '__getitem__', 'data/getitem'),
            (model, 'encode_stacked', 'model/encoder'),
            (model, 'predict_edges_embeddings', 'model/association'),
            (detector, '_window_detections', 'processing/detect_match_index'),
            (detector, 'detection_loss', 'loss/detection'),
            (detector, 'compute_batch_loss', 'loss/association'),
            (detector, 'division_aux_loss', 'loss/division'),
            (detector, 'contrastive_aux_loss', 'loss/contrastive'),
            (detector, 'offset_aux_loss', 'loss/offset'),
            (torch.Tensor, 'backward', 'train/backward'),
            (opt, 'step', 'train/optimizer'),
        ]:
            stack.enter_context(patch.object(obj, name, tagged(getattr(obj, name), label)))
        start = time.perf_counter()
        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            detector.train_epoch(model, loader, opt, device, **kw)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        result['profiled_step_ms'] = (time.perf_counter() - start) * 1000 / args.steps
    result['scopes_ms_per_step'] = {
        item.key: item.cpu_time_total / 1000 / args.steps
        for item in prof.key_averages()
        if '/' in item.key and not item.key.startswith('autograd')
    }
    if device.type == 'cuda':
        result['peak_vram_bytes'] = torch.cuda.max_memory_allocated(device)
        result['scopes_device_ms_per_step'] = {
            item.key: item.device_time_total / 1000 / args.steps
            for item in prof.key_averages()
            if '/' in item.key and not item.key.startswith('autograd')
        }
    prof.export_chrome_trace(str(args.output / 'trace.json'))
    (args.output / 'operators.txt').write_text(
        prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40)
    )
    (args.output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
