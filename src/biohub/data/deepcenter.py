import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import blosc2
import numpy as np
import torch
import zarr
from numcodecs import blosc
from torch.utils.data import Dataset


@dataclass
class FullFrameTrainingConfig:
    seed: int = 2026
    pool_factor: int = 4
    base_channels: int = 24

    gauss_sigma: float = 1.0
    pos_thresh: float = 0.05
    bg_quantile: float = 0.40
    w_pos: float = 12.0
    w_bg: float = 1.0
    w_ignore: float = 0.05

    norm_lo_pct: float = 50.0
    norm_hi_pct: float = 99.5
    norm_clip_lo: float = -0.5
    norm_clip_hi: float = 6.0

    batch_size: int = 8
    epochs: int = 50
    frames_per_movie: int = 0
    movie_limit: int | None = None
    val_fraction: float = 0.10
    num_workers: int = 4

    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    random_flip: bool = True
    brightness_jitter: float = 0.0


def read_zarr_meta(zarr_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    meta = json.loads((zarr_path / '0' / 'zarr.json').read_text())
    return tuple(int(v) for v in meta['shape']), np.dtype(meta['data_type']).newbyteorder('<')


def decompress_blosc(raw: bytes) -> bytes:
    try:
        out = blosc2.decompress(raw)
    except Exception:
        out = blosc.decompress(raw)
    return cast(bytes, out)


def read_frame(zarr_path: Path, t: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    frame_shape = shape[1:]
    chunk = zarr_path / '0' / 'c' / str(t) / '0' / '0' / '0'
    try:
        arr = np.frombuffer(decompress_blosc(chunk.read_bytes()), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    root: Any = zarr.open(zarr_path / '0', mode='r')
    return np.asarray(root[t])


def block_mean_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize_dynamic_range(
    volume: np.ndarray,
    lo_pct: float,
    hi_pct: float,
    clip_lo: float,
    clip_hi: float,
) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(ratio, clip_lo, clip_hi).astype(np.float32)


def make_heatmap(
    pooled_shape: tuple[int, int, int],
    centers_zyx: np.ndarray,
    pool_factor: int,
    sigma: float,
) -> np.ndarray:
    heatmap = np.zeros(pooled_shape, dtype=np.float32)
    if centers_zyx.size == 0:
        return heatmap
    radius = max(1, int(math.ceil(3.0 * sigma)))
    sigma2 = float(sigma) ** 2
    z_max, y_max, x_max = pooled_shape
    for z0, y0, x0 in centers_zyx:
        center = np.array([z0, y0 / pool_factor, x0 / pool_factor], dtype=np.float32)
        zc, yc, xc = [float(v) for v in center]
        z_start = max(0, int(math.floor(zc)) - radius)
        z_stop = min(z_max, int(math.floor(zc)) + radius + 2)
        y_start = max(0, int(math.floor(yc)) - radius)
        y_stop = min(y_max, int(math.floor(yc)) + radius + 2)
        x_start = max(0, int(math.floor(xc)) - radius)
        x_stop = min(x_max, int(math.floor(xc)) + radius + 2)
        if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
            continue
        zz = np.arange(z_start, z_stop, dtype=np.float32)[:, None, None]
        yy = np.arange(y_start, y_stop, dtype=np.float32)[None, :, None]
        xx = np.arange(x_start, x_stop, dtype=np.float32)[None, None, :]
        d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
        blob = np.exp(-0.5 * d2 / max(sigma2, 1e-6)).astype(np.float32)
        view = heatmap[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        np.maximum(view, blob, out=view)
    return heatmap


def positive_unlabeled_weight_map(
    image: np.ndarray,
    heatmap: np.ndarray,
    cfg: FullFrameTrainingConfig,
) -> np.ndarray:
    weights = np.full(heatmap.shape, cfg.w_ignore, dtype=np.float32)
    bg_cutoff = float(np.quantile(image, cfg.bg_quantile))
    weights[image < bg_cutoff] = cfg.w_bg
    weights[heatmap > cfg.pos_thresh] = cfg.w_pos
    return weights


def flip_together(rng: np.random.Generator, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    shape = arrays[0].shape
    axes = tuple(axis for axis in range(len(shape)) if rng.random() < 0.5)
    if axes:
        arrays = tuple(np.flip(arr, axis=axes) for arr in arrays)
    return tuple(np.ascontiguousarray(arr, dtype=np.float32) for arr in arrays)


class FullFrameDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        cfg: FullFrameTrainingConfig,
        training: bool,
    ) -> None:
        self.samples = samples
        self.cfg = cfg
        self.training = training
        self.items: list[tuple[int, int]] = []
        self._norm_cache: dict[tuple[int, int], np.ndarray] = {}
        rng = np.random.default_rng(cfg.seed + (0 if training else 10_000))
        for sample_idx, sample in enumerate(samples):
            n_t = int(sample['shape'][0])
            if cfg.frames_per_movie and cfg.frames_per_movie > 0 and cfg.frames_per_movie < n_t:
                frames = sorted(rng.choice(n_t, size=cfg.frames_per_movie, replace=False).tolist())
            else:
                frames = list(range(n_t))
            self.items.extend((sample_idx, int(t)) for t in frames)
        if not self.items:
            raise ValueError('No training frames were selected.')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx, t = self.items[index]
        sample = self.samples[sample_idx]
        cache_key = (sample_idx, int(t))
        image = self._norm_cache.get(cache_key)
        if image is None:
            frame = read_frame(sample['zarr'], t, sample['shape'], sample['dtype'])
            pooled = block_mean_xy(frame, self.cfg.pool_factor)
            image = normalize_dynamic_range(
                pooled,
                self.cfg.norm_lo_pct,
                self.cfg.norm_hi_pct,
                self.cfg.norm_clip_lo,
                self.cfg.norm_clip_hi,
            )
            self._norm_cache[cache_key] = image
        image = np.array(image, copy=True)
        target = make_heatmap(
            image.shape,
            sample['centers_by_t'].get(t, np.empty((0, 3), dtype=np.float32)),
            self.cfg.pool_factor,
            self.cfg.gauss_sigma,
        )
        weights = positive_unlabeled_weight_map(image, target, self.cfg)

        if self.training:
            rng = np.random.default_rng((self.cfg.seed + 1_000_003 * index) & 0xFFFFFFFF)
            if self.cfg.random_flip:
                image, target, weights = flip_together(rng, image, target, weights)
            if self.cfg.brightness_jitter > 0:
                scale = float(
                    rng.uniform(1.0 - self.cfg.brightness_jitter, 1.0 + self.cfg.brightness_jitter)
                )
                image = np.ascontiguousarray(image * scale, dtype=np.float32)

        return (
            torch.from_numpy(image[None, ...]),
            torch.from_numpy(target[None, ...]),
            torch.from_numpy(weights[None, ...]),
        )
