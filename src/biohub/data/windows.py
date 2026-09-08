import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td
import zarr
from torch.utils.data import Dataset, default_collate

from biohub.data.volume import invert_time_graph, open_dataset
from biohub.features.position import extract_pos_features
from biohub.losses.association import compute_gt_transition_matrix
from biohub.losses.detection import gaussian_heatmap_target
from biohub.utils.seed import SharedEpoch, sample_numpy_rng


@dataclass(frozen=True)
class FrameWindowData:
    t_start: int
    n_frames: int
    pos_feats: list[torch.Tensor]
    coords: list[torch.Tensor]
    node_counts: list[int]
    targets: list[torch.Tensor]


@dataclass(frozen=True)
class VideoMeta:
    zarr_path: Path
    image_shape: tuple[int, ...]
    downsample: tuple[int, ...]
    voxel_size: tuple[float, ...]
    q_low: float
    q_high: float


_ZARR_ARRAYS: dict[str, Any] = {}


def _zarr_array(path: Path):
    key = str(path)
    array = _ZARR_ARRAYS.get(key)
    if array is None:
        array = zarr.open_group(key, mode='r')['0']
        _ZARR_ARRAYS[key] = array
    return array


def get_window_data(
    gt_graph: td.graph.BaseGraph,
    image_shape: tuple[int, ...],
    t_start: int,
    window_size: int = 2,
    downsample: tuple[int, ...] = (1, 1, 1),
    gt_attrs: pl.DataFrame | None = None,
    edge_attrs: pl.DataFrame | None = None,
    frame_cache: dict | None = None,
    transition_cache: dict | None = None,
) -> FrameWindowData | None:
    if gt_attrs is None:
        gt_attrs = gt_graph.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    if edge_attrs is None:
        edge_attrs = gt_graph.edge_attrs(attr_keys=['source_id', 'target_id'])

    ds = np.array(downsample, dtype=np.float32)
    frames = {} if frame_cache is None else frame_cache
    transitions = {} if transition_cache is None else transition_cache

    per_frame_ids: list[np.ndarray] = []
    pos_feats: list[torch.Tensor] = []
    coords_list: list[torch.Tensor] = []
    node_counts: list[int] = []

    for i in range(window_size):
        t = t_start + i
        if t in frames:
            ids, pos, coords = frames[t]
            per_frame_ids.append(ids)
            pos_feats.append(pos)
            coords_list.append(coords)
            node_counts.append(len(ids))
            continue
        gt_t = gt_attrs.filter(pl.col('t') == t)
        if len(gt_t) == 0:
            return None

        gt_coords_t = gt_t.select(['z', 'y', 'x']).to_numpy().astype(np.float32) / ds
        gt_ids = gt_t['node_id'].to_numpy()
        n_gt = len(gt_coords_t)

        full_coords = np.column_stack([np.full(n_gt, t, dtype=np.float32), gt_coords_t])

        pos_feats.append(torch.from_numpy(extract_pos_features(full_coords, image_shape)))
        coords_list.append(torch.from_numpy(gt_coords_t))
        per_frame_ids.append(gt_ids)
        node_counts.append(n_gt)
        frames[t] = (gt_ids, pos_feats[-1], coords_list[-1])

    targets: list[torch.Tensor] = []
    for i in range(window_size - 1):
        key = t_start + i
        if key not in transitions:
            transitions[key] = compute_gt_transition_matrix(
                per_frame_ids[i],
                per_frame_ids[i + 1],
                edge_attrs,
            )
        targets.append(transitions[key])

    return FrameWindowData(
        t_start=t_start,
        n_frames=window_size,
        pos_feats=pos_feats,
        coords=coords_list,
        node_counts=node_counts,
        targets=targets,
    )


def pad_window(
    window: FrameWindowData,
    max_nodes: int,
) -> dict[str, Any]:
    W = window.n_frames
    D = window.pos_feats[0].shape[1]
    M = max_nodes

    pos_feats = torch.zeros(W, M, D, dtype=torch.float32)
    coords = torch.zeros(W, M, 3, dtype=torch.float32)
    masks = torch.zeros(W, M, dtype=torch.bool)
    node_counts = torch.zeros(W, dtype=torch.long)

    for i in range(W):
        n = window.node_counts[i]
        pos_feats[i, :n] = window.pos_feats[i]
        coords[i, :n] = window.coords[i]
        masks[i, :n] = True
        node_counts[i] = n

    targets = torch.zeros(W - 1, M, M, dtype=torch.float32)
    for i in range(W - 1):
        nt = window.node_counts[i]
        nt1 = window.node_counts[i + 1]
        targets[i, :nt, :nt1] = window.targets[i]

    return {
        't_start': window.t_start,
        'n_frames': W,
        'pos_feats': pos_feats,
        'coords': coords,
        'masks': masks,
        'targets': targets,
        'node_counts': node_counts,
    }


def collate_windows(samples: list[dict]) -> dict:
    """Pad node axes to this batch, not the largest movie in the entire fold."""
    size = max(sample['coords'].shape[1] for sample in samples)
    padded = []
    for sample in samples:
        n = size - sample['coords'].shape[1]
        padded.append(
            {
                **sample,
                'coords': F.pad(sample['coords'], (0, 0, 0, n)),
                'pos_feats': F.pad(sample['pos_feats'], (0, 0, 0, n)),
                'masks': F.pad(sample['masks'], (0, n)),
                'targets': F.pad(sample['targets'], (0, n, 0, n)),
            }
        )
    return default_collate(padded)


def compact_window(meta: dict, coords: torch.Tensor, masks: torch.Tensor) -> dict:
    """Keep the prefix-mask contract and reindex both ends of every GT edge."""
    ids = [mask.nonzero(as_tuple=True)[0] for mask in masks]
    out_coords = torch.zeros_like(coords)
    out_pos = torch.zeros_like(meta['pos_feats'])
    out_masks = torch.zeros_like(masks)
    out_targets = torch.zeros_like(meta['targets'])
    for i, idx in enumerate(ids):
        n = len(idx)
        out_coords[i, :n] = coords[i, idx]
        out_pos[i, :n] = meta['pos_feats'][i, idx]
        out_masks[i, :n] = True
        if i < len(ids) - 1:
            out_targets[i, :n, : len(ids[i + 1])] = meta['targets'][i][idx][:, ids[i + 1]]
    return {
        **meta,
        'coords': out_coords,
        'pos_feats': out_pos,
        'masks': out_masks,
        'targets': out_targets,
        'node_counts': masks.sum(dim=1),
    }


class FrameWindowDataset(Dataset):
    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[FrameWindowData]]],
        max_nodes: int | None = None,
        augmentations: list | None = None,
        seed: int = 0,
        batch_padding: bool = False,
        frame_cache_mb: float = 0.0,
        heatmap_sigma: float | None = None,
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.max_nodes = max_nodes
        self.augmentations = augmentations or []
        self.seed = int(seed)
        self.epoch = SharedEpoch(0)
        self.batch_padding = batch_padding
        self.frame_cache_bytes = max(int(frame_cache_mb * 1024**2), 0)
        self._frames: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._frame_bytes = 0
        self.heatmap_sigma = heatmap_sigma

        self._data: list[tuple[FrameWindowData, VideoMeta]] = []
        for video_meta, windows in video_data:
            for window in windows:
                self._data.append((window, video_meta))

    def __len__(self) -> int:
        return len(self._data)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.set(epoch)

    def __getitem__(self, index):
        window, vm = self._data[index]
        size = max(max(window.node_counts), 1) if self.batch_padding else self.max_nodes
        meta = pad_window(window, size)
        t_start = meta['t_start']
        W = meta['n_frames']
        dz, dy, dx = vm.downsample

        z: Any = _zarr_array(vm.zarr_path)
        target_shape = list(vm.image_shape[1:])

        def normalize(raw):
            raw = raw.astype(np.float32)
            result = torch.from_numpy((raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)).clamp(0.0)
            if list(result.shape[1:]) != target_shape:
                result = F.interpolate(
                    result[:, None], size=target_shape, mode='trilinear', align_corners=False
                )[:, 0]
            return result

        if not self.frame_cache_bytes:
            imgs = normalize(z[t_start : t_start + W, ::dz, ::dy, ::dx])
        else:
            frames = []
            for t in range(t_start, t_start + W):
                key = (vm, t)
                frame = self._frames.get(key)
                if frame is None:
                    frame = normalize(z[t : t + 1, ::dz, ::dy, ::dx])[0].contiguous()
                    nbytes = frame.numel() * frame.element_size()
                    if nbytes <= self.frame_cache_bytes:
                        while self._frame_bytes + nbytes > self.frame_cache_bytes:
                            _, old = self._frames.popitem(last=False)
                            self._frame_bytes -= old.numel() * old.element_size()
                        self._frames[key] = frame
                        self._frame_bytes += nbytes
                else:
                    self._frames.move_to_end(key)
                frames.append(frame)
            # Own storage: an in-place augmentation must never modify cached frames.
            imgs = torch.stack(frames)

        if self.augmentations:
            rng = sample_numpy_rng(self.seed, self.epoch.get(), index)
            c, m = meta['coords'], meta['masks']
            for aug in self.augmentations:
                imgs, c, m = aug(imgs, c, m, rng=rng)
            if not torch.equal(m, meta['masks']):
                meta = compact_window(meta, c, m)
            else:
                meta = {**meta, 'coords': c, 'masks': m}

        if self.heatmap_sigma is not None:
            meta['heatmap_target'] = gaussian_heatmap_target(
                meta['coords'], meta['masks'], imgs.shape[1:], self.heatmap_sigma
            )
        return {
            **meta,
            'imgs': imgs.half(),
            'image_shape': torch.tensor(vm.image_shape, dtype=torch.long),
            'voxel_size': torch.tensor(vm.voxel_size, dtype=torch.float32),
            'downsample': torch.tensor(vm.downsample, dtype=torch.float32),
        }


def load_dataset_windows(
    ds_path: Path,
    window_size: int = 2,
    invert_time: bool = False,
    max_frames: int | None = None,
    downsample: tuple[int, ...] = (1, 1, 1),
) -> tuple[VideoMeta, list[FrameWindowData]]:
    ds = open_dataset(
        ds_path, normalize=False, require_tracks=True, load_image=False, downsample=downsample
    )
    if '0.001' not in ds.quantiles or '0.999' not in ds.quantiles:
        raise ValueError(f'Zarr attrs missing image_statistics.quantiles for {ds_path}')

    image_shape = ds.image_shape
    tracks = ds.tracks
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    if image_shape is None or tracks is None or ds.zarr_path is None:
        raise ValueError(f'Missing tracks or image shape for {ds_path}')

    if invert_time:
        tracks = invert_time_graph(tracks, max_t=image_shape[0])

    if max_frames is not None:
        image_shape = (min(max_frames, image_shape[0]), *image_shape[1:])
        tracks = tracks.filter(td.NodeAttr('t') < max_frames).subgraph()

    video_meta = VideoMeta(
        zarr_path=ds.zarr_path,
        image_shape=image_shape,
        downsample=downsample,
        voxel_size=voxel_size,
        q_low=float(ds.quantiles['0.001']),
        q_high=float(ds.quantiles['0.999']),
    )

    windows: list[FrameWindowData] = []
    gt_attrs = tracks.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    edge_attrs = tracks.edge_attrs(attr_keys=['source_id', 'target_id'])
    frame_cache: dict = {}
    transition_cache: dict = {}
    for t in range(image_shape[0] - window_size + 1):
        data = get_window_data(
            tracks,
            image_shape,
            t,
            window_size,
            downsample=downsample,
            gt_attrs=gt_attrs,
            edge_attrs=edge_attrs,
            frame_cache=frame_cache,
            transition_cache=transition_cache,
        )
        if data is not None:
            windows.append(data)

    # Diagnose quantization collisions once per movie, not with torch.unique
    # and GPU -> CPU synchronization on every training batch.
    collisions = 0
    for _, _, coords in frame_cache.values():
        voxels = np.clip(coords.numpy().astype(np.int64), 0, np.asarray(image_shape[1:]) - 1)
        collisions += len(voxels) - len(np.unique(voxels, axis=0))
    if collisions:
        warnings.warn(
            f'{ds_path.name}: {collisions} GT-node voxel collisions after downsampling '
            'across annotated frames; coincident nodes are not separately detectable.',
            stacklevel=2,
        )
    return video_meta, windows
