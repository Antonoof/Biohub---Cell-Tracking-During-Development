from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td
import zarr
from torch.utils.data import Dataset

from biohub.data.volume import invert_time_graph, open_dataset
from biohub.features.position import extract_pos_features
from biohub.losses.association import compute_gt_transition_matrix


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


def get_window_data(
    gt_graph: td.graph.BaseGraph,
    image_shape: tuple[int, ...],
    t_start: int,
    window_size: int = 2,
    downsample: tuple[int, ...] = (1, 1, 1),
) -> FrameWindowData | None:
    gt_attrs = gt_graph.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    edge_attrs = gt_graph.edge_attrs(attr_keys=['source_id', 'target_id'])

    ds = np.array(downsample, dtype=np.float32)

    per_frame_ids: list[np.ndarray] = []
    pos_feats: list[torch.Tensor] = []
    coords_list: list[torch.Tensor] = []
    node_counts: list[int] = []

    for i in range(window_size):
        t = t_start + i
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

    targets: list[torch.Tensor] = []
    for i in range(window_size - 1):
        targets.append(
            compute_gt_transition_matrix(
                per_frame_ids[i],
                per_frame_ids[i + 1],
                edge_attrs,
            )
        )

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


class FrameWindowDataset(Dataset):
    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[FrameWindowData]]],
        max_nodes: int | None = None,
        augmentations: list | None = None,
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.max_nodes = max_nodes
        self.augmentations = augmentations or []

        self._data: list[tuple[dict, VideoMeta]] = []
        for video_meta, windows in video_data:
            for window in windows:
                meta = pad_window(window, max_nodes)
                self._data.append((meta, video_meta))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        meta, vm = self._data[index]
        t_start = meta['t_start']
        W = meta['n_frames']
        dz, dy, dx = vm.downsample

        z: Any = zarr.open_group(str(vm.zarr_path), mode='r')['0']
        target_shape = list(vm.image_shape[1:])

        raw = z[t_start : t_start + W, ::dz, ::dy, ::dx].astype(np.float32)
        imgs = torch.from_numpy((raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)).clamp(0.0)

        if list(imgs.shape[1:]) != target_shape:
            imgs = F.interpolate(
                imgs[:, None],
                size=target_shape,
                mode='trilinear',
                align_corners=False,
            )[:, 0]

        if self.augmentations:
            rng = np.random.default_rng()
            c, m = meta['coords'], meta['masks']
            for aug in self.augmentations:
                imgs, c, m = aug(imgs, c, m, rng=rng)
            meta = {**meta, 'coords': c, 'masks': m}

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
        image_shape = (max_frames, *image_shape[1:])
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
    for t in range(image_shape[0] - window_size + 1):
        data = get_window_data(tracks, image_shape, t, window_size, downsample=downsample)
        if data is not None:
            windows.append(data)

    return video_meta, windows
