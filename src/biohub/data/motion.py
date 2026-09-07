import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import zarr
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from biohub.data.volume import open_dataset


@dataclass
class ProposalVideo:
    stem: str
    zarr_path: Path
    coords: np.ndarray
    offsets: np.ndarray
    fused_det_prob: np.ndarray
    member_det_prob: np.ndarray
    matches: np.ndarray
    gt_edges: set[tuple[int, int]]
    children: dict[int, tuple[int, ...]]
    edges_by_t: dict[int, tuple[tuple[int, int], ...]]
    outgoing: set[int]
    incoming: set[int]
    image_shape_raw: tuple[int, ...]
    image_shape_ds: tuple[int, ...]
    downsample: tuple[int, ...]
    voxel_scale_um: tuple[float, ...]
    q_low: float
    q_high: float
    frozen_sources: frozenset[int]


@dataclass(frozen=True)
class WindowRef:
    video: int
    t: int
    positives: int
    supervised_pairs: int
    embryo: str


def match_frame(
    proposals_zyx: np.ndarray,
    gt_ids: np.ndarray,
    gt_zyx: np.ndarray,
    scale_um: np.ndarray,
    max_um: float,
) -> np.ndarray:
    result = np.full(len(proposals_zyx), -1, dtype=np.int64)
    if not len(proposals_zyx) or not len(gt_zyx):
        return result
    p = proposals_zyx.astype(np.float64) * scale_um
    g = gt_zyx.astype(np.float64) * scale_um
    dist = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    good = dist[rows, cols] <= max_um
    result[rows[good]] = gt_ids[cols[good]]
    return result


def load_proposal_video(
    data_dir: Path, proposal_dir: Path, stem: str, match_um: float
) -> ProposalVideo:
    ppath = proposal_dir / f'{stem}.npz'
    if not ppath.exists():
        raise FileNotFoundError(ppath)
    with np.load(ppath) as item:
        coords = item['coords'].copy()
        offsets = item['frame_offsets'].copy()
        fused_det_prob = item['fused_det_prob'].copy()
        member_det_prob = item['member_det_prob'].copy()
        raw_shape = tuple(int(x) for x in item['image_shape'])
        downsample = tuple(int(x) for x in item['downsample'])
        scale = tuple(float(x) for x in item['voxel_scale_um'])
        frozen = frozenset(int(x) for x in item['frozen_sources'])

    ds = open_dataset(
        data_dir / stem,
        normalize=False,
        require_tracks=True,
        load_image=False,
        downsample=downsample,
    )
    if '0.001' not in ds.quantiles or '0.999' not in ds.quantiles:
        raise ValueError(f'{stem}: missing image quantiles')
    nodes = ds.tracks.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    edges_df = ds.tracks.edge_attrs(attr_keys=['source_id', 'target_id'])
    gt_edges = {
        (int(s), int(t)) for s, t in edges_df.select(['source_id', 'target_id']).iter_rows()
    }
    children_tmp: dict[int, list[int]] = {}
    for source, target in gt_edges:
        children_tmp.setdefault(source, []).append(target)
    children = {source: tuple(targets) for source, targets in children_tmp.items()}
    node_time = {int(nid): int(t) for nid, t in nodes.select(['node_id', 't']).iter_rows()}
    edges_by_t_tmp: dict[int, list[tuple[int, int]]] = {}
    for edge in gt_edges:
        edges_by_t_tmp.setdefault(node_time.get(edge[0], -1), []).append(edge)
    edges_by_t = {t: tuple(items) for t, items in edges_by_t_tmp.items()}
    outgoing = {s for s, _ in gt_edges}
    incoming = {t for _, t in gt_edges}
    T = raw_shape[0]
    matches = np.full(len(coords), -1, dtype=np.int64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    for t in range(T):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        gt = nodes.filter(pl.col('t') == t).sort('node_id')
        if len(gt):
            matches[lo:hi] = match_frame(
                coords[lo:hi, 1:],
                gt['node_id'].to_numpy().astype(np.int64),
                gt.select(['z', 'y', 'x']).to_numpy().astype(np.float32),
                scale_arr,
                match_um,
            )
    ds_shape = (T,) + tuple(int(math.ceil(s / d)) for s, d in zip(raw_shape[1:], downsample))
    return ProposalVideo(
        stem=stem,
        zarr_path=data_dir / f'{stem}.zarr',
        coords=coords,
        offsets=offsets,
        fused_det_prob=fused_det_prob,
        member_det_prob=member_det_prob,
        matches=matches,
        gt_edges=gt_edges,
        children=children,
        edges_by_t=edges_by_t,
        outgoing=outgoing,
        incoming=incoming,
        image_shape_raw=raw_shape,
        image_shape_ds=ds_shape,
        downsample=downsample,
        voxel_scale_um=scale,
        q_low=float(ds.quantiles['0.001']),
        q_high=float(ds.quantiles['0.999']),
        frozen_sources=frozen,
    )


def window_counts(video: ProposalVideo, t: int, max_nodes: int) -> tuple[int, int, int, int]:
    a0, a1 = int(video.offsets[t]), int(video.offsets[t + 1])
    b0, b1 = int(video.offsets[t + 1]), int(video.offsets[t + 2])
    ma = video.matches[a0 : min(a1, a0 + max_nodes)]
    mb = video.matches[b0 : min(b1, b0 + max_nodes)]
    row_active = np.fromiter((int(x) in video.outgoing for x in ma), bool, len(ma))
    col_active = np.fromiter((int(x) in video.incoming for x in mb), bool, len(mb))
    supervised = int(
        row_active.sum() * len(mb)
        + col_active.sum() * len(ma)
        - row_active.sum() * col_active.sum()
    )
    if not supervised:
        return len(ma), len(mb), 0, 0
    mb_set = {int(g) for g in mb if g >= 0}
    positives = sum(
        1 for src in ma if src >= 0 for dst in video.children.get(int(src), ()) if dst in mb_set
    )
    return len(ma), len(mb), int(positives), supervised


class ProposalWindowDataset(Dataset):
    def __init__(
        self,
        videos: list[ProposalVideo],
        max_nodes: int,
        train: bool,
        steps_per_epoch: int | None,
        batch_size: int,
        seed: int,
    ):
        self.videos = videos
        self.max_nodes = max_nodes
        self.train = train
        self.seed = seed
        self.windows: list[WindowRef] = []
        self.by_embryo: dict[str, list[int]] = {'44b6': [], '6bba': []}
        for vi, video in enumerate(videos):
            for t in range(video.image_shape_raw[0] - 1):
                _, _, pos, supervised = window_counts(video, t, max_nodes)
                if supervised:
                    idx = len(self.windows)
                    embryo = video.stem.split('_')[0]
                    self.windows.append(WindowRef(vi, t, pos, supervised, embryo))
                    self.by_embryo.setdefault(embryo, []).append(idx)
        if not self.windows:
            raise RuntimeError('No supervised proposal windows')
        self.virtual_len = (
            steps_per_epoch * batch_size if train and steps_per_epoch else len(self.windows)
        )
        group_counts = {k: len(v) for k, v in self.by_embryo.items()}
        print(
            f'{"train" if train else "val"}: {len(videos)} videos, '
            f'{len(self.windows)} supervised windows, epoch_len={self.virtual_len}, '
            f'by_embryo={group_counts}'
        )

    def __len__(self):
        return self.virtual_len

    def _choose(self, idx: int) -> WindowRef:
        if not self.train:
            return self.windows[idx]
        groups = [v for v in self.by_embryo.values() if v]
        group = random.choice(groups)
        return self.windows[random.choice(group)]

    def __getitem__(self, index: Any) -> Any:
        ref = self._choose(index)
        v = self.videos[ref.video]
        t = ref.t
        ranges = [
            (int(v.offsets[t]), int(v.offsets[t + 1])),
            (int(v.offsets[t + 1]), int(v.offsets[t + 2])),
        ]
        proposal_coords = []
        proposal_matches = []
        proposal_det = []
        proposal_member_det = []
        for lo, hi in ranges:
            hi = min(hi, lo + self.max_nodes)
            proposal_coords.append(v.coords[lo:hi, 1:].astype(np.float32))
            proposal_matches.append(v.matches[lo:hi])
            proposal_det.append(v.fused_det_prob[lo:hi].astype(np.float32))
            proposal_member_det.append(v.member_det_prob[lo:hi].astype(np.float32))

        n0, n1 = map(len, proposal_coords)
        target = np.zeros((n0, n1), dtype=np.float32)
        supervision = np.zeros((n0, n1), dtype=bool)
        row_active = np.fromiter((int(x) in v.outgoing for x in proposal_matches[0]), bool, n0)
        col_active = np.fromiter((int(x) in v.incoming for x in proposal_matches[1]), bool, n1)
        supervision |= row_active[:, None] | col_active[None, :]
        right = {int(g): j for j, g in enumerate(proposal_matches[1]) if g >= 0}
        for i, src in enumerate(proposal_matches[0]):
            if src < 0:
                continue
            for dst in v.children.get(int(src), ()):
                if dst in right:
                    target[i, right[dst]] = 1.0

        root: Any = zarr.open_group(str(v.zarr_path), mode='r')['0']
        dz, dy, dx = v.downsample
        raw = root[t : t + 2, ::dz, ::dy, ::dx].astype(np.float32)
        imgs = torch.from_numpy((raw - v.q_low) / (v.q_high - v.q_low + 1e-6)).clamp(0.0)

        coords = [
            torch.from_numpy(c / np.asarray(v.downsample, np.float32)) for c in proposal_coords
        ]
        if self.train:
            if random.random() < 0.5:
                imgs = imgs.flip(-1)
                for c in coords:
                    c[:, 2] = (v.image_shape_ds[3] - 1) - c[:, 2]
            if random.random() < 0.5:
                imgs = imgs.flip(-2)
                for c in coords:
                    c[:, 1] = (v.image_shape_ds[2] - 1) - c[:, 1]
            imgs = (imgs * random.uniform(0.9, 1.1) + random.uniform(-0.03, 0.03)).clamp(0.0)
        return {
            'imgs': imgs.half(),
            'coords0': coords[0],
            'coords1': coords[1],
            'det0': torch.from_numpy(proposal_det[0]),
            'det1': torch.from_numpy(proposal_det[1]),
            'member_det0': torch.from_numpy(proposal_member_det[0]),
            'member_det1': torch.from_numpy(proposal_member_det[1]),
            'target': torch.from_numpy(target),
            'supervision': torch.from_numpy(supervision),
            'downsample': torch.tensor(v.downsample, dtype=torch.float32),
            'voxel_scale': torch.tensor(v.voxel_scale_um, dtype=torch.float32),
            'image_shape': torch.tensor(v.image_shape_ds, dtype=torch.long),
            'gt_edges_total': torch.tensor(len(v.edges_by_t.get(t, ())), dtype=torch.long),
            'frozen': torch.tensor(t in v.frozen_sources),
            'video_idx': torch.tensor(ref.video, dtype=torch.long),
            'frame': torch.tensor(t, dtype=torch.long),
        }
