#!/usr/bin/env python3
"""Raw Biohub Zarr/GEFF data and sparse-safe targets for Tracker E.

This module intentionally has no dependency on A/B proposals, logits, graphs,
or post-processing.  It reads the competition volumes and sparse GEFF labels
directly and preserves unknown foreground as unknown.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


PRACTICE_STEMS = {
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
}


@dataclass(frozen=True)
class SparseGraph:
    ids: np.ndarray
    times: np.ndarray
    coords_zyx: np.ndarray
    edges: np.ndarray
    parent_of: dict[int, int]
    children_of: dict[int, tuple[int, ...]]
    row_of_id: dict[int, int]
    rows_by_t: dict[int, np.ndarray]


@dataclass(frozen=True)
class RawVideo:
    stem: str
    image_path: Path
    graph_path: Path
    shape_tzyx: tuple[int, int, int, int]
    scale_tzyx: tuple[float, float, float, float]
    q_low: float
    q_high: float
    estimated_number_of_nodes: int
    graph: SparseGraph

    @property
    def spacing_zyx(self) -> np.ndarray:
        return np.asarray(self.scale_tzyx[1:], dtype=np.float32)


@dataclass(frozen=True)
class WindowRef:
    video_index: int
    transition_t: int
    annotated_edges: int
    divisions: int


def _axis_scale(attrs: dict) -> tuple[float, float, float, float]:
    axes = attrs.get("multiscales", [{}])[0].get("datasets", [{}])[0]
    transformations = axes.get("coordinateTransformations", [])
    for item in transformations:
        if item.get("type") == "scale":
            value = tuple(float(v) for v in item["scale"])
            if len(value) == 4:
                return value
    raise ValueError("Zarr multiscale metadata has no four-axis scale")


def _quantiles(attrs: dict) -> tuple[float, float]:
    values = attrs.get("image_statistics", {}).get("quantiles", {})
    low = float(values.get("0.001", values.get("0.01", 0.0)))
    high = float(values.get("0.999", values.get("0.99", low + 1.0)))
    if not high > low:
        raise ValueError(f"Invalid image quantiles: {low}, {high}")
    return low, high


def read_sparse_graph(path: Path) -> SparseGraph:
    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    times = np.asarray(group["nodes/props/t/values"][:], dtype=np.int32)
    coords = np.column_stack(
        [
            np.asarray(group[f"nodes/props/{axis}/values"][:], dtype=np.float32)
            for axis in ("z", "y", "x")
        ]
    )
    edges = np.asarray(group["edges/ids"][:], dtype=np.int64)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    parent_of: dict[int, int] = {}
    child_lists: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        source_i, target_i = int(source), int(target)
        if target_i in parent_of and parent_of[target_i] != source_i:
            raise ValueError(f"{path}: GT child {target_i} has multiple parents")
        parent_of[target_i] = source_i
        child_lists[source_i].append(target_i)
    children_of = {key: tuple(value) for key, value in child_lists.items()}
    row_of_id = {int(node_id): row for row, node_id in enumerate(ids)}
    rows_by_t = {
        int(t): np.flatnonzero(times == t).astype(np.int64)
        for t in np.unique(times)
    }
    missing = sorted(
        {int(v) for v in edges.ravel()} - set(row_of_id)
    )
    if missing:
        raise ValueError(f"{path}: edge endpoints missing from nodes: {missing[:5]}")
    return SparseGraph(
        ids=ids,
        times=times,
        coords_zyx=coords,
        edges=edges,
        parent_of=parent_of,
        children_of=children_of,
        row_of_id=row_of_id,
        rows_by_t=rows_by_t,
    )


def load_video(data_dir: Path, stem: str) -> RawVideo:
    image_path = data_dir / f"{stem}.zarr"
    graph_path = data_dir / f"{stem}.geff"
    if not image_path.exists() or not graph_path.exists():
        raise FileNotFoundError(f"Missing raw pair for {stem}: {image_path}, {graph_path}")
    image_group = zarr.open_group(str(image_path), mode="r")
    image = image_group["0"]
    if len(image.shape) != 4:
        raise ValueError(f"{image_path}: expected TZYX array, got {image.shape}")
    graph_group = zarr.open_group(str(graph_path), mode="r")
    estimated_nodes = int(
        dict(graph_group.attrs)
        .get("geff", {})
        .get("extra", {})
        .get("estimated_number_of_nodes", 0)
    )
    return RawVideo(
        stem=stem,
        image_path=image_path,
        graph_path=graph_path,
        shape_tzyx=tuple(int(v) for v in image.shape),
        scale_tzyx=_axis_scale(dict(image_group.attrs)),
        q_low=_quantiles(dict(image_group.attrs))[0],
        q_high=_quantiles(dict(image_group.attrs))[1],
        estimated_number_of_nodes=estimated_nodes,
        graph=read_sparse_graph(graph_path),
    )


def list_stems(data_dir: Path) -> list[str]:
    image = {path.stem for path in data_dir.glob("*.zarr")}
    graph = {path.stem for path in data_dir.glob("*.geff")}
    missing_graph = sorted(image - graph)
    missing_image = sorted(graph - image)
    if missing_graph or missing_image:
        raise RuntimeError(
            f"Unpaired data: missing_graph={missing_graph[:5]} missing_image={missing_image[:5]}"
        )
    return sorted(image & graph)


def grouped_development_split(
    data_dir: Path,
    seed: int = 7301,
    held_per_embryo: int = 10,
) -> dict[str, list[str]]:
    """One deterministic video-grouped split with practice clips untouched."""
    all_stems = list_stems(data_dir)
    missing_practice = sorted(PRACTICE_STEMS - set(all_stems))
    if missing_practice:
        raise FileNotFoundError(f"Practice stems missing from raw training data: {missing_practice}")
    pool = sorted(set(all_stems) - PRACTICE_STEMS)
    held: list[str] = []
    for offset, embryo in enumerate(("44b6", "6bba")):
        group = [stem for stem in pool if stem.startswith(embryo + "_")]
        rng = random.Random(seed + 1009 * offset)
        rng.shuffle(group)
        if len(group) <= held_per_embryo:
            raise RuntimeError(f"Not enough {embryo} videos for held split: {len(group)}")
        held.extend(sorted(group[:held_per_embryo]))
    train = sorted(set(pool) - set(held))
    if len(train) + len(held) + len(PRACTICE_STEMS) != len(all_stems):
        raise AssertionError("Grouped split accounting failure")
    return {
        "train": train,
        "held": sorted(held),
        "practice": sorted(PRACTICE_STEMS),
    }


def _gaussian_3d(
    shape: tuple[int, int, int],
    center: np.ndarray,
    sigma: np.ndarray,
    truncate: float = 3.0,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    lo = np.maximum(0, np.floor(center - truncate * sigma).astype(int))
    hi = np.minimum(np.asarray(shape), np.ceil(center + truncate * sigma).astype(int) + 1)
    if np.any(hi <= lo):
        empty = np.zeros((0, 0, 0), dtype=np.float32)
        return (slice(0, 0),) * 3, empty
    zz, yy, xx = np.meshgrid(
        np.arange(lo[0], hi[0], dtype=np.float32),
        np.arange(lo[1], hi[1], dtype=np.float32),
        np.arange(lo[2], hi[2], dtype=np.float32),
        indexing="ij",
    )
    d2 = (
        ((zz - center[0]) / sigma[0]) ** 2
        + ((yy - center[1]) / sigma[1]) ** 2
        + ((xx - center[2]) / sigma[2]) ** 2
    )
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), np.exp(-0.5 * d2).astype(np.float32)


def make_sparse_center_targets(
    frames: np.ndarray,
    coords_by_frame: list[np.ndarray],
    raw_shape_zyx: tuple[int, int, int],
    downsample_zyx: tuple[int, int, int],
    spacing_zyx: np.ndarray,
    sigma_um: float,
    reliable_bg_threshold: float,
) -> dict[str, np.ndarray]:
    """Create positive, reliable-background, and unknown detector masks."""
    t_count, z_size, y_size, x_size = frames.shape
    shape = (z_size, y_size, x_size)
    ds = np.asarray(downsample_zyx, dtype=np.float32)
    coarse_spacing = spacing_zyx * ds
    sigma = np.maximum(float(sigma_um) / coarse_spacing, 0.75)
    heatmap = np.zeros((t_count, *shape), dtype=np.float32)
    offsets = np.zeros((t_count, 3, *shape), dtype=np.float32)
    offset_mask = np.zeros((t_count, 1, *shape), dtype=bool)

    for local_t, coords_raw in enumerate(coords_by_frame):
        for coord_raw in coords_raw:
            center = coord_raw.astype(np.float32) / ds
            slices, patch = _gaussian_3d(shape, center, sigma)
            if patch.size:
                heatmap[(local_t, *slices)] = np.maximum(heatmap[(local_t, *slices)], patch)
            anchor = np.rint(center).astype(int)
            anchor = np.minimum(np.maximum(anchor, 0), np.asarray(shape) - 1)
            offsets[local_t, :, anchor[0], anchor[1], anchor[2]] = center - anchor
            offset_mask[local_t, 0, anchor[0], anchor[1], anchor[2]] = True

    positive = heatmap >= 0.10
    # A voxel is reliable negative only when it is robustly dark and outside
    # every annotated Gaussian.  Bright unlabeled foreground remains unknown.
    reliable_background = (frames <= reliable_bg_threshold) & (heatmap < 1e-4)
    unknown = ~(positive | reliable_background)
    if np.any(positive & reliable_background) or np.any(positive & unknown) or np.any(reliable_background & unknown):
        raise AssertionError("Sparse detector masks overlap")
    if not np.all(positive | reliable_background | unknown):
        raise AssertionError("Sparse detector masks do not cover the volume")
    return {
        "heatmap": heatmap,
        "positive_mask": positive,
        "reliable_background_mask": reliable_background,
        "unknown_mask": unknown,
        "offset_target": offsets,
        "offset_mask": offset_mask,
    }


def _transition_counts(video: RawVideo, t: int) -> tuple[int, int]:
    edges = 0
    divisions = 0
    for source, children in video.graph.children_of.items():
        source_row = video.graph.row_of_id[source]
        if int(video.graph.times[source_row]) != t:
            continue
        next_children = [
            child for child in children
            if int(video.graph.times[video.graph.row_of_id[child]]) == t + 1
        ]
        edges += len(next_children)
        divisions += int(len(next_children) == 2)
    return edges, divisions


def _nodes_at(video: RawVideo, t: int) -> tuple[np.ndarray, np.ndarray]:
    rows = video.graph.rows_by_t.get(int(t), np.zeros(0, dtype=np.int64))
    return video.graph.ids[rows].astype(np.int64), video.graph.coords_zyx[rows].astype(np.float32)


def build_transition_targets(
    video: RawVideo,
    transition_t: int,
    candidate_k: int = 16,
    radius_um: float = 20.0,
    pair_k: int = 8,
    pair_cap: int = 64,
) -> dict[str, np.ndarray]:
    """Sparse-safe central-transition parent and division targets.

    Every candidate node in this Gate-2 curriculum is an annotated GT node.
    Therefore alternative annotated parents are confirmed-safe competitors.
    Unknown raw-image peaks are introduced only in the later self-proposal
    curriculum, where they remain unlabeled.
    """
    source_ids, source_coords = _nodes_at(video, transition_t)
    target_ids, target_coords = _nodes_at(video, transition_t + 1)
    spacing = video.spacing_zyx
    source_lookup = {int(node_id): index for index, node_id in enumerate(source_ids)}
    target_lookup = {int(node_id): index for index, node_id in enumerate(target_ids)}

    k = max(1, min(int(candidate_k), max(len(source_ids), 1)))
    parent_index = np.zeros((len(target_ids), k), dtype=np.int64)
    parent_mask = np.zeros((len(target_ids), k), dtype=bool)
    parent_delta = np.zeros((len(target_ids), k, 3), dtype=np.float32)
    parent_distance = np.zeros((len(target_ids), k), dtype=np.float32)
    parent_label = np.full(len(target_ids), -1, dtype=np.int64)

    if len(source_ids) and len(target_ids):
        delta = (source_coords[None] - target_coords[:, None]) * spacing[None, None]
        distance = np.linalg.norm(delta, axis=2)
        order = np.argsort(distance, axis=1)[:, :k]
        parent_index[:] = order
        parent_distance[:] = np.take_along_axis(distance, order, axis=1)
        parent_delta[:] = np.take_along_axis(delta, order[:, :, None], axis=1)
        parent_mask[:] = parent_distance <= radius_um
        for target_row, target_id in enumerate(target_ids):
            true_parent = video.graph.parent_of.get(int(target_id))
            if true_parent is None or true_parent not in source_lookup:
                continue
            true_index = source_lookup[true_parent]
            slots = np.flatnonzero((order[target_row] == true_index) & parent_mask[target_row])
            if len(slots):
                parent_label[target_row] = int(slots[0])

    event_label = np.full(len(source_ids), -1, dtype=np.int8)
    continuation_target = np.full(len(source_ids), -1, dtype=np.int64)
    pair_source: list[int] = []
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_label_group: list[int] = []
    pair_group_source: list[int] = []

    for source_row, source_id in enumerate(source_ids):
        children = [
            child for child in video.graph.children_of.get(int(source_id), ())
            if child in target_lookup
            and int(video.graph.times[video.graph.row_of_id[child]]) == transition_t + 1
        ]
        if len(children) == 1:
            event_label[source_row] = 0
            continuation_target[source_row] = target_lookup[children[0]]
        elif len(children) == 2:
            event_label[source_row] = 1

        if len(target_ids) < 2:
            continue
        distance = np.linalg.norm((target_coords - source_coords[source_row]) * spacing, axis=1)
        near = np.argsort(distance)[: min(pair_k, len(target_ids))]
        local_pairs: list[tuple[int, int, float]] = []
        for i_pos in range(len(near)):
            for j_pos in range(i_pos + 1, len(near)):
                a, b = int(near[i_pos]), int(near[j_pos])
                if distance[a] > radius_um or distance[b] > radius_um:
                    continue
                sister = float(np.linalg.norm((target_coords[a] - target_coords[b]) * spacing))
                local_pairs.append((a, b, float(distance[a] + distance[b] + 0.25 * sister)))
        local_pairs.sort(key=lambda item: item[2])
        local_pairs = local_pairs[:pair_cap]
        if not local_pairs:
            continue
        true_set = {int(v) for v in children} if len(children) == 2 else set()
        true_local = -1
        start = len(pair_source)
        for local_index, (a, b, _) in enumerate(local_pairs):
            pair_source.append(source_row)
            pair_a.append(a)
            pair_b.append(b)
            if true_set and {int(target_ids[a]), int(target_ids[b])} == true_set:
                true_local = local_index
        # Pair supervision exists only for annotated division sources.  The
        # pairs for continuations are retained as event alternatives but are
        # not independently labeled negative outside that confirmed group.
        if event_label[source_row] == 1:
            pair_group_source.append(source_row)
            pair_label_group.append(true_local)
        if len(pair_source) - start != len(local_pairs):
            raise AssertionError("Pair construction accounting error")

    return {
        "source_ids": source_ids,
        "source_coords_raw": source_coords,
        "target_ids": target_ids,
        "target_coords_raw": target_coords,
        "parent_index": parent_index,
        "parent_mask": parent_mask,
        "parent_delta_um": parent_delta,
        "parent_distance_um": parent_distance,
        "parent_label": parent_label,
        "event_label": event_label,
        "continuation_target": continuation_target,
        "pair_source": np.asarray(pair_source, dtype=np.int64),
        "pair_a": np.asarray(pair_a, dtype=np.int64),
        "pair_b": np.asarray(pair_b, dtype=np.int64),
        "pair_group_source": np.asarray(pair_group_source, dtype=np.int64),
        "pair_label_group": np.asarray(pair_label_group, dtype=np.int64),
    }


class TrackerEWindowDataset(Dataset):
    """On-demand raw five-frame windows with central-transition graph targets."""

    def __init__(
        self,
        videos: list[RawVideo],
        downsample_zyx: tuple[int, int, int] = (1, 4, 4),
        window: int = 5,
        sigma_um: float = 2.0,
        reliable_bg_threshold: float = 0.035,
        candidate_k: int = 16,
        radius_um: float = 20.0,
        division_sample_probability: float = 0.5,
        virtual_length: int | None = None,
        train: bool = True,
        seed: int = 7301,
    ) -> None:
        if window != 5:
            raise ValueError("Tracker E Gate-2 currently requires a five-frame window")
        self.videos = videos
        self.downsample = tuple(int(v) for v in downsample_zyx)
        self.window = window
        self.sigma_um = float(sigma_um)
        self.reliable_bg_threshold = float(reliable_bg_threshold)
        self.candidate_k = int(candidate_k)
        self.radius_um = float(radius_um)
        self.division_sample_probability = float(division_sample_probability)
        self.train = bool(train)
        self.seed = int(seed)
        self.refs: list[WindowRef] = []
        self.division_refs: list[int] = []
        for video_index, video in enumerate(videos):
            for t in range(video.shape_tzyx[0] - 1):
                edge_count, division_count = _transition_counts(video, t)
                if edge_count == 0:
                    continue
                index = len(self.refs)
                self.refs.append(WindowRef(video_index, t, edge_count, division_count))
                if division_count:
                    self.division_refs.append(index)
        if not self.refs:
            raise RuntimeError("No annotated transition windows")
        self.virtual_length = int(virtual_length or len(self.refs))

    def __len__(self) -> int:
        return self.virtual_length

    def _choose_ref(self, index: int) -> WindowRef:
        if not self.train:
            return self.refs[index % len(self.refs)]
        rng = random.Random(self.seed + index + random.randrange(1 << 20))
        if self.division_refs and rng.random() < self.division_sample_probability:
            return self.refs[rng.choice(self.division_refs)]
        return self.refs[rng.randrange(len(self.refs))]

    def __getitem__(self, index: int) -> dict:
        ref = self._choose_ref(index)
        video = self.videos[ref.video_index]
        t = ref.transition_t
        frame_ids = [min(max(t + delta, 0), video.shape_tzyx[0] - 1) for delta in (-2, -1, 0, 1, 2)]
        image = zarr.open_group(str(video.image_path), mode="r")["0"]
        dz, dy, dx = self.downsample
        raw = np.stack(
            [np.asarray(image[f, ::dz, ::dy, ::dx], dtype=np.float32) for f in frame_ids]
        )
        frames = np.clip((raw - video.q_low) / (video.q_high - video.q_low + 1e-6), 0.0, 1.5)
        coords_by_frame = [_nodes_at(video, f)[1] for f in frame_ids]
        detector = make_sparse_center_targets(
            frames=frames,
            coords_by_frame=coords_by_frame,
            raw_shape_zyx=video.shape_tzyx[1:],
            downsample_zyx=self.downsample,
            spacing_zyx=video.spacing_zyx,
            sigma_um=self.sigma_um,
            reliable_bg_threshold=self.reliable_bg_threshold,
        )
        transition = build_transition_targets(
            video, t, candidate_k=self.candidate_k, radius_um=self.radius_um
        )
        item: dict[str, object] = {
            "stem": video.stem,
            "transition_t": t,
            "frame_ids": np.asarray(frame_ids, dtype=np.int32),
            "frames": frames.astype(np.float32),
            "spacing_zyx": video.spacing_zyx,
            "downsample_zyx": np.asarray(self.downsample, dtype=np.float32),
        }
        item.update(detector)
        item.update(transition)
        return item


class RawUnlabeledWindowDataset(Dataset):
    """Cache-free raw windows for motion-masked encoder pretraining."""

    def __init__(
        self,
        videos: list[RawVideo],
        downsample_zyx: tuple[int, int, int] = (1, 4, 4),
        window: int = 5,
        virtual_length: int | None = None,
        train: bool = True,
        seed: int = 7301,
    ) -> None:
        self.videos = videos
        self.downsample = tuple(int(v) for v in downsample_zyx)
        self.window = int(window)
        self.train = bool(train)
        self.seed = int(seed)
        self.refs: list[tuple[int, int]] = []
        for video_index, video in enumerate(videos):
            for start in range(max(1, video.shape_tzyx[0] - self.window + 1)):
                self.refs.append((video_index, start))
        if not self.refs:
            raise RuntimeError("No raw pretraining windows")
        self.virtual_length = int(virtual_length or len(self.refs))
        self._arrays: dict[int, object] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def _array(self, video_index: int):
        if video_index not in self._arrays:
            self._arrays[video_index] = zarr.open_group(
                str(self.videos[video_index].image_path), mode="r"
            )["0"]
        return self._arrays[video_index]

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.train:
            rng = random.Random(self.seed + index * 104729)
            video_index, start = self.refs[rng.randrange(len(self.refs))]
        else:
            video_index, start = self.refs[index % len(self.refs)]
        video = self.videos[video_index]
        image = self._array(video_index)
        dz, dy, dx = self.downsample
        raw = np.asarray(
            image[start : start + self.window, ::dz, ::dy, ::dx], dtype=np.float32
        )
        if len(raw) != self.window:
            raise RuntimeError(
                f"{video.stem}: incomplete raw window start={start} shape={raw.shape}"
            )
        frames = np.clip(
            (raw - video.q_low) / (video.q_high - video.q_low + 1e-6), 0.0, 1.5
        ).astype(np.float32)
        return {
            "stem": video.stem,
            "start_t": start,
            "frames": frames,
        }


class SparseDetectorWindowDataset(Dataset):
    """Raw five-frame windows centered on a sparsely annotated frame."""

    def __init__(
        self,
        videos: list[RawVideo],
        downsample_zyx: tuple[int, int, int] = (1, 4, 4),
        sigma_um: float = 2.0,
        reliable_bg_threshold: float = 0.035,
        virtual_length: int | None = None,
        train: bool = True,
        seed: int = 7301,
    ) -> None:
        self.videos = videos
        self.downsample = tuple(int(v) for v in downsample_zyx)
        self.sigma_um = float(sigma_um)
        self.reliable_bg_threshold = float(reliable_bg_threshold)
        self.train = bool(train)
        self.seed = int(seed)
        self.refs: list[tuple[int, int]] = []
        self.refs_by_embryo: dict[str, list[int]] = defaultdict(list)
        for video_index, video in enumerate(videos):
            for center_t in sorted(video.graph.rows_by_t):
                index = len(self.refs)
                self.refs.append((video_index, int(center_t)))
                self.refs_by_embryo[video.stem.split("_")[0]].append(index)
        if not self.refs:
            raise RuntimeError("No sparsely annotated detector frames")
        self.virtual_length = int(virtual_length or len(self.refs))
        self._arrays: dict[int, object] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def _array(self, video_index: int):
        if video_index not in self._arrays:
            self._arrays[video_index] = zarr.open_group(
                str(self.videos[video_index].image_path), mode="r"
            )["0"]
        return self._arrays[video_index]

    def _reference(self, index: int) -> tuple[int, int]:
        if not self.train:
            return self.refs[index % len(self.refs)]
        rng = random.Random(self.seed + index * 130363)
        embryos = sorted(key for key, values in self.refs_by_embryo.items() if values)
        embryo = embryos[rng.randrange(len(embryos))]
        return self.refs[rng.choice(self.refs_by_embryo[embryo])]

    def __getitem__(self, index: int) -> dict[str, object]:
        video_index, center_t = self._reference(index)
        video = self.videos[video_index]
        frame_ids = [
            min(max(center_t + delta, 0), video.shape_tzyx[0] - 1)
            for delta in (-2, -1, 0, 1, 2)
        ]
        image = self._array(video_index)
        dz, dy, dx = self.downsample
        raw = np.stack(
            [np.asarray(image[t, ::dz, ::dy, ::dx], dtype=np.float32) for t in frame_ids]
        )
        frames = np.clip(
            (raw - video.q_low) / (video.q_high - video.q_low + 1e-6), 0.0, 1.5
        ).astype(np.float32)
        coords_by_frame = [_nodes_at(video, frame)[1] for frame in frame_ids]
        targets = make_sparse_center_targets(
            frames=frames,
            coords_by_frame=coords_by_frame,
            raw_shape_zyx=video.shape_tzyx[1:],
            downsample_zyx=self.downsample,
            spacing_zyx=video.spacing_zyx,
            sigma_um=self.sigma_um,
            reliable_bg_threshold=self.reliable_bg_threshold,
        )
        return {
            "stem": video.stem,
            "center_t": center_t,
            "frame_ids": np.asarray(frame_ids, dtype=np.int32),
            "frames": frames,
            "central_gt_coords_raw": coords_by_frame[2].astype(np.float32),
            "spacing_zyx": video.spacing_zyx,
            "downsample_zyx": np.asarray(self.downsample, dtype=np.float32),
            "estimated_nodes_per_frame": float(
                video.estimated_number_of_nodes / max(video.shape_tzyx[0], 1)
            ),
            **targets,
        }


def audit_data(data_dir: Path, stems: list[str] | None = None) -> dict:
    selected = stems or list_stems(data_dir)
    rows = []
    total_divisions = 0
    for stem in selected:
        video = load_video(data_dir, stem)
        degrees = Counter(int(source) for source, _ in video.graph.edges)
        divisions = sum(value == 2 for value in degrees.values())
        total_divisions += divisions
        rows.append(
            {
                "stem": stem,
                "shape": video.shape_tzyx,
                "spacing": video.scale_tzyx,
                "annotated_nodes": int(len(video.graph.ids)),
                "annotated_edges": int(len(video.graph.edges)),
                "annotated_divisions": int(divisions),
            }
        )
    return {
        "version": "tracker-e-raw-coordinate-audit-v1",
        "data_dir": str(data_dir),
        "videos": len(rows),
        "annotated_divisions": int(total_divisions),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_data(args.data, args.stems)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    print(text)


if __name__ == "__main__":
    main()
