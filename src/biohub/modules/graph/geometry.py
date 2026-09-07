import json
import math
from typing import Any, cast

import blosc2
import numpy as np
import zarr


def node_point(upgrade, node: dict) -> tuple[float, float, float]:
    return (float(node['z']), float(node['y']), float(node['x']))


def point_distance_um(upgrade, a, b) -> float:
    dz = (a[0] - b[0]) * upgrade.voxel_scale_um[0]
    dy = (a[1] - b[1]) * upgrade.voxel_scale_um[1]
    dx = (a[2] - b[2]) * upgrade.voxel_scale_um[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def edge_distance_um(upgrade, source: dict, target: dict) -> float:
    return point_distance_um(upgrade, node_point(upgrade, source), node_point(upgrade, target))


def edge_sort_key(upgrade, edge: dict) -> tuple[float, float]:
    prob = edge.get('edge_prob')
    return (float(prob) if prob is not None else 0.0, -float(edge['distance_um']))


def coords_um(upgrade, nodes_by_id: dict, ids) -> np.ndarray:
    out = np.empty((len(ids), 3), dtype=np.float64)
    for index, node_id in enumerate(ids):
        node = nodes_by_id[node_id]
        out[index, 0] = node['z']
        out[index, 1] = node['y']
        out[index, 2] = node['x']
    return out * upgrade.scale


def next_node_id(upgrade, nodes_by_id: dict) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def ids_by_frame(upgrade, nodes_by_id: dict) -> dict[int, list[int]]:
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node['t']), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()
    return ids_by_t


def read_test_frame(upgrade, dataset: str, t: int, frame_cache: dict) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = upgrade.test_dir / f'{dataset}.zarr'
    meta = json.loads((zarr_path / '0' / 'zarr.json').read_text())
    frame_shape = tuple(int(value) for value in meta['shape'])[1:]
    dtype = np.dtype(meta['data_type'])
    frame = None
    try:
        raw = (zarr_path / '0' / 'c' / str(t) / '0' / '0' / '0').read_bytes()
        flat = np.frombuffer(cast(bytes, blosc2.decompress(raw)), dtype=dtype)
        if flat.size == int(np.prod(frame_shape)):
            frame = flat.reshape(frame_shape).copy()
    except Exception:
        frame = None
    if frame is None:
        zarr_arr: Any = zarr.open(zarr_path / '0', mode='r')
        frame = np.asarray(zarr_arr[t])
    frame_cache[t] = frame
    while len(frame_cache) > upgrade.frame_cache_max:
        frame_cache.pop(next(iter(frame_cache)))
    return frame


def sanitized_edge_probs(upgrade, edges: list[dict]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for edge in edges:
        prob = edge.get('edge_prob')
        if prob is None:
            continue
        try:
            value = float(prob)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        key = (int(edge['source_id']), int(edge['target_id']))
        if value > out.get(key, -1.0):
            out[key] = value
    return out
