import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tracksdata as td
import zarr

SOURCE_FEATURES = [
    't_frac',
    'z_boundary',
    'y_frac',
    'x_frac',
    'in_degree',
    'out_degree',
    'incoming_dist',
    'outgoing_dist',
    'incoming_prob',
    'outgoing_prob',
    'velocity_mag',
    'acceleration_mag',
    'shift_mag',
    'density_10',
    'density_15',
    'children_10',
    'children_14',
]
for prefix in ('prev', 'now', 'next'):
    SOURCE_FEATURES += [
        f'{prefix}_mean',
        f'{prefix}_std',
        f'{prefix}_q90',
        f'{prefix}_max',
        f'{prefix}_fg_frac',
        f'{prefix}_elongation',
        f'{prefix}_center',
    ]
SOURCE_FEATURES += ['mean_next_minus_now', 'mean_now_minus_prev', 'elong_next_minus_now']

PAIR_GEOMETRY_FEATURES = [
    'parent_d1',
    'parent_d2',
    'parent_d_mean',
    'parent_d_max',
    'parent_d_asym',
    'sister_dist',
    'midpoint_offset',
    'daughter_angle_cos',
    'radial_sum',
    'child1_claimed',
    'child2_claimed',
    'child1_existing',
    'child2_existing',
    'same_target_component',
    'core_10_14',
    'rescue_annulus',
]
PAIR_IMAGE_FEATURES = []
for prefix in ('child1', 'child2'):
    PAIR_IMAGE_FEATURES += [
        f'{prefix}_mean',
        f'{prefix}_std',
        f'{prefix}_q90',
        f'{prefix}_max',
        f'{prefix}_fg_frac',
        f'{prefix}_elongation',
        f'{prefix}_center',
    ]
PAIR_IMAGE_FEATURES += [
    'child_mean_balance',
    'child_max_balance',
    'child_sum_to_parent',
    'midpoint_intensity',
    'line_mean',
    'line_min',
    'line_valley_ratio',
]
PAIR_FEATURES = SOURCE_FEATURES + PAIR_GEOMETRY_FEATURES + PAIR_IMAGE_FEATURES


def load_graph(path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def weak_components(graph) -> tuple[dict[int, int], dict[int, set[int]]]:
    ids = [int(n) for n in graph.node_ids()]
    parent = {n: n for n in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for s, t in graph.edge_attrs().select(['source_id', 'target_id']).iter_rows():
        union(int(s), int(t))
    comp_of: dict[int, int] = {}
    members: dict[int, set[int]] = defaultdict(set)
    for n in ids:
        root = find(n)
        comp_of[n] = root
        members[root].add(n)
    return comp_of, dict(members)


def graph_data(graph, spacing: np.ndarray):
    attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, 't', 'z', 'y', 'x']).sort(
        td.DEFAULT_ATTR_KEYS.NODE_ID
    )
    nodes = {int(r['node_id']): r for r in attrs.to_dicts()}
    ids_by_t: dict[int, list[int]] = defaultdict(list)
    pos: dict[int, np.ndarray] = {}
    for node_id, r in nodes.items():
        ids_by_t[int(r['t'])].append(node_id)
        pos[node_id] = np.asarray([r['z'], r['y'], r['x']], np.float64) * spacing
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    edge_info: dict[tuple[int, int], tuple[float, float]] = {}
    edges = graph.edge_attrs()
    for r in edges.to_dicts():
        s, t = int(r['source_id']), int(r['target_id'])
        outgoing[s].append(t)
        incoming[t].append(s)
        edge_info[(s, t)] = (float(r.get('edge_prob') or 0.0), float(r.get('edge_dist') or 0.0))
    return nodes, dict(ids_by_t), pos, dict(outgoing), dict(incoming), edge_info


def load_shifts(path: Path) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    rows: Any = frame.itertuples(index=False)
    return {
        int(r.t): np.asarray([r.shift_z_um, r.shift_y_um, r.shift_x_um], np.float64) for r in rows
    }


class VolumeReader:
    def __init__(self, path: Path, max_frames: int = 5):
        group: Any = zarr.open_group(path, mode='r')
        self.array: Any = group['0']
        attrs: Any = dict(group.attrs)
        stats: Any = attrs.get('image_statistics', {})
        q: Any = stats.get('quantiles', {}) if hasattr(stats, 'get') else {}
        self.lo = float(q.get('0.001', q.get('0.0', 0.0)))
        self.hi = float(q.get('0.999', q.get('1.0', 65535.0)))
        self.max_frames = max_frames
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def frame(self, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, self.array.shape[0] - 1))
        if t not in self.cache:
            raw = np.asarray(self.array[t], dtype=np.float32)
            self.cache[t] = np.clip((raw - self.lo) / max(self.hi - self.lo, 1.0), 0.0, 2.0)
            while len(self.cache) > self.max_frames:
                self.cache.popitem(last=False)
        value = self.cache.pop(t)
        self.cache[t] = value
        return value

    @property
    def shape(self):
        return self.array.shape


def crop(frame: np.ndarray, center_zyx: np.ndarray, rz: int = 3, ryx: int = 12) -> np.ndarray:
    z, y, x = [int(round(v)) for v in center_zyx]
    out = np.zeros((2 * rz + 1, 2 * ryx + 1, 2 * ryx + 1), np.float32)
    requested = ((z - rz, z + rz + 1), (y - ryx, y + ryx + 1), (x - ryx, x + ryx + 1))
    source = []
    target = []
    for (lo, hi), size in zip(requested, frame.shape):
        src_lo, src_hi = max(0, lo), min(size, hi)
        if src_hi <= src_lo:
            return out
        dst_lo = src_lo - lo
        source.append(slice(src_lo, src_hi))
        target.append(slice(dst_lo, dst_lo + src_hi - src_lo))
    out[tuple(target)] = frame[tuple(source)]
    return out


def patch_stats(patch: np.ndarray) -> np.ndarray:
    flat = patch.ravel()
    mean, std = float(flat.mean()), float(flat.std())
    q90, maxv = float(np.quantile(flat, 0.90)), float(flat.max())
    fg = float(np.mean(flat > max(0.18, mean + 0.35 * std)))
    center = float(patch[patch.shape[0] // 2, patch.shape[1] // 2, patch.shape[2] // 2])
    weights = np.clip(patch - np.quantile(flat, 0.55), 0, None)
    if float(weights.sum()) > 1e-6:
        zz, yy, xx = np.indices(patch.shape, dtype=np.float32)
        coords = np.column_stack([zz.ravel() * 1.625, yy.ravel() * 0.40625, xx.ravel() * 0.40625])
        w = weights.ravel()
        center_w = np.average(coords, axis=0, weights=w)
        delta = coords - center_w
        cov = (delta * w[:, None]).T @ delta / max(float(w.sum()), 1e-6)
        eig = np.linalg.eigvalsh(cov)
        elong = float(math.sqrt(max(eig[-1], 1e-6) / max(eig[0], 1e-6)))
    else:
        elong = 1.0
    return np.asarray([mean, std, q90, maxv, fg, min(elong, 20.0), center], np.float32)


def sample_intensity(frame: np.ndarray, point_vox: np.ndarray) -> float:
    z, y, x = [int(round(v)) for v in point_vox]
    z = int(np.clip(z, 0, frame.shape[0] - 1))
    y = int(np.clip(y, 0, frame.shape[1] - 1))
    x = int(np.clip(x, 0, frame.shape[2] - 1))
    return float(frame[z, y, x])


def line_features(frame: np.ndarray, a_vox: np.ndarray, b_vox: np.ndarray) -> tuple[float, float]:
    values = [
        sample_intensity(frame, (1 - u) * a_vox + u * b_vox) for u in np.linspace(0.15, 0.85, 9)
    ]
    return float(np.mean(values)), float(np.min(values))


def source_features(
    source: int,
    nodes,
    ids_by_t,
    pos,
    outgoing,
    incoming,
    edge_info,
    shifts,
    reader: VolumeReader,
    spacing: np.ndarray,
    image_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    node = nodes[source]
    t = int(node['t'])
    p = pos[source]
    shape = reader.shape
    inc = incoming.get(source, [])
    out = outgoing.get(source, [])
    in_dist = float(np.linalg.norm(p - pos[inc[0]])) if inc else 0.0
    out_dist = float(np.linalg.norm(pos[out[0]] - p)) if out else 0.0
    in_prob = edge_info.get((inc[0], source), (0.0, 0.0))[0] if inc else 0.0
    out_prob = edge_info.get((source, out[0]), (0.0, 0.0))[0] if out else 0.0
    velocity = p - pos[inc[0]] - shifts.get(t - 1, np.zeros(3)) if inc else np.zeros(3)
    acceleration = np.zeros(3)
    if inc and incoming.get(inc[0], []):
        grand = incoming[inc[0]][0]
        prev_vel = pos[inc[0]] - pos[grand] - shifts.get(t - 2, np.zeros(3))
        acceleration = velocity - prev_vel
    now_ids = ids_by_t.get(t, [])
    dnow = np.asarray([np.linalg.norm(pos[n] - p) for n in now_ids], np.float32)
    next_ids = ids_by_t.get(t + 1, [])
    shift = shifts.get(t, np.zeros(3))
    dnext = np.asarray([np.linalg.norm(pos[n] - p - shift) for n in next_ids], np.float32)
    base = [
        t / max(shape[0] - 1, 1),
        min(
            p[0] / max((shape[1] - 1) * spacing[0], 1),
            1 - p[0] / max((shape[1] - 1) * spacing[0], 1),
        ),
        float(node['y']) / max(shape[2] - 1, 1),
        float(node['x']) / max(shape[3] - 1, 1),
        len(inc),
        len(out),
        in_dist,
        out_dist,
        in_prob,
        out_prob,
        float(np.linalg.norm(velocity)),
        float(np.linalg.norm(acceleration)),
        float(np.linalg.norm(shift)),
        int(np.sum(dnow <= 10.0)) - 1,
        int(np.sum(dnow <= 15.0)) - 1,
        int(np.sum(dnext <= 10.0)),
        int(np.sum(dnext <= 14.0)),
    ]
    center = np.asarray([node['z'], node['y'], node['x']], np.float64)
    patch_values = []
    for dt, key in ((-1, 'prev'), (0, 'now'), (1, 'next')):
        cache_key = (source, dt)
        if cache_key not in image_cache:
            center_dt = center.copy()
            if dt == -1:
                center_dt -= shifts.get(t - 1, np.zeros(3)) / spacing
            elif dt == 1:
                center_dt += shift / spacing
            image_cache[cache_key] = patch_stats(crop(reader.frame(t + dt), center_dt))
        patch_values.append(image_cache[cache_key])
    prev, now, nxt = patch_values
    extra = [float(nxt[0] - now[0]), float(now[0] - prev[0]), float(nxt[5] - now[5])]
    return np.asarray([*base, *prev, *now, *nxt, *extra], np.float32)


def candidate_pairs(source, nodes, ids_by_t, pos, outgoing, incoming, shifts):
    t = int(nodes[source]['t'])
    shift = shifts.get(t, np.zeros(3))
    options = []
    for child in ids_by_t.get(t + 1, []):
        d = float(np.linalg.norm(pos[child] - pos[source] - shift))
        if d <= 14.0:
            options.append((child, d))
    pairs = []
    for i, (a, da) in enumerate(options):
        for b, db in options[i + 1 :]:
            sister = float(np.linalg.norm(pos[a] - pos[b]))
            if sister <= 20.0:
                score = da + db + 0.15 * sister
                pairs.append((score, a, b, da, db, sister))
    pairs.sort(key=lambda x: x[0])
    return pairs


def pair_features(
    source,
    a,
    b,
    da,
    db,
    sister,
    sf,
    nodes,
    pos,
    outgoing,
    incoming,
    shifts,
    comp_of,
    reader: VolumeReader,
    spacing: np.ndarray,
    child_cache: dict[int, np.ndarray],
) -> np.ndarray:
    t = int(nodes[source]['t'])
    shift = shifts.get(t, np.zeros(3))
    pred_parent = pos[source] + shift
    va, vb = pos[a] - pred_parent, pos[b] - pred_parent
    midpoint = 0.5 * (pos[a] + pos[b])
    cos = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-6))
    existing = set(outgoing.get(source, []))
    geom = np.asarray(
        [
            da,
            db,
            0.5 * (da + db),
            max(da, db),
            abs(da - db),
            sister,
            float(np.linalg.norm(midpoint - pred_parent)),
            cos,
            da + db,
            int(bool(incoming.get(a))),
            int(bool(incoming.get(b))),
            int(a in existing),
            int(b in existing),
            int(comp_of[a] == comp_of[b]),
            int(max(da, db) <= 10.0 and sister <= 14.0),
            int(max(da, db) > 10.0 or sister > 14.0),
        ],
        np.float32,
    )
    frame = reader.frame(t + 1)
    for child in (a, b):
        if child not in child_cache:
            n = nodes[child]
            child_cache[child] = patch_stats(
                crop(frame, np.asarray([n['z'], n['y'], n['x']], np.float64))
            )
    sa, sb = child_cache[a], child_cache[b]
    av = np.asarray([nodes[a]['z'], nodes[a]['y'], nodes[a]['x']], np.float64)
    bv = np.asarray([nodes[b]['z'], nodes[b]['y'], nodes[b]['x']], np.float64)
    midpoint_intensity = sample_intensity(frame, 0.5 * (av + bv))
    line_mean, line_min = line_features(frame, av, bv)
    parent_mean = float(sf[SOURCE_FEATURES.index('now_mean')])
    image = np.asarray(
        [
            *sa,
            *sb,
            abs(float(sa[0] - sb[0])),
            abs(float(sa[3] - sb[3])),
            float((sa[0] + sb[0]) / max(parent_mean, 1e-4)),
            midpoint_intensity,
            line_mean,
            line_min,
            float(line_min / max(0.5 * (sa[6] + sb[6]), 1e-4)),
        ],
        np.float32,
    )
    return np.concatenate([sf, geom, image]).astype(np.float32)
