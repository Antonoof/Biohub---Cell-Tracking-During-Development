#!/usr/bin/env python3
"""Train a sparse-safe division-parent classifier and daughter-pair ranker.

Supervision is metric-aligned:

* GT nodes with two children are positive division parents.
* GT nodes with exactly one annotated child are negative parents (the metric
  counts a predicted fork there as a false positive).
* Unmatched/unannotated detections are never used as negatives.
* Daughter-pair negatives are alternatives around an annotated parent; pair
  positives must connect the parent component to both annotated daughter
  lineages under the competition's 7 um matching.

Candidate generation uses a 10/14 um core and a 14/20 um rescue envelope.
The model sees graph/motion/geometry features and compact local-image features
that are all available at competition inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tracksdata as td
import zarr
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


SOURCE_FEATURES = [
    "t_frac", "z_boundary", "y_frac", "x_frac", "in_degree", "out_degree",
    "incoming_dist", "outgoing_dist", "incoming_prob", "outgoing_prob",
    "velocity_mag", "acceleration_mag", "shift_mag", "density_10", "density_15",
    "children_10", "children_14",
]
for prefix in ("prev", "now", "next"):
    SOURCE_FEATURES += [
        f"{prefix}_mean", f"{prefix}_std", f"{prefix}_q90", f"{prefix}_max",
        f"{prefix}_fg_frac", f"{prefix}_elongation", f"{prefix}_center",
    ]
SOURCE_FEATURES += ["mean_next_minus_now", "mean_now_minus_prev", "elong_next_minus_now"]

PAIR_GEOMETRY_FEATURES = [
    "parent_d1", "parent_d2", "parent_d_mean", "parent_d_max", "parent_d_asym",
    "sister_dist", "midpoint_offset", "daughter_angle_cos", "radial_sum",
    "child1_claimed", "child2_claimed", "child1_existing", "child2_existing",
    "same_target_component", "core_10_14", "rescue_annulus",
]
PAIR_IMAGE_FEATURES = []
for prefix in ("child1", "child2"):
    PAIR_IMAGE_FEATURES += [
        f"{prefix}_mean", f"{prefix}_std", f"{prefix}_q90", f"{prefix}_max",
        f"{prefix}_fg_frac", f"{prefix}_elongation", f"{prefix}_center",
    ]
PAIR_IMAGE_FEATURES += [
    "child_mean_balance", "child_max_balance", "child_sum_to_parent",
    "midpoint_intensity", "line_mean", "line_min", "line_valley_ratio",
]
PAIR_FEATURES = SOURCE_FEATURES + PAIR_GEOMETRY_FEATURES + PAIR_IMAGE_FEATURES


def argspec() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path("external/bio_track_repo"))
    p.add_argument("--data", type=Path, default=Path("data/train"))
    p.add_argument("--division-audit", type=Path, default=Path("data/division_candidate_audit_v1"))
    p.add_argument("--cache", type=Path, default=Path("data/division_training_cache_v1.npz"))
    p.add_argument("--output", type=Path, default=Path("data/division_pair_model_v1"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--stems", default="", help="Optional comma-separated smoke-test subset")
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--temporal-radius", type=int, default=2)
    p.add_argument("--max-pairs-positive-source", type=int, default=96)
    p.add_argument("--max-pairs-negative-source", type=int, default=24)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=2031)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


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

    for s, t in graph.edge_attrs().select(["source_id", "target_id"]).iter_rows():
        union(int(s), int(t))
    comp_of: dict[int, int] = {}
    members: dict[int, set[int]] = defaultdict(set)
    for n in ids:
        root = find(n)
        comp_of[n] = root
        members[root].add(n)
    return comp_of, dict(members)


def graph_data(graph, spacing: np.ndarray):
    attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"]
    ).sort(td.DEFAULT_ATTR_KEYS.NODE_ID)
    nodes = {int(r["node_id"]): r for r in attrs.to_dicts()}
    ids_by_t: dict[int, list[int]] = defaultdict(list)
    pos: dict[int, np.ndarray] = {}
    for node_id, r in nodes.items():
        ids_by_t[int(r["t"])].append(node_id)
        pos[node_id] = np.asarray([r["z"], r["y"], r["x"]], np.float64) * spacing
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    edge_info: dict[tuple[int, int], tuple[float, float]] = {}
    edges = graph.edge_attrs()
    for r in edges.to_dicts():
        s, t = int(r["source_id"]), int(r["target_id"])
        outgoing[s].append(t)
        incoming[t].append(s)
        edge_info[(s, t)] = (float(r.get("edge_prob") or 0.0), float(r.get("edge_dist") or 0.0))
    return nodes, dict(ids_by_t), pos, dict(outgoing), dict(incoming), edge_info


def load_shifts(path: Path) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    return {
        int(r.t): np.asarray([r.shift_z_um, r.shift_y_um, r.shift_x_um], np.float64)
        for r in frame.itertuples(index=False)
    }


class VolumeReader:
    def __init__(self, path: Path, max_frames: int = 5):
        group = zarr.open_group(path, mode="r")
        self.array = group["0"]
        q = dict(group.attrs).get("image_statistics", {}).get("quantiles", {})
        self.lo = float(q.get("0.001", q.get("0.0", 0.0)))
        self.hi = float(q.get("0.999", q.get("1.0", 65535.0)))
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
    values = [sample_intensity(frame, (1 - u) * a_vox + u * b_vox) for u in np.linspace(0.15, 0.85, 9)]
    return float(np.mean(values)), float(np.min(values))


def source_features(
    source: int, nodes, ids_by_t, pos, outgoing, incoming, edge_info, shifts,
    reader: VolumeReader, spacing: np.ndarray, image_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    node = nodes[source]
    t = int(node["t"])
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
        min(p[0] / max((shape[1] - 1) * spacing[0], 1), 1 - p[0] / max((shape[1] - 1) * spacing[0], 1)),
        float(node["y"]) / max(shape[2] - 1, 1), float(node["x"]) / max(shape[3] - 1, 1),
        len(inc), len(out), in_dist, out_dist, in_prob, out_prob,
        float(np.linalg.norm(velocity)), float(np.linalg.norm(acceleration)), float(np.linalg.norm(shift)),
        int(np.sum(dnow <= 10.0)) - 1, int(np.sum(dnow <= 15.0)) - 1,
        int(np.sum(dnext <= 10.0)), int(np.sum(dnext <= 14.0)),
    ]
    center = np.asarray([node["z"], node["y"], node["x"]], np.float64)
    patch_values = []
    for dt, key in ((-1, "prev"), (0, "now"), (1, "next")):
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
    t = int(nodes[source]["t"])
    shift = shifts.get(t, np.zeros(3))
    options = []
    for child in ids_by_t.get(t + 1, []):
        d = float(np.linalg.norm(pos[child] - pos[source] - shift))
        if d <= 14.0:
            options.append((child, d))
    pairs = []
    for i, (a, da) in enumerate(options):
        for b, db in options[i + 1:]:
            sister = float(np.linalg.norm(pos[a] - pos[b]))
            if sister <= 20.0:
                score = da + db + 0.15 * sister
                pairs.append((score, a, b, da, db, sister))
    pairs.sort(key=lambda x: x[0])
    return pairs


def pair_features(
    source, a, b, da, db, sister, sf, nodes, pos, outgoing, incoming, shifts,
    comp_of, reader: VolumeReader, spacing: np.ndarray, child_cache: dict[int, np.ndarray],
) -> np.ndarray:
    t = int(nodes[source]["t"])
    shift = shifts.get(t, np.zeros(3))
    pred_parent = pos[source] + shift
    va, vb = pos[a] - pred_parent, pos[b] - pred_parent
    midpoint = 0.5 * (pos[a] + pos[b])
    cos = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-6))
    existing = set(outgoing.get(source, []))
    geom = np.asarray([
        da, db, 0.5 * (da + db), max(da, db), abs(da - db), sister,
        float(np.linalg.norm(midpoint - pred_parent)), cos, da + db,
        int(bool(incoming.get(a))), int(bool(incoming.get(b))), int(a in existing), int(b in existing),
        int(comp_of[a] == comp_of[b]), int(max(da, db) <= 10.0 and sister <= 14.0),
        int(max(da, db) > 10.0 or sister > 14.0),
    ], np.float32)
    frame = reader.frame(t + 1)
    for child in (a, b):
        if child not in child_cache:
            n = nodes[child]
            child_cache[child] = patch_stats(crop(frame, np.asarray([n["z"], n["y"], n["x"]], np.float64)))
    sa, sb = child_cache[a], child_cache[b]
    av = np.asarray([nodes[a]["z"], nodes[a]["y"], nodes[a]["x"]], np.float64)
    bv = np.asarray([nodes[b]["z"], nodes[b]["y"], nodes[b]["x"]], np.float64)
    midpoint_intensity = sample_intensity(frame, 0.5 * (av + bv))
    line_mean, line_min = line_features(frame, av, bv)
    parent_mean = float(sf[SOURCE_FEATURES.index("now_mean")])
    image = np.asarray([
        *sa, *sb,
        abs(float(sa[0] - sb[0])), abs(float(sa[3] - sb[3])),
        float((sa[0] + sb[0]) / max(parent_mean, 1e-4)),
        midpoint_intensity, line_mean, line_min,
        float(line_min / max(0.5 * (sa[6] + sb[6]), 1e-4)),
    ], np.float32)
    return np.concatenate([sf, geom, image]).astype(np.float32)


def division_event_info(graph, gt_graph, comp_of):
    from biohub_tracking.division_metrics import _matched_node_attrs, extract_divisions, match_divisions
    gt_divs = extract_divisions(gt_graph)
    matched = match_divisions(graph, gt_graph, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
    events = []
    for gt_id, gt_div in gt_divs.items():
        attrs = _matched_node_attrs(matched[gt_id])
        children = gt_div.successors(gt_id)
        lineages = []
        for child in children:
            found = {int(child)}
            stack = [int(child)]
            while stack:
                for nxt in gt_div.successors(stack.pop()):
                    nxt = int(nxt)
                    if nxt not in found:
                        found.add(nxt); stack.append(nxt)
            lineages.append(found)
        gt_times = gt_div.node_attrs(attr_keys=["t"]).group_by("t").agg(pl.len().alias("n"))
        one_times = set(int(x) for x in gt_times.filter(pl.col("n") == 1)["t"].to_list())
        divider_t = int(gt_div.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t"]
        ).filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == int(gt_id))["t"][0])
        comp_one: dict[int, bool] = defaultdict(bool)
        comp_lineages: dict[int, int] = defaultdict(int)
        for row in attrs.iter_rows(named=True):
            pred = int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]); root = comp_of[pred]
            if int(row["t"]) in one_times: comp_one[root] = True
            matched_gt = int(row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
            for idx, lineage in enumerate(lineages):
                if matched_gt in lineage: comp_lineages[root] |= 1 << idx
        events.append({
            "gt_id": int(gt_id), "divider_t": divider_t,
            "parent_roots": {r for r, v in comp_one.items() if v},
            "comp_lineages": dict(comp_lineages),
        })
    return events


def build_cache(args) -> None:
    from biohub_tracking.division_metrics import _match_full
    from biohub_tracking.io import open_dataset

    metrics = pd.read_csv(args.division_audit.parent / "audit_907_v1" / "metrics_all.csv")
    stems = metrics.dataset.sort_values().tolist()
    if args.stems:
        wanted = {s.strip() for s in args.stems.split(",") if s.strip()}
        stems = [s for s in stems if s in wanted]
    if args.max_videos:
        stems = stems[:args.max_videos]

    source_x, source_y, source_embryo, source_dataset, source_node, source_event = [], [], [], [], [], []
    pair_x, pair_y, pair_source_idx, pair_event = [], [], [], []
    event_rows = []
    event_counter = 0
    parts_dir = args.cache.parent / f"{args.cache.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(stems, desc="division feature cache"):
        part_path = parts_dir / f"{stem}.pt"
        if part_path.exists() and not args.rebuild_cache:
            part = torch.load(part_path, map_location="cpu", weights_only=False)
            source_base, event_base = len(source_x), event_counter
            source_x.extend(part["source_x"]); source_y.extend(part["source_y"])
            source_embryo.extend(part["source_embryo"]); source_dataset.extend(part["source_dataset"])
            source_node.extend(part["source_node"])
            source_event.extend([event_base + int(e) if int(e) >= 0 else -1 for e in part["source_event"]])
            pair_x.extend(part["pair_x"]); pair_y.extend(part["pair_y"])
            pair_source_idx.extend([source_base + int(i) for i in part["pair_source_idx"]])
            pair_event.extend([event_base + int(e) if int(e) >= 0 else -1 for e in part["pair_event"]])
            for row in part["event_rows"]:
                item = dict(row); item["event_id"] = event_base + int(item["event_id"]); event_rows.append(item)
            event_counter += len(part["event_rows"])
            continue

        source_start, pair_start, event_start = len(source_x), len(pair_x), event_counter
        graph = load_graph(args.division_audit / "pre_safe_graphs" / f"{stem}.geff")
        ds = open_dataset(args.data / stem, normalize=False, require_tracks=True, load_image=False)
        spacing = np.asarray(ds.scale, np.float64)
        nodes, ids_by_t, pos, outgoing, incoming, edge_info = graph_data(graph, spacing)
        shifts = load_shifts(args.division_audit / "registration_shifts" / f"{stem}.csv")
        comp_of, members = weak_components(graph)
        reader = VolumeReader(args.data / f"{stem}.zarr")
        image_cache: dict[tuple[int, int], np.ndarray] = {}
        child_cache: dict[int, np.ndarray] = {}

        matched_full = _match_full(graph, ds.tracks, scale=ds.scale, max_distance=7.0)
        match_attrs = matched_full.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        )
        gt_to_pred = {
            int(r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]): int(r[td.DEFAULT_ATTR_KEYS.NODE_ID])
            for r in match_attrs.to_dicts()
            if r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID] is not None and int(r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]) != -1
        }
        events = division_event_info(graph, ds.tracks, comp_of)
        for event in events:
            event["event_id"] = event_counter
            event_rows.append({"event_id": event_counter, "dataset": stem, "embryo": stem.split("_", 1)[0], "gt_id": event["gt_id"]})
            event_counter += 1

        source_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            for root in event["parent_roots"]:
                for source in members[root]:
                    if abs(int(nodes[source]["t"]) - event["divider_t"]) > args.temporal_radius:
                        continue
                    pairs = candidate_pairs(source, nodes, ids_by_t, pos, outgoing, incoming, shifts)
                    for _, a, b, *_ in pairs:
                        mask = event["comp_lineages"].get(comp_of[a], 0) | event["comp_lineages"].get(comp_of[b], 0)
                        if mask.bit_count() >= 2:
                            source_events[source].append(event)
                            break

        labels: dict[int, int] = {}
        gt_attrs = ds.tracks.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID])
        for gt_id in gt_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list():
            gt_id = int(gt_id)
            degree = ds.tracks.out_degree(gt_id)
            if degree not in (1, 2) or gt_id not in gt_to_pred:
                continue
            pred = gt_to_pred[gt_id]
            labels[pred] = max(labels.get(pred, 0), int(degree >= 2))
        for source in source_events:
            labels[source] = 1

        for source, label in labels.items():
            sf = source_features(source, nodes, ids_by_t, pos, outgoing, incoming, edge_info, shifts, reader, spacing, image_cache)
            idx = len(source_x)
            events_here = source_events.get(source, [])
            source_x.append(sf); source_y.append(label); source_embryo.append(stem.split("_", 1)[0])
            source_dataset.append(stem); source_node.append(source)
            source_event.append(events_here[0]["event_id"] if events_here else -1)

            # A positive parent with no recoverable broad pair is useful to the
            # parent classifier but must not generate false pair negatives.
            if label and not events_here:
                continue
            candidates = candidate_pairs(source, nodes, ids_by_t, pos, outgoing, incoming, shifts)
            pair_rows = []
            for score, a, b, da, db, sister in candidates:
                enabled = []
                for event in events_here:
                    mask = event["comp_lineages"].get(comp_of[a], 0) | event["comp_lineages"].get(comp_of[b], 0)
                    if mask.bit_count() >= 2:
                        enabled.append(event["event_id"])
                y = int(bool(enabled))
                event_id = enabled[0] if enabled else -1
                pair_rows.append((score, y, event_id, a, b, da, db, sister))
            positives = [r for r in pair_rows if r[1]]
            negatives = [r for r in pair_rows if not r[1]]
            cap = args.max_pairs_positive_source if label else args.max_pairs_negative_source
            keep = positives + negatives[:max(0, cap - len(positives))]
            for _, y, event_id, a, b, da, db, sister in keep:
                pair_x.append(pair_features(source, a, b, da, db, sister, sf, nodes, pos, outgoing, incoming, shifts, comp_of, reader, spacing, child_cache))
                pair_y.append(y); pair_source_idx.append(idx); pair_event.append(event_id)

        part = {
            "source_x": np.asarray(source_x[source_start:], np.float32),
            "source_y": np.asarray(source_y[source_start:], np.uint8),
            "source_embryo": list(source_embryo[source_start:]),
            "source_dataset": list(source_dataset[source_start:]),
            "source_node": np.asarray(source_node[source_start:], np.int64),
            "source_event": np.asarray([
                int(e) - event_start if int(e) >= 0 else -1 for e in source_event[source_start:]
            ], np.int32),
            "pair_x": np.asarray(pair_x[pair_start:], np.float32),
            "pair_y": np.asarray(pair_y[pair_start:], np.uint8),
            "pair_source_idx": np.asarray([
                int(i) - source_start for i in pair_source_idx[pair_start:]
            ], np.int32),
            "pair_event": np.asarray([
                int(e) - event_start if int(e) >= 0 else -1 for e in pair_event[pair_start:]
            ], np.int32),
            "event_rows": [
                {**row, "event_id": int(row["event_id"]) - event_start}
                for row in event_rows[event_start:]
            ],
        }
        tmp_path = part_path.with_suffix(".pt.tmp")
        torch.save(part, tmp_path)
        tmp_path.replace(part_path)

    if not source_x or not pair_x:
        raise RuntimeError("Feature cache is empty; choose a subset containing annotated tracks and divisions")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        source_x=np.asarray(source_x, np.float32), source_y=np.asarray(source_y, np.uint8),
        source_embryo=np.asarray(source_embryo), source_dataset=np.asarray(source_dataset),
        source_node=np.asarray(source_node, np.int64), source_event=np.asarray(source_event, np.int32),
        pair_x=np.asarray(pair_x, np.float32), pair_y=np.asarray(pair_y, np.uint8),
        pair_source_idx=np.asarray(pair_source_idx, np.int32), pair_event=np.asarray(pair_event, np.int32),
        event_id=np.asarray([r["event_id"] for r in event_rows], np.int32),
        event_dataset=np.asarray([r["dataset"] for r in event_rows]),
        event_embryo=np.asarray([r["embryo"] for r in event_rows]),
        source_feature_names=np.asarray(SOURCE_FEATURES), pair_feature_names=np.asarray(PAIR_FEATURES),
    )
    print(
        f"Cache: sources={len(source_y):,} positives={sum(source_y):,}; "
        f"pairs={len(pair_y):,} positives={sum(pair_y):,}; events={len(event_rows):,} -> {args.cache}",
        flush=True,
    )


class MLP(nn.Module):
    def __init__(self, n: int, hidden: tuple[int, int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden[0]), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(hidden[0], hidden[1]), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def balanced_loss(logits, y):
    pos = torch.clamp(y.sum(), min=1.0)
    neg = torch.clamp((1 - y).sum(), min=1.0)
    pos_weight = torch.clamp(neg / pos, 1.0, 30.0)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    return (((1 - pt) ** 1.5) * bce).mean()


@torch.no_grad()
def predict(model, x, mean, std, device, batch=8192):
    model.eval(); out = []
    for start in range(0, len(x), batch):
        xb = torch.from_numpy((x[start:start + batch] - mean) / std).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def max_matching(events: list[int], edges: dict[int, set[int]]) -> int:
    match_r: dict[int, int] = {}
    def aug(e, seen):
        for source in edges.get(e, set()):
            if source in seen: continue
            seen.add(source)
            if source not in match_r or aug(match_r[source], seen):
                match_r[source] = e; return True
        return False
    return sum(aug(e, set()) for e in events)


def combined_metric(source_prob, pair_prob, data, source_mask, pair_mask):
    src_indices = np.flatnonzero(source_mask)
    valid_pairs = np.flatnonzero(pair_mask)
    by_source: dict[int, list[int]] = defaultdict(list)
    for pidx in valid_pairs:
        by_source[int(data["pair_source_idx"][pidx])].append(int(pidx))
    val_events = set(int(e) for e, embryo in zip(data["event_id"], data["event_embryo"]) if embryo in set(data["source_embryo"][src_indices]))
    choices = []
    for source in src_indices:
        candidates = by_source.get(int(source), [])
        if not candidates: continue
        best = max(candidates, key=lambda i: pair_prob[i])
        score = float(source_prob[source] * pair_prob[best])
        choices.append((score, int(source), int(data["pair_event"][best])))
    best_result = None
    for threshold in np.linspace(0.02, 0.90, 89):
        selected = [x for x in choices if x[0] >= threshold]
        edges: dict[int, set[int]] = defaultdict(set)
        for _, source, event in selected:
            if event >= 0: edges[event].add(source)
        tp = max_matching(sorted(val_events), edges)
        fp = len(selected) - tp; fn = len(val_events) - tp
        j = tp / max(tp + fp + fn, 1)
        row = {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "jaccard": j,
               "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
        if best_result is None or row["jaccard"] > best_result["jaccard"]:
            best_result = row
    return best_result


def train_fold(data, train_embryo: str, val_embryo: str, args, device):
    sx, sy = data["source_x"].astype(np.float32), data["source_y"].astype(np.float32)
    px, py = data["pair_x"].astype(np.float32), data["pair_y"].astype(np.float32)
    source_train = data["source_embryo"] == train_embryo
    source_val = data["source_embryo"] == val_embryo
    pair_train = source_train[data["pair_source_idx"]]
    pair_val = source_val[data["pair_source_idx"]]
    print(
        f"  rows: source train/val={int(source_train.sum()):,}/{int(source_val.sum()):,}; "
        f"pair train/val={int(pair_train.sum()):,}/{int(pair_val.sum()):,}",
        flush=True,
    )
    sm, ss = sx[source_train].mean(0), sx[source_train].std(0).clip(1e-4)
    pm, ps = px[pair_train].mean(0), px[pair_train].std(0).clip(1e-4)
    source_model = MLP(sx.shape[1], (96, 48)).to(device)
    pair_model = MLP(px.shape[1], (128, 64)).to(device)
    opt = torch.optim.AdamW(
        list(source_model.parameters()) + list(pair_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sloader = DataLoader(TensorDataset(
        torch.from_numpy((sx[source_train] - sm) / ss), torch.from_numpy(sy[source_train]),
    ), batch_size=min(args.batch_size, int(source_train.sum())), shuffle=True)
    ploader = DataLoader(TensorDataset(
        torch.from_numpy((px[pair_train] - pm) / ps), torch.from_numpy(py[pair_train]),
    ), batch_size=min(args.batch_size, int(pair_train.sum())), shuffle=True)
    best, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(args.epochs):
        source_model.train(); pair_model.train(); losses = []
        sit, pit = iter(sloader), iter(ploader)
        steps = max(len(sloader), len(ploader))
        for _ in range(steps):
            try: sb = next(sit)
            except StopIteration: sit = iter(sloader); sb = next(sit)
            try: pb = next(pit)
            except StopIteration: pit = iter(ploader); pb = next(pit)
            sxb, syb = sb[0].to(device), sb[1].to(device)
            pxb, pyb = pb[0].to(device), pb[1].to(device)
            loss = balanced_loss(source_model(sxb), syb) + balanced_loss(pair_model(pxb), pyb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(
                list(source_model.parameters()) + list(pair_model.parameters()), 3.0,
            ); opt.step(); losses.append(float(loss.detach()))
        sp = predict(source_model, sx, sm, ss, device)
        pp = predict(pair_model, px, pm, ps, device)
        metric = combined_metric(sp, pp, data, source_val, pair_val)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metric}
        history.append(row)
        print(f"  {train_embryo}->{val_embryo} epoch={epoch:02d} loss={row['loss']:.4f} "
              f"J={row['jaccard']:.4f} P={row['precision']:.4f} R={row['recall']:.4f} "
              f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']} th={row['threshold']:.2f}", flush=True)
        if row["jaccard"] > best:
            best = row["jaccard"]; stale = 0
            best_state = {
                "source": {k: v.detach().cpu().clone() for k, v in source_model.state_dict().items()},
                "pair": {k: v.detach().cpu().clone() for k, v in pair_model.state_dict().items()},
                "source_mean": sm, "source_std": ss, "pair_mean": pm, "pair_std": ps,
                "metrics": row,
            }
        else:
            stale += 1
            if stale >= args.patience: break
    return best_state, history


def train_full(data, epochs: int, threshold: float, args, device):
    sx, sy = data["source_x"].astype(np.float32), data["source_y"].astype(np.float32)
    px, py = data["pair_x"].astype(np.float32), data["pair_y"].astype(np.float32)
    sm, ss = sx.mean(0), sx.std(0).clip(1e-4); pm, ps = px.mean(0), px.std(0).clip(1e-4)
    source_model = MLP(sx.shape[1], (96, 48)).to(device); pair_model = MLP(px.shape[1], (128, 64)).to(device)
    opt = torch.optim.AdamW(list(source_model.parameters()) + list(pair_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    sloader = DataLoader(TensorDataset(torch.from_numpy((sx-sm)/ss),torch.from_numpy(sy)),batch_size=args.batch_size,shuffle=True)
    ploader = DataLoader(TensorDataset(torch.from_numpy((px-pm)/ps),torch.from_numpy(py)),batch_size=args.batch_size,shuffle=True)
    for epoch in range(epochs):
        source_model.train(); pair_model.train(); sit,pit=iter(sloader),iter(ploader); losses=[]
        for _ in range(max(len(sloader),len(ploader))):
            try: sb=next(sit)
            except StopIteration: sit=iter(sloader);sb=next(sit)
            try: pb=next(pit)
            except StopIteration: pit=iter(ploader);pb=next(pit)
            loss=balanced_loss(source_model(sb[0].to(device)),sb[1].to(device))+balanced_loss(pair_model(pb[0].to(device)),pb[1].to(device))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(source_model.parameters())+list(pair_model.parameters()),3);opt.step();losses.append(float(loss.detach()))
        print(f"  full epoch={epoch:02d}/{epochs-1:02d} loss={np.mean(losses):.4f}",flush=True)
    return {
        "version":"biohub-division-pair-v1",
        "source_model":{k:v.detach().cpu() for k,v in source_model.state_dict().items()},
        "pair_model":{k:v.detach().cpu() for k,v in pair_model.state_dict().items()},
        "source_mean":sm,"source_std":ss,"pair_mean":pm,"pair_std":ps,
        "source_features":SOURCE_FEATURES,"pair_features":PAIR_FEATURES,
        "decision_threshold":threshold,"core_parent_um":10.0,"core_sister_um":14.0,
        "rescue_parent_um":14.0,"rescue_sister_um":20.0,"temporal_radius":args.temporal_radius,
        "source_hidden":[96,48],"pair_hidden":[128,64],
    }


def main():
    args=argspec(); sys.path[:0]=[str(args.repo/"src"),str(args.repo/"scripts")]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.rebuild_cache or not args.cache.exists(): build_cache(args)
    if args.cache_only: return
    # Materialize every NPZ member once. Keeping ``data`` as an NpzFile and
    # indexing a member inside the million-row evaluator would repeatedly
    # decompress the same ZIP member and appear to hang on one CPU core.
    with np.load(args.cache,allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    embryos=sorted(set(str(x) for x in data["source_embryo"]))
    if len(embryos)<2: raise RuntimeError(f"Need both embryos for cross-validation; cache has {embryos}")
    args.output.mkdir(parents=True,exist_ok=True); fold_states=[]; histories=[]
    for train_embryo,val_embryo in (("44b6","6bba"),("6bba","44b6")):
        print(f"\nCROSS-EMBRYO FOLD {train_embryo} -> {val_embryo}",flush=True)
        state,history=train_fold(data,train_embryo,val_embryo,args,device);fold_states.append(state);histories.append({"train":train_embryo,"val":val_embryo,"history":history})
        torch.save(state,args.output/f"fold_{train_embryo}_to_{val_embryo}.pt")
    best_epochs=[int(s["metrics"]["epoch"])+1 for s in fold_states];full_epochs=max(3,int(round(float(np.median(best_epochs)))))
    threshold=float(np.mean([s["metrics"]["threshold"] for s in fold_states]))
    print(f"\nFULL TRAIN epochs={full_epochs} decision_threshold={threshold:.3f}",flush=True)
    final=train_full(data,full_epochs,threshold,args,device)
    final["cross_embryo_metrics"]=[s["metrics"] for s in fold_states]
    final["cache"] = str(args.cache)
    torch.save(final,args.output/"division_pair_model_best.pt")
    (args.output/"metrics.json").write_text(json.dumps(histories,indent=2))
    (args.output/"training_summary.json").write_text(json.dumps({
        "cache":str(args.cache),"sources":int(len(data["source_y"])),"source_positives":int(data["source_y"].sum()),
        "pairs":int(len(data["pair_y"])),"pair_positives":int(data["pair_y"].sum()),"events":int(len(data["event_id"])),
        "fold_metrics":[s["metrics"] for s in fold_states],"full_epochs":full_epochs,"decision_threshold":threshold,
        "checkpoint":str(args.output/"division_pair_model_best.pt"),
    },indent=2))
    print(f"\nTRAINING COMPLETE -> {args.output/'division_pair_model_best.pt'}",flush=True)


if __name__=="__main__": main()
