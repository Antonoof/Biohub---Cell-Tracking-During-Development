"""Deployment runtime for the calibrated EdgeGRAFT component owner.

The frozen model was trained on ambiguous P1/P2 continuation components from
the pre-postprocessing graph.  At serve time this runtime deliberately keeps
that feature substrate, maps it into the finalized graph, and applies only
atomic one-to-one continuation replacements.  Existing forks and their local
neighborhood are immutable.  No node or division transaction is created or
removed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn


SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)
MAPPING_RADIUS_UM = 6.0
CALIBRATED_THRESHOLD = 0.004891517842598334


BASE_FEATURES = [
    "p1_probability", "p2_probability", "p1_present", "p2_present",
    "p1_alternative", "p2_alternative", "p1_margin", "p2_margin",
    "p1_winner", "p2_winner", "p1_source_rank", "p2_source_rank",
    "p1_target_relative", "p2_target_relative", "model_probability_max",
    "model_probability_mean", "model_probability_absdiff", "distance_um",
    "target_union_candidates", "source_density_5um", "source_density_10um",
    "target_density_5um", "target_density_10um", "velocity_residual_um",
    "has_velocity", "is_raw_parent", "raw_edge_probability",
    "raw_parent_probability_gap",
]


def _segment_reduce(values, groups, count, reduce):
    if reduce == "mean":
        output = torch.zeros(
            (count, values.shape[1]), device=values.device, dtype=values.dtype
        )
        output.index_add_(0, groups, values)
        totals = torch.zeros(count, device=values.device, dtype=values.dtype)
        totals.index_add_(
            0, groups,
            torch.ones(len(groups), device=values.device, dtype=values.dtype),
        )
        return output / totals.clamp_min(1).unsqueeze(1)
    output = torch.full(
        (count, values.shape[1]), -torch.inf,
        device=values.device, dtype=values.dtype,
    )
    output.scatter_reduce_(
        0, groups[:, None].expand_as(values), values,
        reduce="amax", include_self=True,
    )
    return output


class ComponentEdgeNet(nn.Module):
    def __init__(self, tab_dim, hidden=96, rounds=3, emb_dim=24):
        super().__init__()
        image_dim = emb_dim * 4
        self.edge = nn.Sequential(
            nn.Linear(tab_dim + image_dim + 6, hidden),
            nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.08),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 5, hidden * 2),
                nn.LayerNorm(hidden * 2), nn.SiLU(), nn.Dropout(0.08),
                nn.Linear(hidden * 2, hidden),
            )
            for _ in range(rounds)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, tab, source_emb, target_emb, structural,
                source_group, target_group):
        image = torch.cat([
            source_emb, target_emb, torch.abs(source_emb - target_emb),
            source_emb * target_emb,
        ], 1)
        hidden = self.edge(torch.cat([tab, image, structural], 1))
        source_count = int(source_group.max()) + 1
        target_count = int(target_group.max()) + 1
        for update, norm in zip(self.updates, self.norms):
            source_mean = _segment_reduce(
                hidden, source_group, source_count, "mean"
            )[source_group]
            source_max = _segment_reduce(
                hidden, source_group, source_count, "max"
            )[source_group]
            target_mean = _segment_reduce(
                hidden, target_group, target_count, "mean"
            )[target_group]
            target_max = _segment_reduce(
                hidden, target_group, target_count, "max"
            )[target_group]
            hidden = norm(hidden + update(torch.cat([
                hidden, source_mean, source_max, target_mean, target_max
            ], 1)))
        return self.out(hidden).squeeze(1)


def _node_position(row: dict[str, Any]) -> np.ndarray:
    return np.asarray((row["z"], row["y"], row["x"]), np.float64)


def _greedy_native_map(native_coords, native_offsets, nodes, spacing):
    """Frame-wise one-to-one mapper used by the evidence-export contract."""
    mapped = np.full(len(native_coords), -1, np.int64)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, row in nodes.items():
        by_frame[int(row["t"])].append(int(node_id))
    for frame in range(len(native_offsets) - 1):
        start, end = int(native_offsets[frame]), int(native_offsets[frame + 1])
        graph_ids = sorted(by_frame.get(frame, ()))
        if end <= start or not graph_ids:
            continue
        native_um = native_coords[start:end, 1:].astype(np.float32) * spacing
        graph_um = np.asarray([
            [round(float(nodes[node]["z"])), round(float(nodes[node]["y"])),
             round(float(nodes[node]["x"]))]
            for node in graph_ids
        ], np.float32) * spacing
        k = min(4, len(graph_um))
        distances, indexes = cKDTree(graph_um).query(native_um, k=k)
        if k == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        options = []
        for native_row in range(len(native_um)):
            for rank in range(k):
                value = float(distances[native_row, rank])
                if value <= MAPPING_RADIUS_UM:
                    options.append((value, native_row, int(indexes[native_row, rank])))
        used_native: set[int] = set()
        used_graph: set[int] = set()
        for _distance, native_row, graph_row in sorted(options):
            if native_row in used_native or graph_row in used_graph:
                continue
            used_native.add(native_row)
            used_graph.add(graph_row)
            mapped[start + native_row] = graph_ids[graph_row]
    return mapped


def _read_evidence(path: str | Path, raw_nodes):
    with np.load(path, allow_pickle=False) as data:
        source = data["source_id"].astype(np.int64, copy=False)
        target = data["target_id"].astype(np.int64, copy=False)
        probability = data["probability"].astype(np.float32, copy=False)
        alternative = data["alternative_parent_probability"].astype(
            np.float32, copy=False
        )
        winner = data["is_target_winner"].astype(np.float32, copy=False)
        rank = data["source_target_rank"].astype(np.int16, copy=False)
        distance = data["distance_um"].astype(np.float32, copy=False)
        native = data["native_node_coords"].astype(np.float32, copy=False)
        offsets = data["native_frame_offsets"].astype(np.int64, copy=False)
        spacing = (
            data["spacing_um"].astype(np.float32, copy=False)
            if "spacing_um" in data else SPACING.astype(np.float32)
        )
        if "mapped_ab_node" in data:
            fused_mapping = data["mapped_ab_node"].astype(np.int64, copy=False)
            if "fused_graph_node_id" in data:
                fused_graph_ids = data["fused_graph_node_id"].astype(
                    np.int64, copy=False
                )
            elif "ab_node_coords" in data:
                fused_coords = data["ab_node_coords"].astype(
                    np.float32, copy=False
                )
                raw_by_identity = {
                    _identity(row): int(node_id)
                    for node_id, row in raw_nodes.items()
                }
                fused_graph_ids = np.asarray([
                    raw_by_identity.get((
                        int(round(float(coord[0]))),
                        round(float(coord[1]), 5),
                        round(float(coord[2]), 5),
                        round(float(coord[3]), 5),
                    ), -1)
                    for coord in fused_coords
                ], np.int64)
            else:
                fused_graph_ids = np.empty(0, np.int64)
            mapped = np.full(len(fused_mapping), -1, np.int64)
            usable = (
                (fused_mapping >= 0) & (fused_mapping < len(fused_graph_ids))
            )
            mapped[usable] = fused_graph_ids[fused_mapping[usable]]
        else:
            mapped = _greedy_native_map(native, offsets, raw_nodes, spacing)
        valid = (
            (source >= 0) & (target >= 0)
            & (source < len(mapped)) & (target < len(mapped))
        )
        rows = []
        for index in np.flatnonzero(valid):
            mapped_source = int(mapped[source[index]])
            mapped_target = int(mapped[target[index]])
            if mapped_source < 0 or mapped_target < 0:
                continue
            # Preserve mapped pre-ILP detector nodes here. They participate in
            # the frozen conflict-component boundary even when the ILP later
            # drops them. Candidate rows are filtered to live raw nodes only
            # after component construction, matching the training extractor.
            rows.append((
                mapped_source, mapped_target, float(probability[index]),
                float(alternative[index]), float(winner[index]),
                float(rank[index]), float(distance[index]),
            ))
    deduped = {}
    for source_id, target_id, probability_value, alternative_value, winner_value, rank_value, distance_value in rows:
        key = (source_id, target_id)
        value = (
            probability_value, alternative_value, winner_value,
            rank_value, distance_value,
        )
        old = deduped.get(key)
        if old is None or value[0] > old[0]:
            deduped[key] = value
    context_rows = [
        (source_id, target_id, probability_value, alternative_value,
         winner_value, distance_value)
        for (
            source_id, target_id, probability_value, alternative_value,
            winner_value, _rank_value, distance_value
        ) in rows
    ]
    return deduped, context_rows


def _conflict_pairs(evidence):
    source_to_target: dict[int, set[int]] = defaultdict(set)
    target_to_source: dict[int, set[int]] = defaultdict(set)
    for model in evidence.values():
        for source, target in model:
            source_to_target[int(source)].add(int(target))
            target_to_source[int(target)].add(int(source))
    seen: set[int] = set()
    keep: set[tuple[int, int]] = set()
    component_of: dict[tuple[int, int], int] = {}
    component = 0
    for initial in source_to_target:
        if initial in seen:
            continue
        stack = [initial]
        sources: set[int] = set()
        targets: set[int] = set()
        while stack:
            source = stack.pop()
            if source in seen:
                continue
            seen.add(source)
            sources.add(source)
            for target in source_to_target[source]:
                targets.add(target)
                for other in target_to_source[target]:
                    if other not in seen:
                        stack.append(other)
        if len(targets) > 1:
            for source in sources:
                for target in source_to_target[source]:
                    keep.add((source, target))
                    component_of[(source, target)] = component
            component += 1
    return keep, component_of


def _density_maps(nodes):
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, row in nodes.items():
        by_frame[int(row["t"])].append(int(node_id))
    result = {}
    for ids in by_frame.values():
        points = np.asarray([_node_position(nodes[node]) for node in ids]) * SPACING
        tree = cKDTree(points)
        near = tree.query_ball_point(points, 5.0, return_length=True) - 1
        broad = tree.query_ball_point(points, 10.0, return_length=True) - 1
        for node, one, two in zip(ids, near, broad):
            result[node] = (float(one), float(two))
    return result


def _best_context(native_rows, need_in, need_out):
    incoming = {}
    outgoing = {}
    for source, target, probability, alternative, winner, distance in sorted(
        native_rows, key=lambda row: -row[2]
    ):
        value = (source, target, probability, alternative, winner, distance)
        if target in need_in and target not in incoming:
            incoming[target] = value
        if source in need_out and source not in outgoing:
            outgoing[source] = value
    return incoming, outgoing


def _cosine(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-6 else 0.0


def _speed_ratio(first, second):
    return float(np.clip(np.log(
        (np.linalg.norm(second) + 1e-3) / (np.linalg.norm(first) + 1e-3)
    ), -4, 4))


def _model_context(prefix, source, target, current_probability, nodes,
                   incoming, outgoing):
    current = nodes[target] - nodes[source]
    previous = incoming.get(source)
    future = outgoing.get(target)
    previous_valid = float(previous is not None and previous[0] in nodes)
    future_valid = float(future is not None and future[1] in nodes)
    previous_delta = (
        nodes[source] - nodes[previous[0]]
        if previous_valid else np.zeros(3, np.float32)
    )
    future_delta = (
        nodes[future[1]] - nodes[target]
        if future_valid else np.zeros(3, np.float32)
    )
    previous_probability = previous[2] if previous_valid else 0.0
    future_probability = future[2] if future_valid else 0.0
    values = {
        f"{prefix}_prev_valid": previous_valid,
        f"{prefix}_prev_probability": previous_probability,
        f"{prefix}_prev_margin": (
            previous[2] - previous[3] if previous_valid else 0.0
        ),
        f"{prefix}_prev_winner": previous[4] if previous_valid else 0.0,
        f"{prefix}_prev_distance": previous[5] if previous_valid else 0.0,
        f"{prefix}_future_valid": future_valid,
        f"{prefix}_future_probability": future_probability,
        f"{prefix}_future_margin": (
            future[2] - future[3] if future_valid else 0.0
        ),
        f"{prefix}_future_winner": future[4] if future_valid else 0.0,
        f"{prefix}_future_distance": future[5] if future_valid else 0.0,
        f"{prefix}_prev_current_residual": (
            float(np.linalg.norm(current - previous_delta))
            if previous_valid else 0.0
        ),
        f"{prefix}_current_future_residual": (
            float(np.linalg.norm(future_delta - current))
            if future_valid else 0.0
        ),
        f"{prefix}_prev_current_cosine": (
            _cosine(previous_delta, current) if previous_valid else 0.0
        ),
        f"{prefix}_current_future_cosine": (
            _cosine(current, future_delta) if future_valid else 0.0
        ),
        f"{prefix}_prev_current_speed_ratio": (
            _speed_ratio(previous_delta, current) if previous_valid else 0.0
        ),
        f"{prefix}_current_future_speed_ratio": (
            _speed_ratio(current, future_delta) if future_valid else 0.0
        ),
        f"{prefix}_path_min": (
            min(previous_probability, current_probability, future_probability)
            if previous_valid and future_valid else 0.0
        ),
        f"{prefix}_path_mean": (
            (previous_probability + current_probability + future_probability) / 3.0
            if previous_valid and future_valid else 0.0
        ),
        f"{prefix}_path_geomean": (
            float(np.cbrt(max(
                previous_probability * current_probability * future_probability, 0.0
            ))) if previous_valid and future_valid else 0.0
        ),
        f"{prefix}_context_both": previous_valid * future_valid,
    }
    return values, (
        int(previous[0]) if previous_valid else -1,
        int(future[1]) if future_valid else -1,
    )


def _identity(row):
    return (
        int(row["t"]), round(float(row["z"]), 5),
        round(float(row["y"]), 5), round(float(row["x"]), 5),
    )


def _raw_to_final_map(raw_nodes, final_nodes):
    result = {
        node: node for node in set(raw_nodes) & set(final_nodes)
        if int(raw_nodes[node]["t"]) == int(final_nodes[node]["t"])
    }
    final_by_key = {}
    duplicates = set()
    for node, row in final_nodes.items():
        key = _identity(row)
        if key in final_by_key:
            duplicates.add(key)
        else:
            final_by_key[key] = node
    for key in duplicates:
        final_by_key.pop(key, None)
    for node, row in raw_nodes.items():
        if node in result:
            continue
        target = final_by_key.get(_identity(row))
        if target is not None:
            result[node] = int(target)
    return result


def _protected_fork_nodes(edges):
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        outgoing[source].add(target)
        incoming[target].add(source)
    protected: set[int] = set()
    for source, children in outgoing.items():
        if len(children) < 2:
            continue
        protected.add(source)
        protected.update(children)
        protected.update(incoming.get(source, ()))
        for child in children:
            protected.update(outgoing.get(child, ()))
    return protected


def _factorize(values):
    lookup = {}
    result = np.empty(len(values), np.int64)
    for index, value in enumerate(values):
        if value not in lookup:
            lookup[value] = len(lookup)
        result[index] = lookup[value]
    return result


class EdgeGraftComponentRuntime:
    def __init__(self, artifact_dir: str | Path,
                 threshold: float = CALIBRATED_THRESHOLD):
        self.artifact_dir = Path(artifact_dir)
        checkpoint = torch.load(
            self.artifact_dir / "component_gate_best.pt",
            map_location="cpu", weights_only=False,
        )
        self.features = list(checkpoint["features"])
        if self.features[:len(BASE_FEATURES)] != BASE_FEATURES or len(self.features) != 72:
            raise RuntimeError("Unexpected EdgeGRAFT checkpoint feature contract")
        self.mean = np.asarray(checkpoint["mean"], np.float32)
        self.std = np.asarray(checkpoint["std"], np.float32)
        self.model = ComponentEdgeNet(len(self.features))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(threshold)

    def _rows(self, raw_nodes, raw_edges, p1_path, p2_path):
        p1, p1_native = _read_evidence(p1_path, raw_nodes)
        p2, p2_native = _read_evidence(p2_path, raw_nodes)
        evidence = {"p1": p1, "p2": p2}
        keep, component_of = _conflict_pairs(evidence)
        candidates: dict[int, set[int]] = defaultdict(set)
        for source, target in keep:
            if source in raw_nodes and target in raw_nodes:
                candidates[target].add(source)
        raw_parent = {}
        raw_probability = {}
        predecessor = {}
        for edge in raw_edges:
            source = int(edge["source_id"])
            target = int(edge["target_id"])
            probability = float(edge.get("edge_prob") or 0.0)
            if probability > raw_probability.get(target, -np.inf):
                raw_parent[target] = source
                raw_probability[target] = probability
            predecessor[target] = source
        density = _density_maps(raw_nodes)
        position = {node: _node_position(row) for node, row in raw_nodes.items()}
        physical = {
            node: (value * SPACING).astype(np.float32)
            for node, value in position.items()
        }
        probability_max = {
            name: defaultdict(float) for name in evidence
        }
        for name, model in evidence.items():
            for (source, target), value in model.items():
                if (source, target) in keep:
                    probability_max[name][target] = max(
                        probability_max[name][target], value[0]
                    )
        need_in = set(source for source, _target in keep)
        need_out = set(target for _source, target in keep)
        contexts = {
            "p1": _best_context(p1_native, need_in, need_out),
            "p2": _best_context(p2_native, need_in, need_out),
        }
        rows = []
        for target, sources in candidates.items():
            for source in sorted(sources):
                first = p1.get((source, target))
                second = p2.get((source, target))
                p1_probability = first[0] if first else 0.0
                p2_probability = second[0] if second else 0.0
                raw_edge_probability = raw_probability.get(target, 0.0)
                distance = float(np.linalg.norm(
                    (position[target] - position[source]) * SPACING
                ))
                previous = predecessor.get(source)
                if previous in position:
                    predicted = position[source] + (
                        position[source] - position[previous]
                    )
                    velocity_residual = float(np.linalg.norm(
                        (position[target] - predicted) * SPACING
                    ))
                    has_velocity = 1.0
                else:
                    velocity_residual = distance
                    has_velocity = 0.0
                source_density = density.get(source, (0.0, 0.0))
                target_density = density.get(target, (0.0, 0.0))
                values = [
                    p1_probability, p2_probability,
                    float(first is not None), float(second is not None),
                    first[1] if first else 0.0,
                    second[1] if second else 0.0,
                    first[0] - first[1] if first else 0.0,
                    second[0] - second[1] if second else 0.0,
                    first[2] if first else 0.0,
                    second[2] if second else 0.0,
                    first[3] if first else 99.0,
                    second[3] if second else 99.0,
                    p1_probability - probability_max["p1"][target],
                    p2_probability - probability_max["p2"][target],
                    max(p1_probability, p2_probability),
                    (p1_probability + p2_probability) / 2.0,
                    abs(p1_probability - p2_probability), distance,
                    float(len(sources)), source_density[0], source_density[1],
                    target_density[0], target_density[1], velocity_residual,
                    has_velocity, float(raw_parent.get(target) == source),
                    raw_edge_probability if raw_parent.get(target) == source else 0.0,
                    max(p1_probability, p2_probability) - raw_edge_probability,
                ]
                features = dict(zip(BASE_FEATURES, values))
                p1_context, p1_identity = _model_context(
                    "ctx_p1", source, target, p1_probability, physical,
                    *contexts["p1"],
                )
                p2_context, p2_identity = _model_context(
                    "ctx_p2", source, target, p2_probability, physical,
                    *contexts["p2"],
                )
                features.update(p1_context)
                features.update(p2_context)
                features.update({
                    "ctx_prev_identity_agreement": float(
                        p1_identity[0] >= 0 and p1_identity[0] == p2_identity[0]
                    ),
                    "ctx_future_identity_agreement": float(
                        p1_identity[1] >= 0 and p1_identity[1] == p2_identity[1]
                    ),
                    "ctx_prev_probability_absdiff": abs(
                        p1_context["ctx_p1_prev_probability"]
                        - p2_context["ctx_p2_prev_probability"]
                    ),
                    "ctx_future_probability_absdiff": abs(
                        p1_context["ctx_p1_future_probability"]
                        - p2_context["ctx_p2_future_probability"]
                    ),
                })
                rows.append({
                    "component": int(component_of[(source, target)]),
                    "source": int(source), "target": int(target),
                    "features": features,
                    "native_probability": max(p1_probability, p2_probability),
                })
        return rows

    @torch.no_grad()
    def _predict(self, rows):
        by_component: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_component[int(row["component"])].append(row)
        components = list(by_component)
        for start in range(0, len(components), 8):
            batch = [
                row for component in components[start:start + 8]
                for row in by_component[component]
            ]
            source_group = _factorize([row["source"] for row in batch])
            target_group = _factorize([row["target"] for row in batch])
            component_group = _factorize([row["component"] for row in batch])
            source_degree = np.bincount(source_group)[source_group].astype(np.float32)
            target_degree = np.bincount(target_group)[target_group].astype(np.float32)
            component_edges = np.bincount(component_group)[component_group].astype(np.float32)
            component_sources = np.asarray([
                len({row["source"] for row in batch
                     if row["component"] == value})
                for value in [row["component"] for row in batch]
            ], np.float32)
            component_targets = np.asarray([
                len({row["target"] for row in batch
                     if row["component"] == value})
                for value in [row["component"] for row in batch]
            ], np.float32)
            structural = np.log1p(np.stack([
                source_degree, target_degree, component_edges,
                component_sources, component_targets,
                source_degree * target_degree,
            ], 1))
            tabular = np.asarray([
                [row["features"][name] for name in self.features]
                for row in batch
            ], np.float32)
            tabular = (tabular - self.mean) / self.std
            zeros = torch.zeros((len(batch), 24), dtype=torch.float32)
            scores = self.model(
                torch.from_numpy(tabular), zeros, zeros,
                torch.from_numpy(structural),
                torch.from_numpy(source_group), torch.from_numpy(target_group),
            ).cpu().numpy()
            for row, score in zip(batch, scores):
                row["score"] = float(score)
        return rows

    def apply(self, raw_nodes, raw_edges, final_nodes, final_edges,
              p1_path, p2_path):
        counters = Counter()
        rows = self._rows(raw_nodes, raw_edges, p1_path, p2_path)
        mapping = _raw_to_final_map(raw_nodes, final_nodes)
        mapped = []
        seen = set()
        current_edges = {
            (int(edge["source_id"]), int(edge["target_id"]))
            for edge in final_edges
        }
        for row in rows:
            source = mapping.get(int(row["source"]))
            target = mapping.get(int(row["target"]))
            if source is None or target is None:
                continue
            key = (int(row["component"]), int(source), int(target))
            if key in seen:
                continue
            seen.add(key)
            result = dict(row)
            result["source"] = int(source)
            result["target"] = int(target)
            result["features"] = dict(row["features"])
            result["features"]["is_raw_parent"] = float(
                (int(source), int(target)) in current_edges
            )
            mapped.append(result)
        counters["raw_rows"] = len(rows)
        counters["mapped_rows"] = len(mapped)
        if not mapped:
            return final_edges, dict(counters)
        self._predict(mapped)

        incoming: dict[int, set[int]] = defaultdict(set)
        outgoing: dict[int, set[int]] = defaultdict(set)
        edge_by_pair = {}
        for edge in final_edges:
            source, target = int(edge["source_id"]), int(edge["target_id"])
            incoming[target].add(source)
            outgoing[source].add(target)
            edge_by_pair[(source, target)] = dict(edge)
        protected = _protected_fork_nodes(final_edges)
        by_component: dict[int, list[dict]] = defaultdict(list)
        for row in mapped:
            by_component[int(row["component"])].append(row)

        for group in by_component.values():
            counters["components"] += 1
            current_parent = {}
            for target in {int(row["target"]) for row in group}:
                parents = incoming.get(target, set())
                if len(parents) != 1:
                    continue
                parent = next(iter(parents))
                if (
                    parent in protected or target in protected
                    or len(outgoing.get(parent, ())) > 1
                ):
                    continue
                current_parent[target] = parent
            represented = {
                (int(row["source"]), int(row["target"])) for row in group
            }
            current_parent = {
                target: source for target, source in current_parent.items()
                if (source, target) in represented
            }
            if not current_parent:
                counters["ineligible"] += 1
                continue
            targets_set = set(current_parent)
            work = [row for row in group if int(row["target"]) in targets_set]
            sources = sorted({int(row["source"]) for row in work})
            targets = sorted({int(row["target"]) for row in work})
            if not sources or len(sources) < len(targets):
                counters["ineligible"] += 1
                continue
            source_index = {value: index for index, value in enumerate(sources)}
            target_index = {value: index for index, value in enumerate(targets)}
            matrix = np.full((len(targets), len(sources)), -1e6, np.float64)
            probability_by_pair = {}
            for row in work:
                source, target = int(row["source"]), int(row["target"])
                i, j = target_index[target], source_index[source]
                matrix[i, j] = max(matrix[i, j], float(row["score"]))
                probability_by_pair[(source, target)] = max(
                    probability_by_pair.get((source, target), 0.0),
                    float(row["native_probability"]),
                )
            base = {(source, target) for target, source in current_parent.items()}
            if len(base) != len(targets):
                counters["ineligible"] += 1
                continue
            base_values = []
            valid_base = True
            for source, target in base:
                column = source_index.get(source)
                if column is None or matrix[target_index[target], column] < -1e5:
                    valid_base = False
                    break
                base_values.append(matrix[target_index[target], column])
            if not valid_base:
                counters["ineligible"] += 1
                continue
            rows_index, columns_index = linear_sum_assignment(
                matrix, maximize=True
            )
            if (
                len(rows_index) != len(targets)
                or np.any(matrix[rows_index, columns_index] < -1e5)
            ):
                counters["ineligible"] += 1
                continue
            proposed = {
                (int(sources[column]), int(targets[row]))
                for row, column in zip(rows_index, columns_index)
            }
            advantage = float(
                (matrix[rows_index, columns_index].sum() - np.sum(base_values))
                / max(len(targets), 1)
            )
            counters["eligible"] += 1
            if proposed == base or advantage < self.threshold:
                continue
            counters["selected"] += 1

            involved_sources = {source for source, _target in proposed | base}
            involved_targets = {target for _source, target in proposed | base}
            if any(len(outgoing.get(source, ())) > 1 for source in involved_sources):
                counters["source_fork"] += 1
                continue
            if any(
                any(len(outgoing.get(parent, ())) > 1
                    for parent in incoming.get(target, ()))
                for target in involved_targets
            ):
                counters["parent_fork"] += 1
                continue
            if any(
                outgoing.get(source, set()) - involved_targets
                for source in involved_sources
            ):
                counters["outside_target"] += 1
                continue
            remove = set()
            add = set()
            for source, target in proposed:
                for parent in incoming.get(target, ()):
                    if parent != source:
                        remove.add((parent, target))
                for child in outgoing.get(source, ()):
                    if child != target:
                        remove.add((source, child))
                if (source, target) not in edge_by_pair:
                    add.add((source, target))
            for pair in remove:
                edge_by_pair.pop(pair, None)
                outgoing[pair[0]].discard(pair[1])
                incoming[pair[1]].discard(pair[0])
            for pair in add:
                edge_by_pair[pair] = {
                    "source_id": pair[0], "target_id": pair[1],
                    "edge_prob": float(probability_by_pair.get(pair, 1.0)),
                }
                outgoing[pair[0]].add(pair[1])
                incoming[pair[1]].add(pair[0])
            if any(len(incoming[target]) > 1 for target in involved_targets):
                raise RuntimeError("EdgeGRAFT transaction produced in-degree > 1")
            if any(len(outgoing[source]) > 1 for source in involved_sources):
                raise RuntimeError("EdgeGRAFT transaction produced out-degree > 1")
            counters["applied"] += 1
            counters["removed"] += len(remove)
            counters["added"] += len(add)

        result = list(edge_by_pair.values())
        if _protected_fork_nodes(result) != protected:
            raise RuntimeError("EdgeGRAFT changed protected fork topology")
        counters["threshold_million"] = int(round(self.threshold * 1_000_000))
        return result, dict(counters)

