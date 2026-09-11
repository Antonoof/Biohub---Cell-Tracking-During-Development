#!/usr/bin/env python3
"""Portable runtime for the clean official-event Biohub V2 division gate.

The runtime consumes the post-link node/edge dictionaries used by the Kaggle
notebook.  It generates the same broad 14/20 um candidate features used for
training, freezes V1's strong daughter-pair scorer, applies a 195-video
leave-four-out parent/time gate, keeps one fork per current lineage, and
transactionally replaces claimed continuation edges.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _load_feature_api(path: Path):
    spec = importlib.util.spec_from_file_location("division_gbm_feature_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import division feature API: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _components(node_ids, edges):
    parent = {int(n): int(n) for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        if source in parent and target in parent:
            union(source, target)
    return {node: find(node) for node in parent}


class DivisionGBMRuntime:
    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir)
        self.spec = json.loads((self.artifact_dir / "deploy_spec.json").read_text())
        self.pair_model = joblib.load(self.artifact_dir / "pair_model.joblib")
        self.v1_source_model = joblib.load(self.artifact_dir / "v1_source_model.joblib")
        self.gate_model = joblib.load(self.artifact_dir / "gate_model.joblib")
        feature_api_path = self.artifact_dir / "train_division_pair_model.py"
        if not feature_api_path.exists():
            feature_api_path = Path(__file__).resolve().with_name("train_division_pair_model.py")
        self.features = _load_feature_api(feature_api_path)
        if list(self.features.SOURCE_FEATURES) != list(self.spec["source_feature_names"]):
            raise RuntimeError("Source feature-name contract mismatch")
        if list(self.features.PAIR_FEATURES) != list(self.spec["pair_feature_names"]):
            raise RuntimeError("Pair feature-name contract mismatch")

    def _graph_data(self, nodes, edges, spacing):
        ids_by_t: dict[int, list[int]] = defaultdict(list)
        pos: dict[int, np.ndarray] = {}
        for node_id, node in nodes.items():
            node_id = int(node_id)
            ids_by_t[int(node["t"])].append(node_id)
            pos[node_id] = np.asarray([node["z"], node["y"], node["x"]], np.float64) * spacing
        outgoing: dict[int, list[int]] = defaultdict(list)
        incoming: dict[int, list[int]] = defaultdict(list)
        edge_info: dict[tuple[int, int], tuple[float, float]] = {}
        for edge in edges:
            source, target = int(edge["source_id"]), int(edge["target_id"])
            outgoing[source].append(target); incoming[target].append(source)
            probability = edge.get("edge_prob")
            try:
                probability = float(probability) if probability is not None else 0.0
            except (TypeError, ValueError):
                probability = 0.0
            distance = edge.get("registered_distance_um", edge.get("distance_um", 0.0))
            edge_info[(source, target)] = (probability, float(distance or 0.0))
        return dict(ids_by_t), pos, dict(outgoing), dict(incoming), edge_info

    def score(
        self,
        dataset_path: Path | str,
        nodes: dict[int, dict[str, Any]],
        edges: list[dict[str, Any]],
        registration_shifts_um: dict[int, np.ndarray] | None = None,
        spacing=(1.625, 0.40625, 0.40625),
        max_pairs_per_source: int = 96,
    ) -> dict[str, Any]:
        spacing = np.asarray(spacing, np.float64)
        shifts = registration_shifts_um or {}
        ids_by_t, pos, outgoing, incoming, edge_info = self._graph_data(nodes, edges, spacing)
        component_of = _components(nodes, edges)
        reader = self.features.VolumeReader(Path(dataset_path))
        image_cache: dict[tuple[int, int], np.ndarray] = {}
        child_cache: dict[int, np.ndarray] = {}

        source_ids = []
        source_x = []
        source_pairs = []
        eligible = [
            source for source in sorted(nodes)
            if len(outgoing.get(source, [])) <= 1
            and ids_by_t.get(int(nodes[source]["t"]) + 1)
        ]
        for source in eligible:
            pairs = self.features.candidate_pairs(
                source, nodes, ids_by_t, pos, outgoing, incoming, shifts,
            )[:max_pairs_per_source]
            if not pairs:
                continue
            sf = self.features.source_features(
                source, nodes, ids_by_t, pos, outgoing, incoming, edge_info,
                shifts, reader, spacing, image_cache,
            )
            source_ids.append(int(source)); source_x.append(sf); source_pairs.append(pairs)

        sx = np.nan_to_num(np.asarray(source_x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        pair_x = []
        pair_owner = []
        pair_nodes = []
        pair_geometry = []
        for source_row, pairs in enumerate(source_pairs):
            source = source_ids[source_row]; sf = sx[source_row]
            for _, a, b, da, db, sister in pairs:
                pair_x.append(self.features.pair_features(
                    source, a, b, da, db, sister, sf, nodes, pos, outgoing,
                    incoming, shifts, component_of, reader, spacing, child_cache,
                ))
                pair_owner.append(source_row); pair_nodes.append((int(a), int(b)))
                pair_geometry.append((float(da), float(db), float(sister)))

        px = np.nan_to_num(np.asarray(pair_x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        owner = np.asarray(pair_owner, np.int32)
        pair_score = self.pair_model.predict_proba(px)[:, 1].astype(np.float32)
        n_sources = len(source_ids)
        pmax = np.zeros(n_sources, np.float32)
        pmean = np.zeros(n_sources, np.float32)
        pcnt = np.zeros(n_sources, np.float32)
        npairs = np.zeros(n_sources, np.float32)
        best_pair = np.full(n_sources, -1, np.int64)
        best_pair_x = np.zeros((n_sources, len(self.features.PAIR_FEATURES)), np.float32)
        order = np.argsort(owner, kind="stable")
        sorted_owner = owner[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1], True]) if len(order) else []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            rows = order[left:right]; source_row = int(sorted_owner[left]); probabilities = pair_score[rows]
            chosen = int(rows[int(np.argmax(probabilities))])
            pmax[source_row] = float(probabilities.max())
            pmean[source_row] = float(probabilities.mean())
            pcnt[source_row] = float(np.count_nonzero(probabilities > 0.5))
            npairs[source_row] = float(len(rows))
            best_pair[source_row] = chosen; best_pair_x[source_row] = px[chosen]

        source_input = np.concatenate([
            sx, best_pair_x, pmax[:, None], pmean[:, None], pcnt[:, None], np.log1p(npairs)[:, None],
        ], axis=1)
        v1_source_score = self.v1_source_model.predict_proba(source_input)[:, 1].astype(np.float32)
        gate_input = np.concatenate([
            sx, best_pair_x, pmax[:, None], v1_source_score[:, None],
        ], axis=1)
        if gate_input.shape[1] != 121:
            raise RuntimeError(f"V2 gate input mismatch: {gate_input.shape}")
        source_score_raw = self.gate_model.predict_proba(gate_input)[:, 1].astype(np.float32)

        # Frozen on 195-video grouped OOF.  The bounded rescue band retains
        # decisive pairs with a modestly lower V2 score while excluding the
        # high-pair geometric confusers that dominated V1 false forks.
        high = source_score_raw >= 0.895
        rescue = (
            (source_score_raw >= 0.800) &
            (pmax >= 0.850) & (pmax <= 0.920) &
            (v1_source_score >= 0.9475)
        )
        passing = high | rescue

        # One fork per currently linked lineage.
        winner_by_tube: dict[int, int] = {}
        for row in np.flatnonzero(passing):
            source = source_ids[int(row)]
            tube = int(component_of.get(source, source))
            previous = winner_by_tube.get(tube)
            if previous is None or source_score_raw[row] > source_score_raw[previous]:
                winner_by_tube[tube] = int(row)
        source_score = np.zeros_like(source_score_raw)
        winners = np.asarray(list(winner_by_tube.values()), np.int64)
        source_score[winners] = 1.0
        return {
            "source_ids": np.asarray(source_ids, np.int64),
            "source_score_raw": source_score_raw,
            "source_score": source_score,
            "v1_source_score": v1_source_score,
            "high_selected": int(np.count_nonzero(high)),
            "rescue_selected": int(np.count_nonzero(rescue & ~high)),
            "passing_sources": int(np.count_nonzero(passing)),
            "best_pair": best_pair,
            "pair_score": pair_score,
            "pair_nodes": pair_nodes,
            "pair_geometry": pair_geometry,
            "eligible_sources": len(eligible),
            "scored_sources": len(source_ids),
            "scored_pairs": len(pair_score),
            "tube_count": len(winner_by_tube),
            # Private deployment fields used by the leakage-safe Model-C
            # decoder.  They expose the exact V2 runtime population and
            # feature order; they do not alter any frozen V2 decision.
            "_source_x": sx,
            "_pair_x": px,
            "_pair_owner": owner,
            "_component_of": component_of,
        }

    def apply(
        self,
        dataset_path: Path | str,
        nodes: dict[int, dict[str, Any]],
        edges: list[dict[str, Any]],
        stats: dict[str, Any],
        registration_shifts_um: dict[int, np.ndarray] | None = None,
        threshold: float = 0.65,
        rescue_delta: float = 0.15,
        steal_delta: float = 0.25,
        spacing=(1.625, 0.40625, 0.40625),
    ) -> list[dict[str, Any]]:
        scored = self.score(dataset_path, nodes, edges, registration_shifts_um, spacing)
        candidates = []
        for source_row, source in enumerate(scored["source_ids"]):
            pair_row = int(scored["best_pair"][source_row])
            if pair_row < 0:
                continue
            score = float(scored["source_score"][source_row])
            gate_score = float(scored["source_score_raw"][source_row])
            a, b = scored["pair_nodes"][pair_row]
            da, db, sister = scored["pair_geometry"][pair_row]
            core = max(da, db) <= 10.0 and sister <= 14.0
            required = threshold if core else threshold + rescue_delta
            if score >= required:
                candidates.append((gate_score, score, int(source), int(a), int(b), core))
        candidates.sort(reverse=True)

        work = [dict(edge) for edge in edges]
        selected_sources: set[int] = set()
        locked_targets: set[int] = set()
        counters = defaultdict(int)
        counters["candidates_passing"] = len(candidates)
        for gate_score, score, source, a, b, core in candidates:
            if source in selected_sources or a in locked_targets or b in locked_targets:
                counters["rejected_locked"] += 1
                continue
            outgoing = [edge for edge in work if int(edge["source_id"]) == source]
            if len(outgoing) >= 2:
                counters["protected_existing_fork"] += 1
                selected_sources.add(source)
                continue
            existing_targets = {int(edge["target_id"]) for edge in outgoing}
            conflicts = [
                edge for edge in work
                if int(edge["target_id"]) in (a, b) and int(edge["source_id"]) != source
            ]
            steal_required = threshold + steal_delta + (0.0 if core else rescue_delta)
            if conflicts and score < steal_required:
                counters["rejected_claimed"] += 1
                continue
            pair = {a, b}
            before = len(work)
            work = [
                edge for edge in work
                if not (int(edge["source_id"]) == source and int(edge["target_id"]) not in pair)
            ]
            counters["removed_source_edges"] += before - len(work)
            if conflicts:
                conflict_ids = {id(edge) for edge in conflicts}
                work = [edge for edge in work if id(edge) not in conflict_ids]
                counters["reassigned_claimed_edges"] += len(conflicts)
            present = {(int(edge["source_id"]), int(edge["target_id"])) for edge in work}
            for target in (a, b):
                if (source, target) not in present:
                    work.append({
                        "source_id": source, "target_id": target,
                        "edge_prob": score, "learned_division": 1,
                        "division_model_score": gate_score,
                    })
                    counters["added_edges"] += 1
            selected_sources.add(source); locked_targets.update((a, b))
            counters["selected_divisions"] += 1
            counters["selected_core" if core else "selected_rescue"] += 1

        stats["gbm_division_eligible_sources"] = int(scored["eligible_sources"])
        stats["gbm_division_scored_sources"] = int(scored["scored_sources"])
        stats["gbm_division_scored_pairs"] = int(scored["scored_pairs"])
        stats["gbm_division_tubes"] = int(scored["tube_count"])
        stats["gbm_division_v2_high_sources"] = int(scored["high_selected"])
        stats["gbm_division_v2_rescue_sources"] = int(scored["rescue_selected"])
        stats["gbm_division_v2_passing_sources"] = int(scored["passing_sources"])
        for key, value in counters.items():
            stats[f"gbm_division_{key}"] = int(value)
        return work
