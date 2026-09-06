"""Atomic graph runtime for the public-.934 source-cardinality head.

The runtime preserves the public P1+P2 graph and uses Division V2 only to
construct its broad 14/20-um candidate population.  Model C, P1, and P2
contribute native association evidence.  The cardinality head chooses one
CONTINUE option or one DIVIDE(a,b) option per source, followed by lineage NMS.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from source_cardinality_runtime import SourceCardinalityRuntime


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SourceCardinalityGraphRuntime:
    """Apply a frozen cardinality decision as one atomic graph transaction."""

    def __init__(
        self,
        v2_runtime,
        artifact_dir: str | Path,
        legacy_artifact_dir: str | Path,
        legacy_fallback_runtime=None,
    ) -> None:
        self.v2 = v2_runtime
        self.artifact_dir = Path(artifact_dir)
        self.legacy_artifact_dir = Path(legacy_artifact_dir)
        self.head = SourceCardinalityRuntime(self.artifact_dir)
        legacy = _load_module(
            "source_cardinality_legacy_native_features",
            self.legacy_artifact_dir / "model_c_safe_runtime.py",
        )
        self.native_features = legacy._native_features
        self.legacy_fallback_runtime = legacy_fallback_runtime
        self.threshold = float(self.head.threshold)

    def _apply(
        self,
        dataset_path,
        model_c_evidence_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes: dict[int, dict[str, Any]],
        edges: list[dict[str, Any]],
        stats: dict[str, Any],
        registration_shifts_um=None,
        spacing=(1.625, 0.40625, 0.40625),
    ) -> list[dict[str, Any]]:
        original = [dict(edge) for edge in edges]
        scored = self.v2.score(
            dataset_path,
            nodes,
            original,
            registration_shifts_um,
            spacing,
        )
        source_ids = np.asarray(scored["source_ids"], np.int64)
        source_x = np.asarray(scored["_source_x"], np.float32)
        pair_x = np.asarray(scored["_pair_x"], np.float32)
        owner = np.asarray(scored["_pair_owner"], np.int32)
        pair_nodes = np.asarray(scored["pair_nodes"], np.int64)
        component_of = scored["_component_of"]
        if not len(source_ids) or not len(pair_nodes):
            stats["source_cardinality_selected_divisions"] = 0
            return original

        spacing_array = np.asarray(spacing, np.float64)
        native_blocks = []
        for label, path in (
            ("model_c", model_c_evidence_path),
            ("public_p1", p1_evidence_path),
            ("public_p2", p2_evidence_path),
        ):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} native evidence: {path}")
            block = self.native_features(
                path,
                nodes,
                source_ids,
                owner,
                pair_nodes[:, 0],
                pair_nodes[:, 1],
                spacing_array,
            )
            if block.shape != (len(pair_nodes), 24):
                raise RuntimeError(f"{label} feature mismatch: {block.shape}")
            native_blocks.append(block)

        pair_input = np.concatenate([pair_x, *native_blocks], axis=1)
        if pair_input.shape[1] != 150:
            raise RuntimeError(f"Cardinality pair feature mismatch: {pair_input.shape}")
        source_probability, best_nodes = self.head.score(
            source_x,
            pair_input,
            owner,
            pair_nodes,
        )

        winner_by_tube: dict[int, int] = {}
        for row in np.flatnonzero(source_probability >= self.threshold):
            source = int(source_ids[row])
            tube = int(component_of.get(source, source))
            previous = winner_by_tube.get(tube)
            if previous is None or source_probability[row] > source_probability[previous]:
                winner_by_tube[tube] = int(row)
        candidates = sorted(
            (
                float(source_probability[row]),
                int(source_ids[row]),
                int(best_nodes[row, 0]),
                int(best_nodes[row, 1]),
                int(component_of.get(int(source_ids[row]), int(source_ids[row]))),
            )
            for row in winner_by_tube.values()
            if int(best_nodes[row, 0]) >= 0 and int(best_nodes[row, 1]) >= 0
        )
        candidates.reverse()

        work = [dict(edge) for edge in original]
        outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for edge in work:
            outgoing[int(edge["source_id"])].append(edge)
        occupied_tubes = {
            int(component_of.get(source, source))
            for source, source_edges in outgoing.items()
            if len(source_edges) >= 2
        }
        locked_targets = {
            int(edge["target_id"])
            for source_edges in outgoing.values()
            if len(source_edges) >= 2
            for edge in source_edges
        }
        counters = defaultdict(int)
        counters["threshold_passing_tubes"] = len(candidates)
        for score, source, a, b, tube in candidates:
            if tube in occupied_tubes:
                counters["rejected_existing_fork_tube"] += 1
                continue
            if a in locked_targets or b in locked_targets:
                counters["rejected_existing_fork_child"] += 1
                continue
            if source not in nodes or a not in nodes or b not in nodes:
                counters["rejected_missing_node"] += 1
                continue
            source_t = int(nodes[source]["t"])
            if int(nodes[a]["t"]) != source_t + 1 or int(nodes[b]["t"]) != source_t + 1:
                counters["rejected_nonadjacent"] += 1
                continue
            pair = {a, b}
            work = [
                edge
                for edge in work
                if not (
                    (int(edge["source_id"]) == source and int(edge["target_id"]) not in pair)
                    or (
                        int(edge["target_id"]) in pair
                        and int(edge["source_id"]) != source
                    )
                )
            ]
            present = {(int(edge["source_id"]), int(edge["target_id"])) for edge in work}
            for target in (a, b):
                if (source, target) not in present:
                    work.append(
                        {
                            "source_id": source,
                            "target_id": target,
                            "edge_prob": score,
                            "learned_division": 1,
                            "source_cardinality_v2": 1,
                            "source_cardinality_score": score,
                        }
                    )
                    counters["added_edges"] += 1
            occupied_tubes.add(tube)
            locked_targets.update(pair)
            counters["selected_divisions"] += 1

        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        for edge in work:
            outdegree[int(edge["source_id"])] += 1
            indegree[int(edge["target_id"])] += 1
        if any(value > 2 for value in outdegree.values()):
            raise RuntimeError("Cardinality transaction produced out-degree > 2")
        if any(value > 1 for value in indegree.values()):
            raise RuntimeError("Cardinality transaction produced in-degree > 1")
        for key, value in counters.items():
            stats[f"source_cardinality_{key}"] = int(value)
        stats["source_cardinality_threshold"] = self.threshold
        return work

    def apply(
        self,
        dataset_path,
        model_c_evidence_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes,
        edges,
        stats,
        registration_shifts_um=None,
        spacing=(1.625, 0.40625, 0.40625),
        **legacy_thresholds,
    ):
        try:
            return self._apply(
                dataset_path,
                model_c_evidence_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                edges,
                stats,
                registration_shifts_um=registration_shifts_um,
                spacing=spacing,
            )
        except Exception as error:
            stats["source_cardinality_fallback"] = 1
            stats["source_cardinality_error"] = f"{type(error).__name__}: {error}"
            if self.legacy_fallback_runtime is None:
                return [dict(edge) for edge in edges]
            return self.legacy_fallback_runtime.apply(
                dataset_path,
                model_c_evidence_path,
                nodes,
                [dict(edge) for edge in edges],
                stats,
                registration_shifts_um=registration_shifts_um,
                spacing=spacing,
                **legacy_thresholds,
            )
