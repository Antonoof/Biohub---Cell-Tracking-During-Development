"""Inference-safe full-population ownership recovery for Multi-UniGRAFT.

The wrapper preserves the proven UG1 -> UG2 -> live V2 specialist order,
reuses that specialist's already-materialized 128-pair evidence, then scores
every eligible source with one of five grouped-video models.  The selected
source-to-two-daughter transaction is atomic.  Existing forks and their
daughter targets are protected; any failure returns the completed specialist
graph unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "steal_distance_delta_um",
    "alternate_minus_current_um",
    "incumbent_edge_prob",
    "alternate_parent_um",
    "steal_distance_ratio",
    "incumbent_back_depth",
    "predicted_midpoint_error_um",
    "source_frame_percentile",
    "source_frame_log_margin",
    "v2_frame_percentile",
    "v2_frame_log_margin",
    "agreement4",
    "agreement5",
    "time_norm",
]


def _canonical(values) -> tuple[int, int]:
    a, b = map(int, values)
    return (a, b) if a <= b else (b, a)


def _edge_maps(edges):
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    probability: dict[tuple[int, int], float] = {}
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        outgoing[source].append(target)
        incoming[target].append(source)
        value = edge.get("edge_prob")
        try:
            probability[(source, target)] = -1.0 if value is None else float(value)
        except (TypeError, ValueError):
            probability[(source, target)] = -1.0
    return dict(outgoing), dict(incoming), probability


def _depth(adjacency: dict[int, list[int]], node: int, cap: int = 16) -> int:
    depth = 0
    current = int(node)
    for _ in range(cap):
        values = adjacency.get(current, ())
        if len(values) != 1:
            break
        current = int(values[0])
        depth += 1
    return depth


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


class FullPopulationOwnershipRuntime:
    """Stack broad ownership recovery after the live V2 specialist."""

    def __init__(self, specialist_runtime, artifact_dir: str | Path) -> None:
        self.specialist = specialist_runtime
        self.artifact_dir = Path(artifact_dir)
        spec = json.loads((self.artifact_dir / "deploy_spec.json").read_text())
        if spec.get("version") != "full-population-ownership-v1":
            raise RuntimeError("Full-population ownership artifact version mismatch")
        if list(spec["features"]) != FEATURES:
            raise RuntimeError("Full-population ownership feature contract mismatch")
        self.threshold = float(spec["threshold"])
        self.models = [
            joblib.load(self.artifact_dir / "models" / f"ownership_fold_{fold}.joblib")
            for fold in range(int(spec["folds"]))
        ]
        if len(self.models) != 5:
            raise RuntimeError("Expected five grouped-video ownership models")

    @staticmethod
    def _model_index(stem: str) -> int:
        # Hidden videos are unseen by every fold model. Stable routing keeps
        # the same single-fold probability scale used by grouped-video OOF.
        digest = hashlib.sha256(stem.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % 5

    def _apply_specialist(
        self,
        dataset_path,
        model_c_evidence_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes,
        original,
        stats,
        registration_shifts_um,
        spacing,
        legacy_thresholds,
    ):
        base = self.specialist
        evidence = base._serving_evidence(
            dataset_path,
            model_c_evidence_path,
            p1_evidence_path,
            p2_evidence_path,
            nodes,
            original,
            registration_shifts_um,
            spacing,
        )
        ug1_edges = base.primary.apply(
            dataset_path,
            model_c_evidence_path,
            p1_evidence_path,
            p2_evidence_path,
            nodes,
            original,
            stats,
            registration_shifts_um=registration_shifts_um,
            spacing=spacing,
            **legacy_thresholds,
        )
        try:
            stats["unigraft2_threshold"] = base.ug2_threshold
            ug12_edges = base.base._merge(
                nodes,
                ug1_edges,
                evidence.get("ug2_candidates", []),
                evidence.get("component_of", {}),
                stats,
            )
        except Exception as error:
            stats["unigraft2_merge_fallback"] = 1
            stats["unigraft2_merge_error"] = f"{type(error).__name__}: {error}"
            ug12_edges = ug1_edges

        first, below = base._candidate_bank(dataset_path, nodes, original, evidence)
        work = base._apply_transactions(nodes, ug12_edges, first, "opposition", stats)
        work = base._apply_transactions(nodes, work, below, "below_gate", stats)
        runner = base._runner_rows(nodes, work, evidence)
        work = base._apply_transactions(nodes, work, runner, "tube_runner", stats)
        base._validate(nodes, work)
        stats["live_v2_global_bundle_enabled"] = 1
        stats["live_v2_global_bundle_no_extra_v2_pass"] = 1
        stats["live_v2_global_bundle_total_applied"] = int(
            stats.get("live_bundle_opposition_applied", 0)
            + stats.get("live_bundle_below_gate_applied", 0)
            + stats.get("live_bundle_tube_runner_applied", 0)
        )
        return work, evidence, ug12_edges

    def _rows(self, stem: str, nodes, edges, evidence, spacing) -> pd.DataFrame:
        outgoing, incoming, probability = _edge_maps(edges)
        spacing_array = np.asarray(spacing, np.float64)
        position = {
            int(node): np.asarray((row["z"], row["y"], row["x"]), np.float64)
            * spacing_array
            for node, row in nodes.items()
        }
        source_ids = np.asarray(evidence["source_ids"], np.int64)
        source_score = np.asarray(evidence["source_score"], np.float64)
        v2_raw = np.asarray(evidence["v2_raw"], np.float64)
        deployed = np.asarray(evidence["deployed_best"], np.int64)
        experts = {
            "model_c": np.asarray(evidence["model_c_best"], np.int64),
            "p1": np.asarray(evidence["p1_best"], np.int64),
            "p2": np.asarray(evidence["p2_best"], np.int64),
            "v2": np.asarray(evidence["v2_best"], np.int64),
        }
        records: list[dict[str, Any]] = []
        for row_index, source_value in enumerate(source_ids):
            source = int(source_value)
            children = outgoing.get(source, ())
            pair = _canonical(deployed[row_index])
            a, b = pair
            if (
                source not in nodes
                or len(children) != 1
                or a < 0
                or b < 0
                or a == b
                or a not in nodes
                or b not in nodes
            ):
                continue
            current = int(children[0])
            if current not in pair:
                continue
            source_time = int(nodes[source]["t"])
            if int(nodes[a]["t"]) != source_time + 1 or int(nodes[b]["t"]) != source_time + 1:
                continue
            alternate = b if current == a else a
            incumbent_values = incoming.get(alternate, ())
            incumbent = int(incumbent_values[0]) if len(incumbent_values) == 1 else -1
            incumbent_safe = incumbent if incumbent in nodes else source
            previous_values = incoming.get(source, ())
            previous = int(previous_values[0]) if len(previous_values) == 1 else source
            incumbent_previous_values = incoming.get(incumbent_safe, ())
            incumbent_previous = (
                int(incumbent_previous_values[0])
                if incumbent >= 0 and len(incumbent_previous_values) == 1
                else incumbent_safe
            )
            source_pos = position[source]
            current_pos = position[current]
            alternate_pos = position[alternate]
            incumbent_pos = position[incumbent_safe]
            previous_pos = position[previous]
            midpoint = (current_pos + alternate_pos) / 2.0
            source_velocity = source_pos - previous_pos
            alternate_parent = _norm(alternate_pos - source_pos)
            current_parent = _norm(current_pos - source_pos)
            incumbent_distance = (
                _norm(alternate_pos - incumbent_pos) if incumbent >= 0 else 99.0
            )
            flags = {
                "deployed": 1,
                **{
                    name: int(_canonical(values[row_index]) == pair)
                    for name, values in experts.items()
                },
            }
            records.append({
                "dataset": stem,
                "source": source,
                "a": a,
                "b": b,
                "source_time": source_time,
                "source_score": float(source_score[row_index]),
                "v2_source_score_raw": float(v2_raw[row_index]),
                "steal_distance_delta_um": alternate_parent - incumbent_distance,
                "alternate_minus_current_um": alternate_parent - current_parent,
                "incumbent_edge_prob": float(
                    probability.get((incumbent, alternate), -1.0)
                ),
                "alternate_parent_um": alternate_parent,
                "steal_distance_ratio": alternate_parent / max(incumbent_distance, 1e-3),
                "incumbent_back_depth": _depth(incoming, incumbent_safe) if incumbent >= 0 else 0,
                "predicted_midpoint_error_um": _norm(
                    midpoint - (source_pos + source_velocity)
                ),
                "agreement4": flags["deployed"] + flags["model_c"] + flags["p1"] + flags["p2"],
                "agreement5": sum(flags.values()),
            })
        if not records:
            return pd.DataFrame(columns=["dataset", "source", "a", "b", "score", *FEATURES])
        frame = pd.DataFrame(records)
        grouped = frame.groupby(["dataset", "source_time"], sort=False)
        frame["source_frame_percentile"] = grouped.source_score.rank(
            method="average", pct=True
        )
        source_log = np.log10(np.clip(frame.source_score.to_numpy(float), 1e-8, 1.0))
        frame["source_frame_log_margin"] = source_log - grouped.source_score.transform(
            lambda values: np.log10(max(float(values.max()), 1e-8))
        )
        frame["v2_frame_percentile"] = grouped.v2_source_score_raw.rank(
            method="average", pct=True
        )
        v2_log = np.log10(np.clip(frame.v2_source_score_raw.to_numpy(float), 1e-8, 1.0))
        frame["v2_frame_log_margin"] = v2_log - grouped.v2_source_score_raw.transform(
            lambda values: np.log10(max(float(values.max()), 1e-8))
        )
        frame["time_norm"] = frame.source_time / max(float(frame.source_time.max()), 1.0)
        matrix = (
            frame[FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(np.float32)
        )
        model_index = self._model_index(stem)
        frame["score"] = self.models[model_index].predict_proba(matrix)[:, 1]
        frame["ownership_model_fold"] = model_index
        return frame.loc[frame.score.ge(self.threshold)].copy()

    @staticmethod
    def _apply_ownership(nodes, edges, rows: pd.DataFrame, stats):
        work = [dict(edge) for edge in edges]
        outgoing, incoming, _ = _edge_maps(work)
        locked_sources = {source for source, children in outgoing.items() if len(children) >= 2}
        locked_targets = {
            child for source in locked_sources for child in outgoing.get(source, ())
        }
        counters: Counter[str] = Counter()
        for row in rows.sort_values("score", ascending=False).itertuples(index=False):
            source, a, b = int(row.source), int(row.a), int(row.b)
            if source in locked_sources or source not in nodes:
                counters["rejected_source"] += 1
                continue
            if a == b or a not in nodes or b not in nodes:
                counters["rejected_daughter"] += 1
                continue
            if a in locked_targets or b in locked_targets:
                counters["rejected_locked_target"] += 1
                continue
            source_time = int(nodes[source]["t"])
            if int(nodes[a]["t"]) != source_time + 1 or int(nodes[b]["t"]) != source_time + 1:
                counters["rejected_nonadjacent"] += 1
                continue
            protected = any(
                parent != source and len(outgoing.get(parent, ())) >= 2
                for child in (a, b)
                for parent in incoming.get(child, ())
            )
            if protected:
                counters["rejected_protected_fork_target"] += 1
                continue
            pair = {a, b}
            kept = []
            for edge in work:
                edge_source = int(edge["source_id"])
                edge_target = int(edge["target_id"])
                remove_source = edge_source == source and edge_target not in pair
                remove_claim = edge_target in pair and edge_source != source
                if remove_source:
                    counters["removed_source_continuation"] += 1
                elif remove_claim:
                    counters["removed_competing_claim"] += 1
                else:
                    kept.append(edge)
            work = kept
            present = {(int(edge["source_id"]), int(edge["target_id"])) for edge in work}
            for child in (a, b):
                if (source, child) not in present:
                    work.append({
                        "source_id": source,
                        "target_id": child,
                        "edge_prob": float(row.score),
                        "learned_division": 1,
                        "full_population_ownership": 1,
                        "full_population_ownership_score": float(row.score),
                    })
                    counters["added_edge"] += 1
            outgoing, incoming, _ = _edge_maps(work)
            if len(outgoing.get(source, ())) != 2:
                raise RuntimeError(f"Atomic ownership transaction failed at {source}")
            locked_sources.add(source)
            locked_targets.update(pair)
            counters["applied"] += 1
        outgoing, incoming, _ = _edge_maps(work)
        if incoming and max(map(len, incoming.values())) > 1:
            raise RuntimeError("Full-population ownership produced in-degree > 1")
        if outgoing and max(map(len, outgoing.values())) > 2:
            raise RuntimeError("Full-population ownership produced out-degree > 2")
        for key, value in counters.items():
            stats[f"ownership_full_{key}"] = int(value)
        stats["ownership_full_selected"] = int(len(rows))
        stats["ownership_full_threshold"] = float(rows.score.min()) if len(rows) else 0.0
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
        original = [dict(edge) for edge in edges]
        try:
            specialist_edges, evidence, feature_edges = self._apply_specialist(
                dataset_path,
                model_c_evidence_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                original,
                stats,
                registration_shifts_um,
                spacing,
                legacy_thresholds,
            )
        except Exception as error:
            stats["ownership_full_specialist_fallback"] = 1
            stats["ownership_full_specialist_error"] = f"{type(error).__name__}: {error}"
            return self.specialist.apply(
                dataset_path,
                model_c_evidence_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                original,
                stats,
                registration_shifts_um=registration_shifts_um,
                spacing=spacing,
                **legacy_thresholds,
            )
        try:
            stem = Path(dataset_path).stem
            rows = self._rows(stem, nodes, feature_edges, evidence, spacing)
            result = self._apply_ownership(nodes, specialist_edges, rows, stats)
            self.specialist._validate(nodes, result)
            stats["ownership_full_population_enabled"] = 1
            stats["ownership_full_model_fold"] = self._model_index(stem)
            stats["ownership_full_no_extra_v2_pass"] = 1
            return result
        except Exception as error:
            stats["ownership_full_fallback"] = 1
            stats["ownership_full_error"] = f"{type(error).__name__}: {error}"
            return specialist_edges


__all__ = ["FullPopulationOwnershipRuntime"]
