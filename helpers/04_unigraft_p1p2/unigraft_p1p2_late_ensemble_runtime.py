"""Independent P1/P2 UniGRAFT branch merged after production UG1.

UG1 and UG2 score the same input graph independently.  UG1 is the frozen
Model-C/V2/source-cardinality transaction.  UG2 uses the same V2 geometry but
only native P1/P2 evidence.  After UG1 completes, UG2 may add a non-conflicting
fork as one atomic parent-to-two-daughter transaction.  Any failure returns the
completed UG1 result unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class OptionHead(nn.Module):
    def __init__(self, source_dim: int, pair_dim: int, hidden_source: int, hidden_pair: int):
        super().__init__()
        self.source_encoder = nn.Sequential(nn.Linear(source_dim, hidden_source), nn.SiLU())
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim, hidden_pair), nn.SiLU())
        self.continue_head = nn.Linear(hidden_source, 1)
        self.divide_head = nn.Linear(hidden_source + hidden_pair, 1)
        self.division_bias = nn.Parameter(torch.zeros(()))


def _grouped_best(owner: np.ndarray, score: np.ndarray, n_sources: int) -> np.ndarray:
    result = np.full(n_sources, -1, np.int64)
    order = np.argsort(owner, kind="stable")
    if not len(order):
        return result
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows = order[left:right]
        result[int(sorted_owner[left])] = int(rows[int(np.argmax(score[rows]))])
    return result


def _cap_options(owner: np.ndarray, maximum: int) -> np.ndarray:
    order = np.argsort(owner, kind="stable")
    if not len(order):
        return order
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return np.sort(
        np.concatenate(
            [order[left : min(right, left + maximum)] for left, right in zip(starts, ends)]
        )
    )


class P1P2OptionRuntime:
    def __init__(self, artifact_dir: str | Path):
        artifact_dir = Path(artifact_dir)
        payload = torch.load(
            artifact_dir / "p1p2_only_source_cardinality_head.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.config = dict(payload["config"])
        expected_mode = "v2_geometry_plus_p1p2_without_model_c"
        if self.config.get("evidence_mode") != expected_mode:
            raise RuntimeError(
                f"UG2 evidence contract mismatch: {self.config.get('evidence_mode')!r}"
            )
        self.threshold = float(self.config["threshold"])
        self.max_pairs = int(self.config["max_pairs"])
        self.model = OptionHead(
            int(self.config["source_dim"]),
            int(self.config["pair_dim"]),
            int(self.config["hidden_source"]),
            int(self.config["hidden_pair"]),
        )
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.source_mean = np.asarray(payload["source_mean"], np.float32)
        self.source_scale = np.asarray(payload["source_scale"], np.float32)
        self.pair_mean = np.asarray(payload["pair_mean"], np.float32)
        self.pair_scale = np.asarray(payload["pair_scale"], np.float32)

    @torch.no_grad()
    def score(
        self,
        source_features: np.ndarray,
        pair_features: np.ndarray,
        pair_owner: np.ndarray,
        pair_nodes: np.ndarray,
        batch_size: int = 250_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.nan_to_num(np.asarray(source_features, np.float32))
        pair = np.nan_to_num(np.asarray(pair_features, np.float32))
        owner = np.asarray(pair_owner, np.int32)
        nodes = np.asarray(pair_nodes, np.int64)
        if source.ndim != 2 or source.shape[1] != len(self.source_mean):
            raise ValueError(f"UG2 source feature mismatch: {source.shape}")
        if pair.ndim != 2 or pair.shape[1] != len(self.pair_mean):
            raise ValueError(f"UG2 pair feature mismatch: {pair.shape}")
        if len(pair) != len(owner) or nodes.shape != (len(pair), 2):
            raise ValueError("UG2 pair owner/node alignment mismatch")
        if len(owner) and (owner.min() < 0 or owner.max() >= len(source)):
            raise ValueError("UG2 pair owner out of bounds")

        retained = _cap_options(owner, self.max_pairs)
        pair, owner, nodes = pair[retained], owner[retained], nodes[retained]
        source_norm = np.clip(
            (source - self.source_mean) / self.source_scale, -10.0, 10.0
        )
        source_h = self.model.source_encoder(torch.from_numpy(source_norm)).cpu()
        continue_logit = self.model.continue_head(source_h).squeeze(-1).numpy()
        pair_logits = np.empty(len(pair), np.float32)
        for left in range(0, len(pair), batch_size):
            right = min(len(pair), left + batch_size)
            block = np.clip(
                (pair[left:right] - self.pair_mean) / self.pair_scale,
                -10.0,
                10.0,
            )
            pair_h = self.model.pair_encoder(torch.from_numpy(block))
            source_block = source_h[
                torch.from_numpy(owner[left:right].astype(np.int64))
            ]
            value = self.model.divide_head(
                torch.cat([source_block, pair_h], dim=1)
            ).squeeze(-1)
            pair_logits[left:right] = (
                value + self.model.division_bias
            ).cpu().numpy()

        n_sources = len(source)
        maximum = np.full(n_sources, -np.inf, np.float32)
        np.maximum.at(maximum, owner, pair_logits)
        total = np.zeros(n_sources, np.float64)
        valid_pair = np.isfinite(maximum[owner])
        np.add.at(
            total,
            owner[valid_pair],
            np.exp(pair_logits[valid_pair] - maximum[owner[valid_pair]]),
        )
        valid_source = total > 0
        log_total = np.full(n_sources, -np.inf, np.float32)
        log_total[valid_source] = (
            maximum[valid_source]
            + np.log(total[valid_source]).astype(np.float32)
        )
        delta = np.clip(log_total - continue_logit, -40.0, 40.0)
        probability = np.zeros(n_sources, np.float32)
        probability[valid_source] = 1.0 / (1.0 + np.exp(-delta[valid_source]))
        best = _grouped_best(owner, pair_logits, n_sources)
        best_nodes = np.full((n_sources, 2), -1, np.int64)
        valid_best = best >= 0
        best_nodes[valid_best] = nodes[best[valid_best]]
        return probability, best_nodes


class IndependentP1P2LateEnsembleRuntime:
    """Run the independent UG2 branch and merge it after complete UG1."""

    def __init__(self, primary_runtime, artifact_dir: str | Path):
        self.primary = primary_runtime
        self.v2 = primary_runtime.v2
        self.native_features = primary_runtime.native_features
        self.ug2 = P1P2OptionRuntime(artifact_dir)
        self.threshold = float(self.ug2.threshold)

    def _score_independent(
        self,
        dataset_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes: dict[int, dict[str, Any]],
        original_edges: list[dict[str, Any]],
        registration_shifts_um,
        spacing,
    ) -> tuple[list[tuple[float, int, int, int, int]], dict[int, int]]:
        scored = self.v2.score(
            dataset_path,
            nodes,
            original_edges,
            registration_shifts_um,
            spacing,
            max_pairs_per_source=self.ug2.max_pairs,
        )
        source_ids = np.asarray(scored["source_ids"], np.int64)
        source_x = np.asarray(scored["_source_x"], np.float32)
        pair_x = np.asarray(scored["_pair_x"], np.float32)
        owner = np.asarray(scored["_pair_owner"], np.int32)
        pair_nodes = np.asarray(scored["pair_nodes"], np.int64)
        component_of = {
            int(key): int(value) for key, value in scored["_component_of"].items()
        }
        if not len(source_ids) or not len(pair_nodes):
            return [], component_of

        spacing_array = np.asarray(spacing, np.float64)
        blocks = [pair_x]
        for label, path in (("public_p1", p1_evidence_path), ("public_p2", p2_evidence_path)):
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
                raise RuntimeError(f"{label} UG2 feature mismatch: {block.shape}")
            blocks.append(block)
        pair_input = np.concatenate(blocks, axis=1)
        if pair_input.shape[1] != int(self.ug2.config["pair_dim"]):
            raise RuntimeError(f"UG2 assembled pair feature mismatch: {pair_input.shape}")
        probability, best_nodes = self.ug2.score(
            source_x, pair_input, owner, pair_nodes
        )

        winner_by_tube: dict[int, int] = {}
        for row in np.flatnonzero(probability >= self.threshold):
            source = int(source_ids[row])
            tube = int(component_of.get(source, source))
            previous = winner_by_tube.get(tube)
            if previous is None or probability[row] > probability[previous]:
                winner_by_tube[tube] = int(row)
        candidates = sorted(
            (
                float(probability[row]),
                int(source_ids[row]),
                int(best_nodes[row, 0]),
                int(best_nodes[row, 1]),
                int(component_of.get(int(source_ids[row]), int(source_ids[row]))),
            )
            for row in winner_by_tube.values()
            if int(best_nodes[row, 0]) >= 0 and int(best_nodes[row, 1]) >= 0
        )
        candidates.reverse()
        return candidates, component_of

    @staticmethod
    def _merge(
        nodes: dict[int, dict[str, Any]],
        ug1_edges: list[dict[str, Any]],
        candidates: list[tuple[float, int, int, int, int]],
        component_of: dict[int, int],
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        work = [dict(edge) for edge in ug1_edges]
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
        counters: dict[str, int] = defaultdict(int)
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
            kept = []
            for edge in work:
                edge_source = int(edge["source_id"])
                edge_target = int(edge["target_id"])
                remove_parent = edge_source == source and edge_target not in pair
                remove_claim = edge_target in pair and edge_source != source
                if remove_parent:
                    counters["removed_parent_edge"] += 1
                elif remove_claim:
                    counters["removed_claim"] += 1
                else:
                    kept.append(edge)
            work = kept
            present = {(int(edge["source_id"]), int(edge["target_id"])) for edge in work}
            for target in (a, b):
                if (source, target) not in present:
                    work.append(
                        {
                            "source_id": source,
                            "target_id": target,
                            "edge_prob": score,
                            "learned_division": 1,
                            "unigraft2_p1p2": 1,
                            "unigraft2_score": score,
                        }
                    )
                    counters["added_edges"] += 1
            occupied_tubes.add(tube)
            locked_targets.update(pair)
            counters["selected_divisions"] += 1

        indegree: dict[int, int] = defaultdict(int)
        outdegree: dict[int, int] = defaultdict(int)
        for edge in work:
            outdegree[int(edge["source_id"])] += 1
            indegree[int(edge["target_id"])] += 1
        if any(value > 2 for value in outdegree.values()):
            raise RuntimeError("UG2 merge produced out-degree > 2")
        if any(value > 1 for value in indegree.values()):
            raise RuntimeError("UG2 merge produced in-degree > 1")
        for key, value in counters.items():
            stats[f"unigraft2_{key}"] = int(value)
        stats["unigraft2_threshold"] = float(candidates and stats.get("unigraft2_threshold", 0.0) or 0.0)
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
        candidates = None
        component_of = None
        try:
            candidates, component_of = self._score_independent(
                dataset_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                original,
                registration_shifts_um,
                spacing,
            )
        except Exception as error:
            stats["unigraft2_score_fallback"] = 1
            stats["unigraft2_score_error"] = f"{type(error).__name__}: {error}"

        ug1_edges = self.primary.apply(
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
        if candidates is None or component_of is None:
            return ug1_edges
        try:
            stats["unigraft2_threshold"] = self.threshold
            return self._merge(nodes, ug1_edges, candidates, component_of, stats)
        except Exception as error:
            stats["unigraft2_merge_fallback"] = 1
            stats["unigraft2_merge_error"] = f"{type(error).__name__}: {error}"
            return ug1_edges

