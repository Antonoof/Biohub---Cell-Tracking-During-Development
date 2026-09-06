"""Inference-safe deployment runtime for the full-population EdgeGRAFT V3 gate."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any
import zlib

import joblib
import numpy as np
import pandas as pd

import edgegraft_component_runtime_v1 as base
import edgegraft_v15_runtime_v1 as shared


class EdgeGraftV3Runtime:
    def __init__(self, ranker_dir: str | Path, metric_dir: str | Path):
        self.ranker_dir = Path(ranker_dir)
        self.metric_dir = Path(metric_dir)
        self.ranker_report = json.loads((self.ranker_dir / "report.json").read_text())
        self.metric_report = json.loads((self.metric_dir / "report.json").read_text())
        self.threshold = float(self.metric_report["frozen_median_threshold"])
        self._base_builder = base.EdgeGraftComponentRuntime.__new__(
            base.EdgeGraftComponentRuntime
        )
        self._models: dict[tuple[str, int], Any] = {}

    def _model(self, family: str, fold: int):
        key = (family, fold)
        if key not in self._models:
            directory = self.ranker_dir if family == "ranker" else self.metric_dir
            self._models[key] = joblib.load(directory / f"fold_{fold}.joblib")
        return self._models[key]

    def _candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges,
                         p1_path, p2_path):
        return shared.EdgeGraftV15Runtime._candidate_frame(
            self, raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )

    def _decisions(self, stem, frame, final_edges, protected):
        return shared.EdgeGraftV15Runtime._decisions(
            self, stem, frame, final_edges, protected
        )

    def _rich_decisions(self, decision, evidence):
        return shared.EdgeGraftV15Runtime._rich_decisions(
            self, decision, evidence
        )

    def apply(self, stem: str, raw_nodes: dict[int, dict], raw_edges: list[dict],
              final_nodes: dict[int, dict], final_edges: list[dict],
              p1_path: str | Path, p2_path: str | Path):
        counters: Counter = Counter()
        fold = int(zlib.crc32(stem.encode("utf-8")) % 5)
        counters["fold"] = fold
        frame = self._candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        counters["candidate_rows"] = len(frame)
        if frame.empty:
            return final_edges, dict(counters)

        rank_features = list(self.ranker_report["features"])
        rank_x = frame.reindex(columns=rank_features).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).to_numpy(np.float32)
        frame["rank_score"] = self._model("ranker", fold).predict_proba(rank_x)[:, 1]
        frame = frame.sort_values("rank_score", ascending=False).drop_duplicates(
            ["source", "target"], keep="first"
        )

        protected = base._protected_fork_nodes(final_edges)
        decision = self._decisions(stem, frame, final_edges, protected)
        counters["decisions"] = len(decision)
        if decision.empty:
            return final_edges, dict(counters)
        rich = self._rich_decisions(decision, frame)
        metric_features = list(self.metric_report["features"])
        metric_x = rich.reindex(columns=metric_features).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).to_numpy(np.float32)
        rich["metric_score"] = self._model("metric", fold).predict_proba(metric_x)[:, 1]
        selected = rich[
            (~rich.protected) & (rich.metric_score >= self.threshold)
        ].sort_values(
            ["metric_score", "top_score", "top_margin"], ascending=False
        )
        counters["selected"] = len(selected)
        chosen = []
        used_sources: set[int] = set()
        for row in selected.itertuples(index=False):
            source = int(row.source)
            if source in used_sources:
                counters["source_conflict"] += 1
                continue
            used_sources.add(source)
            chosen.append(row)
        counters["applied"] = len(chosen)
        if not chosen:
            return final_edges, dict(counters)

        incoming, outgoing = shared._edge_sets(final_edges)
        edge_by_pair = {
            (int(edge["source_id"]), int(edge["target_id"])): dict(edge)
            for edge in final_edges
        }
        remove: set[tuple[int, int]] = set()
        add: set[tuple[int, int]] = set()
        probability = {
            (int(row.source), int(row.target)): float(row.native_probability)
            for row in frame.itertuples(index=False)
        }
        for row in chosen:
            source = int(row.source)
            target = int(row.target)
            current_source = int(row.current_source)
            source_target = int(row.source_current_target)
            if (current_source, target) not in edge_by_pair:
                raise RuntimeError("V3 stale target parent before atomic transaction")
            actual = next(iter(outgoing[source])) if len(outgoing[source]) == 1 else -1
            if len(outgoing[source]) > 1 or actual != source_target:
                raise RuntimeError("V3 stale source child before atomic transaction")
            remove.add((current_source, target))
            if source_target >= 0 and source_target != target:
                remove.add((source, source_target))
            add.add((source, target))
        for edge in remove:
            edge_by_pair.pop(edge, None)
        for source, target in add:
            edge_by_pair.setdefault((source, target), {
                "source_id": source,
                "target_id": target,
                "edge_prob": probability.get((source, target), 1.0),
            })
        result = list(edge_by_pair.values())
        final_incoming, final_outgoing = shared._edge_sets(result)
        if any(len(value) > 1 for value in final_incoming.values()):
            raise RuntimeError("V3 transaction produced in-degree > 1")
        if any(len(value) > 2 for value in final_outgoing.values()):
            raise RuntimeError("V3 transaction produced out-degree > 2")
        if base._protected_fork_nodes(result) != protected:
            raise RuntimeError("V3 changed protected fork topology")
        counters["removed_edges"] = len(remove)
        counters["added_edges"] = len(add)
        return result, dict(counters)


__all__ = ["EdgeGraftV3Runtime"]
