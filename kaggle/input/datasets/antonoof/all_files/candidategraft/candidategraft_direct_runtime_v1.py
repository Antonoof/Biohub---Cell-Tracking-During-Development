#!/usr/bin/env python3
"""Production CandidateGRAFT direct-edge runtime.

Uses the already-exported P1/P2 native association tables.  It adds only a
one-frame edge between two retained nodes when the source has no child and the
target has no parent.  Existing edges and division forks are never removed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "p1", "p2", "pmax", "pmean", "present", "winner_count", "rank_min", "distance",
]


def _degrees(edges):
    incoming, outgoing = Counter(), Counter()
    for edge in edges:
        outgoing[int(edge["source_id"])] += 1
        incoming[int(edge["target_id"])] += 1
    return incoming, outgoing


def _forks(edges):
    _incoming, outgoing = _degrees(edges)
    return {node for node, degree in outgoing.items() if degree >= 2}


class CandidateGraftDirectRuntime:
    def __init__(self, artifact_dir: str | Path, edgegraft_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.edgegraft_dir = Path(edgegraft_dir)
        report = __import__("json").loads((self.artifact_dir / "report.json").read_text())
        if list(report["features"]) != FEATURES:
            raise RuntimeError("Unexpected CandidateGRAFT feature contract")
        self.threshold = float(report["threshold"])
        self.model = joblib.load(self.artifact_dir / "candidategraft_direct.joblib")

        import sys
        edgegraft_dir = str(self.edgegraft_dir)
        if edgegraft_dir not in sys.path:
            sys.path.insert(0, edgegraft_dir)
        import edgegraft_component_runtime_v1 as base
        self._read_evidence = base._read_evidence

    def _population(self, raw_nodes, final_nodes, final_edges, p1_path, p2_path):
        p1, _p1_context = self._read_evidence(p1_path, raw_nodes)
        p2, _p2_context = self._read_evidence(p2_path, raw_nodes)
        final_set = set(map(int, final_nodes))
        current = {
            (int(edge["source_id"]), int(edge["target_id"]))
            for edge in final_edges
        }
        incoming, outgoing = _degrees(final_edges)
        rows = []
        for source, target in set(p1) | set(p2):
            source, target = int(source), int(target)
            if source not in final_set or target not in final_set:
                continue
            if (source, target) in current:
                continue
            if outgoing[source] != 0 or incoming[target] != 0:
                continue
            source_row, target_row = final_nodes[source], final_nodes[target]
            if int(target_row["t"]) != int(source_row["t"]) + 1:
                continue
            first, second = p1.get((source, target)), p2.get((source, target))
            p1_probability = float(first[0]) if first else 0.0
            p2_probability = float(second[0]) if second else 0.0
            rows.append({
                "source": source,
                "target": target,
                "p1": p1_probability,
                "p2": p2_probability,
                "pmax": max(p1_probability, p2_probability),
                "pmean": (p1_probability + p2_probability) / 2.0,
                "present": float(first is not None) + float(second is not None),
                "winner_count": float(first[2] if first else 0.0) + float(second[2] if second else 0.0),
                "rank_min": min(float(first[3]) if first else 99.0, float(second[3]) if second else 99.0),
                "distance": min(float(first[4]) if first else 99.0, float(second[4]) if second else 99.0),
            })
        return pd.DataFrame(rows)

    def apply(self, dataset, raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path):
        del dataset, raw_edges
        protected = _forks(final_edges)
        frame = self._population(raw_nodes, final_nodes, final_edges, p1_path, p2_path)
        counters = Counter(population=int(len(frame)))
        if frame.empty:
            return list(final_edges), dict(counters)
        matrix = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        frame["score"] = self.model.predict_proba(matrix)[:, 1]
        selected = frame[frame.score >= self.threshold].sort_values(
            ["score", "pmax"], ascending=False,
        )
        counters["selected"] = int(len(selected))
        result = [dict(edge) for edge in final_edges]
        incoming, outgoing = _degrees(result)
        final_set = set(map(int, final_nodes))
        for row in selected.itertuples(index=False):
            source, target = int(row.source), int(row.target)
            if source not in final_set or target not in final_set:
                counters["missing_node"] += 1
                continue
            if outgoing[source] != 0 or incoming[target] != 0:
                counters["conflict"] += 1
                continue
            result.append({
                "source_id": source,
                "target_id": target,
                "edge_prob": float(row.pmax),
                "distance_um": float(row.distance),
            })
            outgoing[source] += 1
            incoming[target] += 1
            counters["applied"] += 1
        if any(value > 1 for value in incoming.values()):
            raise RuntimeError("CandidateGRAFT left in-degree > 1")
        if any(value > 2 for value in outgoing.values()):
            raise RuntimeError("CandidateGRAFT left out-degree > 2")
        if _forks(result) != protected:
            raise RuntimeError("CandidateGRAFT changed protected fork topology")
        counters["threshold_million"] = int(round(self.threshold * 1_000_000))
        return result, dict(counters)
