"""V3 runtime preserving dropped proposal nodes as context-only sentinels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import edgegraft_component_runtime_v1 as base
from edgegraft_v3_deploy_runtime_v2 import EdgeGraftV3Runtime as _RuntimeV2


_SENTINEL_BASE = 10_000_000_000


def _read_evidence_with_context_sentinels(path: str | Path, raw_nodes):
    with np.load(path, allow_pickle=False) as data:
        source = data["source_id"].astype(np.int64, copy=False)
        target = data["target_id"].astype(np.int64, copy=False)
        probability = data["probability"].astype(np.float32, copy=False)
        alternative = data["alternative_parent_probability"].astype(np.float32, copy=False)
        winner = data["is_target_winner"].astype(np.float32, copy=False)
        rank = data["source_target_rank"].astype(np.int16, copy=False)
        distance = data["distance_um"].astype(np.float32, copy=False)
        fused_mapping = data["mapped_ab_node"].astype(np.int64, copy=False)
        fused_coords = data["ab_node_coords"].astype(np.float32, copy=False)
        raw_by_identity = {
            base._identity(row): int(node_id) for node_id, row in raw_nodes.items()
        }
        fused_graph_ids = np.asarray([
            raw_by_identity.get(
                (
                    int(round(float(coord[0]))), round(float(coord[1]), 5),
                    round(float(coord[2]), 5), round(float(coord[3]), 5),
                ),
                _SENTINEL_BASE + index,
            )
            for index, coord in enumerate(fused_coords)
        ], np.int64)
        mapped = np.full(len(fused_mapping), -1, np.int64)
        usable = (fused_mapping >= 0) & (fused_mapping < len(fused_graph_ids))
        mapped[usable] = fused_graph_ids[fused_mapping[usable]]
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
            rows.append((
                mapped_source, mapped_target, float(probability[index]),
                float(alternative[index]), float(winner[index]),
                float(rank[index]), float(distance[index]),
            ))
    deduped = {}
    for row in rows:
        source_id, target_id = row[0], row[1]
        value = row[2:]
        old = deduped.get((source_id, target_id))
        if old is None or value[0] > old[0]:
            deduped[(source_id, target_id)] = value
    context_rows = [
        (source_id, target_id, probability_value, alternative_value,
         winner_value, distance_value)
        for source_id, target_id, probability_value, alternative_value,
            winner_value, _rank_value, distance_value in rows
    ]
    return deduped, context_rows


class EdgeGraftV3Runtime(_RuntimeV2):
    def _candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges,
                         p1_path, p2_path):
        original = base._read_evidence
        base._read_evidence = _read_evidence_with_context_sentinels
        try:
            return super()._candidate_frame(
                raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
            )
        finally:
            base._read_evidence = original


__all__ = ["EdgeGraftV3Runtime"]
