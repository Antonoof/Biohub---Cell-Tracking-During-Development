"""V3 runtime preserving both raw-parent and final-current-parent contracts."""

from __future__ import annotations

import numpy as np

import edgegraft_component_runtime_v1 as base
from edgegraft_v3_deploy_runtime_v3 import EdgeGraftV3Runtime as _RuntimeV3


class EdgeGraftV3Runtime(_RuntimeV3):
    def _candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges,
                         p1_path, p2_path):
        frame = super()._candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        if not frame.empty:
            mapping = base._raw_to_final_map(raw_nodes, final_nodes)
            mapped_raw_edges = {
                (int(mapping[source]), int(mapping[target]))
                for edge in raw_edges
                for source, target in [
                    (int(edge["source_id"]), int(edge["target_id"]))
                ]
                if source in mapping and target in mapping
            }
            frame["is_raw_parent"] = np.fromiter(
                (
                    float((int(source), int(target)) in mapped_raw_edges)
                    for source, target in frame[["source", "target"]].itertuples(index=False)
                ),
                np.float32,
                len(frame),
            )
        return frame


__all__ = ["EdgeGraftV3Runtime"]
