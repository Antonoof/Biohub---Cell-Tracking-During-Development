"""V3 runtime with the frozen extraction's explicit current-parent feature."""

from __future__ import annotations

import numpy as np

from edgegraft_v3_deploy_runtime_v1 import EdgeGraftV3Runtime as _RuntimeV1


class EdgeGraftV3Runtime(_RuntimeV1):
    def _candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges,
                         p1_path, p2_path):
        frame = super()._candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        if not frame.empty:
            current = {
                (int(edge["source_id"]), int(edge["target_id"]))
                for edge in final_edges
            }
            frame["is_current_parent"] = np.fromiter(
                (
                    float((int(source), int(target)) in current)
                    for source, target in frame[["source", "target"]].itertuples(index=False)
                ),
                np.float32,
                len(frame),
            )
        return frame


__all__ = ["EdgeGraftV3Runtime"]
