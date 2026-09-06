"""V3 runtime with frozen float32 geometry before relative ranking."""

from __future__ import annotations

import numpy as np

import edgegraft_v15_runtime_v1 as shared
from edgegraft_v3_deploy_runtime_v4 import EdgeGraftV3Runtime as _RuntimeV4


class EdgeGraftV3Runtime(_RuntimeV4):
    def _candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges,
                         p1_path, p2_path):
        frame = super()._candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        if not frame.empty:
            geometry = [column for column in frame.columns if column.startswith("geom_")]
            for column in geometry:
                frame[column] = frame[column].astype(np.float32)
            frame = shared._add_relative(frame)
        return frame


__all__ = ["EdgeGraftV3Runtime"]
