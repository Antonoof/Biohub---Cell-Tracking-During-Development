#!/usr/bin/env python3
"""Build sparse-safe current-population labels for EdgeGRAFT v2.

Maps frozen P1/P2 candidate rows into the exact current UG2/UG3-boundary graph
and labels only targets whose ordinary-continuation GT parent is represented by
a candidate edge. Unknown targets are never converted into negatives. Signed
physical 3-D and motion-consistency features are added without image inference
or graph mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import audit_edgegraft_component_oracle_v1 as edgebase
from replay_edgegraft_component_model_v1 import map_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/tweak/bio/public914_backbone_matched_v1"))
    parser.add_argument("--baseline", type=Path, default=Path("/home/tweak/bio/ug23_boundary_oof_exact_v1/final_candidate"))
    parser.add_argument("--rows", type=Path, default=Path("/home/tweak/bio/public944_p1p2_full_conflicts_all199_v1"))
    parser.add_argument("--data", type=Path, default=Path("/home/tweak/bio/train"))
    parser.add_argument("--output", type=Path, default=Path("/home/tweak/bio/edgegraft_current_labels_v2"))
    parser.add_argument("--match-distance", type=float, default=7.0)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def node_table(graph) -> dict[int, dict]:
    return {int(row["node_id"]): dict(row) for row in graph.node_attrs().to_dicts()}


def physical_position(attrs: dict, scale: np.ndarray) -> np.ndarray:
    return np.asarray([float(attrs["z"]), float(attrs["y"]), float(attrs["x"])], np.float64) * scale


def unique_neighbor(graph, node: int, predecessor: bool) -> int | None:
    values = list(map(int, graph.predecessors(node) if predecessor else graph.successors(node)))
    return values[0] if len(values) == 1 else None


def augment_geometry(frame: pd.DataFrame, raw, raw_nodes: dict[int, dict], scale: np.ndarray) -> pd.DataFrame:
    """Add signed physical geometry without consulting GT."""
    sources = frame.source.to_numpy(np.int64, copy=False)
    targets = frame.target.to_numpy(np.int64, copy=False)
    source_xyz = np.stack([physical_position(raw_nodes[int(n)], scale) for n in sources])
    target_xyz = np.stack([physical_position(raw_nodes[int(n)], scale) for n in targets])
    displacement = target_xyz - source_xyz
    distance = np.linalg.norm(displacement, axis=1)
    unit = displacement / np.maximum(distance[:, None], 1e-6)

    all_xyz = np.stack([physical_position(value, scale) for value in raw_nodes.values()])
    lo = all_xyz.min(axis=0)
    span = np.maximum(all_xyz.max(axis=0) - lo, 1e-6)
    source_norm = (source_xyz - lo) / span
    target_norm = (target_xyz - lo) / span

    previous = {int(node): unique_neighbor(raw, int(node), True) for node in np.unique(sources)}
    future = {int(node): unique_neighbor(raw, int(node), False) for node in np.unique(targets)}
    incoming = np.zeros_like(displacement)
    outgoing = np.zeros_like(displacement)
    incoming_valid = np.zeros(len(frame), np.float32)
    outgoing_valid = np.zeros(len(frame), np.float32)
    for index, (source, target) in enumerate(zip(sources, targets)):
        prev = previous[int(source)]
        if prev is not None and prev in raw_nodes:
            incoming[index] = source_xyz[index] - physical_position(raw_nodes[prev], scale)
            incoming_valid[index] = 1.0
        nxt = future[int(target)]
        if nxt is not None and nxt in raw_nodes:
            outgoing[index] = physical_position(raw_nodes[nxt], scale) - target_xyz[index]
            outgoing_valid[index] = 1.0

    def vector_features(prefix: str, vector: np.ndarray) -> dict[str, np.ndarray]:
        magnitude = np.linalg.norm(vector, axis=1)
        cosine = np.sum(displacement * vector, axis=1) / np.maximum(distance * magnitude, 1e-6)
        residual = displacement - vector
        return {
            f"{prefix}_dz_um": vector[:, 0], f"{prefix}_dy_um": vector[:, 1], f"{prefix}_dx_um": vector[:, 2],
            f"{prefix}_speed_um": magnitude, f"{prefix}_candidate_cosine": np.clip(cosine, -1.0, 1.0),
            f"{prefix}_residual_dz_um": residual[:, 0], f"{prefix}_residual_dy_um": residual[:, 1],
            f"{prefix}_residual_dx_um": residual[:, 2], f"{prefix}_residual_um": np.linalg.norm(residual, axis=1),
        }

    values: dict[str, np.ndarray] = {
        "geom_dz_um": displacement[:, 0], "geom_dy_um": displacement[:, 1], "geom_dx_um": displacement[:, 2],
        "geom_unit_z": unit[:, 0], "geom_unit_y": unit[:, 1], "geom_unit_x": unit[:, 2],
        "geom_distance_um_exact": distance,
        "geom_source_z_norm": source_norm[:, 0], "geom_source_y_norm": source_norm[:, 1], "geom_source_x_norm": source_norm[:, 2],
        "geom_target_z_norm": target_norm[:, 0], "geom_target_y_norm": target_norm[:, 1], "geom_target_x_norm": target_norm[:, 2],
        "geom_incoming_valid": incoming_valid, "geom_outgoing_valid": outgoing_valid,
    }
    values.update(vector_features("geom_incoming", incoming))
    values.update(vector_features("geom_outgoing", outgoing))
    return frame.assign(**{key: value.astype(np.float32) for key, value in values.items()})


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    sys.path.insert(0, "/home/tweak/bio/kaggle-cell-tracking-competition-patched/src")
    sys.path.insert(0, "/home/tweak/bio_track_repo/src")
    from biohub_tracking.io import open_dataset

    raw_dir = args.root / "repo/predictions/tweak/public914_train_raw/split_0"
    stems = sorted(path.stem for path in args.baseline.glob("*.geff"))
    if not stems:
        raise RuntimeError(f"No baseline graphs found: {args.baseline}")

    records: list[dict] = []
    totals: Counter = Counter()
    for index, stem in enumerate(stems, 1):
        baseline = edgebase.load_graph(args.baseline / f"{stem}.geff")
        raw = edgebase.load_graph(raw_dir / f"{stem}.geff")
        gt = edgebase.load_graph(args.data / f"{stem}.geff")
        dataset = open_dataset(args.data / f"{stem}.zarr", require_tracks=False, load_image=False, device="cpu")
        scale = np.asarray(dataset.scale, np.float64)
        truth = edgebase.matched_truth(baseline, gt, scale, args.match_distance)
        truth_by_target = {int(target): int(source) for source, target in truth}

        mapping = edgebase.raw_to_final_map(raw, baseline)
        raw_frame = pd.read_parquet(args.rows / f"{stem}.parquet")
        raw_frame = augment_geometry(raw_frame, raw, node_table(raw), scale)
        frame = map_rows(raw_frame, mapping)
        known_targets = set(truth_by_target) & set(map(int, frame.target.unique()))
        frame = frame[frame.target.isin(known_targets)].copy()
        frame["y"] = np.fromiter(
            (int(truth_by_target[int(target)] == int(source)) for source, target in frame[["source", "target"]].itertuples(index=False)),
            np.int8, len(frame),
        )
        represented_targets = set(map(int, frame.loc[frame.y == 1, "target"].unique()))
        frame = frame[frame.target.isin(represented_targets)].copy()
        current = edgebase.edge_set(baseline)
        frame["is_current_parent"] = np.fromiter(
            (float((int(source), int(target)) in current) for source, target in frame[["source", "target"]].itertuples(index=False)),
            np.float32, len(frame),
        )
        frame["video_fold"] = int(zlib.crc32(stem.encode("utf-8")) % args.folds)
        frame.to_parquet(args.output / f"{stem}.parquet", index=False)

        positive_targets = int(frame.loc[frame.y == 1, "target"].nunique())
        current_correct = int(frame.loc[(frame.y == 1) & (frame.is_current_parent == 1), "target"].nunique())
        record = {"dataset": stem, "fold": int(frame.video_fold.iloc[0]) if len(frame) else -1,
                  "truth_edges": len(truth), "represented_targets": positive_targets,
                  "candidate_rows": len(frame), "current_correct": current_correct}
        records.append(record)
        totals["videos"] += 1
        for key in ("truth_edges", "represented_targets", "candidate_rows", "current_correct"):
            totals[key] += int(record[key])
        print(f"[{index:03d}/{len(stems)}] {stem}: truth={len(truth):,} represented={positive_targets:,} rows={len(frame):,} current={current_correct:,}", flush=True)

    payload = {
        "version": "edgegraft-current-labels-v2",
        "contract": {"unknown_targets_are_negatives": False, "ordinary_continuation_only": True,
                     "current_population": str(args.baseline), "candidate_population": str(args.rows), "folds": args.folds},
        "videos": records, "totals": {key: int(value) for key, value in totals.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
