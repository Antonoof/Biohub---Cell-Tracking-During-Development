#!/usr/bin/env python3
"""Audit native P1/P2 candidates for EndpointGRAFT and CandidateGRAFT.

The raw native populations are mapped into the stable public-graph node IDs.
Candidate construction is label-free. GT is attached only after construction
for grouped-video coverage diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td
from scipy.spatial import cKDTree


BIO = Path("data")
WORKSPACE = Path(".")
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))

from augment_public_p1p2_temporal_rows_v1 import MODEL_FOLDERS  # noqa: E402
from extract_public_p1p2_full_conflicts_v1 import mapped_all  # noqa: E402
from smoke_edgegraft_v3_deploy_parity_v1 import load  # noqa: E402
from train_public_p1p2_dense_ranker_v1 import dedupe  # noqa: E402


SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=BIO / "public914_backbone_matched_v1",
    )
    parser.add_argument(
        "--final-dir", type=Path,
        default=BIO / "live_v2_bundle_boundary_ug12_edgegraft_v3_oof_v4/graphs",
    )
    parser.add_argument("--data-dir", type=Path, default=BIO / "train")
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=BIO / "native_endpoint_candidate_graft_v1",
    )
    return parser.parse_args()


def graph_object(path: Path):
    value = td.graph.IndexedRXGraph.from_geff(path)
    return value[0] if isinstance(value, tuple) else value


def pred_to_gt(graph, gt, scale, max_distance: float) -> dict[int, int]:
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph.match(
                gt,
                matching=DistanceMatching(
                    max_distance=max_distance,
                    scale=tuple(float(value) for value in scale),
                ),
            )
    finally:
        set_options(show_progress=previous)
    attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    result = {}
    for row in attrs.iter_rows(named=True):
        matched = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        if matched is not None and int(matched) != -1:
            result[int(row[td.DEFAULT_ATTR_KEYS.NODE_ID])] = int(matched)
    return result


def degree_maps(edges: list[dict]) -> tuple[Counter, Counter]:
    incoming, outgoing = Counter(), Counter()
    for edge in edges:
        outgoing[int(edge["source_id"])] += 1
        incoming[int(edge["target_id"])] += 1
    return incoming, outgoing


def combine_evidence(root: Path, stem: str) -> dict[tuple[int, int], dict[str, float]]:
    with np.load(root / "public914_proposals" / f"{stem}.npz", allow_pickle=False) as values:
        graph_node_id = values["graph_node_id"].astype(np.int64, copy=False)
    evidence = {
        name: dedupe(mapped_all(root / folder / f"{stem}.npz", graph_node_id))
        for name, folder in MODEL_FOLDERS.items()
    }
    result: dict[tuple[int, int], dict[str, float]] = {}
    for pair in set(evidence["p1"]) | set(evidence["p2"]):
        a, b = evidence["p1"].get(pair), evidence["p2"].get(pair)
        p1, p2 = (float(a[0]) if a else 0.0), (float(b[0]) if b else 0.0)
        result[pair] = {
            "p1": p1,
            "p2": p2,
            "pmax": max(p1, p2),
            "pmean": (p1 + p2) / 2.0,
            "present": float(a is not None) + float(b is not None),
            "winner_count": float(a[2] if a else 0.0) + float(b[2] if b else 0.0),
            "rank_min": min(float(a[3]) if a else 99.0, float(b[3]) if b else 99.0),
            "distance": min(float(a[4]) if a else 99.0, float(b[4]) if b else 99.0),
        }
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 1.0


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    sys.path.insert(0, "external/bio_track_repo/src")
    from biohub_tracking.io import open_dataset

    raw_dir = args.root / "repo/predictions/tweak/public914_train_raw/split_0"
    paths = sorted(args.final_dir.glob("*.geff"))
    if args.limit:
        paths = paths[: args.limit]
    endpoint_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    per_video: list[dict[str, object]] = []

    for index, final_path in enumerate(paths, 1):
        stem = final_path.stem
        raw_path = raw_dir / final_path.name
        raw_nodes, raw_edges = load(raw_path)
        final_nodes, final_edges = load(final_path)
        native = combine_evidence(args.root, stem)
        final_set = set(final_nodes)
        final_edges_set = {
            (int(edge["source_id"]), int(edge["target_id"])) for edge in final_edges
        }
        final_in, final_out = degree_maps(final_edges)

        incoming: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
        for (source, target), evidence in native.items():
            if source not in raw_nodes or target not in raw_nodes:
                continue
            if int(raw_nodes[target]["t"]) - int(raw_nodes[source]["t"]) != 1:
                continue
            outgoing[source].append((target, evidence))
            incoming[target].append((source, evidence))
        for values in incoming.values():
            values.sort(key=lambda item: (item[1]["pmax"], item[1]["present"]), reverse=True)
        for values in outgoing.values():
            values.sort(key=lambda item: (item[1]["pmax"], item[1]["present"]), reverse=True)

        raw_graph = graph_object(raw_path)
        final_graph = graph_object(final_path)
        gt_graph = graph_object(args.data_dir / final_path.name)
        dataset = open_dataset(
            args.data_dir / f"{stem}.zarr",
            require_tracks=False,
            load_image=False,
            device="cpu",
        )
        scale = np.asarray(dataset.scale, np.float64)
        raw_match = pred_to_gt(raw_graph, gt_graph, scale, args.max_distance)
        final_match = pred_to_gt(final_graph, gt_graph, scale, args.max_distance)
        final_matched_gt = set(final_match.values())
        gt_edges = {(int(source), int(target)) for source, target in gt_graph.edge_list()}

        frame_points: dict[int, cKDTree] = {}
        grouped: dict[int, list[np.ndarray]] = defaultdict(list)
        for row in final_nodes.values():
            grouped[int(row["t"])].append(
                np.asarray([row["z"], row["y"], row["x"]], np.float64) * scale
            )
        for frame, points in grouped.items():
            frame_points[frame] = cKDTree(np.stack(points))

        counts = Counter()
        for middle in sorted(set(raw_nodes) - final_set):
            middle_row = raw_nodes[middle]
            frame = int(middle_row["t"])
            middle_xyz = np.asarray(
                [middle_row["z"], middle_row["y"], middle_row["x"]], np.float64,
            ) * scale
            tree = frame_points.get(frame)
            nearest = float(tree.query(middle_xyz)[0]) if tree is not None else 99.0
            density10 = int(tree.query_ball_point(middle_xyz, 10.0, return_length=True)) if tree is not None else 0
            for source, left in incoming.get(middle, ())[:4]:
                if source not in final_set or final_out[source] != 0:
                    continue
                for target, right in outgoing.get(middle, ())[:4]:
                    if target not in final_set or final_in[target] != 0:
                        continue
                    source_row, target_row = final_nodes[source], final_nodes[target]
                    source_xyz = np.asarray(
                        [source_row["z"], source_row["y"], source_row["x"]], np.float64,
                    ) * scale
                    target_xyz = np.asarray(
                        [target_row["z"], target_row["y"], target_row["x"]], np.float64,
                    ) * scale
                    first, second = middle_xyz - source_xyz, target_xyz - middle_xyz
                    gt_source, gt_middle, gt_target = (
                        raw_match.get(source), raw_match.get(middle), raw_match.get(target)
                    )
                    known = all(value is not None for value in (gt_source, gt_middle, gt_target))
                    positive = bool(
                        known
                        and (gt_source, gt_middle) in gt_edges
                        and (gt_middle, gt_target) in gt_edges
                        and gt_middle not in final_matched_gt
                    )
                    endpoint_rows.append({
                        "dataset": stem,
                        "source": source,
                        "middle": middle,
                        "target": target,
                        "left_p1": left["p1"], "left_p2": left["p2"],
                        "right_p1": right["p1"], "right_p2": right["p2"],
                        "pmin": min(left["pmax"], right["pmax"]),
                        "pmean": (left["pmax"] + right["pmax"]) / 2.0,
                        "agreement_min": min(left["present"], right["present"]),
                        "winner_min": min(left["winner_count"], right["winner_count"]),
                        "rank_max": max(left["rank_min"], right["rank_min"]),
                        "distance_max_um": max(float(np.linalg.norm(first)), float(np.linalg.norm(second))),
                        "midpoint_residual_um": float(np.linalg.norm(middle_xyz - (source_xyz + target_xyz) / 2.0)),
                        "velocity_cosine": cosine(first, second),
                        "nearest_final_same_frame_um": nearest,
                        "density10": density10,
                        "label_known": known,
                        "label_positive": positive if known else None,
                    })
                    counts["endpoint_candidates"] += 1
                    counts["endpoint_known"] += int(known)
                    counts["endpoint_positive"] += int(positive)

        for (source, target), evidence in native.items():
            source_final, target_final = source in final_set, target in final_set
            if source_final == target_final:
                continue
            if source_final:
                if final_out[source] != 0 or target not in raw_nodes:
                    continue
                anchor, missing, transaction_type = source, target, "missing_target"
            else:
                if target not in final_set or final_in[target] != 0 or source not in raw_nodes:
                    continue
                anchor, missing, transaction_type = target, source, "missing_source"
            missing_row = raw_nodes[missing]
            frame = int(missing_row["t"])
            missing_xyz = np.asarray(
                [missing_row["z"], missing_row["y"], missing_row["x"]], np.float64,
            ) * scale
            tree = frame_points.get(frame)
            nearest = float(tree.query(missing_xyz)[0]) if tree is not None else 99.0
            density10 = int(tree.query_ball_point(missing_xyz, 10.0, return_length=True)) if tree is not None else 0
            gt_source, gt_target = raw_match.get(source), raw_match.get(target)
            known = gt_source is not None and gt_target is not None
            missing_gt = gt_target if transaction_type == "missing_target" else gt_source
            positive = bool(
                known
                and (gt_source, gt_target) in gt_edges
                and missing_gt not in final_matched_gt
            )
            leaf_rows.append({
                "dataset": stem,
                "source": source,
                "target": target,
                "anchor": anchor,
                "missing": missing,
                "transaction_type": transaction_type,
                **evidence,
                "nearest_final_same_frame_um": nearest,
                "density10": density10,
                "label_known": known,
                "label_positive": positive if known else None,
            })
            counts["leaf_candidates"] += 1
            counts["leaf_known"] += int(known)
            counts["leaf_positive"] += int(positive)

        for (source, target), evidence in native.items():
            if source not in final_set or target not in final_set:
                continue
            if (source, target) in final_edges_set:
                continue
            if final_out[source] != 0 or final_in[target] != 0:
                continue
            gt_source, gt_target = raw_match.get(source), raw_match.get(target)
            known = gt_source is not None and gt_target is not None
            positive = bool(known and (gt_source, gt_target) in gt_edges)
            direct_rows.append({
                "dataset": stem,
                "source": source,
                "target": target,
                **evidence,
                "label_known": known,
                "label_positive": positive if known else None,
            })
            counts["direct_candidates"] += 1
            counts["direct_known"] += int(known)
            counts["direct_positive"] += int(positive)

        per_video.append({"dataset": stem, **counts})
        if index % 10 == 0 or index == len(paths):
            print(f"[{index:03d}/{len(paths)}] {stem}: {dict(counts)}", flush=True)

    endpoint = pd.DataFrame(endpoint_rows)
    leaf = pd.DataFrame(leaf_rows)
    direct = pd.DataFrame(direct_rows)
    endpoint.to_parquet(args.output / "endpoint_bridges.parquet", index=False)
    leaf.to_parquet(args.output / "endpoint_leaves.parquet", index=False)
    direct.to_parquet(args.output / "direct_edges.parquet", index=False)
    pd.DataFrame(per_video).fillna(0).to_csv(args.output / "per_video.csv", index=False)
    summary = {
        "version": "native-endpoint-candidate-graft-v1",
        "videos": len(paths),
        "endpoint_candidates": len(endpoint),
        "endpoint_known": int(endpoint.label_known.sum()) if len(endpoint) else 0,
        "endpoint_positive": int(endpoint.label_positive.fillna(False).sum()) if len(endpoint) else 0,
        "leaf_candidates": len(leaf),
        "leaf_known": int(leaf.label_known.sum()) if len(leaf) else 0,
        "leaf_positive": int(leaf.label_positive.fillna(False).sum()) if len(leaf) else 0,
        "direct_candidates": len(direct),
        "direct_known": int(direct.label_known.sum()) if len(direct) else 0,
        "direct_positive": int(direct.label_positive.fillna(False).sum()) if len(direct) else 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
