#!/usr/bin/env python3
"""Add incumbent/alternate ownership and trajectory geometry to the UG1/UG2 bank.

The V1 bank stores sorted parent distances, which discards which daughter is
the currently owned continuation.  This enrichment restores that identity and
measures the cost of atomically stealing the alternate daughter from its
incumbent continuation.  It reads no GT; diagnostic labels are merely carried
through from the frozen V1 bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BIO = Path("data")
WORKSPACE = Path(".")
INPUT = BIO / "ug12_serving_geometry_bank_v1/serving_geometry.parquet"
SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)


def graph_contract(stem: str, panel: str) -> Path:
    if panel == "train175":
        return BIO / "ug3_joint_prelock_oof_v1/prefinal_current" / f"{stem}.geff"
    return BIO / "ug3_joint_prelock_held20_full_serving_v1/prefinal_current" / f"{stem}.geff"


def safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", a, b)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)


def safe_norm(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=1)


def build_depth(values: np.ndarray, count: np.ndarray, cap: int = 16) -> np.ndarray:
    depth = np.zeros(len(values), dtype=np.int16)
    for node in range(len(values)):
        current = node
        for _ in range(cap):
            if current < 0 or current >= len(values) or count[current] != 1:
                break
            current = int(values[current])
            depth[node] += 1
    return depth


def enrich_video(local: pd.DataFrame, load) -> pd.DataFrame:
    stem = str(local.dataset.iloc[0])
    panel = str(local.panel.iloc[0])
    nodes, edges = load(graph_contract(stem, panel))
    maximum = max(nodes) + 1
    pos = np.full((maximum, 3), np.nan, dtype=np.float64)
    for node, row in nodes.items():
        pos[int(node)] = np.asarray((row["z"], row["y"], row["x"]), np.float64) * SPACING
    succ = np.full(maximum, -1, dtype=np.int64)
    pred = np.full(maximum, -1, dtype=np.int64)
    succ_count = np.zeros(maximum, dtype=np.int16)
    pred_count = np.zeros(maximum, dtype=np.int16)
    outgoing_prob = np.full(maximum, np.nan, dtype=np.float64)
    incoming_prob = np.full(maximum, np.nan, dtype=np.float64)
    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        succ_count[source] += 1
        pred_count[target] += 1
        if succ_count[source] == 1:
            succ[source] = target
            outgoing_prob[source] = np.nan if edge["edge_prob"] is None else float(edge["edge_prob"])
        if pred_count[target] == 1:
            pred[target] = source
            incoming_prob[target] = np.nan if edge["edge_prob"] is None else float(edge["edge_prob"])
    back_depth = build_depth(pred, pred_count)
    forward_depth = build_depth(succ, succ_count)

    source = local.source.to_numpy(np.int64)
    a = local.a.to_numpy(np.int64)
    b = local.b.to_numpy(np.int64)
    current = local.current_child.to_numpy(np.int64)
    alternate = np.where(current == a, b, a)
    incumbent = np.where(pred_count[alternate] == 1, pred[alternate], -1)
    valid_incumbent = incumbent >= 0
    incumbent_safe = np.where(valid_incumbent, incumbent, source)
    previous = np.where(pred_count[source] == 1, pred[source], source)
    incumbent_previous = np.where(
        valid_incumbent & (pred_count[incumbent_safe] == 1), pred[incumbent_safe], incumbent_safe
    )
    current_next = np.where(succ_count[current] == 1, succ[current], current)
    alternate_next = np.where(succ_count[alternate] == 1, succ[alternate], alternate)

    source_pos = pos[source]
    current_pos = pos[current]
    alternate_pos = pos[alternate]
    incumbent_pos = pos[incumbent_safe]
    previous_pos = pos[previous]
    incumbent_previous_pos = pos[incumbent_previous]
    current_next_pos = pos[current_next]
    alternate_next_pos = pos[alternate_next]

    source_velocity = source_pos - previous_pos
    current_displacement = current_pos - source_pos
    alternate_displacement = alternate_pos - source_pos
    split_axis = alternate_pos - current_pos
    incumbent_velocity = incumbent_pos - incumbent_previous_pos
    incumbent_displacement = alternate_pos - incumbent_pos
    midpoint = (current_pos + alternate_pos) / 2.0
    predicted_parent = source_pos + source_velocity
    current_forward = current_next_pos - current_pos
    alternate_forward = alternate_next_pos - alternate_pos

    result = local.copy()
    result["current_is_a"] = (current == a).astype(np.int8)
    result["current_parent_um"] = safe_norm(current_displacement)
    result["alternate_parent_um"] = safe_norm(alternate_displacement)
    result["alternate_minus_current_um"] = result.alternate_parent_um - result.current_parent_um
    result["alternate_over_current"] = result.alternate_parent_um / np.maximum(result.current_parent_um, 1e-3)
    result["current_forward_depth"] = forward_depth[current]
    result["alternate_forward_depth"] = forward_depth[alternate]
    result["forward_depth_difference"] = result.alternate_forward_depth - result.current_forward_depth
    result["incumbent_exists"] = valid_incumbent.astype(np.int8)
    result["incumbent_is_source"] = (incumbent == source).astype(np.int8)
    result["incumbent_back_depth"] = np.where(valid_incumbent, back_depth[incumbent_safe], 0)
    result["incumbent_to_alternate_um"] = np.where(
        valid_incumbent, safe_norm(incumbent_displacement), 99.0
    )
    result["source_to_incumbent_um"] = np.where(
        valid_incumbent, safe_norm(source_pos - incumbent_pos), 99.0
    )
    result["steal_distance_delta_um"] = (
        result.alternate_parent_um - result.incumbent_to_alternate_um
    )
    result["steal_distance_ratio"] = result.alternate_parent_um / np.maximum(
        result.incumbent_to_alternate_um, 1e-3
    )
    result["source_previous_step_um"] = safe_norm(source_velocity)
    result["current_step_um"] = safe_norm(current_displacement)
    result["alternate_step_um"] = safe_norm(alternate_displacement)
    result["parent_current_cosine"] = safe_cosine(source_velocity, current_displacement)
    result["parent_alternate_cosine"] = safe_cosine(source_velocity, alternate_displacement)
    result["parent_split_axis_abs_cosine"] = np.abs(safe_cosine(source_velocity, split_axis))
    result["predicted_midpoint_error_um"] = safe_norm(midpoint - predicted_parent)
    result["incumbent_previous_step_um"] = np.where(
        valid_incumbent, safe_norm(incumbent_velocity), 0.0
    )
    result["incumbent_continuation_cosine"] = np.where(
        valid_incumbent, safe_cosine(incumbent_velocity, incumbent_displacement), 0.0
    )
    result["current_next_step_um"] = safe_norm(current_forward)
    result["alternate_next_step_um"] = safe_norm(alternate_forward)
    result["current_forward_cosine"] = safe_cosine(current_displacement, current_forward)
    result["alternate_forward_cosine"] = safe_cosine(alternate_displacement, alternate_forward)
    result["current_edge_prob"] = np.nan_to_num(incoming_prob[current], nan=-1.0)
    result["incumbent_edge_prob"] = np.nan_to_num(incoming_prob[alternate], nan=-1.0)
    result["incumbent_edge_prob_minus_current"] = (
        result.incumbent_edge_prob - result.current_edge_prob
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=BIO / "ug12_ownership_geometry_bank_v2"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(WORKSPACE / "scripts"))
    from smoke_edgegraft_v3_deploy_parity_v1 import load

    frame = pd.read_parquet(INPUT)
    frames = []
    contracts = list(frame.groupby(["panel", "dataset"], sort=True))
    for index, ((panel, stem), local) in enumerate(contracts, 1):
        frames.append(enrich_video(local.reset_index(drop=True), load))
        if index % 10 == 0 or index == len(contracts):
            print(f"[{index:03d}/{len(contracts)}] {stem}: rows={len(local):,}", flush=True)
    enriched = pd.concat(frames, ignore_index=True)
    enriched.to_parquet(args.output / "ownership_geometry.parquet", index=False)
    payload = {
        "version": "ug12-ownership-geometry-bank-v2",
        "input": str(INPUT),
        "rows": int(len(enriched)),
        "videos": int(enriched.dataset.nunique()),
        "physical_spacing_zyx_um": SPACING.tolist(),
        "gt_read_by_builder": False,
        "daughter_identity_preserved": True,
        "incumbent_ownership_features": True,
        "trajectory_features": True,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
