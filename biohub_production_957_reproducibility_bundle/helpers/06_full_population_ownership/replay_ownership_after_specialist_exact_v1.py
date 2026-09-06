#!/usr/bin/env python3
"""Exact stacked audit: specialist-final graph plus clean ownership expert.

The ownership expert was selected by grouped-video OOF on annotated rows only.
This audit restores its exact candidate rows (including daughter IDs), applies
them atomically to the already-finalized live-v2 specialist graph, and scores
only the affected videos with the host-patched metric.  No threshold is tuned
here and no unknown source is assigned a label.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BIO = Path("/home/tweak/bio")
WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
BANK = BIO / "ug12_ownership_geometry_bank_v2/ownership_geometry.parquet"
OOF = BIO / "ownership_toprank_stage1_v1/oof_predictions.parquet"
CONTROL = BIO / "live_v2_global_bundle_ug12_matched_v1/train175/final"
CONTROL_SUMMARY = BIO / "live_v2_global_bundle_ug12_matched_v1/train175/exact/summary.json"
OUTPUT = BIO / "ownership_after_specialist_exact_v1"


def selected_rows() -> pd.DataFrame:
    bank = pd.read_parquet(BANK)
    known = bank.loc[bank.source_label.notna()].copy().reset_index(drop=True)
    oof = pd.read_parquet(OOF).reset_index(drop=True)
    if len(known) != len(oof):
        raise RuntimeError(f"OOF/bank row mismatch: {len(oof)} != {len(known)}")
    for column in ("dataset", "source"):
        if not np.array_equal(known[column].to_numpy(), oof[column].to_numpy()):
            raise RuntimeError(f"OOF/bank ordering mismatch in {column}")
    selected = known.loc[oof.selected.astype(bool)].copy()
    selected["score"] = oof.loc[oof.selected.astype(bool), "score"].to_numpy()
    if not (
        selected.source_label.eq(1).all()
        and selected.best_pair_label.eq(1).all()
        and selected.known_positive.eq(1).all()
    ):
        raise RuntimeError("Selected ownership rows are not exact known-positive pairs")
    if selected.duplicated(["dataset", "source"]).any():
        raise RuntimeError("More than one selected transaction for a source")
    return selected


def apply_transaction(graph, row: pd.Series) -> dict[str, int]:
    counters: Counter[str] = Counter()
    table = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).to_dicts()
    times = {int(item["node_id"]): int(item["t"]) for item in table}
    positions = {
        int(item["node_id"]): np.asarray([item["z"], item["y"], item["x"]], np.float32)
        for item in table
    }
    nodes = set(times)
    edge_keys = set(graph.edge_attr_keys())
    source, a, b = int(row.source), int(row.a), int(row.b)
    if source not in nodes or a not in nodes or b not in nodes:
        counters["rejected_missing_node"] += 1
        return dict(counters)
    if graph.out_degree(source) >= 2:
        counters["rejected_existing_fork"] += 1
        return dict(counters)
    if times[a] != times[source] + 1 or times[b] != times[source] + 1:
        counters["rejected_nonadjacent"] += 1
        return dict(counters)
    if a == b:
        counters["rejected_duplicate_daughter"] += 1
        return dict(counters)

    initial_in = {node: int(graph.in_degree(node)) for node in nodes}
    initial_out = {node: int(graph.out_degree(node)) for node in nodes}
    pair = {a, b}
    for child in list(graph.successors(source)):
        child = int(child)
        if child not in pair:
            graph.remove_edge(source, child)
            counters["removed_source_continuation"] += 1
    for child in pair:
        for parent in list(graph.predecessors(child)):
            parent = int(parent)
            if parent != source:
                if graph.out_degree(parent) >= 2:
                    counters["rejected_protected_fork_target"] += 1
                    return dict(counters)
                graph.remove_edge(parent, child)
                counters["removed_competing_claim"] += 1
    for child in pair:
        if graph.has_edge(source, child):
            continue
        attrs = {}
        if "edge_prob" in edge_keys:
            attrs["edge_prob"] = float(row.score)
        if "edge_dist" in edge_keys:
            attrs["edge_dist"] = float(np.linalg.norm(positions[source] - positions[child]))
        graph.add_edge(source, child, attrs)
        counters["added_edge"] += 1
    if graph.out_degree(source) != 2:
        raise RuntimeError(f"Atomic ownership transaction failed for {row.dataset}:{source}")
    if any(graph.in_degree(node) > max(1, initial_in[node]) for node in nodes):
        raise RuntimeError(f"In-degree violation after {row.dataset}:{source}")
    if any(graph.out_degree(node) > max(2, initial_out[node]) for node in nodes):
        raise RuntimeError(f"Out-degree violation after {row.dataset}:{source}")
    counters["applied"] += 1
    return dict(counters)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    graph_dir = OUTPUT / "graphs"
    graph_dir.mkdir()
    sys.path[:0] = [
        str(WORKSPACE / "scripts"),
        str(BIO / "kaggle-cell-tracking-competition-patched/src"),
        "/home/tweak/bio_track_repo/src",
    ]
    import replay_model_c_full_population_gate as graph_io
    from biohub_tracking.io import open_dataset
    from biohub_tracking.metrics import EvaluationResult, evaluate, node_recall, per_sample_metrics
    from replay_legacy_v2_light_geometry_all195_v1 import exact_row
    from tracking_cellmot import division_metrics

    selected = selected_rows()
    selected.to_csv(OUTPUT / "selected_transactions.csv", index=False)
    aggregate: Counter[str] = Counter()
    records: list[dict] = []
    for stem, local in selected.groupby("dataset", sort=True):
        path = CONTROL / f"{stem}.geff"
        before_graph = graph_io.load_graph(path)
        after_graph = before_graph.copy()
        local_counts: Counter[str] = Counter()
        for _, row in local.sort_values("score", ascending=False).iterrows():
            local_counts.update(apply_transaction(after_graph, row))
        aggregate.update(local_counts)
        gt_path = BIO / "train" / f"{stem}.geff"
        gt = graph_io.load_graph(gt_path)
        dataset = open_dataset(
            BIO / "train" / f"{stem}.zarr", require_tracks=False,
            load_image=False, device="cpu",
        )
        before, before_detail = exact_row(
            before_graph, gt, dataset.scale, gt_path, evaluate, node_recall,
            EvaluationResult, per_sample_metrics, division_metrics, 7.0,
        )
        after, after_detail = exact_row(
            after_graph, gt, dataset.scale, gt_path, evaluate, node_recall,
            EvaluationResult, per_sample_metrics, division_metrics, 7.0,
        )
        graph_io.save_graph(after_graph, graph_dir / f"{stem}.geff")
        record = {
            "dataset": stem,
            "proposed": int(len(local)),
            **{key: int(value) for key, value in local_counts.items()},
            "before_edge": float(before["adj_edge_jaccard"]),
            "after_edge": float(after["adj_edge_jaccard"]),
            "delta_edge": float(after["adj_edge_jaccard"] - before["adj_edge_jaccard"]),
            "before_tp": int(before_detail["division_tp"]),
            "after_tp": int(after_detail["division_tp"]),
            "before_fp": int(before_detail["division_fp"]),
            "after_fp": int(after_detail["division_fp"]),
            "before_fn": int(before_detail["division_fn"]),
            "after_fn": int(after_detail["division_fn"]),
        }
        records.append(record)
        print(
            f"{stem}: proposed={len(local)} applied={local_counts.get('applied', 0)} "
            f"division {record['before_tp']}/{record['before_fp']}/{record['before_fn']} -> "
            f"{record['after_tp']}/{record['after_fp']}/{record['after_fn']} "
            f"edge={record['delta_edge']:+.6f}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT / "per_video.csv", index=False)
    control_payload = json.loads(CONTROL_SUMMARY.read_text())["summary"]
    delta_tp = int((frame.after_tp - frame.before_tp).sum())
    delta_fp = int((frame.after_fp - frame.before_fp).sum())
    delta_fn = int((frame.after_fn - frame.before_fn).sum())
    tp = int(control_payload["division_tp"]) + delta_tp
    fp = int(control_payload["division_fp"]) + delta_fp
    fn = int(control_payload["division_fn"]) + delta_fn
    before_j = float(control_payload["division_jaccard"])
    after_j = tp / max(tp + fp + fn, 1)
    mean_edge_delta = float(frame.delta_edge.sum() / int(control_payload["n_adj"]))
    before_edge = float(control_payload["adj_edge_jaccard"])
    after_edge = before_edge + mean_edge_delta
    payload = {
        "version": "ownership-after-specialist-exact-v1",
        "contract": {
            "control": "live-v2 specialist exact final train-175 graph",
            "ownership": "grouped-video OOF top-rank stage1; 11 selected known-positive rows",
            "operation": "authoritative atomic pair replacement after specialist finalization",
            "metric": "host-patched exact graph metric on every changed video",
            "unknown_sources_used_as_negatives": False,
        },
        "transactions": int(len(selected)),
        "videos": int(selected.dataset.nunique()),
        "aggregate": dict(aggregate),
        "exact_delta": {
            "division_tp": delta_tp, "division_fp": delta_fp, "division_fn": delta_fn,
            "division_jaccard": after_j - before_j,
            "adjusted_edge_jaccard": mean_edge_delta,
            "composite_proxy": mean_edge_delta + 0.1 * (after_j - before_j),
        },
        "control": {
            "division_tp": int(control_payload["division_tp"]),
            "division_fp": int(control_payload["division_fp"]),
            "division_fn": int(control_payload["division_fn"]),
            "division_jaccard": before_j,
            "adjusted_edge_jaccard": before_edge,
        },
        "candidate": {
            "division_tp": tp, "division_fp": fp, "division_fn": fn,
            "division_jaccard": after_j,
            "adjusted_edge_jaccard": after_edge,
            "score_proxy": after_edge + 0.1 * after_j,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
