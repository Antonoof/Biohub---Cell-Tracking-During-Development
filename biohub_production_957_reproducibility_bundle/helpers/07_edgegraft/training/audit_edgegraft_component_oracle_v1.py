#!/usr/bin/env python3
"""Materialize the exact EdgeGRAFT continuation-component oracle.

This is a diagnostic ceiling, not a deployable selector.  Candidate edges are
generated exclusively from the frozen P1/P2 native evidence plus the current
production graph.  Ground truth is consulted only after candidate generation
to choose the best ordinary-continuation transaction inside each ambiguous
bipartite component.

Safety contract:

* existing division sources and daughters are immutable;
* a whole ambiguous component is skipped if it touches an existing fork;
* replacements are applied atomically and remain one-to-one;
* nodes are never added or removed;
* only adjacent-frame ordinary continuation edges are changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

import materialize_trackgraft_v3_held20 as identity_safe
from audit_trackgraft_mixed_edge_oracle_v1 import mapped_candidates
from train_public949_transition_owner_router_v1 import graph_nodes, load_graph


DEFAULT_ROOT = Path("/home/tweak/bio/public914_backbone_matched_v1")
DEFAULT_FINAL = Path("/home/tweak/bio/ug23_boundary_oof_exact_v1/final_candidate")
DEFAULT_DATA = Path("/home/tweak/bio/train")
DEFAULT_OUTPUT = Path("/home/tweak/bio/edgegraft_component_oracle_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--max-distance", type=float, default=14.0)
    parser.add_argument("--match-distance", type=float, default=7.0)
    return parser.parse_args()


def edge_set(graph) -> set[tuple[int, int]]:
    return {(int(source), int(target)) for source, target in graph.edge_list()}


def add_edge(graph, source: int, target: int) -> None:
    attrs: dict[str, float] = {}
    for key in graph.edge_attr_keys():
        lower = str(key).lower()
        attrs[key] = 1.0 if ("prob" in lower or "predict" in lower) else 0.0
    graph.add_edge(int(source), int(target), attrs)


def raw_to_final_map(raw, final) -> dict[int, int]:
    raw_nodes = graph_nodes(raw)
    final_nodes = graph_nodes(final)
    _, raw_id_key, _ = identity_safe.identity_maps(raw)
    _, _, final_key_id = identity_safe.identity_maps(final)
    result = {
        node: node
        for node in set(raw_nodes) & set(final_nodes)
        if int(raw_nodes[node]["t"]) == int(final_nodes[node]["t"])
    }
    result.update(
        {
            raw_id: final_key_id[key]
            for raw_id, key in raw_id_key.items()
            if raw_id not in result and key in final_key_id
        }
    )
    return result


def broad_final_candidates(
    root: Path,
    stem: str,
    raw,
    final,
    threshold: float,
    max_distance: float,
) -> set[tuple[int, int]]:
    raw_nodes = graph_nodes(raw)
    final_nodes = graph_nodes(final)
    mapping = raw_to_final_map(raw, final)
    with np.load(root / "public914_proposals" / f"{stem}.npz") as data:
        graph_node_id = data["graph_node_id"].astype(np.int64, copy=False)

    result: set[tuple[int, int]] = set()
    for directory in (
        "public_primary_native_evidence_all199_v1",
        "public_secondary_native_evidence_all199_v1",
    ):
        candidates = mapped_candidates(
            root / directory / f"{stem}.npz",
            graph_node_id,
            raw_nodes,
            threshold,
            max_distance,
        )
        for raw_target, raw_sources in candidates.items():
            target = mapping.get(int(raw_target))
            if target is None:
                continue
            for raw_source in raw_sources:
                source = mapping.get(int(raw_source))
                if source is None:
                    continue
                if int(final_nodes[target]["t"]) != int(final_nodes[source]["t"]) + 1:
                    continue
                result.add((source, target))
    return result


def candidate_components(
    candidates: set[tuple[int, int]],
) -> list[tuple[set[int], set[int], set[tuple[int, int]]]]:
    """Connected components of a bipartite source/target candidate graph."""
    source_to_target: dict[int, set[int]] = defaultdict(set)
    target_to_source: dict[int, set[int]] = defaultdict(set)
    for source, target in candidates:
        source_to_target[source].add(target)
        target_to_source[target].add(source)

    output = []
    seen_sources: set[int] = set()
    seen_targets: set[int] = set()
    for start in source_to_target:
        if start in seen_sources:
            continue
        sources: set[int] = set()
        targets: set[int] = set()
        queue: deque[tuple[str, int]] = deque([("s", start)])
        while queue:
            kind, node = queue.popleft()
            if kind == "s":
                if node in seen_sources:
                    continue
                seen_sources.add(node)
                sources.add(node)
                queue.extend(("t", target) for target in source_to_target[node])
            else:
                if node in seen_targets:
                    continue
                seen_targets.add(node)
                targets.add(node)
                queue.extend(("s", source) for source in target_to_source[node])
        edges = {
            (source, target)
            for source in sources
            for target in source_to_target[source]
            if target in targets
        }
        output.append((sources, targets, edges))
    return output


def matched_truth(final, gt, scale, max_distance):
    from tracking_cellmot.division_metrics import _match_full

    final_nodes = graph_nodes(final)
    matched = _match_full(final, gt, scale, max_distance)
    gt_to_pred = {
        int(row["match_node_id"]): int(row["node_id"])
        for row in matched.node_attrs().to_dicts()
        if int(row["match_node_id"]) >= 0
    }
    truth: set[tuple[int, int]] = set()
    for gt_source, gt_target in gt.edge_list():
        gt_source, gt_target = int(gt_source), int(gt_target)
        if gt.out_degree(gt_source) != 1:
            continue
        if gt_source not in gt_to_pred or gt_target not in gt_to_pred:
            continue
        source, target = gt_to_pred[gt_source], gt_to_pred[gt_target]
        if int(final_nodes[target]["t"]) == int(final_nodes[source]["t"]) + 1:
            truth.add((source, target))
    return truth


def protected_fork_nodes(graph) -> set[int]:
    protected: set[int] = set()
    for source in map(int, graph.node_ids()):
        if graph.out_degree(source) < 2:
            continue
        protected.add(source)
        protected.update(map(int, graph.successors(source)))
        protected.update(map(int, graph.predecessors(source)))
        for child in map(int, graph.successors(source)):
            protected.update(map(int, graph.successors(child)))
    return protected


def apply_component_oracle(graph, candidates, truth) -> Counter:
    counters: Counter = Counter()
    current = edge_set(graph)
    protected = protected_fork_nodes(graph)
    adjacent_current = {
        edge for edge in current
        if edge[0] in graph_nodes(graph) and edge[1] in graph_nodes(graph)
    }
    all_candidates = set(candidates) | adjacent_current

    for sources, targets, component_edges in candidate_components(all_candidates):
        counters["components"] += 1
        if len(component_edges) <= 1:
            counters["singleton_components"] += 1
            continue
        if (sources | targets) & protected:
            counters["protected_components"] += 1
            continue

        desired = component_edges & truth
        missing = desired - current
        if not missing:
            counters["no_recoverable_truth"] += 1
            continue

        # Correct continuation truth is one-to-one.  Apply all correct edges in
        # the component simultaneously, while leaving unrelated/unknown edges
        # outside their incident sources and targets untouched.
        desired_sources = {source for source, _ in desired}
        desired_targets = {target for _, target in desired}
        if len(desired_sources) != len(desired) or len(desired_targets) != len(desired):
            raise RuntimeError("ordinary continuation truth is not one-to-one")

        remove: set[tuple[int, int]] = set()
        for source, target in desired:
            remove.update(
                (source, int(child))
                for child in graph.successors(source)
                if int(child) != target
            )
            remove.update(
                (int(parent), target)
                for parent in graph.predecessors(target)
                if int(parent) != source
            )

        # A transaction may not touch an existing fork through an edge that was
        # not discovered in the candidate component.
        touched = desired_sources | desired_targets
        if touched & protected:
            counters["protected_transactions"] += 1
            continue
        if any(graph.out_degree(source) > 1 for source in desired_sources):
            counters["fork_source_transactions"] += 1
            continue

        before = edge_set(graph)
        for source, target in remove:
            if graph.has_edge(source, target):
                graph.remove_edge(source, target)
        for source, target in desired:
            if not graph.has_edge(source, target):
                add_edge(graph, source, target)

        if any(graph.in_degree(target) > 1 for target in desired_targets):
            raise RuntimeError("EdgeGRAFT oracle produced in-degree > 1")
        if any(graph.out_degree(source) > 1 for source in desired_sources):
            raise RuntimeError("EdgeGRAFT oracle produced out-degree > 1")

        after = edge_set(graph)
        current = after
        counters["applied_components"] += 1
        counters["recoverable_truth_edges"] += len(missing)
        counters["edges_removed"] += len(before - after)
        counters["edges_added"] += len(after - before)

    return counters


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True)

    sys.path.insert(0, "/home/tweak/bio/kaggle-cell-tracking-competition-patched/src")
    sys.path.insert(0, "/home/tweak/bio_track_repo/src")
    from biohub_tracking.io import open_dataset, save_graph

    raw_dir = args.root / "repo/predictions/tweak/public914_train_raw/split_0"
    stems = sorted(path.stem for path in args.final.glob("*.geff"))
    if not stems:
        raise RuntimeError(f"No production graphs found in {args.final}")

    records = []
    aggregate: Counter = Counter()
    for index, stem in enumerate(stems, 1):
        final = load_graph(args.final / f"{stem}.geff")
        raw = load_graph(raw_dir / f"{stem}.geff")
        gt = load_graph(args.data / f"{stem}.geff")
        dataset = open_dataset(
            args.data / f"{stem}.zarr",
            require_tracks=False,
            load_image=False,
            device="cpu",
        )
        candidates = broad_final_candidates(
            args.root,
            stem,
            raw,
            final,
            args.candidate_threshold,
            args.max_distance,
        )
        truth = matched_truth(
            final,
            gt,
            np.asarray(dataset.scale, np.float64),
            args.match_distance,
        )
        before = edge_set(final)
        counts = apply_component_oracle(final, candidates, truth)
        after = edge_set(final)
        counts["candidate_edges"] = len(candidates)
        counts["truth_edges"] = len(truth)
        counts["final_edge_delta"] = len(after) - len(before)
        record = {"dataset": stem, **{key: int(value) for key, value in counts.items()}}
        records.append(record)
        aggregate.update(counts)
        save_graph(final, graph_dir / f"{stem}.geff")
        print(
            f"[{index:03d}/{len(stems)}] {stem}: "
            f"components={counts['components']} applied={counts['applied_components']} "
            f"recoverable={counts['recoverable_truth_edges']} "
            f"-{counts['edges_removed']} +{counts['edges_added']}",
            flush=True,
        )

    payload = {
        "version": "edgegraft-component-oracle-v1",
        "warning": "GT oracle only; never deploy this artifact",
        "contract": {
            "input_graph": str(args.final),
            "candidate_threshold": args.candidate_threshold,
            "max_distance": args.max_distance,
            "match_distance": args.match_distance,
            "division_components_immutable": True,
            "nodes_unchanged": True,
            "atomic_components": True,
        },
        "videos": records,
        "totals": {key: int(value) for key, value in aggregate.items()},
    }
    (args.output / "oracle.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
