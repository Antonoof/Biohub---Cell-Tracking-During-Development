#!/usr/bin/env python3
"""Train a sparse-safe Model-C evidence decoder on top of frozen Division V2.

The raw Model-C edge tensor is *not* a division classifier: its source-axis
softmax answers which parent wins each target.  Treating two >0.5 edges as a
fork floods the graph.  This trainer instead:

1. keeps Division V2's broad 14/20 um source/pair candidate population;
2. aligns Model C's native-TTA detections to the frozen A+B nodes;
3. appends Model-C competition evidence to every V2 daughter pair;
4. trains a daughter-pair reranker and a parent/time gate using only annotated
   divisions and confirmed annotated continuations;
5. uses grouped OOF predictions for a calibration diagnostic;
6. calibrates the deploy threshold on the disjoint 20-video validation split;
7. evaluates once on the four practice clips, which are untouched by Model C,
   the decoder, and threshold selection.

Unannotated sources are never converted into negative labels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import tracksdata as td
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


C_FEATURE_NAMES = [
    "c_source_mapped",
    "c_source_map_distance_um",
    "c_edge_a_present",
    "c_edge_b_present",
    "c_edge_a_probability",
    "c_edge_b_probability",
    "c_edge_min_probability",
    "c_edge_mean_probability",
    "c_edge_max_probability",
    "c_edge_a_parent_margin",
    "c_edge_b_parent_margin",
    "c_edge_min_parent_margin",
    "c_edge_winner_count",
    "c_edge_a_distance_um",
    "c_edge_b_distance_um",
    "c_edge_a_rank",
    "c_edge_b_rank",
    "c_source_evidence_count",
    "c_source_winner_count",
    "c_source_probability_ge_05_count",
    "c_source_winner_probability_ge_05_count",
    "c_source_max_probability",
    "c_target_a_map_distance_um",
    "c_target_b_map_distance_um",
]

CTC_FEATURE_NAMES = [
    "ctc_pair_valid",
    "ctc_pair_is_top1",
    "ctc_pair_inverse_rank",
    "ctc_pair_softmax",
    "ctc_pair_centered_logit",
    "ctc_source_top1_softmax",
    "ctc_source_top1_top2_margin",
    "ctc_source_normalized_entropy",
    "ctc_source_geometry_agreement",
    "ctc_source_valid_pair_fraction",
]

PUBLIC_EVIDENCE_LABELS = ("public_primary", "public_secondary")
PUBLIC_FEATURE_NAMES = [
    f"{label}_{name.removeprefix('c_')}"
    for label in PUBLIC_EVIDENCE_LABELS
    for name in C_FEATURE_NAMES
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--event-cache",
        type=Path,
        default=Path("data/division_official_event_cache_v2_parts"),
    )
    p.add_argument(
        "--full-population-cache",
        type=Path,
        default=Path("data/division_v3_full_population_cache"),
    )
    p.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/division_candidate_audit_v1/pre_safe_graphs"),
        help="Exact graphs used to construct the official-event V2 cache.",
    )
    p.add_argument(
        "--train-evidence",
        type=Path,
        default=Path("data/model_c_native_division_evidence_train175_bestpair"),
    )
    p.add_argument(
        "--held-evidence",
        type=Path,
        default=Path("data/model_c_native_division_evidence_held20_bestpair"),
    )
    p.add_argument(
        "--practice-evidence",
        type=Path,
        default=Path("data/model_c_native_division_evidence_practice4_bestpair"),
    )
    p.add_argument(
        "--split",
        type=Path,
        default=Path("data/division_balanced_175_20_split.json"),
    )
    p.add_argument(
        "--v2-artifact",
        type=Path,
        default=Path("/mnt/c/Kaggle/biohub-division-parent-gate-v2"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_c_v2_event_decoder_v1"),
    )
    p.add_argument(
        "--ctc-evidence",
        type=Path,
        default=None,
        help=(
            "Optional fold-clean CTC pair evidence. When supplied with a "
            "held-only 190/9 split, run the Model-C control versus Model-C+CTC "
            "without using V2 model scores as features."
        ),
    )
    p.add_argument(
        "--public-primary-evidence",
        type=Path,
        default=None,
        help=(
            "Native evidence exported from the public .934 primary backbone. "
            "Requires --public-secondary-evidence and enables a direct "
            "Model-C control versus Model-C+public-backbone comparison."
        ),
    )
    p.add_argument(
        "--public-secondary-evidence",
        type=Path,
        default=None,
        help="Native evidence exported from the public .934 secondary backbone.",
    )
    p.add_argument("--min-public-oof-gain", type=float, default=0.005)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--max-pair-negatives", type=int, default=180_000)
    p.add_argument("--min-ctc-oof-gain", type=float, default=0.01)
    return p.parse_args()


@dataclass
class VideoRows:
    stem: str
    embryo: str
    source_x: np.ndarray
    source_y: np.ndarray
    source_node: np.ndarray
    source_tube: np.ndarray
    source_event: np.ndarray
    pair_x: np.ndarray
    pair_y: np.ndarray
    pair_owner: np.ndarray
    pair_a: np.ndarray
    pair_b: np.ndarray
    c_pair_x: np.ndarray
    ctc_pair_x: np.ndarray
    public_pair_x: np.ndarray
    event_total: int


def safe_probability(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(
        np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    )[:, 1].astype(np.float32)


def grouped_best_pair(owner: np.ndarray, score: np.ndarray, n_sources: int) -> np.ndarray:
    result = np.full(n_sources, -1, np.int64)
    order = np.argsort(owner, kind="stable")
    if not len(order):
        return result
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows = order[left:right]
        result[int(sorted_owner[left])] = int(rows[int(np.argmax(score[rows]))])
    return result


def base_v2_scores(cache: dict, pair_model, v1_model, gate_model):
    sx = np.nan_to_num(cache["source_x"].astype(np.float32), nan=0.0)
    px = np.nan_to_num(cache["pair_x"].astype(np.float32), nan=0.0)
    owner = cache["pair_source_idx"].astype(np.int32)
    pair_score = (
        safe_probability(pair_model, px)
        if len(px)
        else np.empty(0, np.float32)
    )
    n_sources = len(sx)
    pmax = np.zeros(n_sources, np.float32)
    pmean = np.zeros(n_sources, np.float32)
    pcnt = np.zeros(n_sources, np.float32)
    npairs = np.zeros(n_sources, np.float32)
    best = np.full(n_sources, -1, np.int64)
    best_x = np.zeros((n_sources, px.shape[1]), np.float32)
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_owner[1:] != sorted_owner[:-1]]
    ) if len(order) else np.empty(0, np.int64)
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows = order[left:right]
        source_row = int(sorted_owner[left])
        probabilities = pair_score[rows]
        pair_row = int(rows[int(np.argmax(probabilities))])
        best[source_row] = pair_row
        pmax[source_row] = float(probabilities.max())
        pmean[source_row] = float(probabilities.mean())
        pcnt[source_row] = float(np.count_nonzero(probabilities > 0.5))
        npairs[source_row] = float(len(rows))
        best_x[source_row] = px[pair_row]
    v1_input = np.concatenate([
        sx, best_x, pmax[:, None], pmean[:, None],
        pcnt[:, None], np.log1p(npairs)[:, None],
    ], axis=1)
    v1 = safe_probability(v1_model, v1_input)
    gate_input = np.concatenate([sx, best_x, pmax[:, None], v1[:, None]], axis=1)
    gate = safe_probability(gate_model, gate_input)
    return pair_score, v1, gate


def source_tubes(full_cache: Path, stem: str, source_nodes: np.ndarray) -> np.ndarray:
    root = full_cache / stem
    ids = np.load(root / "source_id.npy", mmap_mode="r")
    tubes = np.load(root / "source_tube.npy", mmap_mode="r")
    order = np.argsort(ids, kind="stable")
    sorted_ids = np.asarray(ids[order])
    where = np.searchsorted(sorted_ids, source_nodes)
    valid = (where < len(sorted_ids))
    valid &= sorted_ids[np.minimum(where, len(sorted_ids) - 1)] == source_nodes
    result = source_nodes.astype(np.int64, copy=True)
    result[valid] = np.asarray(tubes[order[where[valid]]], np.int64)
    return result


def _query_edges(
    query_source: np.ndarray,
    query_target: np.ndarray,
    sorted_keys: np.ndarray,
    values: dict[str, np.ndarray],
    key_stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    query_key = query_source.astype(np.int64) * key_stride + query_target.astype(np.int64)
    where = np.searchsorted(sorted_keys, query_key)
    present = where < len(sorted_keys)
    present &= sorted_keys[np.minimum(where, len(sorted_keys) - 1)] == query_key
    out = {}
    for key, value in values.items():
        fill = np.zeros(len(query_key), np.float32)
        fill[present] = value[where[present]]
        out[key] = fill
    return present, out


def load_graph_coordinates(path: Path) -> dict[int, tuple[int, np.ndarray]]:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    attrs = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    return {
        int(row["node_id"]): (
            int(row["t"]),
            np.asarray([row["z"], row["y"], row["x"]], np.float32),
        )
        for row in attrs.iter_rows(named=True)
    }


def model_c_features(
    evidence_path: Path,
    graph_coordinates: dict[int, tuple[int, np.ndarray]],
    source_node,
    pair_owner,
    pair_a,
    pair_b,
):
    with np.load(evidence_path, allow_pickle=False) as z:
        source_native = z["source_id"].astype(np.int64)
        target_native = z["target_id"].astype(np.int64)
        probability = z["probability"].astype(np.float32)
        alternative = z["alternative_parent_probability"].astype(np.float32)
        winner = z["is_target_winner"].astype(np.float32)
        edge_distance = z["distance_um"].astype(np.float32)
        rank = z["source_target_rank"].astype(np.float32)
        native_coords = z["native_node_coords"].astype(np.float32)
        native_offsets = z["native_frame_offsets"].astype(np.int64)
        spacing = z["spacing_um"].astype(np.float32)

    n_sources = len(source_node)
    n_pairs = len(pair_owner)
    source_frame = np.full(n_sources, -1, np.int16)
    source_position = np.zeros((n_sources, 3), np.float32)
    for row, node in enumerate(source_node):
        value = graph_coordinates.get(int(node))
        if value is not None:
            source_frame[row], source_position[row] = value
    pair_a_position = np.zeros((n_pairs, 3), np.float32)
    pair_b_position = np.zeros((n_pairs, 3), np.float32)
    pair_a_valid = np.zeros(n_pairs, bool)
    pair_b_valid = np.zeros(n_pairs, bool)
    for row, (a_node, b_node) in enumerate(zip(pair_a, pair_b)):
        a_value = graph_coordinates.get(int(a_node))
        b_value = graph_coordinates.get(int(b_node))
        if a_value is not None:
            pair_a_valid[row] = True
            pair_a_position[row] = a_value[1]
        if b_value is not None:
            pair_b_valid[row] = True
            pair_b_position[row] = b_value[1]

    source_to_native = np.full(n_sources, -1, np.int64)
    source_map_distance = np.zeros(n_sources, np.float32)
    frames = len(native_offsets) - 1
    rows_by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(source_frame):
        if frame >= 0:
            rows_by_frame.setdefault(int(frame), []).append(row)
    for frame, source_rows_list in rows_by_frame.items():
        if frame < 0 or frame >= frames:
            continue
        n0, n1 = int(native_offsets[frame]), int(native_offsets[frame + 1])
        if n1 <= n0:
            continue
        source_rows = np.asarray(source_rows_list, np.int64)
        query = source_position[source_rows] * spacing
        tree = cKDTree(native_coords[n0:n1, 1:] * spacing)
        distance, index = tree.query(query, k=1)
        valid = distance <= 6.0
        source_to_native[source_rows[valid]] = n0 + index[valid].astype(np.int64)
        source_map_distance[source_rows[valid]] = distance[valid].astype(np.float32)

    # Group exported evidence by native source and V2 pairs by source row.
    evidence_order = np.argsort(source_native, kind="stable")
    evidence_sorted_source = source_native[evidence_order]
    evidence_starts = np.flatnonzero(
        np.r_[True, evidence_sorted_source[1:] != evidence_sorted_source[:-1]]
    ) if len(evidence_order) else np.empty(0, np.int64)
    evidence_ends = np.r_[evidence_starts[1:], len(evidence_order)]
    evidence_groups = {
        int(evidence_sorted_source[left]): evidence_order[left:right]
        for left, right in zip(evidence_starts, evidence_ends)
    }
    pair_order = np.argsort(pair_owner, kind="stable")
    pair_sorted_owner = pair_owner[pair_order]
    pair_starts = np.flatnonzero(
        np.r_[True, pair_sorted_owner[1:] != pair_sorted_owner[:-1]]
    ) if len(pair_order) else np.empty(0, np.int64)
    pair_ends = np.r_[pair_starts[1:], len(pair_order)]
    pair_groups = {
        int(pair_sorted_owner[left]): pair_order[left:right]
        for left, right in zip(pair_starts, pair_ends)
    }

    present_a = np.zeros(n_pairs, bool)
    present_b = np.zeros(n_pairs, bool)
    pa = np.zeros(n_pairs, np.float32)
    pb = np.zeros(n_pairs, np.float32)
    alt_a = np.zeros(n_pairs, np.float32)
    alt_b = np.zeros(n_pairs, np.float32)
    win_a = np.zeros(n_pairs, np.float32)
    win_b = np.zeros(n_pairs, np.float32)
    dist_a = np.zeros(n_pairs, np.float32)
    dist_b = np.zeros(n_pairs, np.float32)
    rank_a = np.zeros(n_pairs, np.float32)
    rank_b = np.zeros(n_pairs, np.float32)
    target_map_a = np.zeros(n_pairs, np.float32)
    target_map_b = np.zeros(n_pairs, np.float32)
    source_evidence_count = np.zeros(n_sources, np.float32)
    source_winner_count = np.zeros(n_sources, np.float32)
    source_p50_count = np.zeros(n_sources, np.float32)
    source_winner_p50_count = np.zeros(n_sources, np.float32)
    source_max_probability = np.zeros(n_sources, np.float32)

    for source_row, owned_pairs in pair_groups.items():
        native_source = int(source_to_native[source_row])
        evidence_rows = evidence_groups.get(native_source)
        if evidence_rows is None or not len(evidence_rows):
            continue
        source_evidence_count[source_row] = len(evidence_rows)
        source_winner_count[source_row] = float(winner[evidence_rows].sum())
        source_p50_count[source_row] = float(
            np.count_nonzero(probability[evidence_rows] >= 0.5)
        )
        source_winner_p50_count[source_row] = float(np.count_nonzero(
            (probability[evidence_rows] >= 0.5) & (winner[evidence_rows] > 0.5)
        ))
        source_max_probability[source_row] = float(
            probability[evidence_rows].max()
        )
        native_targets = target_native[evidence_rows]
        target_positions = native_coords[native_targets, 1:] * spacing
        tree = cKDTree(target_positions)
        query_a = pair_a_position[owned_pairs] * spacing
        query_b = pair_b_position[owned_pairs] * spacing
        map_a, index_a = tree.query(query_a, k=1)
        map_b, index_b = tree.query(query_b, k=1)
        ok_a = (map_a <= 6.0) & pair_a_valid[owned_pairs]
        ok_b = (map_b <= 6.0) & pair_b_valid[owned_pairs]
        rows_a = evidence_rows[index_a.astype(np.int64)]
        rows_b = evidence_rows[index_b.astype(np.int64)]
        present_a[owned_pairs] = ok_a
        present_b[owned_pairs] = ok_b
        target_map_a[owned_pairs] = map_a.astype(np.float32)
        target_map_b[owned_pairs] = map_b.astype(np.float32)
        for ok, rows, p_out, alt_out, win_out, dist_out, rank_out in (
            (ok_a, rows_a, pa, alt_a, win_a, dist_a, rank_a),
            (ok_b, rows_b, pb, alt_b, win_b, dist_b, rank_b),
        ):
            selected_pairs = owned_pairs[ok]
            selected_evidence = rows[ok]
            p_out[selected_pairs] = probability[selected_evidence]
            alt_out[selected_pairs] = alternative[selected_evidence]
            win_out[selected_pairs] = winner[selected_evidence]
            dist_out[selected_pairs] = edge_distance[selected_evidence]
            rank_out[selected_pairs] = rank[selected_evidence]

    margin_a, margin_b = pa - alt_a, pb - alt_b
    pair_source_rows = pair_owner.astype(np.int64)
    c = np.column_stack([
        (source_to_native[pair_source_rows] >= 0).astype(np.float32),
        source_map_distance[pair_source_rows],
        present_a.astype(np.float32),
        present_b.astype(np.float32),
        pa,
        pb,
        np.minimum(pa, pb),
        0.5 * (pa + pb),
        np.maximum(pa, pb),
        margin_a,
        margin_b,
        np.minimum(margin_a, margin_b),
        win_a + win_b,
        dist_a,
        dist_b,
        rank_a,
        rank_b,
        source_evidence_count[pair_source_rows],
        source_winner_count[pair_source_rows],
        source_p50_count[pair_source_rows],
        source_winner_p50_count[pair_source_rows],
        source_max_probability[pair_source_rows],
        target_map_a,
        target_map_b,
    ]).astype(np.float32)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def load_video(stem: str, args, evidence_root: Path | None) -> VideoRows:
    cache = torch.load(
        args.event_cache / f"{stem}.pt", map_location="cpu", weights_only=False
    )
    source_node = cache["source_node"].astype(np.int64)
    owner = cache["pair_source_idx"].astype(np.int32)
    needs_coordinates = (
        evidence_root is not None
        or args.public_primary_evidence is not None
        or args.public_secondary_evidence is not None
    )
    graph_coordinates = (
        load_graph_coordinates(args.graph_dir / f"{stem}.geff")
        if needs_coordinates else {}
    )
    if evidence_root is None:
        c_pair_x = np.zeros((len(owner), 0), np.float32)
    else:
        # Latent-aware caches may retain unmatched native Model-C detections
        # as candidate-only daughter nodes.  They are absent from the frozen
        # continuation graph by design, so restore their coordinates solely
        # for Model-C evidence alignment.  Older caches remain unchanged.
        latent_node = cache.get("latent_node")
        latent_coords = cache.get("latent_node_coords")
        if latent_node is not None and latent_coords is not None:
            latent_node = np.asarray(latent_node, np.int64)
            latent_coords = np.asarray(latent_coords, np.float32)
            if latent_coords.shape != (len(latent_node), 4):
                raise RuntimeError(
                    f"Latent-node coordinate mismatch for {stem}: "
                    f"{latent_node.shape} / {latent_coords.shape}"
                )
            for node, coordinate in zip(latent_node, latent_coords):
                graph_coordinates[int(node)] = (
                    int(coordinate[0]),
                    coordinate[1:].astype(np.float32),
                )
        c_pair_x = model_c_features(
            evidence_root / f"{stem}.npz",
            graph_coordinates,
            source_node,
            owner,
            cache["pair_a"].astype(np.int64),
            cache["pair_b"].astype(np.int64),
        )
    public_roots = (
        args.public_primary_evidence,
        args.public_secondary_evidence,
    )
    if all(root is None for root in public_roots):
        public_pair_x = np.zeros((len(owner), 0), np.float32)
    elif any(root is None for root in public_roots):
        raise ValueError(
            "Public evidence is atomic: both primary and secondary roots "
            "must be supplied."
        )
    else:
        public_blocks = []
        for label, root in zip(PUBLIC_EVIDENCE_LABELS, public_roots):
            path = root / f"{stem}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {label} evidence for {stem}: {path}"
                )
            block = model_c_features(
                path,
                graph_coordinates,
                source_node,
                owner,
                cache["pair_a"].astype(np.int64),
                cache["pair_b"].astype(np.int64),
            )
            if block.shape != (len(owner), len(C_FEATURE_NAMES)):
                raise RuntimeError(
                    f"{stem}: {label} feature shape {block.shape}, expected "
                    f"({len(owner)}, {len(C_FEATURE_NAMES)})"
                )
            public_blocks.append(block)
        public_pair_x = np.concatenate(public_blocks, axis=1)
    if args.ctc_evidence is None:
        ctc_pair_x = np.zeros((len(owner), 0), np.float32)
    else:
        ctc_path = args.ctc_evidence / f"{stem}.npz"
        if not ctc_path.exists():
            raise FileNotFoundError(f"Missing CTC evidence: {ctc_path}")
        with np.load(ctc_path, allow_pickle=False) as ctc:
            ctc_pair_x = ctc["ctc_pair_features"].astype(np.float32)
            ctc_names = list(map(str, ctc["feature_names"]))
            if ctc_names != CTC_FEATURE_NAMES:
                raise RuntimeError(
                    f"CTC feature schema mismatch for {stem}: {ctc_names}"
                )
            for key, expected in (
                ("source_node", source_node),
                ("pair_source_idx", owner),
                ("pair_a", cache["pair_a"].astype(np.int64)),
                ("pair_b", cache["pair_b"].astype(np.int64)),
            ):
                if not np.array_equal(ctc[key], expected):
                    raise RuntimeError(
                        f"CTC evidence row alignment failed for {stem}: {key}"
                    )
    return VideoRows(
        stem=stem,
        embryo=stem.split("_", 1)[0],
        source_x=np.nan_to_num(cache["source_x"].astype(np.float32), nan=0.0),
        source_y=cache["source_y"].astype(np.int8),
        source_node=source_node,
        source_tube=source_tubes(args.full_population_cache, stem, source_node),
        source_event=cache["source_event"].astype(np.int32),
        pair_x=np.nan_to_num(cache["pair_x"].astype(np.float32), nan=0.0),
        pair_y=cache["pair_y"].astype(np.int8),
        pair_owner=owner,
        pair_a=cache["pair_a"].astype(np.int64),
        pair_b=cache["pair_b"].astype(np.int64),
        c_pair_x=c_pair_x,
        ctc_pair_x=np.nan_to_num(
            ctc_pair_x, nan=0.0, posinf=0.0, neginf=0.0
        ),
        public_pair_x=np.nan_to_num(
            public_pair_x, nan=0.0, posinf=0.0, neginf=0.0
        ),
        event_total=len(cache["event_rows"]),
    )


def pair_training_rows(
    videos: list[VideoRows],
    max_neg: int,
    seed: int,
    use_c: bool,
    use_ctc: bool = False,
    use_public: bool = False,
):
    xs, ys, groups = [], [], []
    rng = np.random.default_rng(seed)
    for video_index, video in enumerate(videos):
        parts = [video.pair_x]
        if use_c:
            parts.append(video.c_pair_x)
        if use_ctc:
            parts.append(video.ctc_pair_x)
        if use_public:
            parts.append(video.public_pair_x)
        x = np.concatenate(parts, axis=1)
        positive = np.flatnonzero(video.pair_y == 1)
        positive_sources = np.unique(video.pair_owner[positive])
        hard = []
        for source in positive_sources:
            rows = np.flatnonzero(video.pair_owner == source)
            negatives = rows[video.pair_y[rows] == 0]
            if len(negatives):
                # ``candidate_pairs`` is already ordered by its geometric
                # score.  Preserve that leakage-free order for hard negatives.
                hard.extend(negatives[:48].tolist())
        continuation_sources = np.flatnonzero(video.source_y == 0)
        continuation_rows = np.flatnonzero(
            np.isin(video.pair_owner, continuation_sources)
        )
        if len(continuation_rows):
            hard.extend(
                continuation_rows[: min(1500, len(continuation_rows))].tolist()
            )
        selected = np.unique(np.r_[positive, np.asarray(hard, np.int64)])
        xs.append(x[selected])
        ys.append(video.pair_y[selected])
        groups.append(np.full(len(selected), video_index, np.int32))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    group = np.concatenate(groups)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if len(negative) > max_neg:
        negative = rng.choice(negative, max_neg, replace=False)
    keep = np.r_[positive, negative]
    rng.shuffle(keep)
    return x[keep], y[keep], group[keep]


def fit_pair_model(
    videos: list[VideoRows],
    args,
    seed: int,
    use_c: bool = True,
    use_ctc: bool = False,
    use_public: bool = False,
):
    x, y, _ = pair_training_rows(
        videos, args.max_pair_negatives, seed, use_c, use_ctc, use_public
    )
    weight = np.ones(len(y), np.float32)
    weight[y == 1] = min(75.0, max(1.0, (y == 0).sum() / max((y == 1).sum(), 1)))
    model = HistGradientBoostingClassifier(
        learning_rate=0.065,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def source_matrix(
    video: VideoRows,
    corrected_pair_model,
    use_c: bool = True,
    use_ctc: bool = False,
    use_public: bool = False,
):
    pair_parts = [video.pair_x]
    if use_c:
        pair_parts.append(video.c_pair_x)
    if use_ctc:
        pair_parts.append(video.ctc_pair_x)
    if use_public:
        pair_parts.append(video.public_pair_x)
    pair_input = np.concatenate(pair_parts, axis=1)
    corrected_pair_score = (
        safe_probability(corrected_pair_model, pair_input)
        if len(pair_input)
        else np.empty(0, np.float32)
    )
    best = grouped_best_pair(
        video.pair_owner, corrected_pair_score, len(video.source_x)
    )
    pair_width = video.pair_x.shape[1]
    c_width = video.c_pair_x.shape[1]
    ctc_width = video.ctc_pair_x.shape[1]
    public_width = video.public_pair_x.shape[1]
    best_pair_x = np.zeros((len(best), pair_width), np.float32)
    best_c_x = np.zeros((len(best), c_width), np.float32)
    best_ctc_x = np.zeros((len(best), ctc_width), np.float32)
    best_public_x = np.zeros((len(best), public_width), np.float32)
    best_score = np.zeros(len(best), np.float32)
    best_pair_label = np.zeros(len(best), np.int8)
    for source_row, pair_row in enumerate(best):
        if pair_row < 0:
            continue
        best_pair_x[source_row] = video.pair_x[pair_row]
        best_c_x[source_row] = video.c_pair_x[pair_row]
        best_ctc_x[source_row] = video.ctc_pair_x[pair_row]
        best_public_x[source_row] = video.public_pair_x[pair_row]
        best_score[source_row] = corrected_pair_score[pair_row]
        best_pair_label[source_row] = video.pair_y[pair_row]
    parts = [video.source_x, best_pair_x, best_score[:, None]]
    if use_c:
        parts.append(best_c_x)
    if use_ctc:
        parts.append(best_ctc_x)
    if use_public:
        parts.append(best_public_x)
    x = np.concatenate(parts, axis=1)
    return x, best, best_pair_label


def fit_source_model(
    videos: list[VideoRows],
    pair_model,
    seed: int,
    use_c: bool = True,
    use_ctc: bool = False,
    use_public: bool = False,
):
    xs, ys = [], []
    for video in videos:
        x, _, _ = source_matrix(
            video, pair_model, use_c=use_c, use_ctc=use_ctc,
            use_public=use_public,
        )
        xs.append(x)
        ys.append(video.source_y)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.ones(len(y), np.float32)
    weight[y == 1] = min(100.0, max(1.0, (y == 0).sum() / max((y == 1).sum(), 1)))
    model = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=260,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def score_videos(
    videos,
    pair_model,
    source_model,
    use_c: bool = True,
    use_ctc: bool = False,
    use_public: bool = False,
):
    scored = []
    for video in videos:
        source_x, best_pair, pair_label = source_matrix(
            video, pair_model, use_c=use_c, use_ctc=use_ctc,
            use_public=use_public,
        )
        score = safe_probability(source_model, source_x)
        scored.append((video, score, best_pair, pair_label))
    return scored


def metric(scored, threshold: float):
    tp = fp = 0
    total_events = sum(video.event_total for video, *_ in scored)
    found_events = set()
    selected = 0
    for video, score, best_pair, pair_label in scored:
        winner_by_tube = {}
        for row in np.flatnonzero(score >= threshold):
            tube = int(video.source_tube[row])
            old = winner_by_tube.get(tube)
            if old is None or score[row] > score[old]:
                winner_by_tube[tube] = int(row)
        locked_targets: set[int] = set()
        ordered_rows = sorted(
            winner_by_tube.values(), key=lambda row: float(score[row]), reverse=True
        )
        for row in ordered_rows:
            pair_row = int(best_pair[row])
            if pair_row < 0:
                continue
            targets = {
                int(video.pair_a[pair_row]), int(video.pair_b[pair_row])
            }
            if locked_targets & targets:
                continue
            locked_targets.update(targets)
            selected += 1
            if video.source_y[row] == 1 and pair_label[row] == 1:
                event = (video.stem, int(video.source_event[row]))
                if event not in found_events:
                    found_events.add(event)
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
    fn = total_events - tp
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "selected": int(selected),
        "jaccard": tp / max(tp + fp + fn, 1),
    }


def threshold_sweep(scored):
    """Pooled sweep — prefer threshold_sweep_by_fold for honest selection."""
    candidates = np.unique(np.r_[
        np.linspace(0.05, 0.99, 95),
        np.linspace(0.990, 0.999, 19),
    ])
    results = [metric(scored, value) for value in candidates]
    return max(results, key=lambda row: (row["jaccard"], row["tp"], -row["fp"]))


def threshold_sweep_by_fold(scored_with_fold):
    """Mean Jaccard across folds; never optimize pooled OOF directly."""
    candidates = np.unique(np.r_[
        np.linspace(0.05, 0.99, 95),
        np.linspace(0.990, 0.999, 19),
    ])
    by_fold: dict[int, list] = {}
    for fold_i, row in scored_with_fold:
        by_fold.setdefault(int(fold_i), []).append(row)
    best = None
    best_key = None
    sweep = []
    for value in candidates:
        fold_js = []
        for fold_i, rows in sorted(by_fold.items()):
            fold_js.append(metric(rows, value)["jaccard"])
        mean_j = float(np.mean(fold_js)) if fold_js else 0.0
        std_j = float(np.std(fold_js)) if fold_js else 0.0
        entry = {
            "threshold": float(value),
            "mean_jaccard": mean_j,
            "std_jaccard": std_j,
            "per_fold": fold_js,
        }
        sweep.append(entry)
        key = (mean_j, -std_j)
        if best_key is None or key > best_key:
            best_key = key
            best = entry
    assert best is not None
    # Compat shape with threshold_sweep / metric consumers.
    return {
        "threshold": best["threshold"],
        "jaccard": best["mean_jaccard"],
        "mean_jaccard": best["mean_jaccard"],
        "std_jaccard": best["std_jaccard"],
        "per_fold": best["per_fold"],
        "policy": "mean_over_folds",
        "sweep_top": sweep[-5:],
    }


def frozen_v2_scores_for_videos(videos, args):
    pair_model = joblib.load(args.v2_artifact / "pair_model.joblib")
    v1_model = joblib.load(args.v2_artifact / "v1_source_model.joblib")
    gate_model = joblib.load(args.v2_artifact / "gate_model.joblib")
    result = []
    for video in videos:
        cache = torch.load(
            args.event_cache / f"{video.stem}.pt",
            map_location="cpu",
            weights_only=False,
        )
        pair_score, v1, gate = base_v2_scores(
            cache, pair_model, v1_model, gate_model
        )
        best = grouped_best_pair(
            video.pair_owner, pair_score, len(video.source_x)
        )
        pmax = np.zeros(len(video.source_x), np.float32)
        for row, pair_row in enumerate(best):
            if pair_row >= 0:
                pmax[row] = pair_score[pair_row]
        passing = (
            (gate >= 0.895)
            | (
                (gate >= 0.800)
                & (pmax >= 0.850)
                & (pmax <= 0.920)
                & (v1 >= 0.9475)
            )
        )
        binary = np.zeros(len(video.source_x), np.float32)
        winner_by_tube = {}
        for row in np.flatnonzero(passing):
            tube = int(video.source_tube[row])
            old = winner_by_tube.get(tube)
            if old is None or gate[row] > gate[old]:
                winner_by_tube[tube] = int(row)
        binary[np.asarray(list(winner_by_tube.values()), np.int64)] = 1.0
        pair_label = np.zeros(len(video.source_x), np.int8)
        valid = best >= 0
        pair_label[valid] = video.pair_y[best[valid]]
        result.append((video, binary, best, pair_label))
    return result


def additive_union_metric_from_scores(
    frozen_v2_scored,
    addition_scored,
    addition_threshold: float,
):
    union = []
    for base_row, add_row in zip(frozen_v2_scored, addition_scored):
        video, base_score, base_best, _ = base_row
        _, add_score, add_best, _ = add_row
        score = np.zeros(len(video.source_x), np.float32)
        best = add_best.copy()
        add_selected = add_score >= addition_threshold
        score[add_selected] = 1.0
        base_selected = base_score >= 0.5
        # Frozen V2 has priority within a lineage and at shared children.
        score[base_selected] = 2.0
        best[base_selected] = base_best[base_selected]
        pair_label = np.zeros(len(video.source_x), np.int8)
        valid = best >= 0
        pair_label[valid] = video.pair_y[best[valid]]
        union.append((video, score, best, pair_label))
    return metric(union, 0.5)


def train_family(
    train,
    held,
    practice,
    args,
    groups,
    use_c: bool,
    label: str,
    use_ctc: bool = False,
    use_public: bool = False,
):
    splitter = GroupKFold(n_splits=args.folds)
    oof_parts = []
    oof_with_fold = []
    dummy = np.zeros(len(train))
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(dummy, groups=groups)):
        fit_videos = [train[i] for i in fit_idx]
        val_videos = [train[i] for i in val_idx]
        print(
            f"{label} fold {fold}: fit={len(fit_videos)} held={len(val_videos)}",
            flush=True,
        )
        pair = fit_pair_model(
            fit_videos,
            args,
            args.seed + fold,
            use_c=use_c,
            use_ctc=use_ctc,
            use_public=use_public,
        )
        source = fit_source_model(
            fit_videos,
            pair,
            args.seed + 100 + fold,
            use_c=use_c,
            use_ctc=use_ctc,
            use_public=use_public,
        )
        scored = score_videos(
            val_videos,
            pair,
            source,
            use_c=use_c,
            use_ctc=use_ctc,
            use_public=use_public,
        )
        oof_parts.extend(scored)
        for row in scored:
            oof_with_fold.append((fold, row))
    frozen_threshold = threshold_sweep_by_fold(oof_with_fold)
    print(f"{label} OOF {json.dumps({k: frozen_threshold[k] for k in frozen_threshold if k != 'sweep_top'})}", flush=True)

    # Durable OOF scores for downstream / audits.
    oof_rows = []
    for fold_i, (video, score, best_pair, pair_label) in oof_with_fold:
        for row in range(len(score)):
            oof_rows.append(
                {
                    "movie": video.stem,
                    "fold_id": f"gkf5_{fold_i}",
                    "scheme": "gkf_movie",
                    "source_row": int(row),
                    "y_score": float(score[row]),
                    "y_true": int(video.source_y[row]) if hasattr(video, "source_y") else None,
                    "pair_label": int(pair_label[row]),
                    "best_pair": int(best_pair[row]),
                    "embryo": video.stem.split("_", 1)[0],
                    "family": label,
                }
            )
    oof_path = args.output / f"oof_{label.replace('+', '_').replace(' ', '_').lower()}.parquet"
    try:
        import pandas as pd

        pd.DataFrame(oof_rows).to_parquet(oof_path, index=False)
        print(f"{label} wrote OOF {oof_path} rows={len(oof_rows)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        (args.output / f"oof_{label.replace('+', '_').replace(' ', '_').lower()}.json").write_text(
            json.dumps({"error": str(exc), "n_rows": len(oof_rows)}) + "\n"
        )
    final_pair = fit_pair_model(
        train,
        args,
        args.seed + 500,
        use_c=use_c,
        use_ctc=use_ctc,
        use_public=use_public,
    )
    final_source = fit_source_model(
        train,
        final_pair,
        args.seed + 600,
        use_c=use_c,
        use_ctc=use_ctc,
        use_public=use_public,
    )
    held_scored = score_videos(
        held,
        final_pair,
        final_source,
        use_c=use_c,
        use_ctc=use_ctc,
        use_public=use_public,
    )
    held_result = metric(held_scored, frozen_threshold["threshold"])
    held_calibration = threshold_sweep(held_scored)
    per_embryo = {}
    for embryo in ("44b6", "6bba"):
        subset = [row for row in held_scored if row[0].embryo == embryo]
        per_embryo[embryo] = metric(
            subset, frozen_threshold["threshold"]
        )
    print(f"{label} HELD FROZEN {json.dumps(held_result)}", flush=True)
    if practice:
        practice_scored = score_videos(
            practice,
            final_pair,
            final_source,
            use_c=use_c,
            use_ctc=use_ctc,
            use_public=use_public,
        )
        practice_result = metric(
            practice_scored, frozen_threshold["threshold"]
        )
        print(
            f"{label} PRACTICE DIRECT @ OOF-FROZEN "
            f"{json.dumps(practice_result)}",
            flush=True,
        )
    else:
        practice_scored = []
        practice_result = None
    return {
        "label": label,
        "use_model_c": use_c,
        "use_ctc_pair_ranker": use_ctc,
        "use_public_backbone_evidence": use_public,
        "pair_model": final_pair,
        "source_model": final_source,
        "oof_frozen_threshold": frozen_threshold,
        "held_frozen": held_result,
        "held_per_embryo": per_embryo,
        # Compat key now aliases OOF-frozen (held sweep is diagnostic-only).
        "held_calibrated_threshold": frozen_threshold,
        "held_calibration_diagnostic_only": held_calibration,
        "practice_final_test": practice_result,
        "practice_scored": practice_scored,
    }


def model_c_evidence_root(stem: str, args) -> Path:
    roots = (
        args.train_evidence,
        args.held_evidence,
        args.practice_evidence,
    )
    matches = [root for root in roots if (root / f"{stem}.npz").exists()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Model-C evidence file for {stem}, "
            f"found {len(matches)}"
        )
    return matches[0]


def run_ctc_comparison(args, split: dict) -> None:
    held_names = list(map(str, split["held"]))
    all_names = sorted(path.stem for path in args.event_cache.glob("*.pt"))
    train_names = sorted(set(all_names) - set(held_names))
    if len(train_names) != 190 or len(held_names) != 9:
        raise RuntimeError(
            f"CTC comparison expects 190/9, got "
            f"{len(train_names)}/{len(held_names)}"
        )
    print(
        "Loading fold-clean CTC evidence comparison: "
        f"train={len(train_names)} held={len(held_names)}",
        flush=True,
    )
    train = [
        load_video(stem, args, model_c_evidence_root(stem, args))
        for stem in train_names
    ]
    held = [
        load_video(stem, args, model_c_evidence_root(stem, args))
        for stem in held_names
    ]
    groups = np.asarray([video.stem for video in train])
    control = train_family(
        train,
        held,
        [],
        args,
        groups,
        use_c=True,
        use_ctc=False,
        label="MODEL-C CONTROL",
    )
    combined = train_family(
        train,
        held,
        [],
        args,
        groups,
        use_c=True,
        use_ctc=True,
        label="MODEL-C+CTC",
    )
    oof_control = control["oof_frozen_threshold"]["jaccard"]
    oof_combined = combined["oof_frozen_threshold"]["jaccard"]
    held_control = control["held_frozen"]["jaccard"]
    held_combined = combined["held_frozen"]["jaccard"]
    oof_gain = oof_combined - oof_control
    held_gain = held_combined - held_control
    embryo_safe = all(
        combined["held_per_embryo"][embryo]["jaccard"]
        >= control["held_per_embryo"][embryo]["jaccard"]
        for embryo in ("44b6", "6bba")
    )
    promoted = bool(
        oof_gain >= args.min_ctc_oof_gain
        and held_gain >= 0.0
        and embryo_safe
    )
    joblib.dump(
        combined["pair_model"],
        args.output / "pair_model.joblib",
        compress=3,
    )
    joblib.dump(
        combined["source_model"],
        args.output / "source_model.joblib",
        compress=3,
    )
    joblib.dump(
        control["pair_model"],
        args.output / "control_pair_model.joblib",
        compress=3,
    )
    joblib.dump(
        control["source_model"],
        args.output / "control_source_model.joblib",
        compress=3,
    )
    for result in (control, combined):
        result.pop("pair_model")
        result.pop("source_model")
        result.pop("practice_scored")
    summary = {
        "version": "model-c-ctc-v2-event-decoder-v1",
        "method": (
            "Exact Model-C/V2 geometry decoder recipe plus fold-clean, "
            "source-normalized CTC daughter-pair evidence"
        ),
        "split": {
            "train_videos": len(train_names),
            "held_videos": len(held_names),
            "practice_in_train": sorted(
                {
                    "44b6_0113de3b",
                    "44b6_0b24845f",
                    "6bba_05b6850b",
                    "6bba_05db0fb1",
                }
                & set(train_names)
            ),
            "held_video_names": held_names,
        },
        "feature_contract": {
            "v2_geometry_features": 78,
            "model_c_features": len(C_FEATURE_NAMES),
            "ctc_features": len(CTC_FEATURE_NAMES),
            "pair_input_features": 78
            + len(C_FEATURE_NAMES)
            + len(CTC_FEATURE_NAMES),
            "source_input_features": 41
            + 78
            + 1
            + len(C_FEATURE_NAMES)
            + len(CTC_FEATURE_NAMES),
            "v2_model_scores_used_as_features": False,
            "ctc_feature_names": CTC_FEATURE_NAMES,
        },
        "control": control,
        "combined": combined,
        "oof_jaccard_gain": oof_gain,
        "held9_jaccard_gain": held_gain,
        "promotion_rules": {
            "minimum_oof_jaccard_gain": args.min_ctc_oof_gain,
            "held9_non_regression": True,
            "held9_each_embryo_non_regression": True,
            "unknown_sources_as_negatives": False,
            "held9_used_for_threshold_selection": False,
        },
        "promoted_to_graph_replay": promoted,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


def run_public_evidence_comparison(args, split: dict) -> None:
    if args.public_primary_evidence is None or args.public_secondary_evidence is None:
        raise ValueError(
            "Both --public-primary-evidence and "
            "--public-secondary-evidence are required."
        )
    train_names = [str(x).removesuffix(".zarr") for x in split["train"]]
    held_names = [str(x).removesuffix(".zarr") for x in split["held"]]
    practice_names = [str(x).removesuffix(".zarr") for x in split["practice"]]
    print(
        "Loading public-backbone evidence comparison: "
        f"train={len(train_names)} held={len(held_names)} "
        f"practice={len(practice_names)}",
        flush=True,
    )
    train = [
        load_video(stem, args, args.train_evidence)
        for stem in train_names
    ]
    held = [
        load_video(stem, args, args.held_evidence)
        for stem in held_names
    ]
    practice = [
        load_video(stem, args, args.practice_evidence)
        for stem in practice_names
    ]
    groups = np.asarray([video.stem for video in train])
    control = train_family(
        train,
        held,
        practice,
        args,
        groups,
        use_c=True,
        use_public=False,
        label="PUBLIC-GRAPH MODEL-C CONTROL",
    )
    combined = train_family(
        train,
        held,
        practice,
        args,
        groups,
        use_c=True,
        use_public=True,
        label="MODEL-C+PUBLIC-BACKBONE-EVIDENCE",
    )
    oof_gain = (
        combined["oof_frozen_threshold"]["jaccard"]
        - control["oof_frozen_threshold"]["jaccard"]
    )
    held_gain = (
        combined["held_frozen"]["jaccard"]
        - control["held_frozen"]["jaccard"]
    )
    embryo_safe = all(
        combined["held_per_embryo"][embryo]["jaccard"]
        >= control["held_per_embryo"][embryo]["jaccard"]
        for embryo in ("44b6", "6bba")
    )
    promoted = bool(
        oof_gain >= args.min_public_oof_gain
        and held_gain >= 0.0
        and embryo_safe
    )
    joblib.dump(
        combined["pair_model"],
        args.output / "pair_model.joblib",
        compress=3,
    )
    joblib.dump(
        combined["source_model"],
        args.output / "source_model.joblib",
        compress=3,
    )
    joblib.dump(
        control["pair_model"],
        args.output / "control_pair_model.joblib",
        compress=3,
    )
    joblib.dump(
        control["source_model"],
        args.output / "control_source_model.joblib",
        compress=3,
    )
    for result in (control, combined):
        result.pop("pair_model")
        result.pop("source_model")
        result.pop("practice_scored")
    summary = {
        "version": "model-c-public934-evidence-decoder-v1",
        "method": (
            "Exact proven Model-C/V2 decoder recipe with native evidence "
            "from the public .934 primary and independent-seed secondary "
            "backbones."
        ),
        "split": {
            "train_videos": len(train_names),
            "held_videos": len(held_names),
            "practice_videos": len(practice_names),
            "held_video_names": held_names,
        },
        "feature_contract": {
            "v2_geometry_features": 78,
            "model_c_features": len(C_FEATURE_NAMES),
            "public_primary_features": len(C_FEATURE_NAMES),
            "public_secondary_features": len(C_FEATURE_NAMES),
            "pair_input_features": (
                78 + len(C_FEATURE_NAMES) + len(PUBLIC_FEATURE_NAMES)
            ),
            "source_input_features": (
                41 + 78 + 1 + len(C_FEATURE_NAMES)
                + len(PUBLIC_FEATURE_NAMES)
            ),
            "v2_model_scores_used_as_features": False,
            "public_feature_names": PUBLIC_FEATURE_NAMES,
        },
        "control": control,
        "combined": combined,
        "oof_jaccard_gain": oof_gain,
        "held20_jaccard_gain": held_gain,
        "promotion_rules": {
            "minimum_oof_jaccard_gain": args.min_public_oof_gain,
            "held20_non_regression": True,
            "held20_each_embryo_non_regression": True,
            "unknown_sources_as_negatives": False,
            "threshold_selected_on_held20": False,
        },
        "promoted_to_graph_replay": promoted,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (
        args.public_primary_evidence is not None
        or args.public_secondary_evidence is not None
    ):
        run_public_evidence_comparison(args, split)
        return
    if args.ctc_evidence is not None:
        run_ctc_comparison(args, split)
        return
    train_names = [str(x).removesuffix(".zarr") for x in split["train"]]
    held_names = [str(x).removesuffix(".zarr") for x in split["held"]]
    practice_names = [str(x).removesuffix(".zarr") for x in split["practice"]]
    print(f"Loading {len(train_names)} training videos", flush=True)
    train = [
        load_video(stem, args, args.train_evidence)
        for stem in train_names
    ]
    print(f"Loading {len(held_names)} untouched held videos", flush=True)
    held = [
        load_video(stem, args, args.held_evidence)
        for stem in held_names
    ]
    print(f"Loading {len(practice_names)} untouched practice videos", flush=True)
    practice = [
        load_video(stem, args, args.practice_evidence)
        for stem in practice_names
    ]

    groups = np.asarray([video.stem for video in train])
    frozen_v2_practice = frozen_v2_scores_for_videos(practice, args)
    control = train_family(
        train, held, practice, args, groups,
        use_c=False, label="V2-FEATURE CONTROL"
    )
    combined = train_family(
        train, held, practice, args, groups,
        use_c=True, label="V2+MODEL-C"
    )
    frozen_v2_practice_metric = metric(frozen_v2_practice, 0.5)
    control_union = additive_union_metric_from_scores(
        frozen_v2_practice,
        control["practice_scored"],
        control["oof_frozen_threshold"]["threshold"],
    )
    combined_union = additive_union_metric_from_scores(
        frozen_v2_practice,
        combined["practice_scored"],
        combined["oof_frozen_threshold"]["threshold"],
    )
    print(
        "PRACTICE ADDITIVE "
        + json.dumps({
            "frozen_v2": frozen_v2_practice_metric,
            "control_union": control_union,
            "model_c_union": combined_union,
        }),
        flush=True,
    )
    joblib.dump(combined["pair_model"], args.output / "pair_model.joblib", compress=3)
    joblib.dump(combined["source_model"], args.output / "source_model.joblib", compress=3)
    joblib.dump(control["pair_model"], args.output / "control_pair_model.joblib", compress=3)
    joblib.dump(control["source_model"], args.output / "control_source_model.joblib", compress=3)
    for result in (control, combined):
        result.pop("pair_model")
        result.pop("source_model")
        result.pop("practice_scored")
    c_adds_value = (
        combined_union["jaccard"] > frozen_v2_practice_metric["jaccard"]
        and combined_union["jaccard"] >= control_union["jaccard"]
    )
    summary = {
        "version": "model-c-v2-event-decoder-v1",
        "method": "native Model-C TTA evidence appended to frozen Division-V2 candidates",
        "train_videos": len(train),
        "held_videos": len(held),
        "practice_videos": len(practice),
        "model_c_feature_names": C_FEATURE_NAMES,
        "pair_input_features": 78 + len(C_FEATURE_NAMES),
        "source_input_features": 41 + 78 + 1 + len(C_FEATURE_NAMES),
        "control": control,
        "combined": combined,
        "practice_additive": {
            "frozen_v2": frozen_v2_practice_metric,
            "control_union": control_union,
            "model_c_union": combined_union,
        },
        "model_c_adds_practice_value": c_adds_value,
        "promotion_gate": (
            "Model C must add value over frozen V2 and beat the identically "
            "trained V2-feature control on the untouched four practice clips "
            "before official graph replay"
        ),
        "sparse_safety": {
            "unannotated_sources_as_negatives": False,
            "practice_clips_used_for_final_gate": True,
            "threshold_selected_on_practice": False,
            "threshold_selected_on_validation_held20": False,
            "threshold_selected_on_oof_only": True,
            "practice_used_for_training_or_threshold": False,
            "frozen_v2_probabilities_used_as_features": False,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
