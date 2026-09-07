#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]


def _probability(model, x: np.ndarray) -> np.ndarray:
    if not len(x):
        return np.empty(0, np.float32)
    return model.predict_proba(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))[:, 1].astype(
        np.float32
    )


def _grouped_best(owner: np.ndarray, score: np.ndarray, n_sources: int) -> np.ndarray:
    best = np.full(n_sources, -1, np.int64)
    for row, source in enumerate(owner):
        source = int(source)
        previous = int(best[source])
        if previous < 0 or score[row] > score[previous]:
            best[source] = row
    return best


def native_features(
    evidence_path: Path,
    nodes: dict[int, dict[str, Any]],
    source_node: np.ndarray,
    pair_owner: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    with np.load(evidence_path, allow_pickle=False) as z:
        source_native = z['source_id'].astype(np.int64)
        target_native = z['target_id'].astype(np.int64)
        probability = z['probability'].astype(np.float32)
        alternative = z['alternative_parent_probability'].astype(np.float32)
        winner = z['is_target_winner'].astype(np.float32)
        edge_distance = z['distance_um'].astype(np.float32)
        rank = z['source_target_rank'].astype(np.float32)
        native_coords = z['native_node_coords'].astype(np.float32)
        native_offsets = z['native_frame_offsets'].astype(np.int64)

    n_sources = len(source_node)
    n_pairs = len(pair_owner)
    source_frame = np.full(n_sources, -1, np.int16)
    source_position = np.zeros((n_sources, 3), np.float32)
    for row, node_id in enumerate(source_node):
        node = nodes.get(int(node_id))
        if node is not None:
            source_frame[row] = int(node['t'])
            source_position[row] = (node['z'], node['y'], node['x'])

    pair_a_position = np.zeros((n_pairs, 3), np.float32)
    pair_b_position = np.zeros((n_pairs, 3), np.float32)
    pair_a_valid = np.zeros(n_pairs, bool)
    pair_b_valid = np.zeros(n_pairs, bool)
    for row, (a_node, b_node) in enumerate(zip(pair_a, pair_b)):
        a_value = nodes.get(int(a_node))
        b_value = nodes.get(int(b_node))
        if a_value is not None:
            pair_a_valid[row] = True
            pair_a_position[row] = (a_value['z'], a_value['y'], a_value['x'])
        if b_value is not None:
            pair_b_valid[row] = True
            pair_b_position[row] = (b_value['z'], b_value['y'], b_value['x'])

    source_to_native = np.full(n_sources, -1, np.int64)
    source_map_distance = np.zeros(n_sources, np.float32)
    rows_by_frame: dict[int, list[int]] = defaultdict(list)
    for row, frame in enumerate(source_frame):
        if frame >= 0:
            rows_by_frame[int(frame)].append(row)
    for frame, source_rows_list in rows_by_frame.items():
        if frame + 1 >= len(native_offsets):
            continue
        n0, n1 = int(native_offsets[frame]), int(native_offsets[frame + 1])
        if n1 <= n0:
            continue
        source_rows = np.asarray(source_rows_list, np.int64)
        tree = cKDTree(native_coords[n0:n1, 1:] * spacing)
        distance, index = tree.query(source_position[source_rows] * spacing, k=1)
        valid = distance <= 6.0
        source_to_native[source_rows[valid]] = n0 + index[valid].astype(np.int64)
        source_map_distance[source_rows[valid]] = distance[valid].astype(np.float32)

    evidence_order = np.argsort(source_native, kind='stable')
    sorted_source = source_native[evidence_order]
    starts = (
        np.flatnonzero(np.r_[True, sorted_source[1:] != sorted_source[:-1]])
        if len(evidence_order)
        else np.empty(0, np.int64)
    )
    ends = np.r_[starts[1:], len(evidence_order)]
    evidence_groups = {
        int(sorted_source[left]): evidence_order[left:right] for left, right in zip(starts, ends)
    }
    pair_order = np.argsort(pair_owner, kind='stable')
    sorted_owner = pair_owner[pair_order]
    starts = (
        np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
        if len(pair_order)
        else np.empty(0, np.int64)
    )
    ends = np.r_[starts[1:], len(pair_order)]
    pair_groups = {
        int(sorted_owner[left]): pair_order[left:right] for left, right in zip(starts, ends)
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
    evidence_count = np.zeros(n_sources, np.float32)
    winner_count = np.zeros(n_sources, np.float32)
    p50_count = np.zeros(n_sources, np.float32)
    winner_p50_count = np.zeros(n_sources, np.float32)
    max_probability = np.zeros(n_sources, np.float32)

    for source_row, owned_pairs in pair_groups.items():
        native_source = int(source_to_native[source_row])
        evidence_rows = evidence_groups.get(native_source)
        if evidence_rows is None or not len(evidence_rows):
            continue
        evidence_count[source_row] = len(evidence_rows)
        winner_count[source_row] = float(winner[evidence_rows].sum())
        p50_count[source_row] = np.count_nonzero(probability[evidence_rows] >= 0.5)
        winner_p50_count[source_row] = np.count_nonzero(
            (probability[evidence_rows] >= 0.5) & (winner[evidence_rows] > 0.5)
        )
        max_probability[source_row] = float(probability[evidence_rows].max())
        native_targets = target_native[evidence_rows]
        tree = cKDTree(native_coords[native_targets, 1:] * spacing)
        map_a, index_a = tree.query(pair_a_position[owned_pairs] * spacing, k=1)
        map_b, index_b = tree.query(pair_b_position[owned_pairs] * spacing, k=1)
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

    margin_a = pa - alt_a
    margin_b = pb - alt_b
    source_rows = pair_owner.astype(np.int64)
    result = np.column_stack(
        [
            (source_to_native[source_rows] >= 0).astype(np.float32),
            source_map_distance[source_rows],
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
            evidence_count[source_rows],
            winner_count[source_rows],
            p50_count[source_rows],
            winner_p50_count[source_rows],
            max_probability[source_rows],
            target_map_a,
            target_map_b,
        ]
    ).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class CombinedModelCPrimaryRuntime:
    def __init__(self, v2_runtime, artifact_dir: Path | str, threshold: float = 0.96):
        self.v2 = v2_runtime
        self.artifact_dir = Path(artifact_dir)
        self.pair_model = joblib.load(self.artifact_dir / 'decoder' / 'pair_model.joblib')
        self.source_model = joblib.load(self.artifact_dir / 'decoder' / 'source_model.joblib')
        self.threshold = float(threshold)

    def _apply_combined(
        self,
        dataset_path,
        evidence_path,
        nodes,
        edges,
        stats,
        registration_shifts_um=None,
        spacing=(1.625, 0.40625, 0.40625),
        v2_threshold=0.65,
        v2_rescue_delta=0.15,
        v2_steal_delta=0.25,
    ):
        spacing_array = np.asarray(spacing, np.float64)
        original_edges = [dict(edge) for edge in edges]
        scored = self.v2.score(
            dataset_path,
            nodes,
            original_edges,
            registration_shifts_um,
            spacing,
        )

        source_ids = scored['source_ids'].astype(np.int64)
        source_x = scored['_source_x'].astype(np.float32)
        pair_x = scored['_pair_x'].astype(np.float32)
        owner = scored['_pair_owner'].astype(np.int32)
        component_of = scored['_component_of']
        pair_nodes = np.asarray(scored['pair_nodes'], np.int64)
        if not len(source_ids) or not len(pair_nodes):
            stats['model_c_primary_selected_divisions'] = 0
            return original_edges

        c_pair_x = native_features(
            Path(evidence_path),
            nodes,
            source_ids,
            owner,
            pair_nodes[:, 0],
            pair_nodes[:, 1],
            spacing_array,
        )
        pair_input = np.concatenate([pair_x, c_pair_x], axis=1)
        if pair_input.shape[1] != 102:
            raise RuntimeError(f'Model-C pair feature mismatch: {pair_input.shape}')
        pair_score = _probability(self.pair_model, pair_input)
        best = _grouped_best(owner, pair_score, len(source_ids))
        best_pair_x = np.zeros((len(source_ids), 78), np.float32)
        best_c_x = np.zeros((len(source_ids), 24), np.float32)
        best_score = np.zeros(len(source_ids), np.float32)
        for source_row, pair_row in enumerate(best):
            if pair_row < 0:
                continue
            best_pair_x[source_row] = pair_x[pair_row]
            best_c_x[source_row] = c_pair_x[pair_row]
            best_score[source_row] = pair_score[pair_row]
        source_input = np.concatenate(
            [source_x, best_pair_x, best_score[:, None], best_c_x], axis=1
        )
        if source_input.shape[1] != 144:
            raise RuntimeError(f'Model-C source feature mismatch: {source_input.shape}')
        source_score = _probability(self.source_model, source_input)

        winner_by_tube: dict[int, int] = {}
        for row in np.flatnonzero(source_score >= self.threshold):
            tube = int(component_of.get(int(source_ids[row]), int(source_ids[row])))
            previous = winner_by_tube.get(tube)
            if previous is None or source_score[row] > source_score[previous]:
                winner_by_tube[tube] = int(row)
        candidates = sorted(
            [
                (
                    float(source_score[row]),
                    int(source_ids[row]),
                    int(pair_nodes[best[row], 0]),
                    int(pair_nodes[best[row], 1]),
                    int(component_of.get(int(source_ids[row]), int(source_ids[row]))),
                )
                for row in winner_by_tube.values()
                if best[row] >= 0
            ],
            reverse=True,
        )

        work = [dict(edge) for edge in original_edges]
        outgoing: dict[int, list[dict]] = defaultdict(list)
        for edge in work:
            outgoing[int(edge['source_id'])].append(edge)
        occupied_tubes = {
            int(component_of.get(source, source))
            for source, source_edges in outgoing.items()
            if len(source_edges) >= 2
        }
        locked_targets = {
            int(edge['target_id'])
            for source_edges in outgoing.values()
            if len(source_edges) >= 2
            for edge in source_edges
        }
        counters = defaultdict(int)
        counters['threshold_passing_tubes'] = len(candidates)
        for score, source, a, b, tube in candidates:
            if tube in occupied_tubes:
                counters['rejected_existing_fork_tube'] += 1
                continue
            if a in locked_targets or b in locked_targets:
                counters['rejected_existing_fork_child'] += 1
                continue
            if source not in nodes or a not in nodes or b not in nodes:
                counters['rejected_missing_node'] += 1
                continue
            source_t = int(nodes[source]['t'])
            if int(nodes[a]['t']) != source_t + 1 or int(nodes[b]['t']) != source_t + 1:
                counters['rejected_nonadjacent'] += 1
                continue

            pair = {a, b}
            work = [
                edge
                for edge in work
                if not (
                    (int(edge['source_id']) == source and int(edge['target_id']) not in pair)
                    or (int(edge['target_id']) in pair and int(edge['source_id']) != source)
                )
            ]
            present = {(int(edge['source_id']), int(edge['target_id'])) for edge in work}
            for target in (a, b):
                if (source, target) not in present:
                    work.append(
                        {
                            'source_id': source,
                            'target_id': target,
                            'edge_prob': score,
                            'learned_division': 1,
                            'model_c_combined_primary': 1,
                            'model_c_decoder_score': score,
                        }
                    )
                    counters['added_edges'] += 1
            occupied_tubes.add(tube)
            locked_targets.update(pair)
            counters['selected_divisions'] += 1

        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        for edge in work:
            source = int(edge['source_id'])
            target = int(edge['target_id'])
            outdegree[source] += 1
            indegree[target] += 1
        if any(value > 2 for value in outdegree.values()):
            raise RuntimeError('Model-C transaction produced out-degree > 2')
        if any(value > 1 for value in indegree.values()):
            raise RuntimeError('Model-C transaction produced in-degree > 1')
        for key, value in counters.items():
            stats[f'model_c_primary_{key}'] = int(value)
        stats['model_c_primary_threshold'] = float(self.threshold)
        stats['model_c_primary_legacy_v2_used'] = 0
        return work

    def apply(
        self,
        dataset_path,
        evidence_path,
        nodes,
        edges,
        stats,
        registration_shifts_um=None,
        spacing=(1.625, 0.40625, 0.40625),
        v2_threshold=0.65,
        v2_rescue_delta=0.15,
        v2_steal_delta=0.25,
    ):
        try:
            return self._apply_combined(
                dataset_path,
                evidence_path,
                nodes,
                edges,
                stats,
                registration_shifts_um=registration_shifts_um,
                spacing=spacing,
                v2_threshold=v2_threshold,
                v2_rescue_delta=v2_rescue_delta,
                v2_steal_delta=v2_steal_delta,
            )
        except Exception as error:
            stats['model_c_primary_fallback_to_v2'] = 1
            stats['model_c_primary_error'] = f'{type(error).__name__}: {error}'
            return self.v2.apply(
                dataset_path,
                nodes,
                [dict(edge) for edge in edges],
                stats,
                registration_shifts_um=registration_shifts_um,
                threshold=v2_threshold,
                rescue_delta=v2_rescue_delta,
                steal_delta=v2_steal_delta,
                spacing=spacing,
            )
