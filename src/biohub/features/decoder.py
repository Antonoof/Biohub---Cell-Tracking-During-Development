from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

C_FEATURE_NAMES = [
    'c_source_mapped',
    'c_source_map_distance_um',
    'c_edge_a_present',
    'c_edge_b_present',
    'c_edge_a_probability',
    'c_edge_b_probability',
    'c_edge_min_probability',
    'c_edge_mean_probability',
    'c_edge_max_probability',
    'c_edge_a_parent_margin',
    'c_edge_b_parent_margin',
    'c_edge_min_parent_margin',
    'c_edge_winner_count',
    'c_edge_a_distance_um',
    'c_edge_b_distance_um',
    'c_edge_a_rank',
    'c_edge_b_rank',
    'c_source_evidence_count',
    'c_source_winner_count',
    'c_source_probability_ge_05_count',
    'c_source_winner_probability_ge_05_count',
    'c_source_max_probability',
    'c_target_a_map_distance_um',
    'c_target_b_map_distance_um',
]

CTC_FEATURE_NAMES = [
    'ctc_pair_valid',
    'ctc_pair_is_top1',
    'ctc_pair_inverse_rank',
    'ctc_pair_softmax',
    'ctc_pair_centered_logit',
    'ctc_source_top1_softmax',
    'ctc_source_top1_top2_margin',
    'ctc_source_normalized_entropy',
    'ctc_source_geometry_agreement',
    'ctc_source_valid_pair_fraction',
]

PUBLIC_EVIDENCE_LABELS = ('public_primary', 'public_secondary')
PUBLIC_FEATURE_NAMES = [
    f'{label}_{name.removeprefix("c_")}'
    for label in PUBLIC_EVIDENCE_LABELS
    for name in C_FEATURE_NAMES
]


def model_c_features(
    evidence_path: Path,
    graph_coordinates: dict[int, tuple[int, np.ndarray]],
    source_node,
    pair_owner,
    pair_a,
    pair_b,
):
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
        spacing = z['spacing_um'].astype(np.float32)

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

    evidence_order = np.argsort(source_native, kind='stable')
    evidence_sorted_source = source_native[evidence_order]
    evidence_starts = (
        np.flatnonzero(np.r_[True, evidence_sorted_source[1:] != evidence_sorted_source[:-1]])
        if len(evidence_order)
        else np.empty(0, np.int64)
    )
    evidence_ends = np.r_[evidence_starts[1:], len(evidence_order)]
    evidence_groups = {
        int(evidence_sorted_source[left]): evidence_order[left:right]
        for left, right in zip(evidence_starts, evidence_ends)
    }
    pair_order = np.argsort(pair_owner, kind='stable')
    pair_sorted_owner = pair_owner[pair_order]
    pair_starts = (
        np.flatnonzero(np.r_[True, pair_sorted_owner[1:] != pair_sorted_owner[:-1]])
        if len(pair_order)
        else np.empty(0, np.int64)
    )
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
        source_p50_count[source_row] = float(np.count_nonzero(probability[evidence_rows] >= 0.5))
        source_winner_p50_count[source_row] = float(
            np.count_nonzero((probability[evidence_rows] >= 0.5) & (winner[evidence_rows] > 0.5))
        )
        source_max_probability[source_row] = float(probability[evidence_rows].max())
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
    c = np.column_stack(
        [
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
        ]
    ).astype(np.float32)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
