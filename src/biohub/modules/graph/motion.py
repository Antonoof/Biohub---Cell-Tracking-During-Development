import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

from biohub.modules.graph.geometry import coords_um, ids_by_frame


def _frame_registration(
    source_pos: np.ndarray, target_pos: np.ndarray, gate_um: float
) -> np.ndarray:
    if len(source_pos) < 5 or len(target_pos) < 5:
        return np.zeros(3)
    distance, index = cKDTree(target_pos).query(source_pos, k=1)
    keep = distance <= gate_um
    if int(keep.sum()) < 5:
        return np.zeros(3)
    return np.median(target_pos[index[keep]] - source_pos[keep], axis=0)


def motion_relink_edges(upgrade, nodes_by_id: dict, stats, learned_edge_probs=None) -> list[dict]:
    if not upgrade.motion_relink or not nodes_by_id:
        return []

    ids_by_t = ids_by_frame(upgrade, nodes_by_id)
    if max((len(ids) for ids in ids_by_t.values()), default=0) > upgrade.relink_max_frame_nodes:
        stats['motion_relink_skipped_large_frame'] = 1
        return []

    probs = learned_edge_probs or {}
    position: dict[int, np.ndarray] = {}
    density: dict[int, np.ndarray] = {}
    for t, ids in ids_by_t.items():
        coords = coords_um(upgrade, nodes_by_id, ids)
        position[t] = coords
        counts = cKDTree(coords).query_ball_point(coords, 15.0, return_length=True)
        density[t] = np.maximum(counts.astype(np.float64) - 1.0, 0.0)

    previous_position: dict[int, np.ndarray] = {}
    previous_history: dict[int, list] = {}
    selected: list[dict] = []

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue

        source_pos = position[t]
        target_pos = position[t + 1]
        velocity = np.zeros_like(source_pos)
        step_velocity = np.zeros_like(source_pos)
        has_predecessor = np.zeros(len(source_ids), dtype=np.float64)
        for index, node_id in enumerate(source_ids):
            previous = previous_position.get(node_id)
            if previous is not None:
                step_velocity[index] = source_pos[index] - previous
                velocity[index] = step_velocity[index]
                has_predecessor[index] = 1.0
            if upgrade.relink_motion_steps > 1:
                chain = previous_history.get(node_id)
                if chain:
                    points = [source_pos[index], *chain]
                    deltas = [
                        points[offset] - points[offset + 1] for offset in range(len(points) - 1)
                    ]
                    velocity[index] = np.mean(
                        deltas[: upgrade.relink_motion_steps],
                        axis=0,
                    )
                    has_predecessor[index] = 1.0
        shift = (
            _frame_registration(source_pos, target_pos, upgrade.relink_relaxed_um)
            if upgrade.relink_frame_registration
            else np.zeros(3)
        )
        if upgrade.relink_frame_registration:
            stats['relink_registered_frames'] += 1
            stats['relink_registration_shift_um_sum'] += float(np.linalg.norm(shift))
        extrapolation = upgrade.relink_velocity_axes * velocity
        predicted = source_pos + extrapolation + upgrade.relink_registration_weight * shift
        corrector_velocity = step_velocity if upgrade.relink_corrector_one_step else velocity
        velocity_magnitude = np.linalg.norm(corrector_velocity, axis=1)
        z_boundary_src = np.clip(source_pos[:, 0] / upgrade.z_extent_um, 0.0, 1.0)
        z_boundary_src = np.minimum(z_boundary_src, 1.0 - z_boundary_src)
        z_boundary_tgt = np.clip(target_pos[:, 0] / upgrade.z_extent_um, 0.0, 1.0)
        z_boundary_tgt = np.minimum(z_boundary_tgt, 1.0 - z_boundary_tgt)

        def assign_pass(
            source_sel: np.ndarray, target_sel: np.ndarray, gate_um: float, tight_bonus: float = 0.0
        ):
            if source_sel.size == 0 or target_sel.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            local_source = source_pos[source_sel]
            local_target = target_pos[target_sel]
            neighbours = cKDTree(local_target).query_ball_point(local_source, gate_um)
            counts = np.fromiter((len(item) for item in neighbours), np.int64, len(neighbours))
            if not counts.any():
                return np.empty(0, np.int64), np.empty(0, np.int64)
            rows = np.repeat(np.arange(len(source_sel), dtype=np.int64), counts)
            cols = np.fromiter(
                (index for item in neighbours for index in item), np.int64, int(counts.sum())
            )
            delta = local_target[cols] - local_source[rows]
            raw = np.linalg.norm(delta, axis=1)
            inside = raw <= gate_um
            rows, cols, delta, raw = rows[inside], cols[inside], delta[inside], raw[inside]
            if rows.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)

            global_source = source_sel[rows]
            global_target = target_sel[cols]
            motion = np.linalg.norm(local_target[cols] - predicted[global_source], axis=1)
            probability = np.fromiter(
                (
                    probs.get((source_ids[s], target_ids[g]), 0.0)
                    for s, g in zip(global_source, global_target)
                ),
                np.float64,
                rows.size,
            )
            base_cost = motion + upgrade.relink_raw_weight * raw
            values = base_cost - upgrade.relink_learned_bonus * probability

            if upgrade.motion_corrector is not None:
                absolute = np.abs(delta)
                registered_delta = delta - shift
                registered = np.linalg.norm(registered_delta, axis=1)
                features = np.column_stack(
                    [
                        base_cost,
                        raw,
                        registered,
                        motion,
                        absolute,
                        np.abs(registered_delta),
                        corrector_velocity[global_source],
                        velocity_magnitude[global_source],
                        np.tile(shift, (rows.size, 1)),
                        np.full(rows.size, float(np.linalg.norm(shift))),
                        density[t][global_source],
                        density[t + 1][global_target],
                        z_boundary_src[global_source],
                        z_boundary_tgt[global_target],
                        has_predecessor[global_source],
                    ]
                )
                residual = (
                    _motion_cost_residual(upgrade, features) * upgrade.motion_corrector_strength
                )
                values = values - residual
                stats['motion_corrector_candidates'] += int(rows.size)
                stats['motion_corrector_abs_residual_sum'] += float(np.abs(residual).sum())

            if tight_bonus > 0.0:
                values = values - tight_bonus * (raw <= upgrade.relink_tight_um)

            if upgrade.relink_max_match_cost > 0.0:
                judged = values if upgrade.relink_cap_includes_learned else base_cost
                keep = judged <= upgrade.relink_max_match_cost
                rejected = int(keep.size - keep.sum())
                if rejected:
                    stats['relink_cost_cap_rejected'] += rejected
                    rows, cols, values = rows[keep], cols[keep], values[keep]
                    if rows.size == 0:
                        stats['relink_cost_cap_emptied_frames'] += 1
                        return np.empty(0, np.int64), np.empty(0, np.int64)

            big = gate_um * 1000.0 + 1.0
            cost = np.full((source_sel.size, target_sel.size), big, dtype=np.float64)
            cost[rows, cols] = values

            if upgrade.relink_orphan_prior:
                surviving = (
                    probability[keep] if upgrade.relink_max_match_cost > 0.0 else probability
                )
                best = np.zeros(target_sel.size, np.float64)
                np.maximum.at(best, cols, surviving)
                orphan = upgrade.relink_orphan_base_um * np.clip(
                    upgrade.relink_orphan_floor
                    + (1.0 - upgrade.relink_orphan_floor) * upgrade.relink_orphan_scale * best,
                    0.0,
                    1.0,
                )
                cost = np.vstack([cost, np.tile(orphan, (source_sel.size, 1))])

            row_index, col_index = linear_sum_assignment(cost)
            if upgrade.relink_orphan_prior:
                real = row_index < source_sel.size
                stats['relink_orphan_declined'] += int((~real).sum())
                row_index, col_index = row_index[real], col_index[real]
            matched = cost[row_index, col_index] < big
            return source_sel[row_index[matched]], target_sel[col_index[matched]]

        open_sources = np.ones(len(source_ids), dtype=bool)
        open_targets = np.ones(len(target_ids), dtype=bool)
        if upgrade.relink_joint_assignment:
            schedule = (('joint', upgrade.relink_relaxed_um, upgrade.relink_tight_bonus_um),)
        else:
            schedule = (
                ('tight', upgrade.relink_tight_um, 0.0),
                ('relaxed', upgrade.relink_relaxed_um, 0.0),
            )
        for pass_name, gate_um, tight_bonus in schedule:
            matched_source, matched_target = assign_pass(
                np.flatnonzero(open_sources),
                np.flatnonzero(open_targets),
                gate_um,
                tight_bonus,
            )
            if matched_source.size == 0:
                continue
            open_sources[matched_source] = False
            open_targets[matched_target] = False

            raw = np.linalg.norm(target_pos[matched_target] - source_pos[matched_source], axis=1)
            if pass_name == 'joint':
                inside_tight = int((raw <= upgrade.relink_tight_um).sum())
                stats['motion_relink_tight_edges'] += inside_tight
                stats['motion_relink_relaxed_edges'] += int(matched_source.size) - inside_tight
                stats['motion_relink_joint_frames'] += 1
            else:
                stats[f'motion_relink_{pass_name}_edges'] += int(matched_source.size)

            motion = np.linalg.norm(target_pos[matched_target] - predicted[matched_source], axis=1)
            for offset in range(matched_source.size):
                source_id = source_ids[int(matched_source[offset])]
                target_id = target_ids[int(matched_target[offset])]
                selected.append(
                    {
                        'source_id': source_id,
                        'target_id': target_id,
                        'edge_prob': probs.get((source_id, target_id), 0.0),
                        'distance_um': float(raw[offset]),
                        'motion_distance_um': float(motion[offset]),
                        'motion_relinked': 1,
                        'motion_pass': pass_name,
                    }
                )
                source_point = source_pos[int(matched_source[offset])]
                previous_position[target_id] = source_point
                if upgrade.relink_motion_steps > 1:
                    previous_history[target_id] = [
                        source_point,
                        *previous_history.get(source_id, []),
                    ][: upgrade.relink_motion_steps]
        stats['motion_relink_frames'] += 1

    stats['motion_relink_edges'] = len(selected)
    return selected


def _motion_cost_residual(upgrade, features: np.ndarray) -> np.ndarray:
    if upgrade.motion_corrector is None or features.size == 0:
        return np.zeros(len(features), dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    x = (x - upgrade.motion_corrector['mean']) / upgrade.motion_corrector['std']
    scale = upgrade.motion_corrector['residual_scale']
    with torch.no_grad():
        raw = upgrade.motion_corrector['model'](x)
        residual = scale * torch.tanh(raw / scale)
    return residual.detach().cpu().numpy().astype(np.float64)
