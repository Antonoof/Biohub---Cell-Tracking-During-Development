import numpy as np

from biohub.modules.edgegraft.geometry import cosine, speed_ratio

BASE_FEATURES = [
    'p1_probability',
    'p2_probability',
    'p1_present',
    'p2_present',
    'p1_alternative',
    'p2_alternative',
    'p1_margin',
    'p2_margin',
    'p1_winner',
    'p2_winner',
    'p1_source_rank',
    'p2_source_rank',
    'p1_target_relative',
    'p2_target_relative',
    'model_probability_max',
    'model_probability_mean',
    'model_probability_absdiff',
    'distance_um',
    'target_union_candidates',
    'source_density_5um',
    'source_density_10um',
    'target_density_5um',
    'target_density_10um',
    'velocity_residual_um',
    'has_velocity',
    'is_raw_parent',
    'raw_edge_probability',
    'raw_parent_probability_gap',
]


def best_context(native_rows, need_in, need_out):
    incoming = {}
    outgoing = {}
    for source, target, probability, alternative, winner, distance in sorted(
        native_rows, key=lambda row: -row[2]
    ):
        value = (source, target, probability, alternative, winner, distance)
        if target in need_in and target not in incoming:
            incoming[target] = value
        if source in need_out and source not in outgoing:
            outgoing[source] = value
    return incoming, outgoing


def model_context(prefix, source, target, current_probability, nodes, incoming, outgoing):
    current = nodes[target] - nodes[source]
    previous = incoming.get(source)
    future = outgoing.get(target)
    previous_valid = float(previous is not None and previous[0] in nodes)
    future_valid = float(future is not None and future[1] in nodes)
    previous_delta = (
        nodes[source] - nodes[previous[0]] if previous_valid else np.zeros(3, np.float32)
    )
    future_delta = nodes[future[1]] - nodes[target] if future_valid else np.zeros(3, np.float32)
    previous_probability = previous[2] if previous_valid else 0.0
    future_probability = future[2] if future_valid else 0.0
    values = {
        f'{prefix}_prev_valid': previous_valid,
        f'{prefix}_prev_probability': previous_probability,
        f'{prefix}_prev_margin': (previous[2] - previous[3] if previous_valid else 0.0),
        f'{prefix}_prev_winner': previous[4] if previous_valid else 0.0,
        f'{prefix}_prev_distance': previous[5] if previous_valid else 0.0,
        f'{prefix}_future_valid': future_valid,
        f'{prefix}_future_probability': future_probability,
        f'{prefix}_future_margin': (future[2] - future[3] if future_valid else 0.0),
        f'{prefix}_future_winner': future[4] if future_valid else 0.0,
        f'{prefix}_future_distance': future[5] if future_valid else 0.0,
        f'{prefix}_prev_current_residual': (
            float(np.linalg.norm(current - previous_delta)) if previous_valid else 0.0
        ),
        f'{prefix}_current_future_residual': (
            float(np.linalg.norm(future_delta - current)) if future_valid else 0.0
        ),
        f'{prefix}_prev_currentcosine': (
            cosine(previous_delta, current) if previous_valid else 0.0
        ),
        f'{prefix}_current_futurecosine': (cosine(current, future_delta) if future_valid else 0.0),
        f'{prefix}_prev_currentspeed_ratio': (
            speed_ratio(previous_delta, current) if previous_valid else 0.0
        ),
        f'{prefix}_current_futurespeed_ratio': (
            speed_ratio(current, future_delta) if future_valid else 0.0
        ),
        f'{prefix}_path_min': (
            min(previous_probability, current_probability, future_probability)
            if previous_valid and future_valid
            else 0.0
        ),
        f'{prefix}_path_mean': (
            (previous_probability + current_probability + future_probability) / 3.0
            if previous_valid and future_valid
            else 0.0
        ),
        f'{prefix}_path_geomean': (
            float(
                np.cbrt(max(previous_probability * current_probability * future_probability, 0.0))
            )
            if previous_valid and future_valid
            else 0.0
        ),
        f'{prefix}_context_both': previous_valid * future_valid,
    }
    return values, (
        int(previous[0]) if previous_valid else -1,
        int(future[1]) if future_valid else -1,
    )
