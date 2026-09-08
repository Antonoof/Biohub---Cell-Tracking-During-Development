from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

from biohub.features.position import extract_pos_features
from biohub.modules.detect.config import PredictConfig
from biohub.modules.detect.peaks import detect_cells_pooled, index_zyx

_ACTIVE_DETECT_CFG = PredictConfig()


def biohub_savez(file, **arrays):
    cfg = _ACTIVE_DETECT_CFG
    if cfg.uncompressed_evidence:
        total = sum(int(getattr(value, 'nbytes', 0)) for value in arrays.values())
        cap = int(cfg.uncompressed_evidence_max_bytes or 0)
        if cap <= 0 or total <= cap:
            return np.savez(file, **arrays)
    return np.savez_compressed(file, **arrays)


def new_native_evidence_state():
    return {
        'coords_by_t': {},
        'offsets': {},
        'node_count': 0,
        'seen_frames': set(),
        'seen_pairs': set(),
        'evidence': {
            key: []
            for key in (
                'source_id',
                'target_id',
                'probability',
                'alternative_parent_probability',
                'is_target_winner',
                'distance_um',
                'source_target_rank',
            )
        },
    }


def detect_native_evidence_frames(
    state,
    detection,
    frame_indices,
    threshold,
    pool_kernel,
    downsample,
):
    scale = np.asarray(downsample, np.float32)
    for local_frame, absolute_frame in enumerate(frame_indices):
        if absolute_frame in state['seen_frames']:
            continue
        coord = detect_cells_pooled(
            detection[local_frame][0],
            absolute_frame,
            threshold,
            pool_kernel,
            refine=_ACTIVE_DETECT_CFG.subvoxel_refinement,
        ).astype(np.float32)
        coord[:, 1:] *= scale
        start = int(state['node_count'])
        state['coords_by_t'][absolute_frame] = coord
        state['offsets'][absolute_frame] = (start, start + len(coord))
        state['node_count'] = start + len(coord)
        state['seen_frames'].add(absolute_frame)


def append_native_evidence_pair(
    state,
    model,
    feature_maps,
    f_idx,
    t_src,
    t_tgt,
    downsample,
    spacing,
    window_shape,
    device,
    radius_um,
    topk_per_source,
    min_probability,
):
    if (t_src, t_tgt) in state['seen_pairs']:
        return
    state['seen_pairs'].add((t_src, t_tgt))
    source_native = state['coords_by_t'].get(t_src)
    target_native = state['coords_by_t'].get(t_tgt)
    if source_native is None or target_native is None:
        return
    if not len(source_native) or not len(target_native):
        return
    downsample_array = np.asarray(downsample, np.float32)
    source_raw = source_native[:, 1:].astype(np.float32)
    target_raw = target_native[:, 1:].astype(np.float32)
    source_downsampled = index_zyx(model, source_raw / downsample_array)
    target_downsampled = index_zyx(model, target_raw / downsample_array)
    n_source, n_target = len(source_downsampled), len(target_downsampled)
    source_tensor = torch.from_numpy(source_downsampled).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target_downsampled).unsqueeze(0).to(device)
    source_relative = np.column_stack(
        [
            np.full(n_source, f_idx, np.float32),
            source_downsampled,
        ]
    )
    target_relative = np.column_stack(
        [
            np.full(n_target, f_idx + 1, np.float32),
            target_downsampled,
        ]
    )
    source_position = (
        torch.from_numpy(extract_pos_features(source_relative, window_shape).astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )
    target_position = (
        torch.from_numpy(extract_pos_features(target_relative, window_shape).astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )
    source_mask = torch.ones(1, n_source, dtype=torch.bool, device=device)
    target_mask = torch.ones(1, n_target, dtype=torch.bool, device=device)
    source_features = model.index_features(
        feature_maps[:, f_idx],
        source_tensor,
        source_mask,
    )
    target_features = model.index_features(
        feature_maps[:, f_idx + 1],
        target_tensor,
        target_mask,
    )
    downsample_tensor = torch.as_tensor(
        downsample_array,
        dtype=torch.float32,
        device=device,
    )
    logits = model.predict_edges(
        source_features,
        target_features,
        source_tensor * downsample_tensor,
        target_tensor * downsample_tensor,
        source_position,
        target_position,
        source_mask,
        target_mask,
    )[0]
    probability = torch.softmax(logits, dim=0).float()
    spacing_array = np.asarray(spacing, np.float32)
    source_um = torch.as_tensor(
        source_raw * spacing_array,
        dtype=torch.float32,
        device=device,
    )
    target_um = torch.as_tensor(
        target_raw * spacing_array,
        dtype=torch.float32,
        device=device,
    )
    distance = torch.cdist(source_um, target_um)
    eligible = distance <= radius_um
    masked = probability.masked_fill(~eligible, -1.0)
    k = min(topk_per_source, n_target)
    values, targets = torch.topk(masked, k, dim=1, sorted=True)
    keep = values >= min_probability
    if n_source == 1:
        best_probability = probability[0]
        best_parent = torch.zeros_like(best_probability, dtype=torch.long)
        second_probability = torch.zeros_like(best_probability)
    else:
        top2 = torch.topk(probability, 2, dim=0, sorted=True)
        best_probability = top2.values[0]
        best_parent = top2.indices[0]
        second_probability = top2.values[1]
    source_rows = (
        torch.arange(
            n_source,
            device=device,
        )
        .unsqueeze(1)
        .expand_as(targets)
    )
    picked_source = source_rows[keep]
    picked_target = targets[keep]
    target_best_parent = best_parent[picked_target]
    alternative = torch.where(
        target_best_parent == picked_source,
        second_probability[picked_target],
        best_probability[picked_target],
    )
    source_start = int(state['offsets'][t_src][0])
    target_start = int(state['offsets'][t_tgt][0])
    evidence = state['evidence']
    evidence['source_id'].append((picked_source + source_start).cpu().numpy().astype(np.int64))
    evidence['target_id'].append((picked_target + target_start).cpu().numpy().astype(np.int64))
    evidence['probability'].append(values[keep].cpu().numpy().astype(np.float32))
    evidence['alternative_parent_probability'].append(alternative.cpu().numpy().astype(np.float32))
    evidence['is_target_winner'].append(
        (target_best_parent == picked_source).cpu().numpy().astype(np.uint8)
    )
    evidence['distance_um'].append(
        distance[picked_source, picked_target].cpu().numpy().astype(np.float32)
    )
    rank_grid = (
        torch.arange(
            k,
            device=device,
        )
        .unsqueeze(0)
        .expand_as(targets)
    )
    evidence['source_target_rank'].append(rank_grid[keep].cpu().numpy().astype(np.int16))


def _edgegraft_greedy_native_map(
    native_coords,
    native_offsets,
    fused_coords,
    fused_offsets,
    spacing,
    max_um=6.0,
):
    mapped = np.full(len(native_coords), -1, np.int64)
    distance = np.full(len(native_coords), np.inf, np.float32)
    frame_count = min(len(native_offsets), len(fused_offsets)) - 1
    for frame in range(frame_count):
        n0, n1 = int(native_offsets[frame]), int(native_offsets[frame + 1])
        f0, f1 = int(fused_offsets[frame]), int(fused_offsets[frame + 1])
        if n1 <= n0 or f1 <= f0:
            continue
        native_um = native_coords[n0:n1, 1:].astype(np.float32) * spacing
        fused_um = fused_coords[f0:f1, 1:].astype(np.float32) * spacing
        k = min(4, len(fused_um))
        query_distance, query_index = cKDTree(fused_um).query(native_um, k=k)
        if k == 1:
            query_distance = query_distance[:, None]
            query_index = query_index[:, None]
        options = []
        for native_row in range(len(native_um)):
            for rank in range(k):
                value = float(query_distance[native_row, rank])
                if value <= max_um:
                    options.append((value, native_row, int(query_index[native_row, rank])))
        used_native = set()
        used_fused = set()
        for value, native_row, fused_row in sorted(options):
            if native_row in used_native or fused_row in used_fused:
                continue
            used_native.add(native_row)
            used_fused.add(fused_row)
            mapped[n0 + native_row] = f0 + fused_row
            distance[n0 + native_row] = value
    return mapped, distance


def edgegraft_append_graph_ids(path, coords, graph):
    if path is None or not Path(path).is_file():
        return
    by_identity = {}
    for row in graph.node_attrs().iter_rows(named=True):
        key = (
            int(row['t']),
            round(float(row['z']), 5),
            round(float(row['y']), 5),
            round(float(row['x']), 5),
        )
        if key in by_identity:
            raise RuntimeError(f'Duplicate fused graph identity: {key}')
        by_identity[key] = int(row['node_id'])
    graph_ids = np.asarray(
        [
            by_identity[
                (
                    int(round(float(coord[0]))),
                    round(float(coord[1]), 5),
                    round(float(coord[2]), 5),
                    round(float(coord[3]), 5),
                )
            ]
            for coord in coords
        ],
        np.int64,
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays['fused_graph_node_id'] = graph_ids
    temporary = Path(path).with_suffix('.edgegraft.tmp.npz')
    biohub_savez(temporary, **arrays)
    temporary.replace(path)


def save_native_evidence(state, output_path, frames, spacing, fused_detector_coords):
    if output_path is None:
        return
    empty_coords = np.empty((0, 4), np.float32)
    chunks = [state['coords_by_t'].get(frame, empty_coords) for frame in range(frames)]
    counts = np.asarray([len(chunk) for chunk in chunks], np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    native_coords = (
        np.concatenate(chunks).astype(np.float32, copy=False)
        if chunks
        else np.empty((0, 4), np.float32)
    )
    fused_coords = np.asarray(fused_detector_coords, np.float32)
    fused_frames = (
        np.rint(fused_coords[:, 0]).astype(np.int64, copy=False)
        if len(fused_coords)
        else np.empty(0, np.int64)
    )
    if len(fused_frames) and (
        np.any(fused_frames < 0)
        or np.any(fused_frames >= frames)
        or np.any(fused_frames[1:] < fused_frames[:-1])
    ):
        raise RuntimeError('Fused detector coordinates are not frame ordered')
    fused_counts = np.bincount(fused_frames, minlength=frames)[:frames]
    fused_offsets = np.concatenate([[0], np.cumsum(fused_counts, dtype=np.int64)])
    mapped_fused, map_distance = _edgegraft_greedy_native_map(
        native_coords,
        offsets,
        fused_coords,
        fused_offsets,
        np.asarray(spacing, np.float32),
        max_um=6.0,
    )
    dtypes = {
        'source_id': np.int64,
        'target_id': np.int64,
        'probability': np.float32,
        'alternative_parent_probability': np.float32,
        'is_target_winner': np.uint8,
        'distance_um': np.float32,
        'source_target_rank': np.int16,
    }
    arrays = {}
    for key, evidence_chunks in state['evidence'].items():
        arrays[key] = (
            np.concatenate(evidence_chunks).astype(dtypes[key], copy=False)
            if evidence_chunks
            else np.empty(0, dtypes[key])
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        biohub_savez(
            handle,
            **arrays,
            native_node_coords=native_coords,
            native_frame_offsets=offsets,
            mapped_ab_node=mapped_fused,
            map_distance_um=map_distance,
            ab_node_coords=fused_coords,
            ab_frame_offsets=fused_offsets,
            spacing_um=np.asarray(spacing, np.float32),
        )
    temporary.replace(output_path)
