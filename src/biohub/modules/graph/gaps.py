import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from scipy.spatial.distance import cdist

from biohub.modules.graph.geometry import (
    coords_um,
    edge_distance_um,
    ids_by_frame,
    next_node_id,
    node_point,
    point_distance_um,
    read_test_frame,
)


def _dc_pool_frame_xy(volume, factor):
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume, cfg):
    volume = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(volume, float(getattr(cfg, 'norm_lo_pct', 50.0))))
    hi = float(np.percentile(volume, float(getattr(cfg, 'norm_hi_pct', 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    ratio = (volume - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, 'norm_clip_lo', -0.5)),
        float(getattr(cfg, 'norm_clip_hi', 6.0)),
    ).astype(np.float32)


def _deepcenter_heatmap(upgrade, dataset, t, frame_cache, heatmap_cache):
    key = (dataset, int(t))
    if key in heatmap_cache:
        return heatmap_cache[key]
    bundle = upgrade.load_deepcenter()
    if bundle is None:
        return None
    cfg = bundle['cfg']
    pool_factor = int(getattr(cfg, 'pool_factor', 4))
    volume = read_test_frame(upgrade, dataset, int(t), frame_cache)
    image = _dc_normalize_dynamic_range(_dc_pool_frame_xy(volume, pool_factor), cfg)
    with torch.no_grad():
        tensor = torch.from_numpy(image[None, None]).to(
            device=bundle['device'], dtype=torch.float32
        )
        heatmap = torch.sigmoid(bundle['model'](tensor))[0, 0].cpu().numpy()
    heatmap = heatmap.astype(np.float32, copy=False)
    heatmap_cache[key] = heatmap
    while len(heatmap_cache) > max(1, upgrade.deepcenter_score_cache_max_frames):
        heatmap_cache.pop(next(iter(heatmap_cache)))
    return heatmap


def deepcenter_accept_gap(upgrade, dataset, t, point, frame_cache, heatmap_cache, stats):
    stats['deepcenter_gap_checked'] += 1
    heatmap = _deepcenter_heatmap(upgrade, dataset, int(t), frame_cache, heatmap_cache)
    if heatmap is None or not heatmap.size:
        stats['deepcenter_gap_missing'] += 1
        return True
    cfg = upgrade.deepcenter_bundle['cfg']
    factor = max(1, int(getattr(cfg, 'pool_factor', 4)))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / factor))
    x = int(round(float(point[2]) / factor))
    z0, z1 = (
        max(0, z - upgrade.deepcenter_score_win_z),
        min(heatmap.shape[0], z + upgrade.deepcenter_score_win_z + 1),
    )
    y0, y1 = (
        max(0, y - upgrade.deepcenter_score_win_yx),
        min(heatmap.shape[1], y + upgrade.deepcenter_score_win_yx + 1),
    )
    x0, x1 = (
        max(0, x - upgrade.deepcenter_score_win_yx),
        min(heatmap.shape[2], x + upgrade.deepcenter_score_win_yx + 1),
    )
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if not patch.size:
        stats['deepcenter_gap_missing'] += 1
        return True
    score = float(np.max(patch))
    if not np.isfinite(score):
        stats['deepcenter_gap_missing'] += 1
        return True
    if score < upgrade.deepcenter_gap_threshold:
        stats['deepcenter_gap_rejected'] += 1
        return False
    stats['deepcenter_gap_accepted'] += 1
    return True


def deepcenter_accept_safe_division(
    upgrade,
    dataset,
    t,
    point,
    frame_cache,
    heatmap_cache,
    stats,
):
    if not upgrade.deepcenter_safe_div_veto:
        return True
    stats['deepcenter_safe_div_checked'] += 1
    heatmap = _deepcenter_heatmap(upgrade, dataset, int(t), frame_cache, heatmap_cache)
    if heatmap is None or not heatmap.size:
        stats['deepcenter_safe_div_missing'] += 1
        return True
    cfg = upgrade.deepcenter_bundle['cfg']
    factor = max(1, int(getattr(cfg, 'pool_factor', 4)))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / factor))
    x = int(round(float(point[2]) / factor))
    z0, z1 = (
        max(0, z - upgrade.deepcenter_score_win_z),
        min(heatmap.shape[0], z + upgrade.deepcenter_score_win_z + 1),
    )
    y0, y1 = (
        max(0, y - upgrade.deepcenter_score_win_yx),
        min(heatmap.shape[1], y + upgrade.deepcenter_score_win_yx + 1),
    )
    x0, x1 = (
        max(0, x - upgrade.deepcenter_score_win_yx),
        min(heatmap.shape[2], x + upgrade.deepcenter_score_win_yx + 1),
    )
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if not patch.size:
        stats['deepcenter_safe_div_missing'] += 1
        return True
    score = float(np.max(patch))
    if not np.isfinite(score):
        stats['deepcenter_safe_div_missing'] += 1
        return True
    if score < upgrade.deepcenter_safe_div_threshold:
        stats['deepcenter_safe_div_rejected'] += 1
        return False
    stats['deepcenter_safe_div_accepted'] += 1
    return True


def refine_synthetic_midpoint(upgrade, dataset, t, midpoint, frame_cache, stats):
    if not upgrade.gap_refine_synthetic or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(upgrade, dataset, t, frame_cache)
        z, y, x = (int(round(value)) for value in midpoint)
        z0, z1 = (
            max(0, z - upgrade.gap_refine_win_z),
            min(frame.shape[0], z + upgrade.gap_refine_win_z + 1),
        )
        y0, y1 = (
            max(0, y - upgrade.gap_refine_win_yx),
            min(frame.shape[1], y + upgrade.gap_refine_win_yx + 1),
        )
        x0, x1 = (
            max(0, x - upgrade.gap_refine_win_yx),
            min(frame.shape[2], x + upgrade.gap_refine_win_yx + 1),
        )
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats['gap_refine_failed'] += 1
            return midpoint
        weights = np.maximum(patch - float(np.percentile(patch, 20.0)), 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats['gap_refine_failed'] += 1
            return midpoint
        refined = (
            float((weights.sum(axis=(1, 2)) * np.arange(z0, z1)).sum() / total),
            float((weights.sum(axis=(0, 2)) * np.arange(y0, y1)).sum() / total),
            float((weights.sum(axis=(0, 1)) * np.arange(x0, x1)).sum() / total),
        )
        if point_distance_um(upgrade, refined, midpoint) > upgrade.gap_refine_max_shift_um:
            stats['gap_refine_rejected_shift'] += 1
            return midpoint
        stats['gap_refined_synthetic'] += 1
        return refined
    except Exception:
        stats['gap_refine_failed'] += 1
        return midpoint


def close_single_frame_gaps(
    upgrade,
    nodes_by_id,
    edges,
    stats,
    dataset=None,
    frame_cache=None,
    deepcenter_enabled=False,
    deepcenter_heatmap_cache=None,
):
    if not upgrade.gap_close or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge['source_id']) for edge in edges}
    incoming = {int(edge['target_id']) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t = ids_by_frame(upgrade, nodes_by_id)
    for node_id, node in nodes_by_id.items():
        t = int(node['t'])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)
    for bucket in (ends_by_t, starts_by_t, isolated_by_t):
        for ids in bucket.values():
            ids.sort()

    max_synthetic = min(
        upgrade.gap_max_added_abs,
        max(1, int(round(len(nodes_by_id) * upgrade.gap_max_added_frac)))
        if upgrade.gap_max_added_frac > 0
        else 0,
    )
    next_id = next_node_id(upgrade, nodes_by_id)
    frame_cache = {} if frame_cache is None else frame_cache
    deepcenter_heatmap_cache = {} if deepcenter_heatmap_cache is None else deepcenter_heatmap_cache
    used_starts: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict] = []

    spacing_cache: dict[int, np.ndarray] = {}

    def frame_spacing(t: int) -> np.ndarray:
        cached = spacing_cache.get(t)
        if cached is not None:
            return cached
        ids = all_ids_by_t.get(t, [])
        if len(ids) <= 1:
            result = np.full(len(ids), upgrade.gap_density_reference_um)
        else:
            coords = coords_um(upgrade, nodes_by_id, ids)
            k = min(len(ids), max(2, upgrade.gap_density_neighbors + 1))
            distances = np.atleast_2d(cKDTree(coords).query(coords, k=k)[0])[:, 1:]
            distances = np.where(np.isfinite(distances), distances, np.nan)
            with np.errstate(invalid='ignore'):
                result = np.nanmedian(distances, axis=1)
            result = np.where(np.isfinite(result), result, upgrade.gap_density_reference_um)
        spacing_cache[t] = result
        return result

    isolated_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def isolated_pool(t: int):
        cached = isolated_cache.get(t)
        if cached is None:
            ids = np.asarray(
                [i for i in isolated_by_t.get(t, []) if i not in incoming],
                dtype=np.int64,
            )
            cached = (ids, coords_um(upgrade, nodes_by_id, ids), np.ones(ids.size, dtype=bool))
            isolated_cache[t] = cached
        return cached

    def retire_from_pool(node_id: int, frame: int) -> None:
        cached = isolated_cache.get(frame)
        if cached is None:
            return
        hit = np.flatnonzero(cached[0] == node_id)
        if hit.size:
            cached[2][hit[0]] = False

    threshold_um = upgrade.gap_close_um * 2.0
    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = [
            node_id
            for node_id in starts_by_t.get(t + 2, [])
            if node_id not in used_starts and node_id not in incoming
        ]
        if not end_ids or not start_ids:
            continue

        distance = cdist(
            coords_um(upgrade, nodes_by_id, end_ids), coords_um(upgrade, nodes_by_id, start_ids)
        )
        adaptive = np.full_like(distance, threshold_um)
        if upgrade.gap_density_adaptive:
            end_index = {node_id: i for i, node_id in enumerate(all_ids_by_t[t])}
            start_index = {node_id: i for i, node_id in enumerate(all_ids_by_t[t + 2])}
            source_spacing = frame_spacing(t)[[end_index[i] for i in end_ids]]
            target_spacing = frame_spacing(t + 2)[[start_index[j] for j in start_ids]]
            local = 0.5 * (source_spacing[:, None] + target_spacing[None, :])
            step = np.clip(
                upgrade.gap_density_gain * (local - upgrade.gap_density_reference_um),
                -upgrade.gap_density_max_step_delta_um,
                upgrade.gap_density_max_step_delta_um,
            )
            adaptive = threshold_um + 2.0 * step

        base_allowed = distance <= threshold_um
        allowed = distance <= adaptive
        stats['gap_density_candidates_expanded'] += int((allowed & ~base_allowed).sum())
        stats['gap_density_candidates_restricted'] += int((base_allowed & ~allowed).sum())
        stats['gap_candidates'] += int(allowed.sum())
        if not allowed.any():
            continue

        big = float(np.max(adaptive)) * 1000.0 + 1.0
        row_index, col_index = linear_sum_assignment(np.where(allowed, distance, big))

        for r, c in zip(row_index, col_index):
            if not allowed[r, c]:
                continue
            source_id = end_ids[int(r)]
            target_id = start_ids[int(c)]
            if source_id in outgoing or target_id in used_starts:
                continue
            if not base_allowed[r, c]:
                stats['gap_density_selected_outside_base'] += 1

            source = nodes_by_id[source_id]
            target = nodes_by_id[target_id]
            mid_t = int(source['t']) + 1
            mid_point = (
                (float(source['z']) + float(target['z'])) / 2.0,
                (float(source['y']) + float(target['y'])) / 2.0,
                (float(source['x']) + float(target['x'])) / 2.0,
            )

            middle_id = None
            if upgrade.gap_reuse_existing:
                pool_ids, pool_coords, alive = isolated_pool(mid_t)
                if alive.any():
                    reuse = np.where(
                        alive,
                        np.linalg.norm(pool_coords - np.asarray(mid_point) * upgrade.scale, axis=1),
                        np.inf,
                    )
                    best = int(np.argmin(reuse))
                    if reuse[best] <= upgrade.gap_reuse_um:
                        middle_id = int(pool_ids[best])
                        alive[best] = False
                        stats['gap_reused_existing'] += 1

            if middle_id is None:
                if synthetic_added >= max_synthetic:
                    stats['gap_skipped_node_cap'] += 1
                    continue
                middle_id = next_id
                next_id += 1
                refined = refine_synthetic_midpoint(
                    upgrade, dataset, mid_t, mid_point, frame_cache, stats
                )
                nodes_by_id[middle_id] = {
                    'node_id': middle_id,
                    't': mid_t,
                    'z': refined[0],
                    'y': refined[1],
                    'x': refined[2],
                    'gap_synthetic': 1,
                }
                synthetic_added += 1
                stats['gap_inserted_synthetic'] += 1

            middle = nodes_by_id[middle_id]
            gap_span_um = float(distance[r, c])
            synthetic_middle = int(middle.get('gap_synthetic', 0)) == 1
            if deepcenter_enabled and not synthetic_middle:
                stats['deepcenter_gap_bypassed_observed_node'] += 1
            elif (
                deepcenter_enabled
                and synthetic_middle
                and gap_span_um < upgrade.deepcenter_gap_confirm_min_span_um
            ):
                stats['deepcenter_gap_bypassed_short_span'] += 1
            elif (
                deepcenter_enabled
                and synthetic_middle
                and not deepcenter_accept_gap(
                    upgrade,
                    dataset,
                    mid_t,
                    node_point(upgrade, middle),
                    frame_cache,
                    deepcenter_heatmap_cache,
                    stats,
                )
            ):
                nodes_by_id.pop(middle_id, None)
                synthetic_added = max(0, synthetic_added - 1)
                stats['gap_inserted_synthetic'] = max(0, stats['gap_inserted_synthetic'] - 1)
                continue

            new_edges.append(
                {
                    'source_id': source_id,
                    'target_id': middle_id,
                    'edge_prob': None,
                    'distance_um': edge_distance_um(upgrade, source, middle),
                    'gap_closed': 1,
                }
            )
            new_edges.append(
                {
                    'source_id': middle_id,
                    'target_id': target_id,
                    'edge_prob': None,
                    'distance_um': edge_distance_um(upgrade, middle, target),
                    'gap_closed': 1,
                }
            )
            outgoing.update((source_id, middle_id))
            incoming.update((middle_id, target_id))
            used_starts.add(target_id)
            retire_from_pool(target_id, int(target['t']))
            stats['gap_pairs_selected'] += 1
            stats['gap_added_edges'] += 2

    stats['gap_added_nodes'] = stats['gap_inserted_synthetic']
    return nodes_by_id, ([*edges, *new_edges] if new_edges else edges)
