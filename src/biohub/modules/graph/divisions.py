import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

from biohub.modules.graph.gaps import deepcenter_accept_safe_division
from biohub.modules.graph.geometry import (
    coords_um,
    edge_distance_um,
    ids_by_frame,
    next_node_id,
    node_point,
)


def evidence_path(upgrade, root: Path, dataset: str) -> Path:
    path = root / f'{dataset}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing native evidence: {path}')
    return path


def _load_native_endpoint_bundle(path: Path):
    if path is None or not Path(path).is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data['native_node_coords'], np.float64)
        offsets = np.asarray(data['native_frame_offsets'], np.int64)
        source_id = (
            np.asarray(data['source_id'], np.int64)
            if 'source_id' in data.files
            else np.empty(0, np.int64)
        )
        target_id = (
            np.asarray(data['target_id'], np.int64)
            if 'target_id' in data.files
            else np.empty(0, np.int64)
        )
        probability = (
            np.asarray(data['probability'], np.float64)
            if 'probability' in data.files
            else np.empty(0, np.float64)
        )
    association: dict[tuple[int, int], float] = {}
    for src, tgt, prob in zip(source_id, target_id, probability):
        key = (int(src), int(tgt))
        value = float(prob)
        if value > association.get(key, -1.0):
            association[key] = value
    return {'coords': coords, 'offsets': offsets, 'association': association}


def _native_frame_points(upgrade, bundle, frame: int):
    if bundle is None:
        return np.empty(0, np.int64), np.empty((0, 3), np.float64)
    offsets = bundle['offsets']
    if frame < 0 or frame + 1 >= len(offsets):
        return np.empty(0, np.int64), np.empty((0, 3), np.float64)
    start, stop = int(offsets[frame]), int(offsets[frame + 1])
    if stop <= start:
        return np.empty(0, np.int64), np.empty((0, 3), np.float64)
    ids = np.arange(start, stop, dtype=np.int64)
    zyx = bundle['coords'][start:stop, 1:].astype(np.float64, copy=False)
    return ids, zyx * upgrade.scale


def _nearest_native(ids, points_um, query_um, max_um: float) -> int:
    if ids.size == 0:
        return -1
    distance, index = cKDTree(points_um).query(query_um, k=1)
    if not np.isfinite(distance) or float(distance) > max_um:
        return -1
    return int(ids[int(index)])


def restore_ownership_endpoints(upgrade, dataset, nodes_by_id, edges, stats):
    injected: set[int] = set()
    if not upgrade.ownership_endpoint_graft or dataset is None or not nodes_by_id:
        return nodes_by_id, edges, injected

    p1_path = upgrade.p1_evidence_dir / f'{dataset}.npz'
    p2_path = upgrade.p2_evidence_dir / f'{dataset}.npz'
    p1 = _load_native_endpoint_bundle(p1_path)
    p2 = _load_native_endpoint_bundle(p2_path)
    if p1 is None or p2 is None:
        stats['ownership_endpoint_missing_evidence'] = 1
        return nodes_by_id, edges, injected

    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge['source_id'])].append(int(edge['target_id']))
    ids_by_t = ids_by_frame(upgrade, nodes_by_id)
    sources_by_t: dict[int, list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
    for source_id, children in outgoing.items():
        if len(children) != 1 or source_id not in nodes_by_id:
            continue
        child_id = int(children[0])
        child = nodes_by_id.get(child_id)
        source = nodes_by_id.get(source_id)
        if source is None or child is None:
            continue
        if int(child['t']) != int(source['t']) + 1:
            continue
        source_um = np.asarray(node_point(upgrade, source), np.float64) * upgrade.scale
        child_um = np.asarray(node_point(upgrade, child), np.float64) * upgrade.scale
        sources_by_t.setdefault(int(source['t']), []).append(
            (int(source_id), child_id, source_um, child_um)
        )
    if not sources_by_t:
        return nodes_by_id, edges, injected

    used_sources: set[int] = set()
    next_id = next_node_id(upgrade, nodes_by_id)
    for source_t, sources in sources_by_t.items():
        target_t = source_t + 1
        fused_ids = ids_by_t.get(target_t, [])
        fused_um = (
            coords_um(upgrade, nodes_by_id, fused_ids)
            if fused_ids
            else np.empty((0, 3), np.float64)
        )
        fused_tree = cKDTree(fused_um) if len(fused_um) else None
        p1_ids, p1_um = _native_frame_points(upgrade, p1, target_t)
        p2_ids, p2_um = _native_frame_points(upgrade, p2, target_t)
        src_p1_ids, src_p1_um = _native_frame_points(upgrade, p1, source_t)
        src_p2_ids, src_p2_um = _native_frame_points(upgrade, p2, source_t)
        if p1_ids.size == 0 or p2_ids.size == 0:
            continue
        p2_tree = cKDTree(p2_um)
        p2_dist, p2_index = p2_tree.query(p1_um, k=1)
        order = np.argsort(p2_dist)
        used_p2: set[int] = set()
        for row in order:
            agree = float(p2_dist[row])
            p2_row = int(p2_index[row])
            if agree > upgrade.ownership_endpoint_agree_um or p2_row in used_p2:
                continue
            used_p2.add(p2_row)
            point = 0.5 * (p1_um[row] + p2_um[p2_row])
            stats['ownership_endpoint_consensus'] += 1
            if fused_tree is not None:
                existing = float(fused_tree.query(point, k=1)[0])
                if existing <= upgrade.ownership_endpoint_existing_um:
                    stats['ownership_endpoint_already_present'] += 1
                    continue
            best = None
            for source_id, child_id, source_um, child_um in sources:
                if source_id in used_sources:
                    continue
                parent = float(np.linalg.norm(point - source_um))
                sister = float(np.linalg.norm(point - child_um))
                if (
                    parent > upgrade.ownership_endpoint_parent_max_um
                    or sister > upgrade.ownership_endpoint_sister_max_um
                    or sister < upgrade.ownership_endpoint_existing_um
                ):
                    continue
                p1_src = _nearest_native(
                    src_p1_ids, src_p1_um, source_um, upgrade.ownership_endpoint_agree_um
                )
                p2_src = _nearest_native(
                    src_p2_ids, src_p2_um, source_um, upgrade.ownership_endpoint_agree_um
                )
                native_prob = max(
                    p1['association'].get((p1_src, int(p1_ids[row])), -1.0),
                    p2['association'].get((p2_src, int(p2_ids[p2_row])), -1.0),
                )
                if native_prob < upgrade.ownership_endpoint_min_prob and agree > 2.0:
                    continue
                rank = (parent, -float(native_prob), source_id)
                if best is None or rank < best[0]:
                    best = (rank, source_id, parent, sister, native_prob)
            if best is None:
                stats['ownership_endpoint_no_source'] += 1
                continue
            _, source_id, parent, sister, native_prob = best
            voxel = point / upgrade.scale
            node_id = next_id
            next_id += 1
            nodes_by_id[node_id] = {
                'node_id': node_id,
                't': int(target_t),
                'z': float(voxel[0]),
                'y': float(voxel[1]),
                'x': float(voxel[2]),
                'ownership_endpoint_restored': 1,
                'ownership_endpoint_source': int(source_id),
                'ownership_endpoint_agree_um': float(agree),
                'ownership_endpoint_parent_um': float(parent),
                'ownership_endpoint_sister_um': float(sister),
                'ownership_endpoint_native_prob': float(native_prob),
            }
            injected.add(node_id)
            used_sources.add(source_id)
            stats['ownership_endpoint_inserted'] += 1
            ids_by_t.setdefault(int(target_t), []).append(node_id)
            if fused_tree is not None or fused_ids:
                fused_ids = ids_by_t[int(target_t)]
                fused_um = coords_um(upgrade, nodes_by_id, fused_ids)
                fused_tree = cKDTree(fused_um)

    stats['ownership_endpoint_sources_considered'] = sum(
        len(group) for group in sources_by_t.values()
    )
    return nodes_by_id, edges, injected


def retract_unused_ownership_endpoints(upgrade, nodes_by_id, edges, injected, stats):
    if not injected:
        stats['ownership_endpoint_kept'] = 0
        stats['ownership_endpoint_retracted'] = 0
        return nodes_by_id, edges
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge['source_id'])].append(int(edge['target_id']))
        incoming[int(edge['target_id'])].append(int(edge['source_id']))
    fork_daughters = {
        int(child)
        for source, children in outgoing.items()
        if len(children) >= 2
        for child in children
    }
    keep = {node_id for node_id in injected if node_id in fork_daughters}
    drop = set(injected) - keep
    if drop:
        nodes_by_id = {
            node_id: node for node_id, node in nodes_by_id.items() if node_id not in drop
        }
        edges = [
            edge
            for edge in edges
            if int(edge['source_id']) not in drop and int(edge['target_id']) not in drop
        ]
    stats['ownership_endpoint_kept'] = len(keep)
    stats['ownership_endpoint_retracted'] = len(drop)
    return nodes_by_id, edges


def apply_division_decoder(upgrade, dataset, nodes_by_id, edges, stats):
    if upgrade.division_runtime is None:
        upgrade.init_division_runtime()
    return upgrade.division_runtime.apply(
        upgrade.test_dir / f'{dataset}.zarr',
        evidence_path(upgrade, upgrade.model_c_evidence_dir, dataset),
        evidence_path(upgrade, upgrade.p1_evidence_dir, dataset),
        evidence_path(upgrade, upgrade.p2_evidence_dir, dataset),
        nodes_by_id,
        edges,
        stats,
        spacing=upgrade.voxel_scale_um,
        v2_threshold=upgrade.gbm_threshold,
        v2_rescue_delta=upgrade.gbm_rescue_delta,
        v2_steal_delta=upgrade.gbm_steal_delta,
    )


def add_safe_divisions_postlink(
    upgrade,
    nodes_by_id,
    edges,
    stats,
    dataset=None,
    frame_cache=None,
    deepcenter_heatmap_cache=None,
):
    if not upgrade.safe_divisions or not edges or not nodes_by_id:
        return edges
    frame_cache = {} if frame_cache is None else frame_cache
    deepcenter_heatmap_cache = {} if deepcenter_heatmap_cache is None else deepcenter_heatmap_cache

    out_by_source: dict[int, list[dict]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge['source_id']), []).append(edge)
        incoming.add(int(edge['target_id']))

    ids_by_t = ids_by_frame(upgrade, nodes_by_id)
    existing_edges = {(int(edge['source_id']), int(edge['target_id'])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * upgrade.safe_div_global_frac_cap)))
    added: list[dict] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [
            node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1
        ]
        candidate_ids = [
            node_id
            for node_id in child_frame_ids
            if node_id not in incoming and node_id not in used_targets
        ]
        if not source_ids or not candidate_ids:
            continue

        candidate_positions = coords_um(upgrade, nodes_by_id, candidate_ids)
        candidate_tree = cKDTree(candidate_positions)
        frame_cap = max(1, int(round(len(source_ids) * upgrade.safe_div_frame_frac_cap)))
        proposals: list[tuple[float, int, int, float, float]] = []

        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge['target_id'])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child['t']) != t + 1:
                continue
            child_distance = edge_distance_um(upgrade, source, existing_child)
            if child_distance > upgrade.safe_div_existing_child_max_um:
                continue

            mutual_nn_id = None
            if upgrade.safe_div_require_mutual_nn:
                _, nn_index = candidate_tree.query(
                    np.asarray(node_point(upgrade, existing_child)) * upgrade.scale
                )
                mutual_nn_id = candidate_ids[int(nn_index)]

            source_point_um = np.asarray(node_point(upgrade, source)) * upgrade.scale
            near = candidate_tree.query_ball_point(source_point_um, upgrade.safe_div_max_um)
            for candidate_index in near:
                candidate_id = candidate_ids[int(candidate_index)]
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_distance = edge_distance_um(upgrade, source, candidate)
                sister_distance = edge_distance_um(upgrade, existing_child, candidate)
                if sister_distance > upgrade.safe_div_sister_max_um:
                    continue

                if upgrade.safe_div_require_mutual_nn and candidate_id != mutual_nn_id:
                    stats['safe_division_mutual_nn_rejected'] += 1
                    continue

                if upgrade.safe_div_require_divergence:
                    child_successors = out_by_source.get(existing_child_id, [])
                    candidate_successors = out_by_source.get(candidate_id, [])
                    if len(child_successors) != 1 or len(candidate_successors) != 1:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    child_grandchild = nodes_by_id.get(int(child_successors[0]['target_id']))
                    candidate_grandchild = nodes_by_id.get(
                        int(candidate_successors[0]['target_id'])
                    )
                    if (
                        child_grandchild is None
                        or candidate_grandchild is None
                        or int(child_grandchild['t']) != t + 2
                        or int(candidate_grandchild['t']) != t + 2
                    ):
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    grandchild_distance = edge_distance_um(
                        upgrade, child_grandchild, candidate_grandchild
                    )
                    if grandchild_distance - sister_distance < upgrade.safe_div_diverge_um:
                        stats['safe_division_divergence_rejected'] += 1
                        continue

                stats['safe_division_geometric_candidates'] += 1
                if not deepcenter_accept_safe_division(
                    upgrade,
                    dataset,
                    int(candidate['t']),
                    node_point(upgrade, candidate),
                    frame_cache,
                    deepcenter_heatmap_cache,
                    stats,
                ):
                    continue
                if upgrade.safe_div_sister_symmetry_tau > 0.0:
                    symmetry_denominator = max((child_distance + parent_distance) / 2.0, 1e-6)
                    if (
                        abs(child_distance - parent_distance) / symmetry_denominator
                        > upgrade.safe_div_sister_symmetry_tau
                    ):
                        stats['safe_division_symmetry_rejected'] += 1
                        continue
                score = parent_distance + upgrade.safe_div_sister_score_weight * sister_distance
                proposals.append((score, source_id, candidate_id, parent_distance, sister_distance))

        stats['safe_division_candidates'] += len(proposals)
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_distance, _ in proposals:
            if len(added) >= global_cap:
                stats['safe_division_skipped_cap'] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            added.append(
                {
                    'source_id': source_id,
                    'target_id': candidate_id,
                    'edge_prob': None,
                    'distance_um': parent_distance,
                    'safe_division': 1,
                    'public_wide_safe_division': 1,
                }
            )
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1

    if not added:
        return edges
    stats['safe_divisions_added'] = len(added)
    stats['public_wide_safe_div_pre_ug_added'] = len(added)
    return [*edges, *added]


def load_retention_guard_frames(upgrade, dataset: str) -> set[int]:
    guarded: set[int] = set()
    for path in Path(upgrade.working_dir).glob('retention_guard_*.jsonl'):
        try:
            with path.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(row.get('dataset')) == str(dataset) and bool(
                        row.get('use_primary', False)
                    ):
                        guarded.add(int(row['frame']))
        except FileNotFoundError:
            continue
    return guarded


def _edge_transition_is_guarded(edge, nodes_by_id, guarded_frames: set[int]) -> bool:
    if not guarded_frames:
        return False
    source = nodes_by_id.get(int(edge['source_id']))
    target = nodes_by_id.get(int(edge['target_id']))
    if source is None or target is None:
        return False
    return int(source['t']) in guarded_frames or int(target['t']) in guarded_frames


def preserve_guarded_transition_ownership(
    upgrade,
    before_edges,
    after_edges,
    nodes_by_id,
    guarded_frames,
    stats,
    prefix,
):
    before = [dict(edge) for edge in before_edges]
    after = [dict(edge) for edge in after_edges]
    if not guarded_frames:
        stats[f'{prefix}_guarded_frames'] = 0
        stats[f'{prefix}_guard_reverted_edges'] = 0
        return after
    before_guarded = {
        (int(edge['source_id']), int(edge['target_id'])): edge
        for edge in before
        if _edge_transition_is_guarded(edge, nodes_by_id, guarded_frames)
    }
    after_guarded = {
        (int(edge['source_id']), int(edge['target_id'])): edge
        for edge in after
        if _edge_transition_is_guarded(edge, nodes_by_id, guarded_frames)
    }
    if before_guarded == after_guarded:
        stats[f'{prefix}_guarded_frames'] = len(guarded_frames)
        stats[f'{prefix}_guard_reverted_edges'] = 0
        return after
    kept = [
        edge for edge in after if not _edge_transition_is_guarded(edge, nodes_by_id, guarded_frames)
    ]
    result = [*kept, *before_guarded.values()]
    stats[f'{prefix}_guarded_frames'] = len(guarded_frames)
    stats[f'{prefix}_guard_reverted_edges'] = len(set(before_guarded) ^ set(after_guarded))
    stats[f'{prefix}_guard_reverted_transactions'] = 1
    return result


def capture_full_ownership_contract(upgrade, edges):
    tagged_sources = {
        int(edge['source_id'])
        for edge in edges
        if int(edge.get('full_population_ownership', 0) or 0) == 1
    }
    outgoing = defaultdict(set)
    for edge in edges:
        outgoing[int(edge['source_id'])].add(int(edge['target_id']))
    return {
        source: frozenset(outgoing.get(source, ()))
        for source in tagged_sources
        if len(outgoing.get(source, ())) == 2
    }


def validate_full_ownership_contract(upgrade, contract, nodes_by_id, edges, stats):
    outgoing = defaultdict(set)
    for edge in edges:
        outgoing[int(edge['source_id'])].add(int(edge['target_id']))
    missing = []
    for source, daughters in contract.items():
        if source not in nodes_by_id or any(child not in nodes_by_id for child in daughters):
            missing.append(source)
            continue
        if outgoing.get(source, set()) != set(daughters):
            missing.append(source)
    stats['ownership_endpoint_contract_sources'] = len(contract)
    stats['ownership_endpoint_contract_failures'] = len(missing)
    if missing:
        raise RuntimeError(
            f'A later stage changed full-population ownership endpoints: {missing[:5]}'
        )
