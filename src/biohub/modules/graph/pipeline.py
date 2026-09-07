import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from biohub.modules.graph.cleanup import (
    cleanup_after_edgegraft,
    enforce_node_budget,
    filter_short_track_components,
    linefit_smooth_output_graph,
    validate_final_graph_contract,
)
from biohub.modules.graph.divisions import (
    add_safe_divisions_postlink,
    apply_division_decoder,
    capture_full_ownership_contract,
    evidence_path,
    load_retention_guard_frames,
    preserve_guarded_transition_ownership,
    restore_ownership_endpoints,
    retract_unused_ownership_endpoints,
    validate_full_ownership_contract,
)
from biohub.modules.graph.export import graph_nodes_edges, write_dataset_shard
from biohub.modules.graph.gaps import close_single_frame_gaps
from biohub.modules.graph.geometry import edge_distance_um, edge_sort_key, sanitized_edge_probs
from biohub.modules.graph.motion import motion_relink_edges


def filter_output_graph(
    upgrade,
    nodes_by_id,
    raw_edges,
    dataset=None,
    division_mode='combined',
    raw_node_attrs=None,
    raw_edge_attrs=None,
):
    stats: dict[str, Any] = defaultdict(int, upgrade.base_stats)
    stats['raw_edges'] = len(raw_edges)
    _stage_t = time.monotonic()
    detected_nodes = len(nodes_by_id)
    if raw_node_attrs is None:
        raw_node_attrs = {node_id: dict(node) for node_id, node in nodes_by_id.items()}
    if raw_edge_attrs is None:
        raw_edge_attrs = [dict(edge) for edge in raw_edges]

    edges: list[dict] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge['source_id']))
        target = nodes_by_id.get(int(edge['target_id']))
        if source is None or target is None:
            continue
        if upgrade.enforce_next_frame and int(target['t']) != int(source['t']) + 1:
            stats['dropped_nonconsecutive_edges'] += 1
            continue
        distance = edge_distance_um(upgrade, source, target)
        edge['distance_um'] = distance
        if upgrade.edge_max_um > 0 and distance > upgrade.edge_max_um:
            stats['dropped_long_edges'] += 1
            continue
        edges.append(edge)

    _stage_t = upgrade.mark_stage(stats, 'edge_prefilter', _stage_t)
    if upgrade.motion_relink:
        motion_edges = motion_relink_edges(
            upgrade, nodes_by_id, stats, sanitized_edge_probs(upgrade, edges)
        )
        if motion_edges:
            stats['motion_relink_replaced_raw_edges'] = len(edges)
            edges = motion_edges
        else:
            stats['motion_relink_fallback_raw'] = 1

    _stage_t = upgrade.mark_stage(stats, 'motion_relink', _stage_t)
    if upgrade.single_parent_repair and edges:
        best_by_target: dict[int, dict] = {}
        for edge in edges:
            target_id = int(edge['target_id'])
            previous = best_by_target.get(target_id)
            if previous is None or edge_sort_key(upgrade, edge) > edge_sort_key(upgrade, previous):
                best_by_target[target_id] = edge
        kept = {id(edge) for edge in best_by_target.values()}
        stats['dropped_multi_parent_edges'] = len(edges) - len(kept)
        edges = [edge for edge in edges if id(edge) in kept]

    _stage_t = upgrade.mark_stage(stats, 'single_parent', _stage_t)
    repair_frame_cache = {}
    deepcenter_heatmap_cache = {}
    nodes_by_id, edges = close_single_frame_gaps(
        upgrade,
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        frame_cache=repair_frame_cache,
        deepcenter_enabled=(division_mode == 'combined' and upgrade.deepcenter_gap_veto),
        deepcenter_heatmap_cache=deepcenter_heatmap_cache,
    )

    _stage_t = upgrade.mark_stage(stats, 'gap_close', _stage_t)
    edges = add_safe_divisions_postlink(
        upgrade,
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        frame_cache=repair_frame_cache,
        deepcenter_heatmap_cache=deepcenter_heatmap_cache,
    )
    public_safe_pairs = {
        (int(edge['source_id']), int(edge['target_id']))
        for edge in edges
        if int(edge.get('public_wide_safe_division', 0)) == 1
    }

    injected_endpoints: set[int] = set()
    if division_mode == 'combined' and upgrade.combined_decoder:
        if dataset is None:
            raise RuntimeError('The combined decoder needs a dataset id')
        nodes_by_id, edges, injected_endpoints = restore_ownership_endpoints(
            upgrade,
            dataset,
            nodes_by_id,
            edges,
            stats,
        )
        edges = apply_division_decoder(upgrade, dataset, nodes_by_id, edges, stats)
        nodes_by_id, edges = retract_unused_ownership_endpoints(
            upgrade,
            nodes_by_id,
            edges,
            injected_endpoints,
            stats,
        )
    post_ug_pairs = {(int(edge['source_id']), int(edge['target_id'])) for edge in edges}
    stats['public_wide_safe_div_post_ug_retained'] = sum(
        pair in post_ug_pairs for pair in public_safe_pairs
    )
    stats['public_wide_safe_div_post_ug_removed'] = (
        len(public_safe_pairs) - stats['public_wide_safe_div_post_ug_retained']
    )

    _stage_t = upgrade.mark_stage(stats, 'division', _stage_t)
    guarded_frames = (
        load_retention_guard_frames(upgrade, str(dataset))
        if division_mode == 'combined' and dataset is not None
        else set()
    )
    stats['retention_guard_frames_loaded'] = len(guarded_frames)
    ownership_endpoint_contract = capture_full_ownership_contract(upgrade, edges)

    if division_mode == 'combined' and upgrade.edgegraft_enabled:
        if dataset is None:
            raise RuntimeError('EdgeGRAFT needs a dataset id')
        if upgrade.edgegraft_runtime is None:
            upgrade.init_edgegraft_runtime()
        edgegraft_started = time.monotonic()
        edgegraft_before = [dict(edge) for edge in edges]
        edges, edgegraft_stats = upgrade.edgegraft_runtime.apply(
            dataset,
            raw_node_attrs,
            raw_edge_attrs,
            nodes_by_id,
            edges,
            evidence_path(upgrade, upgrade.p1_evidence_dir, dataset),
            evidence_path(upgrade, upgrade.p2_evidence_dir, dataset),
        )
        edges = preserve_guarded_transition_ownership(
            upgrade,
            edgegraft_before,
            edges,
            nodes_by_id,
            guarded_frames,
            stats,
            'edgegraft',
        )
        stats.update({f'edgegraft_{key}': value for key, value in edgegraft_stats.items()})
        stats['edgegraft_seconds'] = time.monotonic() - edgegraft_started
        stats['edgegraft_v3_full_population_atomic_upgrade'] = 1
        edges = cleanup_after_edgegraft(upgrade, nodes_by_id, edges, stats)
    else:
        stats['edgegraft_skipped_for_fallback'] = int(upgrade.edgegraft_enabled)

    _stage_t = upgrade.mark_stage(stats, 'edgegraft', _stage_t)
    if division_mode == 'combined' and upgrade.candidategraft_enabled:
        if dataset is None:
            raise RuntimeError('Pre-pruning CandidateGRAFT needs a dataset id')
        if upgrade.candidategraft_runtime is None:
            upgrade.init_candidategraft_runtime()
        candidate_pre_started = time.monotonic()
        candidate_pre_before = [dict(edge) for edge in edges]
        edges, candidate_pre_stats = upgrade.candidategraft_runtime.apply(
            dataset,
            raw_node_attrs,
            raw_edge_attrs,
            nodes_by_id,
            edges,
            evidence_path(upgrade, upgrade.p1_evidence_dir, dataset),
            evidence_path(upgrade, upgrade.p2_evidence_dir, dataset),
        )
        edges = preserve_guarded_transition_ownership(
            upgrade,
            candidate_pre_before,
            edges,
            nodes_by_id,
            guarded_frames,
            stats,
            'candidategraft_pre',
        )
        stats.update(
            {f'candidategraft_pre_{key}': value for key, value in candidate_pre_stats.items()}
        )
        stats['candidategraft_pre_seconds'] = time.monotonic() - candidate_pre_started
        stats['candidategraft_prepruning_add_only'] = 1
    _stage_t = upgrade.mark_stage(stats, 'candidategraft_pre', _stage_t)
    if upgrade.prune_isolated:
        incident = {int(e['source_id']) for e in edges} | {int(e['target_id']) for e in edges}
        if incident:
            kept_nodes = {i: node for i, node in nodes_by_id.items() if i in incident}
            stats['pruned_isolated_nodes'] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [
                edge
                for edge in edges
                if int(edge['source_id']) in nodes_by_id and int(edge['target_id']) in nodes_by_id
            ]

    _stage_t = upgrade.mark_stage(stats, 'prune', _stage_t)
    nodes_by_id, edges = filter_short_track_components(upgrade, nodes_by_id, edges, stats)
    nodes_by_id, edges = enforce_node_budget(upgrade, nodes_by_id, edges, stats, detected_nodes)
    nodes_by_id = linefit_smooth_output_graph(upgrade, nodes_by_id, edges, stats)
    _stage_t = upgrade.mark_stage(stats, 'component_filters', _stage_t)

    if division_mode == 'combined' and upgrade.candidategraft_enabled:
        if dataset is None:
            raise RuntimeError('CandidateGRAFT needs a dataset id')
        if upgrade.candidategraft_runtime is None:
            upgrade.init_candidategraft_runtime()
        candidate_started = time.monotonic()
        candidate_final_before = [dict(edge) for edge in edges]
        edges, candidate_stats = upgrade.candidategraft_runtime.apply(
            dataset,
            raw_node_attrs,
            raw_edge_attrs,
            nodes_by_id,
            edges,
            evidence_path(upgrade, upgrade.p1_evidence_dir, dataset),
            evidence_path(upgrade, upgrade.p2_evidence_dir, dataset),
        )
        edges = preserve_guarded_transition_ownership(
            upgrade,
            candidate_final_before,
            edges,
            nodes_by_id,
            guarded_frames,
            stats,
            'candidategraft_final',
        )
        stats.update({f'candidategraft_{key}': value for key, value in candidate_stats.items()})
        stats['candidategraft_seconds'] = time.monotonic() - candidate_started
        stats['candidategraft_direct_add_only'] = 1
    else:
        stats['candidategraft_skipped_for_fallback'] = int(upgrade.candidategraft_enabled)
    _stage_t = upgrade.mark_stage(stats, 'candidategraft', _stage_t)
    final_pairs = {(int(edge['source_id']), int(edge['target_id'])) for edge in edges}
    stats['public_wide_safe_div_final_retained'] = sum(
        pair in final_pairs for pair in public_safe_pairs
    )
    stats['public_wide_safe_div_final_removed'] = (
        len(public_safe_pairs) - stats['public_wide_safe_div_final_retained']
    )
    validate_full_ownership_contract(
        upgrade, ownership_endpoint_contract, nodes_by_id, edges, stats
    )
    validate_final_graph_contract(upgrade, nodes_by_id, edges, stats)
    return nodes_by_id, edges, dict(stats)


def process_graph(upgrade, geff_path: Path, division_mode: str) -> dict:
    dataset = Path(geff_path).stem
    read_started = time.monotonic()
    nodes_by_id, raw_edges = graph_nodes_edges(upgrade, Path(geff_path))
    read_seconds = time.monotonic() - read_started
    if upgrade.skip_redundant_raw_copies:
        raw_node_attrs = None
        raw_edge_attrs = None
    else:
        raw_node_attrs = {node_id: dict(node) for node_id, node in nodes_by_id.items()}
        raw_edge_attrs = [dict(edge) for edge in raw_edges]
    raw_nodes = len(nodes_by_id)

    started = time.monotonic()
    nodes_by_id, edges, stats = filter_output_graph(
        upgrade,
        nodes_by_id,
        raw_edges,
        dataset=dataset,
        division_mode=division_mode,
        raw_node_attrs=raw_node_attrs,
        raw_edge_attrs=raw_edge_attrs,
    )
    seconds = time.monotonic() - started
    if not nodes_by_id:
        raise AssertionError(f'{dataset}: post-processing removed every node')

    write_started = time.monotonic()
    node_count, edge_count, divisions = write_dataset_shard(upgrade, dataset, nodes_by_id, edges)
    if upgrade.stage_timing:
        stats['t_graph_read'] = read_seconds
        stats['t_shard_write'] = time.monotonic() - write_started
    stats.update(
        {
            'dataset': dataset,
            'raw_nodes': raw_nodes,
            'nodes': node_count,
            'edges': edge_count,
            'division_like_sources': divisions,
            'edge_to_node_ratio': edge_count / max(node_count, 1),
            'division_mode': division_mode,
            'process_seconds': seconds,
        }
    )
    return stats


def emergency_shard(upgrade, geff_path: Path) -> dict:
    dataset = Path(geff_path).stem
    nodes_by_id, raw_edges = graph_nodes_edges(upgrade, Path(geff_path))

    best_by_target: dict[int, dict] = {}
    for original in raw_edges:
        source = nodes_by_id.get(int(original['source_id']))
        target = nodes_by_id.get(int(original['target_id']))
        if source is None or target is None:
            continue
        if int(target['t']) != int(source['t']) + 1:
            continue
        edge = dict(original)
        edge['distance_um'] = edge_distance_um(upgrade, source, target)
        if upgrade.edge_max_um > 0 and edge['distance_um'] > upgrade.edge_max_um:
            continue
        target_id = int(edge['target_id'])
        previous = best_by_target.get(target_id)
        if previous is None or edge_sort_key(upgrade, edge) > edge_sort_key(upgrade, previous):
            best_by_target[target_id] = edge

    by_source: dict[int, list[dict]] = {}
    for edge in best_by_target.values():
        by_source.setdefault(int(edge['source_id']), []).append(edge)
    edges = []
    for group in by_source.values():
        group.sort(key=lambda edge: edge_sort_key(upgrade, edge), reverse=True)
        edges.extend(group[:2])

    incident = {int(e['source_id']) for e in edges} | {int(e['target_id']) for e in edges}
    nodes_by_id = {i: node for i, node in nodes_by_id.items() if i in incident}
    edges = [
        edge
        for edge in edges
        if int(edge['source_id']) in nodes_by_id and int(edge['target_id']) in nodes_by_id
    ]
    if not nodes_by_id:
        raise AssertionError(f'{dataset}: emergency shard is empty')

    node_count, edge_count, divisions = write_dataset_shard(upgrade, dataset, nodes_by_id, edges)
    return {
        'dataset': dataset,
        'raw_nodes': node_count,
        'nodes': node_count,
        'edges': edge_count,
        'division_like_sources': divisions,
        'edge_to_node_ratio': edge_count / max(node_count, 1),
        'division_mode': 'emergency',
        'process_seconds': 0.0,
    }
