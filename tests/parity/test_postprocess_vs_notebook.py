from biohub.data.synthetic import continuation_pair, fork_pair
from biohub.infer.config import TrackingConfig, build_tracking_paths, resolve_bundle_paths
from biohub.modules.graph.upgrade import GraphUpgrade


def _tracking_config(tmp_path, **overrides) -> TrackingConfig:
    bundle = tmp_path / 'bundle'
    for name in (
        'model_c',
        'deepcenter',
        'motion_corrector',
        'source_cardinality',
        'unigraft_p1p2',
        'live_v2_bundle',
        'ownership',
        'edgegraft',
        'candidategraft',
    ):
        (bundle / name).mkdir(parents=True)
    payload = {
        'experiment_tag': 'test',
        'voxel_scale_um': (1.625, 0.40625, 0.40625),
        'graph': {
            'deepcenter_gap_veto': False,
            'deepcenter_safe_div_veto': False,
            'gap_refine_synthetic': False,
            'motion_relink': True,
            'safe_divisions': True,
        },
        'division': {'combined_decoder': False},
        'models': {
            'motion_corrector_enabled': False,
            'edgegraft_enabled': False,
            'candidategraft_enabled': False,
        },
        'paths': build_tracking_paths(tmp_path / 'work', tmp_path / 'movies'),
        'bundle': resolve_bundle_paths(bundle),
    }
    payload.update(overrides)
    return TrackingConfig.model_validate(payload)


def _as_dicts(graph):
    nodes = {
        int(node_id): {
            'node_id': int(node_id),
            't': int(t),
            'z': float(z),
            'y': float(y),
            'x': float(x),
        }
        for node_id, t, z, y, x in zip(
            graph.node_ids.tolist(),
            graph.t.tolist(),
            graph.z.tolist(),
            graph.y.tolist(),
            graph.x.tolist(),
            strict=True,
        )
    }
    edges = [
        {'source_id': int(src), 'target_id': int(tgt), 'edge_prob': 0.9}
        for src, tgt in zip(graph.source_ids.tolist(), graph.target_ids.tolist(), strict=True)
    ]
    return nodes, edges


def test_baseline_filter_matches_production_math(tmp_path) -> None:
    upgrade = GraphUpgrade(_tracking_config(tmp_path))
    for factory in (continuation_pair, fork_pair):
        pred, _ = factory()
        nodes, edges = _as_dicts(pred)
        ours_nodes, ours_edges, ours_stats = upgrade.filter_output_graph(
            {key: dict(value) for key, value in nodes.items()},
            [dict(edge) for edge in edges],
            dataset=None,
            division_mode='baseline',
        )
        assert ours_nodes
        assert ours_edges
        assert ours_stats['raw_edges'] == len(edges)
