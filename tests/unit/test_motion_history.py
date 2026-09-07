from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from biohub.data.motion import _proposal_learned_edges
from biohub.metrics.official import supervised_edge_score
from biohub.modules.graph.motion import motion_relink_edges
from biohub.train.motion import (
    _commit_motion_cache,
    assert_cache_manifest,
    cache_manifest,
    score_motion_rollout,
    serving_motion_runtime,
    video_rows,
)


def _video():
    coords = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 20, 0],
            [1, 0, 20, 0],
            [1, 0, 0, 0],
            [2, 0, 20, 0],
            [2, 0, 0, 0],
        ],
        np.float64,
    )
    matches = np.asarray([1, 2, 1, 2, 1, 2], np.int64)
    n = len(coords)
    return SimpleNamespace(
        voxel_scale_um=np.ones(3),
        image_shape_raw=(3, 8, 32, 32),
        offsets=np.asarray([0, 2, 4, 6], np.int32),
        coords=coords,
        matches=matches,
        gt_edges=[(1, 1), (2, 2)],
        outgoing={1, 2},
        incoming={1, 2},
        children={1: (1,), 2: (2,)},
        edges_by_t={0: (0,), 1: (0,)},
        frozen_sources=set(),
        member_det_prob=np.full((n, 2), 0.5),
        fused_det_prob=np.full(n, 0.5),
    )


def _args():
    return SimpleNamespace(relaxed=20.0, tight=6.2, velocity_weight=0.52, negative_ratio=20)


def test_val_motion_uses_predicted_predecessor() -> None:
    video = _video()
    args = _args()
    train_rows, _meta, _gid = video_rows(video, 0, True, args)
    val_rows, _meta, _gid = video_rows(video, 0, False, args)
    train_vel = np.concatenate([row[0][:, 10:14] for row in train_rows])
    val_vel = np.concatenate([row[0][:, 10:14] for row in val_rows])
    assert train_vel.shape == val_vel.shape
    assert not np.allclose(train_vel, val_vel)


def test_motion_cache_manifest_mismatch(tmp_path) -> None:
    args = SimpleNamespace(
        cache=tmp_path,
        proposals=tmp_path / 'proposals',
        data=tmp_path / 'data',
        tight=6.2,
        relaxed=9.5,
        velocity_weight=0.52,
        negative_ratio=20,
    )
    args.proposals.mkdir()
    args.data.mkdir()
    payload = cache_manifest(['a', 'b'], 'train', args)
    (tmp_path / 'train_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError, match='fingerprint mismatch'):
        assert_cache_manifest(['a', 'b'], 'train', args)
    (tmp_path / 'train_manifest.json').write_text(
        __import__('json').dumps(payload, indent=2, sort_keys=True) + '\n'
    )
    assert_cache_manifest(['a', 'b'], 'train', args)


def test_motion_cache_manifest_hashes_proposal_bytes(tmp_path) -> None:
    proposals = tmp_path / 'proposals'
    proposals.mkdir()
    (proposals / 'a.npz').write_bytes(b'old-proposal')
    args = SimpleNamespace(
        cache=tmp_path,
        proposals=proposals,
        data=tmp_path / 'data',
        tight=6.2,
        relaxed=9.5,
        velocity_weight=0.52,
        negative_ratio=20,
    )
    args.data.mkdir()
    payload = cache_manifest(['a'], 'train', args)
    (tmp_path / 'train_manifest.json').write_text(
        __import__('json').dumps(payload, indent=2, sort_keys=True) + '\n'
    )
    assert payload['proposal_sha256']['a']
    assert_cache_manifest(['a'], 'train', args)
    (proposals / 'a.npz').write_bytes(b'new-proposal')
    with pytest.raises(RuntimeError, match='fingerprint mismatch'):
        assert_cache_manifest(['a'], 'train', args)


def test_motion_rollout_predecessor_differs_from_gt_parent() -> None:
    nodes_by_id = {
        0: {'t': 0, 'z': 0.0, 'y': 0.0, 'x': 0.0},
        1: {'t': 0, 'z': 0.0, 'y': 10.0, 'x': 0.0},
        2: {'t': 1, 'z': 0.0, 'y': 10.0, 'x': 0.0},
        3: {'t': 1, 'z': 0.0, 'y': 0.0, 'x': 0.0},
        4: {'t': 2, 'z': 0.0, 'y': 10.0, 'x': 0.0},
        5: {'t': 2, 'z': 0.0, 'y': 0.0, 'x': 0.0},
    }
    selected = motion_relink_edges(serving_motion_runtime(None), nodes_by_id, defaultdict(int))
    pairs = {(int(edge['source_id']), int(edge['target_id'])) for edge in selected}
    gt_index_pairs = {(0, 2), (1, 3), (2, 4), (3, 5)}
    assert pairs != gt_index_pairs
    payload = {
        'nodes_by_id': nodes_by_id,
        'matches': np.asarray([1, 2, 1, 2, 1, 2], np.int64),
        'gt_edges': [(1, 1), (2, 2)],
    }
    rollout = score_motion_rollout([payload], None, geometry_only=True)
    assert rollout['jaccard'] < 1.0
    with pytest.raises(RuntimeError, match='learned edge probabilities'):
        score_motion_rollout([payload], None)


def test_rollout_scorer_counts_unmatched_supervised_fp() -> None:
    matches = np.asarray([1, 2, -1, 3, 4], np.int64)
    gt_edges = {(1, 2), (3, 4)}
    pred = [(0, 1), (0, 2)]
    score = supervised_edge_score(pred, matches, gt_edges)
    assert score['tp'] == 1
    assert score['fp'] == 1
    assert score['fn'] == 1
    assert score['jaccard'] == pytest.approx(1 / 3)


def test_rollout_scorer_ignores_unsupervised_matched_pair() -> None:
    matches = np.asarray([1, 2, 5, 6], np.int64)
    gt_edges = {(1, 2)}
    pred = [(0, 1), (2, 3)]
    score = supervised_edge_score(pred, matches, gt_edges)
    assert score['tp'] == 1
    assert score['fp'] == 0
    assert score['fn'] == 0
    assert score['jaccard'] == pytest.approx(1.0)


def test_learned_edge_probability_switches_serving_relink() -> None:
    nodes_by_id = {
        0: {'t': 0, 'z': 0.0, 'y': 0.0, 'x': 0.0},
        1: {'t': 0, 'z': 0.0, 'y': 2.0, 'x': 0.0},
        2: {'t': 1, 'z': 0.0, 'y': 2.0, 'x': 0.0},
        3: {'t': 1, 'z': 0.0, 'y': 0.0, 'x': 0.0},
    }
    upgrade = serving_motion_runtime(None)
    geometry = motion_relink_edges(upgrade, nodes_by_id, defaultdict(int), {})
    geometry_pairs = {(int(edge['source_id']), int(edge['target_id'])) for edge in geometry}
    biased = motion_relink_edges(
        upgrade,
        nodes_by_id,
        defaultdict(int),
        {(0, 2): 1.0, (1, 3): 1.0},
    )
    biased_pairs = {(int(edge['source_id']), int(edge['target_id'])) for edge in biased}
    assert geometry_pairs != biased_pairs
    payload = {
        'nodes_by_id': nodes_by_id,
        'matches': np.asarray([1, 2, 1, 2], np.int64),
        'gt_edges': [(1, 1), (2, 2)],
        'edges': [
            {'source_id': 0, 'target_id': 2, 'edge_prob': 1.0},
            {'source_id': 1, 'target_id': 3, 'edge_prob': 1.0},
        ],
    }
    with_probs = score_motion_rollout([payload], None)
    without = dict(payload)
    without['edges'] = []
    empty = score_motion_rollout([without], None)
    assert empty != with_probs
    missing = dict(payload)
    missing.pop('edges')
    with pytest.raises(RuntimeError, match='learned edge probabilities'):
        score_motion_rollout([missing], None)
    none_edges: dict = dict(payload)
    none_edges['edges'] = None
    with pytest.raises(RuntimeError, match='learned edge probabilities'):
        score_motion_rollout([none_edges], None)
    geometry = score_motion_rollout([payload], None, geometry_only=True)
    assert geometry['geometry_only'] is True
    assert with_probs['geometry_only'] is False
    geometry_metrics = {key: value for key, value in geometry.items() if key != 'geometry_only'}
    serving_metrics = {key: value for key, value in with_probs.items() if key != 'geometry_only'}
    empty_metrics = {key: value for key, value in empty.items() if key != 'geometry_only'}
    assert geometry_metrics != serving_metrics
    assert geometry_metrics == empty_metrics


def test_proposal_learned_edges_distinguishes_missing_schema(tmp_path) -> None:
    missing = tmp_path / 'old.npz'
    np.savez(missing, coords=np.zeros((1, 4), np.float32))
    with np.load(missing) as item:
        assert _proposal_learned_edges(item) is None
    empty = tmp_path / 'empty.npz'
    np.savez(
        empty,
        edge_source=np.zeros(0, np.int64),
        edge_target=np.zeros(0, np.int64),
        edge_prob=np.zeros(0, np.float32),
    )
    with np.load(empty) as item:
        assert _proposal_learned_edges(item) == ()
    present = tmp_path / 'present.npz'
    np.savez(
        present,
        edge_source=np.asarray([0], np.int64),
        edge_target=np.asarray([1], np.int64),
        edge_prob=np.asarray([0.9], np.float32),
    )
    with np.load(present) as item:
        learned = _proposal_learned_edges(item)
        assert learned is not None
        assert len(learned) == 1
        assert learned[0][:2] == (0, 1)
        assert learned[0][2] == pytest.approx(0.9)


def test_assert_cache_manifest_rejects_missing_rollout_edges(tmp_path) -> None:
    args = SimpleNamespace(
        cache=tmp_path,
        proposals=tmp_path / 'proposals',
        data=tmp_path / 'data',
        tight=6.2,
        relaxed=9.5,
        velocity_weight=0.52,
        negative_ratio=20,
    )
    args.proposals.mkdir()
    args.data.mkdir()
    payload = cache_manifest(['a'], 'val', args)
    (tmp_path / 'val_manifest.json').write_text(
        __import__('json').dumps(payload, indent=2, sort_keys=True) + '\n'
    )
    (tmp_path / 'val_rollout.pkl').write_bytes(
        __import__('pickle').dumps([{'stem': 'a', 'edges': None}])
    )
    with pytest.raises(RuntimeError, match='learned edge probabilities'):
        assert_cache_manifest(['a'], 'val', args)


def _cache_arrays():
    n = 2
    return [
        np.zeros((n, 3), np.float32),
        np.zeros(n, np.float32),
        np.ones(n, np.float32),
        np.zeros(n, np.int32),
        np.zeros(n, np.int32),
        np.ones(n, np.int32),
        np.zeros(n, np.float32),
    ]


def test_failed_motion_cache_rebuild_leaves_previous_files(tmp_path) -> None:
    cache = tmp_path / 'cache'
    cache.mkdir()
    old_npz = b'old-npz-bytes'
    old_manifest = '{"name": "old"}\n'
    old_rollout = __import__('pickle').dumps(
        [{'stem': 'old', 'edges': [{'source_id': 0, 'target_id': 1, 'edge_prob': 1.0}]}]
    )
    (cache / 'val.npz').write_bytes(old_npz)
    (cache / 'val_manifest.json').write_text(old_manifest)
    (cache / 'val_rollout.pkl').write_bytes(old_rollout)
    with pytest.raises(RuntimeError, match='learned edge probability'):
        _commit_motion_cache(
            cache,
            'val',
            arrays=_cache_arrays(),
            group_meta=np.zeros((1, 4), np.int32),
            manifest={'name': 'new'},
            rollouts=[{'stem': 'a', 'edges': None}],
        )
    assert (cache / 'val.npz').read_bytes() == old_npz
    assert (cache / 'val_manifest.json').read_text() == old_manifest
    assert (cache / 'val_rollout.pkl').read_bytes() == old_rollout
    leftover = list(cache.glob('.staging-val-*'))
    assert leftover == []


def test_motion_cache_commit_replaces_npz_manifest_and_rollout_together(tmp_path) -> None:
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'val.npz').write_bytes(b'old-npz-bytes')
    (cache / 'val_manifest.json').write_text('{"name": "old"}\n')
    (cache / 'val_rollout.pkl').write_bytes(__import__('pickle').dumps([{'stem': 'old'}]))
    rollouts = [
        {
            'stem': 'a',
            'edges': [{'source_id': 0, 'target_id': 1, 'edge_prob': 0.5}],
        }
    ]
    _commit_motion_cache(
        cache,
        'val',
        arrays=_cache_arrays(),
        group_meta=np.zeros((1, 4), np.int32),
        manifest={'name': 'new', 'stems': ['a']},
        rollouts=rollouts,
    )
    loaded = np.load(cache / 'val.npz')
    assert loaded['labels'].shape == (2,)
    assert __import__('json').loads((cache / 'val_manifest.json').read_text()) == {
        'name': 'new',
        'stems': ['a'],
    }
    assert __import__('pickle').loads((cache / 'val_rollout.pkl').read_bytes()) == rollouts
    leftover = list(cache.glob('.staging-val-*'))
    assert leftover == []
