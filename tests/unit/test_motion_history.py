from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from biohub.modules.graph.motion import motion_relink_edges
from biohub.train.motion import (
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
    rollout = score_motion_rollout([payload], None)
    assert rollout['jaccard'] < 1.0
