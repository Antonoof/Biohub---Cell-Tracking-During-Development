import numpy as np

from biohub.contracts import GraphState
from biohub.data.graph import fingerprint, load_graph_npz, save_graph_npz, validate_topology
from biohub.data.synthetic import continuation_pair, fork_pair


def _codes(graph: GraphState) -> set[str]:
    return {issue.code for issue in validate_topology(graph)}


def test_topology_accepts_valid_fork() -> None:
    pred, _ = fork_pair()
    assert validate_topology(pred) == []


def test_topology_rejects_in_degree() -> None:
    pred, _ = continuation_pair()
    extra = GraphState(
        movie_id=pred.movie_id,
        node_ids=np.concatenate([pred.node_ids, np.array([99], dtype=np.int64)]),
        t=np.concatenate([pred.t, np.array([0], dtype=np.int64)]),
        z=np.concatenate([pred.z, pred.z[:1]]),
        y=np.concatenate([pred.y, pred.y[:1]]),
        x=np.concatenate([pred.x, pred.x[:1]]),
        source_ids=np.concatenate([pred.source_ids, np.array([99], dtype=np.int64)]),
        target_ids=np.concatenate([pred.target_ids, pred.node_ids[1:2]]),
    )
    assert 'in_degree' in _codes(extra)


def test_topology_rejects_nonconsecutive_frame() -> None:
    pred, _ = continuation_pair()
    pred.source_ids = pred.source_ids.copy()
    pred.target_ids = pred.target_ids.copy()
    pred.source_ids[0] = pred.node_ids[0]
    pred.target_ids[0] = pred.node_ids[2]
    assert 'nonconsecutive_frame' in _codes(pred)


def test_topology_rejects_duplicate_daughters() -> None:
    pred, _ = continuation_pair()
    pred.source_ids = np.array([pred.node_ids[0], pred.node_ids[0]], dtype=np.int64)
    pred.target_ids = np.array([pred.node_ids[1], pred.node_ids[1]], dtype=np.int64)
    codes = _codes(pred)
    assert 'duplicate_edge' in codes
    assert 'duplicate_daughter' in codes


def test_topology_rejects_out_degree() -> None:
    pred, _ = fork_pair()
    pred.source_ids = np.concatenate([pred.source_ids, pred.source_ids[:1]])
    extra_target = int(pred.node_ids.max()) + 1
    pred.node_ids = np.concatenate([pred.node_ids, np.array([extra_target], dtype=np.int64)])
    pred.t = np.concatenate([pred.t, np.array([2], dtype=np.int64)])
    pred.z = np.concatenate([pred.z, pred.z[:1]])
    pred.y = np.concatenate([pred.y, pred.y[:1]])
    pred.x = np.concatenate([pred.x, pred.x[:1]])
    pred.target_ids = np.concatenate([pred.target_ids, np.array([extra_target], dtype=np.int64)])
    pred.source_ids[-1] = pred.node_ids[1]
    assert 'out_degree' in _codes(pred)


def test_npz_roundtrip_preserves_coords_and_attrs(tmp_path) -> None:
    pred, _ = continuation_pair()
    pred.node_attrs['score'] = pred.z * 0 + 0.9
    path = tmp_path / 'graph.npz'
    save_graph_npz(pred, path)
    loaded = load_graph_npz(path)
    assert loaded.n_nodes == pred.n_nodes
    assert loaded.n_edges == pred.n_edges
    assert fingerprint(loaded) == fingerprint(pred)
    assert loaded.movie_id == pred.movie_id
    np.testing.assert_allclose(loaded.node_attrs['score'], pred.node_attrs['score'])
