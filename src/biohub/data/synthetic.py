import numpy as np

from biohub.contracts import GraphState
from biohub.data.coordinates import VOXEL_SCALE_ZYX


def _graph(
    movie_id: str,
    nodes: list[tuple[int, int, float, float, float]],
    edges: list[tuple[int, int]],
    estimated: float | None = None,
) -> GraphState:
    node_ids, ts, zs, ys, xs = zip(*nodes, strict=True)
    src, tgt = zip(*edges, strict=True) if edges else ((), ())
    return GraphState(
        movie_id=movie_id,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        t=np.asarray(ts, dtype=np.int64),
        z=np.asarray(zs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        x=np.asarray(xs, dtype=np.float64),
        source_ids=np.asarray(src, dtype=np.int64),
        target_ids=np.asarray(tgt, dtype=np.int64),
        estimated_number_of_nodes=estimated,
    )


def continuation_pair(offset_y_um: float = 0.0) -> tuple[GraphState, GraphState]:
    dy = offset_y_um / VOXEL_SCALE_ZYX[1]
    gt = _graph(
        'synth_continuation',
        [(1, 0, 10, 40, 40), (2, 1, 10, 42, 40), (3, 2, 10, 44, 40)],
        [(1, 2), (2, 3)],
        estimated=3,
    )
    pred = _graph(
        'synth_continuation',
        [(11, 0, 10, 40 + dy, 40), (12, 1, 10, 42 + dy, 40), (13, 2, 10, 44 + dy, 40)],
        [(11, 12), (12, 13)],
        estimated=3,
    )
    return pred, gt


def fork_pair() -> tuple[GraphState, GraphState]:
    gt = _graph(
        'synth_fork',
        [
            (1, 0, 10, 40, 40),
            (2, 1, 10, 42, 40),
            (3, 2, 10, 50, 40),
            (4, 2, 10, 34, 40),
            (5, 3, 10, 54, 40),
            (6, 3, 10, 30, 40),
        ],
        [(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)],
        estimated=6,
    )
    pred = gt.copy()
    pred.movie_id = 'synth_fork'
    return pred, gt


def late_fork_pair() -> tuple[GraphState, GraphState]:
    gt = _graph(
        'synth_late_fork',
        [
            (1, 0, 10, 40, 40),
            (2, 1, 10, 42, 40),
            (3, 2, 10, 50, 40),
            (4, 2, 10, 34, 40),
            (5, 3, 10, 54, 40),
            (6, 3, 10, 30, 40),
        ],
        [(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)],
        estimated=6,
    )
    pred = _graph(
        'synth_late_fork',
        [
            (11, 0, 10, 40, 40),
            (12, 1, 10, 42, 40),
            (13, 2, 10, 46, 40),
            (15, 3, 10, 54, 40),
            (16, 3, 10, 30, 40),
        ],
        [(11, 12), (12, 13), (13, 15), (13, 16)],
        estimated=6,
    )
    return pred, gt


def missing_middle() -> tuple[GraphState, GraphState]:
    gt = _graph(
        'synth_gap',
        [(1, 0, 10, 40, 40), (2, 1, 10, 42, 40), (3, 2, 10, 44, 40)],
        [(1, 2), (2, 3)],
        estimated=3,
    )
    pred = _graph(
        'synth_gap',
        [(11, 0, 10, 40, 40), (13, 2, 10, 44, 40)],
        [],
        estimated=3,
    )
    return pred, gt


def sparse_unknown_region() -> tuple[GraphState, GraphState]:
    gt = _graph(
        'synth_sparse',
        [(1, 0, 10, 40, 40), (2, 1, 10, 42, 40)],
        [(1, 2)],
        estimated=10,
    )
    pred = _graph(
        'synth_sparse',
        [
            (11, 0, 10, 40, 40),
            (12, 1, 10, 42, 40),
            (21, 0, 10, 200, 200),
            (22, 1, 10, 202, 200),
        ],
        [(11, 12), (21, 22)],
        estimated=10,
    )
    return pred, gt


def valid_false_parent() -> tuple[GraphState, GraphState]:
    gt = _graph(
        'synth_fp',
        [(1, 0, 10, 40, 40), (2, 1, 10, 42, 40), (3, 0, 10, 80, 40)],
        [(1, 2)],
        estimated=3,
    )
    pred = _graph(
        'synth_fp',
        [(11, 0, 10, 40, 40), (12, 1, 10, 42, 40), (13, 0, 10, 80, 40)],
        [(13, 12)],
        estimated=3,
    )
    return pred, gt
