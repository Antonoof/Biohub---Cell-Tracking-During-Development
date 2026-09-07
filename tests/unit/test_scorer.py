import pytest

from biohub.data.synthetic import (
    continuation_pair,
    fork_pair,
    late_fork_pair,
    sparse_unknown_region,
    valid_false_parent,
)
from biohub.metrics.scorer import score_graph_pair

pytest.importorskip('tracksdata')


def test_one_to_one_matching_within_seven_um() -> None:
    pred, gt = continuation_pair(offset_y_um=0.0)
    close, _ = continuation_pair(offset_y_um=3.0)
    far, _ = continuation_pair(offset_y_um=10.0)
    exact = score_graph_pair(pred, gt, n_total=3)
    near = score_graph_pair(close, gt, n_total=3)
    missed = score_graph_pair(far, gt, n_total=3)
    assert exact.edge_tp == 2
    assert exact.edge_fp == 0
    assert exact.edge_fn == 0
    assert near.edge_tp == 2
    assert missed.edge_tp == 0
    assert missed.edge_fn == 2


def test_unannotated_extra_track_is_ignored() -> None:
    pred, gt = sparse_unknown_region()
    row = score_graph_pair(pred, gt, n_total=10)
    assert row.edge_tp == 1
    assert row.edge_fp == 0
    assert row.edge_fn == 0
    assert row.num_pred_nodes == 4


def test_wrong_parent_on_annotated_target_is_fp() -> None:
    pred, gt = valid_false_parent()
    row = score_graph_pair(pred, gt, n_total=3)
    assert row.edge_tp == 0
    assert row.edge_fp >= 1
    assert row.edge_fn == 1


def test_local_fork_identity() -> None:
    pred, gt = fork_pair()
    row = score_graph_pair(pred, gt, n_total=6)
    assert row.division_tp == 1
    assert row.division_fn == 0


def test_late_fork_can_recover_gt_split() -> None:
    pred, gt = late_fork_pair()
    row = score_graph_pair(pred, gt, n_total=6)
    assert row.division_tp + row.division_fn == 1
