import numpy as np

from biohub.data.coordinates import (
    MATCH_MAX_DISTANCE_UM,
    VOXEL_SCALE_ZYX,
    pairwise_distance_um,
    um_to_voxels,
    voxels_to_um,
)


def test_voxel_to_um_scale() -> None:
    points = voxels_to_um([2], [16], [16])
    np.testing.assert_allclose(points, [[3.25, 6.5, 6.5]])


def test_voxel_um_roundtrip() -> None:
    original = np.array([[2.0, 16.0, 16.0]])
    back = um_to_voxels(voxels_to_um(original[:, 0], original[:, 1], original[:, 2]))
    np.testing.assert_allclose(back, original)


def test_seven_micron_gate_in_xy() -> None:
    a = np.array([[0.0, 0.0, 0.0]])
    inside = np.array([[0.0, MATCH_MAX_DISTANCE_UM / VOXEL_SCALE_ZYX[1] - 1e-6, 0.0]])
    outside = np.array([[0.0, MATCH_MAX_DISTANCE_UM / VOXEL_SCALE_ZYX[1] + 1e-6, 0.0]])
    assert pairwise_distance_um(a, inside)[0, 0] < MATCH_MAX_DISTANCE_UM
    assert pairwise_distance_um(a, outside)[0, 0] > MATCH_MAX_DISTANCE_UM
