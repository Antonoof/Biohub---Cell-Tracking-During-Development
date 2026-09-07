import numpy as np
from numpy.typing import ArrayLike, NDArray

from biohub.constants import MATCH_MAX_DISTANCE_UM, VOXEL_SCALE_ZYX

AXES = ('z', 'y', 'x')
COORDINATE_UNIT = 'original_voxel'

__all__ = [
    'AXES',
    'COORDINATE_UNIT',
    'MATCH_MAX_DISTANCE_UM',
    'VOXEL_SCALE_ZYX',
    'pairwise_distance_um',
    'um_to_voxels',
    'voxels_to_um',
]


def voxels_to_um(
    z: ArrayLike,
    y: ArrayLike,
    x: ArrayLike,
    scale: tuple[float, float, float] = VOXEL_SCALE_ZYX,
) -> NDArray[np.float64]:
    z_arr = np.asarray(z, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    stacked = np.stack([z_arr, y_arr, x_arr], axis=-1)
    return stacked * np.asarray(scale, dtype=np.float64)


def um_to_voxels(
    points_um: ArrayLike,
    scale: tuple[float, float, float] = VOXEL_SCALE_ZYX,
) -> NDArray[np.float64]:
    points = np.asarray(points_um, dtype=np.float64)
    return points / np.asarray(scale, dtype=np.float64)


def pairwise_distance_um(
    a_zyx_vox: ArrayLike,
    b_zyx_vox: ArrayLike,
    scale: tuple[float, float, float] = VOXEL_SCALE_ZYX,
) -> NDArray[np.float64]:
    a = np.asarray(a_zyx_vox, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    b = np.asarray(b_zyx_vox, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    delta = a[:, None, :] - b[None, :, :]
    return np.sqrt((delta * delta).sum(axis=-1))
