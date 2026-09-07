import numpy as np
import torch


def quantile_normalize(
    image: np.ndarray,
    gamma: float = 1.0,
    subsample_factor: int = 50,
    q_min: float = 0.001,
    q_max: float = 0.999,
    clip_min: float = 0.0,
    clip_max: float = 4.0,
) -> np.ndarray:
    image = image.astype(np.float32)
    q1, q2 = np.quantile(image.ravel()[::subsample_factor], [q_min, q_max])
    image_normalized = (image - q1) / (q2 - q1 + 1e-6)
    image_normalized = np.clip(image_normalized, 0.0, None)
    image_normalized = image_normalized**gamma
    image_normalized = np.clip(image_normalized, clip_min, clip_max)

    return image_normalized


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-6)


def rescale_coords_to_isotropic(
    coords: np.ndarray,
    scale: tuple[float, float, float],
) -> np.ndarray:
    scale_arr = np.array(scale)
    target_scale = scale_arr.min()
    zoom_factors = scale_arr / target_scale

    coords_iso = coords.astype(np.float32).copy()
    coords_iso[:, 1] *= zoom_factors[0]
    coords_iso[:, 2] *= zoom_factors[1]
    coords_iso[:, 3] *= zoom_factors[2]

    return coords_iso


def rescale_coords_from_isotropic(
    coords: np.ndarray,
    scale: tuple[float, float, float],
) -> np.ndarray:
    scale_arr = np.array(scale)
    target_scale = scale_arr.min()
    zoom_factors = scale_arr / target_scale

    coords_aniso = coords.astype(np.float32).copy()
    coords_aniso[:, 1] /= zoom_factors[0]
    coords_aniso[:, 2] /= zoom_factors[1]
    coords_aniso[:, 3] /= zoom_factors[2]

    return coords_aniso


def resample_image_to_isotropic(
    image: np.ndarray,
    scale: tuple[float, float, float],
    target_scale: tuple[float, float, float] | None = None,
) -> np.ndarray:
    assert image.ndim == 4, f'Expected 4D image (T, Z, Y, X), got {image.ndim}D'

    scale_arr = np.array(scale)
    if target_scale is None:
        target_arr = np.full(3, scale_arr.min())
    else:
        target_arr = np.array(target_scale)
    zoom_factors = scale_arr / target_arr

    new_spatial_shape = (np.array(image.shape[1:]) * zoom_factors).astype(int).tolist()

    tensor = torch.from_numpy(image).cuda().float()
    tensor = tensor.unsqueeze(1)
    tensor = torch.nn.functional.interpolate(
        tensor, size=new_spatial_shape, mode='trilinear', align_corners=False
    )
    tensor = tensor.squeeze(1)

    return tensor.cpu().numpy()


def nms_3d(coords: np.ndarray, scores: np.ndarray, min_distance: float, scale: tuple) -> np.ndarray:
    if len(coords) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(scores)[::-1]
    coords_sorted = coords[order].astype(np.float64)

    scale_arr = np.array(scale, dtype=np.float64)
    coords_physical = coords_sorted * scale_arr

    min_dist_sq = min_distance**2

    keep = []
    suppressed = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        diff = coords_physical[i + 1 :] - coords_physical[i]
        dist_sq = (diff**2).sum(axis=1)
        suppressed[i + 1 :] |= dist_sq < min_dist_sq

    return np.array(keep, dtype=np.int64)
