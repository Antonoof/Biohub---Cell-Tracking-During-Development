import numpy as np
import torch
import torch.nn.functional as F


def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _subvoxel_offsets(
    logits: torch.Tensor,
    peak_idx: torch.Tensor,
    max_shift: float = 0.49,
) -> torch.Tensor:
    offsets = torch.zeros(peak_idx.shape, dtype=torch.float32, device=logits.device)
    for axis in range(3):
        size = logits.shape[axis]
        if size < 3:
            continue
        interior = (peak_idx[:, axis] > 0) & (peak_idx[:, axis] < size - 1)
        if not bool(interior.any()):
            continue
        rows = torch.nonzero(interior, as_tuple=False)[:, 0]
        centre = peak_idx[rows]
        lower = centre.clone()
        upper = centre.clone()
        lower[:, axis] = centre[:, axis] - 1
        upper[:, axis] = centre[:, axis] + 1
        value_c = logits[centre[:, 0], centre[:, 1], centre[:, 2]].float()
        value_l = logits[lower[:, 0], lower[:, 1], lower[:, 2]].float()
        value_r = logits[upper[:, 0], upper[:, 1], upper[:, 2]].float()
        denominator = value_l - 2.0 * value_c + value_r
        shift = torch.where(
            denominator.abs() > 1e-6,
            0.5 * (value_l - value_r) / denominator,
            torch.zeros_like(denominator),
        )
        offsets[rows, axis] = shift.clamp(-max_shift, max_shift)
    return offsets


def detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
    refine: bool = True,
) -> np.ndarray:
    logits = det_logits.unsqueeze(0)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    refined = peak_idx.float()
    if refine:
        refined = refined + _subvoxel_offsets(det_logits[0], peak_idx)
    coords = refined.cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.float32)
