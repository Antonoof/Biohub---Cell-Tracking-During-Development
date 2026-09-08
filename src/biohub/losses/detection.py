import warnings

import torch
import torch.nn.functional as F

DET_LOSSES = ('weighted_bce', 'focal', 'gaussian_heatmap')


def _binary_target(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)
    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = int(nt[b].item())
        if n_gt <= 0:
            continue
        gt_coords = coords[b, :n_gt]
        zi = gt_coords[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt_coords[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt_coords[:, 2].long().clamp(0, spatial[2] - 1)
        n_unique = len(torch.unique(torch.stack([zi, yi, xi], dim=1), dim=0))
        if n_unique < n_gt:
            warnings.warn(
                f'Sample {b}: {n_gt - n_unique}/{n_gt} GT nodes collapsed to '
                f'duplicate voxels after downsampling — these are undetectable.',
                stacklevel=2,
            )
        target[b, zi, yi, xi] = 1.0
    return logits, target


def compute_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.1,
) -> torch.Tensor:
    B = det_logits.shape[0]
    logits, target = _binary_target(det_logits, coords, mask)
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * (logits.ndim - 1)
    w_pos = (1.0 / n_pos).reshape(shape)
    w_neg = (neg_weight / n_neg).reshape(shape)
    weight = torch.where(target == 1.0, w_pos, w_neg)
    return (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            weight=weight,
            reduction='sum',
        )
        / B
    )


def focal_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    neg_weight: float = 0.1,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    B = det_logits.shape[0]
    logits, target = _binary_target(det_logits, coords, mask)
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * (logits.ndim - 1)
    alpha = torch.where(
        target == 1.0, (1.0 / n_pos).reshape(shape), (neg_weight / n_neg).reshape(shape)
    )
    return (alpha * ((1 - p_t) ** focal_gamma) * bce).sum() / B


def gaussian_heatmap_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    heatmap_sigma: float = 1.0,
) -> torch.Tensor:
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)
    zz, yy, xx = torch.meshgrid(
        torch.arange(spatial[0], device=logits.device, dtype=logits.dtype),
        torch.arange(spatial[1], device=logits.device, dtype=logits.dtype),
        torch.arange(spatial[2], device=logits.device, dtype=logits.dtype),
        indexing='ij',
    )
    sigma2 = 2.0 * max(float(heatmap_sigma), 1e-6) ** 2
    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = int(nt[b].item())
        if n_gt <= 0:
            continue
        for coord in coords[b, :n_gt]:
            blob = torch.exp(
                -((zz - coord[0]) ** 2 + (yy - coord[1]) ** 2 + (xx - coord[2]) ** 2) / sigma2
            )
            target[b] = torch.maximum(target[b], blob)
    return F.mse_loss(torch.sigmoid(logits), target)


def detection_loss(
    kind: str,
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    neg_weight: float = 0.1,
    heatmap_sigma: float = 1.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if kind == 'weighted_bce':
        return compute_detection_loss(det_logits, coords, mask, neg_weight)
    if kind == 'focal':
        return focal_detection_loss(
            det_logits, coords, mask, neg_weight=neg_weight, focal_gamma=focal_gamma
        )
    if kind == 'gaussian_heatmap':
        return gaussian_heatmap_loss(det_logits, coords, mask, heatmap_sigma=heatmap_sigma)
    raise ValueError(f'Unknown det_loss {kind!r}; expected one of {DET_LOSSES}')
