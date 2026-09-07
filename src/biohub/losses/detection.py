import warnings

import torch
import torch.nn.functional as F


def compute_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.1,
) -> torch.Tensor:
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)

    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = nt[b]
        if n_gt <= 0:
            continue
        gt_coords = coords[b, : nt[b]]
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

    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * len(spatial)
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
