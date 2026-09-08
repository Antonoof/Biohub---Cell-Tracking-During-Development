import math

import torch
import torch.nn.functional as F

from biohub.augmentations.proba import AugmentRng, skip_augment


def blur_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    sigma: float = 0.8,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba) or sigma <= 0:
        return imgs, coords, masks
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, dtype=imgs.dtype, device=imgs.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    spatial = imgs.reshape(-1, 1, imgs.shape[-2], imgs.shape[-1])

    horizontal = F.conv2d(
        F.pad(spatial, (radius, radius, 0, 0), mode='replicate'), kernel_1d.view(1, 1, 1, -1)
    )
    blurred = F.conv2d(
        F.pad(horizontal, (0, 0, radius, radius), mode='replicate'), kernel_1d.view(1, 1, -1, 1)
    )
    return blurred.reshape(imgs.shape), coords, masks
