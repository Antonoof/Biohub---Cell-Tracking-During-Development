import numpy as np
import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def noise_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    std: float = 0.05,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    noise = rng.normal(0.0, std, size=tuple(imgs.shape)).astype(np.float32)
    return imgs + torch.from_numpy(noise).to(device=imgs.device, dtype=imgs.dtype), coords, masks
