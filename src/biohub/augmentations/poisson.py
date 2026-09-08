import numpy as np
import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def poisson_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    scale: float = 30.0,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba) or scale <= 0:
        return imgs, coords, masks
    lam = np.clip(imgs.detach().cpu().numpy(), 0.0, None) * float(scale)
    sampled = rng.poisson(lam).astype(np.float32) / float(scale)
    noisy = torch.from_numpy(sampled).to(device=imgs.device, dtype=imgs.dtype)
    return noisy, coords, masks
