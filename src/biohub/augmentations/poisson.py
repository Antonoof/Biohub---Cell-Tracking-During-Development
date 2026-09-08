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
    array = np.array(imgs.detach().cpu().numpy(), dtype=np.float64, copy=True)
    np.maximum(array, 0.0, out=array)
    array *= float(scale)
    sampled = np.asarray(rng.poisson(array), dtype=np.float32)
    noisy = torch.from_numpy(sampled)
    noisy.mul_(1.0 / float(scale))
    return noisy.to(device=imgs.device, dtype=imgs.dtype), coords, masks
