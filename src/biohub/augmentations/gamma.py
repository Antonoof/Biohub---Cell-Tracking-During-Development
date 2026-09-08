import numpy as np
import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def gamma_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    gamma_range: float = 0.2,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    gamma = float(rng.uniform(1.0 - gamma_range, 1.0 + gamma_range))
    if imgs.device.type == 'cpu' and imgs.dtype == torch.float32 and not imgs.requires_grad:
        result = np.maximum(imgs.numpy(), 0.0)
        np.power(result, np.float32(gamma), out=result)
        return torch.from_numpy(result), coords, masks
    return imgs.clamp(min=0.0).pow(gamma), coords, masks
