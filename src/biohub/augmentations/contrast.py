import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def contrast_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    contrast_range: float = 0.2,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    factor = float(rng.uniform(1.0 - contrast_range, 1.0 + contrast_range))
    mean = imgs.mean()
    return (imgs - mean) * factor + mean, coords, masks
