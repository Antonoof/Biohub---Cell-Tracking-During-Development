import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def haze_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    amount: float = 0.1,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    mix = float(rng.uniform(0.0, amount))
    level = float(rng.uniform(0.0, 1.0))
    return imgs * (1.0 - mix) + mix * level, coords, masks
