import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def bleach_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    strength: float = 0.4,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    depth = int(imgs.shape[-3])
    amount = float(rng.uniform(0.0, strength))
    z = torch.linspace(0.0, 1.0, depth, dtype=imgs.dtype, device=imgs.device)
    gain = torch.exp(-amount * z).view(1, depth, 1, 1)
    return imgs * gain, coords, masks
