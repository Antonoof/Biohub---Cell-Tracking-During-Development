import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def cutout_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    holes: int = 1,
    size: int = 4,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    out = imgs.clone()
    depth, height, width = (int(v) for v in imgs.shape[-3:])
    hole_size = max(int(size), 1)
    for _ in range(max(int(holes), 0)):
        z0 = int(rng.integers(0, max(depth, 1)))
        y0 = int(rng.integers(0, max(height, 1)))
        x0 = int(rng.integers(0, max(width, 1)))
        out[..., z0 : z0 + hole_size, y0 : y0 + hole_size, x0 : x0 + hole_size] = 0
    return out, coords, masks
