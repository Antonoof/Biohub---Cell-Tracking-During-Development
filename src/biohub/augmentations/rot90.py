import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def rot90_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    height = int(imgs.shape[-2])
    width = int(imgs.shape[-1])
    k = int(rng.integers(0, 4))
    if k == 0:
        return imgs, coords, masks
    if k % 2 == 1 and height != width:
        return imgs, coords, masks
    rotated = torch.rot90(imgs, k, dims=(-2, -1))
    out = coords.clone()
    y = coords[..., 1]
    x = coords[..., 2]
    if k == 1:
        out[..., 1] = x
        out[..., 2] = height - 1 - y
    elif k == 2:
        out[..., 1] = height - 1 - y
        out[..., 2] = width - 1 - x
    else:
        out[..., 1] = width - 1 - x
        out[..., 2] = y
    return rotated, out, masks
