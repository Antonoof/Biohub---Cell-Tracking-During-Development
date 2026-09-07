import numpy as np
import torch


def flip_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flip_mask = rng.random(3) < 0.5
    dims_to_flip = [1 + dim for dim, flip in enumerate(flip_mask) if flip]
    if not dims_to_flip:
        return imgs, coords, masks
    imgs = imgs.flip(dims=dims_to_flip)
    coords = coords.clone()
    shape = imgs.shape[1:]
    for dim in range(3):
        if flip_mask[dim]:
            dim_coords = coords[..., dim]
            dim_coords[masks] = shape[dim] - dim_coords[masks] - 1
            coords[..., dim] = dim_coords
    return imgs, coords, masks
