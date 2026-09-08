import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def flip_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
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
