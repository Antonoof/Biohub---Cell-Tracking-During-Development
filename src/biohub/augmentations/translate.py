import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def translate_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    px: int = 4,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    shift = max(int(px), 0)
    dy = int(rng.integers(-shift, shift + 1))
    dx = int(rng.integers(-shift, shift + 1))
    if dy == 0 and dx == 0:
        return imgs, coords, masks
    height = int(imgs.shape[-2])
    width = int(imgs.shape[-1])
    shifted = torch.zeros_like(imgs)
    y_src_start = max(0, -dy)
    y_src_end = min(height, height - dy)
    x_src_start = max(0, -dx)
    x_src_end = min(width, width - dx)
    y_dst_start = max(0, dy)
    x_dst_start = max(0, dx)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)
    if y_src_end > y_src_start and x_src_end > x_src_start:
        shifted[..., y_dst_start:y_dst_end, x_dst_start:x_dst_end] = imgs[
            ..., y_src_start:y_src_end, x_src_start:x_src_end
        ]
    out = coords.clone()
    out[..., 1] = coords[..., 1] + dy
    out[..., 2] = coords[..., 2] + dx
    valid = masks.clone()
    valid &= (out[..., 1] >= 0) & (out[..., 1] < height)
    valid &= (out[..., 2] >= 0) & (out[..., 2] < width)
    return shifted, out, valid
