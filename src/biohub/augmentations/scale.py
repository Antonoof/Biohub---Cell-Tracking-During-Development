import torch
import torch.nn.functional as F

from biohub.augmentations.proba import AugmentRng, skip_augment


def scale_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    scale_range: float = 0.15,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    factor = float(rng.uniform(1.0 - scale_range, 1.0 + scale_range))
    if abs(factor - 1.0) < 1e-6:
        return imgs, coords, masks
    height = int(imgs.shape[-2])
    width = int(imgs.shape[-1])
    scaled = F.interpolate(
        imgs.reshape(-1, 1, height, width),
        scale_factor=factor,
        mode='bilinear',
        align_corners=False,
        recompute_scale_factor=True,
    )
    new_h, new_w = int(scaled.shape[-2]), int(scaled.shape[-1])
    canvas = torch.zeros(scaled.shape[0], 1, height, width, dtype=imgs.dtype)
    y0 = max((new_h - height) // 2, 0)
    x0 = max((new_w - width) // 2, 0)
    y1 = y0 + min(height, new_h)
    x1 = x0 + min(width, new_w)
    dy = max((height - new_h) // 2, 0)
    dx = max((width - new_w) // 2, 0)
    canvas[:, :, dy : dy + (y1 - y0), dx : dx + (x1 - x0)] = scaled[:, :, y0:y1, x0:x1]
    out = coords.clone()

    out[..., 1] = (coords[..., 1] + 0.5) * (new_h / height) - 0.5 - y0 + dy
    out[..., 2] = (coords[..., 2] + 0.5) * (new_w / width) - 0.5 - x0 + dx
    valid = masks.clone()
    valid &= (out[..., 1] >= 0) & (out[..., 1] < height)
    valid &= (out[..., 2] >= 0) & (out[..., 2] < width)
    return canvas.reshape(imgs.shape), out, valid
