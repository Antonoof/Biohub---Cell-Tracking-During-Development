import math

import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def resample_time(imgs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    frames = []
    last = int(imgs.shape[0]) - 1
    for pos in positions.tolist():
        lo = int(math.floor(pos))
        hi = min(lo + 1, last)
        lo = min(max(lo, 0), last)
        alpha = float(pos) - lo
        frames.append(imgs[lo] * (1.0 - alpha) + imgs[hi] * alpha)
    return torch.stack(frames, dim=0)


def time_stretch_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    scale: float = 0.25,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba) or imgs.shape[0] < 2:
        return imgs, coords, masks
    width = int(imgs.shape[0])
    factor = float(rng.uniform(1.0 - scale, 1.0 + scale))
    center = (width - 1) / 2.0
    src = torch.linspace(0.0, float(width - 1), width)
    warped = ((src - center) / max(factor, 1e-3) + center).clamp(0.0, float(width - 1))
    return resample_time(imgs, warped), coords, masks


def time_warp_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    magnitude: float = 0.2,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba) or imgs.shape[0] < 2:
        return imgs, coords, masks
    width = int(imgs.shape[0])
    noise = [float(rng.uniform(-magnitude, magnitude)) for _ in range(width)]
    grid = [float(i) + noise[i] * (width - 1) for i in range(width)]
    grid[0] = 0.0
    grid[-1] = float(width - 1)
    for i in range(1, width):
        grid[i] = max(grid[i], grid[i - 1] + 1e-3)
    scale = (width - 1) / max(grid[-1], 1e-3)
    warped = torch.tensor([min(max(g * scale, 0.0), float(width - 1)) for g in grid])
    return resample_time(imgs, warped), coords, masks
