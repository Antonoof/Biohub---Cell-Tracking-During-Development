import numpy as np
import torch


def brightness_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
    shift_range: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift = rng.uniform(-shift_range, shift_range)
    return imgs + shift, coords, masks
