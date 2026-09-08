import numpy as np
import torch

from biohub.augmentations.proba import AugmentRng, skip_augment


def noise_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: AugmentRng,
    std: float = 0.05,
    proba: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if skip_augment(rng, proba):
        return imgs, coords, masks
    if isinstance(rng, np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        if imgs.device.type == 'cpu' and imgs.dtype == torch.float32 and not imgs.requires_grad:
            # Child RNG preserves the parent stream used by subsequent augmentations.
            noise_np = np.random.default_rng(seed).standard_normal(imgs.shape, dtype=np.float32)
            noise_np *= np.float32(std)
            noise_np += imgs.numpy()
            return torch.from_numpy(noise_np), coords, masks
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        noise = torch.randn(imgs.shape, generator=generator, dtype=torch.float32) * float(std)
        return imgs + noise.to(device=imgs.device, dtype=imgs.dtype), coords, masks
    noise = np.asarray(rng.normal(0.0, std, size=tuple(imgs.shape)), dtype=np.float32)
    return imgs + torch.from_numpy(noise).to(device=imgs.device, dtype=imgs.dtype), coords, masks
