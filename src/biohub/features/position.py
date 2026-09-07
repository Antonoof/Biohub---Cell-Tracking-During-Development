import numpy as np
import torch

POS_EMBED_DIM = 8


def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = POS_EMBED_DIM,
) -> np.ndarray:
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def pos_embed_torch(
    coords: torch.Tensor,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = POS_EMBED_DIM,
) -> torch.Tensor:
    shape_t = torch.tensor(image_shape, dtype=torch.float32, device=coords.device)
    norms = coords / shape_t.clamp(min=1)
    freqs = (
        2.0 ** torch.arange(pos_embed_dim // 2, device=coords.device, dtype=torch.float32)
    ) * torch.pi
    parts = []
    for ax in range(4):
        angles = norms[..., ax].unsqueeze(-1) * freqs
        parts.extend([angles.sin(), angles.cos()])
    return torch.cat(parts, dim=-1)
