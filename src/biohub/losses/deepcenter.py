import torch
import torch.nn.functional as F


def weighted_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    weighted = loss * weights
    return weighted.sum() / torch.clamp(weights.sum(), min=1.0)
