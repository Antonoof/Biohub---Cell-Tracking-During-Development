import torch
import torch.nn.functional as F


def balanced_focal_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = torch.clamp(target.sum(), min=1.0)
    neg = torch.clamp((1 - target).sum(), min=1.0)
    pos_weight = torch.clamp(neg / pos, 1.0, 30.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction='none'
    )
    probability = torch.sigmoid(logits)
    pt = probability * target + (1 - probability) * (1 - target)
    return (((1 - pt) ** 1.5) * bce).mean()
