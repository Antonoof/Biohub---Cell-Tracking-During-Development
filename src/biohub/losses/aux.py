import torch
import torch.nn.functional as F

from biohub.modules.detect.peaks import subvoxel_offsets


def division_aux_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or target.numel() == 0:
        return logits.sum() * 0
    gt = (target.sum(dim=1) > 1).float()
    mass = torch.softmax(logits, dim=0).sum(dim=1)
    pred = torch.sigmoid(mass - 1.0)
    return F.binary_cross_entropy(pred, gt)


def contrastive_aux_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if query.numel() == 0 or key.numel() == 0 or not (target > 0.5).any():
        return query.sum() * 0
    qn = F.normalize(query, dim=-1)
    kn = F.normalize(key, dim=-1)
    logits = qn @ kn.transpose(-2, -1) / max(float(temperature), 1e-6)
    positives = target > 0.5
    logp = F.log_softmax(logits, dim=-1)
    n_pos = positives.sum(dim=-1).clamp(min=1)
    nll = -(logp * positives.float()).sum(dim=-1) / n_pos
    rows = positives.any(dim=-1)
    if not rows.any():
        return query.sum() * 0
    return nll[rows].mean()


def offset_aux_loss(
    offset_pred: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    target: str = 'frac',
    det_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    B = offset_pred.shape[0]
    spatial = offset_pred.shape[2:]
    losses = []
    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = int(nt[b].item())
        if n_gt <= 0:
            losses.append(offset_pred[b].sum() * 0)
            continue
        gt = coords[b, :n_gt]
        zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
        pred = offset_pred[b, :, zi, yi, xi].T
        if target == 'parabolic':
            if det_logits is None:
                raise ValueError('offset_target=parabolic requires det_logits')
            peak_idx = torch.stack((zi, yi, xi), dim=-1)
            frac = subvoxel_offsets(det_logits[b, 0], peak_idx)
        elif target == 'frac':
            frac = gt - torch.stack([zi, yi, xi], dim=-1).to(dtype=gt.dtype)
        else:
            raise ValueError(f'Unknown offset_target {target!r}')
        losses.append((pred - frac).abs().mean())
    return torch.stack(losses).mean()
