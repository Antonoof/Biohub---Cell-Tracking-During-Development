import torch
import torch.nn.functional as F

from biohub.modules.detect.peaks import subvoxel_offsets_batched


def division_aux_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.numel() == 0 or target.numel() == 0:
        return logits.sum() * 0
    with torch.autocast(logits.device.type, enabled=False):
        if logits.ndim == 2:
            logits, target = logits.unsqueeze(0), target.unsqueeze(0)
        if src_mask is None:
            src_mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
        if tgt_mask is None:
            tgt_mask = torch.ones(
                logits.shape[0], logits.shape[2], device=logits.device, dtype=torch.bool
            )
        valid = src_mask.unsqueeze(-1) & tgt_mask.unsqueeze(1)
        safe = logits.float().masked_fill(~src_mask.unsqueeze(-1), -float('inf'))
        safe = torch.where(src_mask.any(dim=1)[:, None, None], safe, 0.0)
        mass = torch.softmax(safe, dim=1).masked_fill(~valid, 0.0).sum(dim=-1)
        gt = (target.masked_fill(~valid, 0).sum(dim=-1) > 1).float()
        per_row = F.binary_cross_entropy_with_logits(mass - 1.0, gt, reduction='none')
        rows = src_mask & tgt_mask.any(dim=1, keepdim=True)
        return ((per_row * rows).sum(dim=1) / rows.sum(dim=1).clamp(min=1)).mean()


def contrastive_aux_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    target: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if query.numel() == 0 or key.numel() == 0:
        return query.sum() * 0
    with torch.autocast(query.device.type, enabled=False):
        if query.ndim == 2:
            query, key, target = query.unsqueeze(0), key.unsqueeze(0), target.unsqueeze(0)
        if src_mask is None:
            src_mask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        if tgt_mask is None:
            tgt_mask = torch.ones(key.shape[:2], device=key.device, dtype=torch.bool)
        qn = F.normalize(query.float(), dim=-1)
        kn = F.normalize(key.float(), dim=-1)
        logits = qn @ kn.transpose(-2, -1) / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~tgt_mask.unsqueeze(1), -float('inf'))
        logits = torch.where(tgt_mask.any(dim=1)[:, None, None], logits, 0.0)
        positives = (target > 0.5) & src_mask.unsqueeze(-1) & tgt_mask.unsqueeze(1)
        logp = F.log_softmax(logits, dim=-1)
        n_pos = positives.sum(dim=-1).clamp(min=1)
        nll = -logp.masked_fill(~positives, 0.0).sum(dim=-1) / n_pos
        rows = positives.any(dim=-1)
        return ((nll * rows).sum(dim=-1) / rows.sum(dim=-1).clamp(min=1)).mean()


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
    with torch.autocast(offset_pred.device.type, enabled=False):
        offset_pred = offset_pred.float()
        coords = coords.float()
        if target == 'parabolic':
            if det_logits is None:
                raise ValueError('offset_target=parabolic requires det_logits')
            det_logits = det_logits.float()
            valid = mask.bool()
            batch_idx = torch.arange(B, device=mask.device).unsqueeze(1).expand_as(mask)[valid]
            gt = coords[valid]
            zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
            yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
            xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
            pred = offset_pred[batch_idx, :, zi, yi, xi]
            frac = subvoxel_offsets_batched(
                det_logits[:, 0], torch.stack((zi, yi, xi), dim=-1), batch_idx
            )
            err = (pred - frac).abs().sum(dim=-1)
            summed = offset_pred.new_zeros(B)
            if batch_idx.numel():
                summed.scatter_add_(0, batch_idx, err)
            denom = mask.sum(dim=1).to(dtype=summed.dtype) * offset_pred.shape[1]
            return (summed / denom.clamp(min=1)).mean()
        if target != 'frac':
            raise ValueError(f'Unknown offset_target {target!r}')
        valid = mask.bool()
        batch_idx = torch.arange(B, device=mask.device).unsqueeze(1).expand_as(mask)[valid]
        gt = coords[valid]
        zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
        pred = offset_pred[batch_idx, :, zi, yi, xi]
        frac = gt - torch.stack((zi, yi, xi), dim=-1)
        summed = offset_pred.new_zeros(B)
        if batch_idx.numel():
            summed.scatter_add_(0, batch_idx, (pred - frac).abs().sum(dim=-1))
        denom = mask.sum(dim=1).to(dtype=summed.dtype) * offset_pred.shape[1]
        return (summed / denom.clamp(min=1)).mean()
