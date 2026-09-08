import torch
import torch.nn.functional as F

DET_LOSSES = ('weighted_bce', 'focal', 'gaussian_heatmap')
_HEATMAP_GRIDS: dict[tuple, torch.Tensor] = {}


def _binary_target(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)
    batch_idx = torch.arange(B, device=coords.device)[:, None].expand_as(mask)[mask]
    gt_coords = coords[mask]
    zi = gt_coords[:, 0].long().clamp(0, spatial[0] - 1)
    yi = gt_coords[:, 1].long().clamp(0, spatial[1] - 1)
    xi = gt_coords[:, 2].long().clamp(0, spatial[2] - 1)
    target[batch_idx, zi, yi, xi] = 1.0
    return logits, target


def compute_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.1,
) -> torch.Tensor:
    B = det_logits.shape[0]
    logits, target = _binary_target(det_logits, coords, mask)
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * (logits.ndim - 1)
    w_pos = (1.0 / n_pos).reshape(shape)
    w_neg = (neg_weight / n_neg).reshape(shape)
    weight = torch.where(target == 1.0, w_pos, w_neg)
    return (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            weight=weight,
            reduction='sum',
        )
        / B
    )


def focal_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    neg_weight: float = 0.1,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    B = det_logits.shape[0]
    logits, target = _binary_target(det_logits, coords, mask)
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * (logits.ndim - 1)
    alpha = torch.where(
        target == 1.0, (1.0 / n_pos).reshape(shape), (neg_weight / n_neg).reshape(shape)
    )
    return (alpha * ((1 - p_t) ** focal_gamma) * bce).sum() / B


def gaussian_heatmap_target(coords, mask, spatial, sigma=1.0) -> torch.Tensor:
    coords_f = coords.detach().float()
    mask_b = mask.bool()
    depth, height, width = (int(v) for v in spatial)
    device = coords_f.device
    with torch.autocast(device.type, enabled=False):
        if coords_f.shape[1] == 0:
            return coords_f.new_zeros(coords_f.shape[0], depth, height, width)
        key = (depth, height, width, device.type, device.index)
        grid = _HEATMAP_GRIDS.get(key)
        if grid is None or grid.device != device:
            zz = torch.arange(depth, device=device, dtype=torch.float32)
            yy = torch.arange(height, device=device, dtype=torch.float32)
            xx = torch.arange(width, device=device, dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing='ij'), dim=-1).reshape(-1, 3)
            _HEATMAP_GRIDS[key] = grid
        points = coords_f.masked_fill(~mask_b.unsqueeze(-1), 1.0e6)
        batch, size = coords_f.shape[0], grid.shape[0]
        nearest_sq = coords_f.new_empty(batch, size)
        chunk = 32768
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            delta = grid[start:stop].view(1, -1, 1, 3) - points.unsqueeze(1)
            nearest_sq[:, start:stop] = delta.square().sum(dim=-1).min(dim=-1).values
        empty = ~mask_b.any(dim=1)
        nearest_sq = nearest_sq.masked_fill(empty.unsqueeze(-1), 0)
        sigma2 = 2.0 * max(float(sigma), 1e-6) ** 2
        return torch.exp(-nearest_sq / sigma2).reshape(batch, depth, height, width)


def gaussian_heatmap_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    heatmap_sigma: float = 1.0,
    heatmap_target: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.autocast(det_logits.device.type, enabled=False):
        spatial = det_logits.shape[2:]
        logits = det_logits[:, 0].float()
        if heatmap_target is None:
            heatmap_target = gaussian_heatmap_target(coords, mask, spatial, heatmap_sigma)
        target = heatmap_target.to(device=logits.device, dtype=torch.float32, non_blocking=True)
        return F.mse_loss(torch.sigmoid(logits), target)


def detection_loss(
    kind: str,
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    neg_weight: float = 0.1,
    heatmap_sigma: float = 1.0,
    focal_gamma: float = 2.0,
    heatmap_target: torch.Tensor | None = None,
) -> torch.Tensor:
    det_logits = det_logits.float()
    if kind == 'weighted_bce':
        return compute_detection_loss(det_logits, coords, mask, neg_weight)
    if kind == 'focal':
        return focal_detection_loss(
            det_logits, coords, mask, neg_weight=neg_weight, focal_gamma=focal_gamma
        )
    if kind == 'gaussian_heatmap':
        return gaussian_heatmap_loss(
            det_logits, coords, mask, heatmap_sigma=heatmap_sigma, heatmap_target=heatmap_target
        )
    raise ValueError(f'Unknown det_loss {kind!r}; expected one of {DET_LOSSES}')
