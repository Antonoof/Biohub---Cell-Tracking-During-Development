import math

import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

DET_LOSSES = ('weighted_bce', 'focal', 'gaussian_heatmap', 'pu_bce', 'pu_heatmap')
HEATMAP_POS_THRESHOLD = 0.5
PU_POS_THRESHOLD = 0.01
PU_DARK_QUANTILE = 0.40
PU_POS_WEIGHT = 12.0
PU_DARK_WEIGHT = 1.0
PU_UNCERTAIN_WEIGHT = 0.05
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


def _center_frames(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        return images
    if images.ndim == 5:
        return images[:, images.shape[1] // 2]
    raise ValueError(f'Expected image batch (B,Z,Y,X) or (B,W,Z,Y,X); got {tuple(images.shape)}')


def pu_loss_weights(
    images: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_threshold: float = PU_POS_THRESHOLD,
    dark_quantile: float = PU_DARK_QUANTILE,
    positive_weight: float = PU_POS_WEIGHT,
    dark_weight: float = PU_DARK_WEIGHT,
    uncertain_weight: float = PU_UNCERTAIN_WEIGHT,
) -> torch.Tensor:
    center = _center_frames(images).to(device=target.device, dtype=torch.float32)
    if center.shape[-3:] != target.shape[-3:]:
        raise ValueError(
            f'Image spatial {tuple(center.shape[-3:])} != target {tuple(target.shape[-3:])}'
        )
    flat = center.reshape(center.shape[0], -1)
    dark_cutoff = torch.quantile(flat.float(), dark_quantile, dim=1).reshape(
        center.shape[0], *([1] * (center.ndim - 1))
    )
    weights = torch.full_like(target, float(uncertain_weight))
    weights[center <= dark_cutoff] = float(dark_weight)
    weights[target > float(positive_threshold)] = float(positive_weight)
    return weights


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
    return (alpha * ((1 - p_t).clamp_min(1e-8) ** focal_gamma) * bce).sum() / B


def _gaussian_heatmap_splat(coords, mask, depth, height, width, sigma):
    batch = coords.shape[0]
    device = coords.device
    sigma = max(float(sigma), 1e-6)
    radius = max(1, math.ceil(6.08 * sigma))
    out = coords.new_zeros(batch, depth, height, width)
    batch_idx, node_idx = mask.nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return out
    offs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    offset = torch.stack(torch.meshgrid(offs, offs, offs, indexing='ij'), dim=-1).reshape(-1, 3)
    points = coords[batch_idx, node_idx]
    dest = points.round().unsqueeze(1) + offset.unsqueeze(0)
    zi, yi, xi = dest.unbind(-1)
    valid = (zi >= 0) & (zi < depth) & (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
    dist_sq = (dest - points.unsqueeze(1)).square().sum(dim=-1)
    values = torch.exp(-dist_sq / (2.0 * sigma * sigma)).masked_fill(~valid, 0.0)
    sample = batch_idx.unsqueeze(1).expand_as(values)
    index = (
        (sample.long() * depth + zi.long().clamp(0, depth - 1)) * height
        + yi.long().clamp(0, height - 1)
    ) * width + xi.long().clamp(0, width - 1)
    out.view(-1).scatter_reduce_(
        0, index.reshape(-1), values.reshape(-1), reduce='amax', include_self=True
    )
    return out


def gaussian_heatmap_target(coords, mask, spatial, sigma=1.0) -> torch.Tensor:
    coords_f = coords.detach().float()
    mask_b = mask.bool()
    depth, height, width = (int(v) for v in spatial)
    device = coords_f.device
    with torch.autocast(device.type, enabled=False):
        if coords_f.shape[1] == 0:
            return coords_f.new_zeros(coords_f.shape[0], depth, height, width)
        if device.type != 'cpu':
            return _gaussian_heatmap_splat(coords_f, mask_b, depth, height, width, sigma)
        key = (depth, height, width, device.type, device.index)
        grid = _HEATMAP_GRIDS.get(key)
        if grid is None or grid.device != device:
            zz = torch.arange(depth, device=device, dtype=torch.float32)
            yy = torch.arange(height, device=device, dtype=torch.float32)
            xx = torch.arange(width, device=device, dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing='ij'), dim=-1).reshape(-1, 3)
            _HEATMAP_GRIDS.clear()
            _HEATMAP_GRIDS[key] = grid
        batch, size = coords_f.shape[0], grid.shape[0]
        nearest_sq = coords_f.new_full((batch, size), float('inf'))
        grid_np = grid.numpy()
        for b in range(batch):
            points = coords_f[b, mask_b[b]].numpy()
            if len(points):
                distance, _ = KDTree(points, leafsize=16).query(grid_np, workers=1)
                nearest_sq[b] = torch.from_numpy(distance * distance)
        sigma2 = 2.0 * max(float(sigma), 1e-6) ** 2
        return torch.exp(-nearest_sq / sigma2).reshape(batch, depth, height, width)


def gaussian_heatmap_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    heatmap_sigma: float = 1.0,
    heatmap_target: torch.Tensor | None = None,
    neg_weight: float = 0.01,
) -> torch.Tensor:
    with torch.autocast(det_logits.device.type, enabled=False):
        spatial = det_logits.shape[2:]
        logits = det_logits[:, 0].float()
        batch = logits.shape[0]
        if heatmap_target is None:
            heatmap_target = gaussian_heatmap_target(coords, mask, spatial, heatmap_sigma)
        target = heatmap_target.to(device=logits.device, dtype=torch.float32, non_blocking=True)
        pos = target > HEATMAP_POS_THRESHOLD
        n_pos = pos.reshape(batch, -1).sum(dim=1).clamp(min=1)
        n_neg = (target.numel() // batch - n_pos).clamp(min=1)
        shape = (batch,) + (1,) * (logits.ndim - 1)
        weight = torch.where(
            pos,
            (1.0 / n_pos).reshape(shape),
            (neg_weight / n_neg).reshape(shape),
        )
        return (
            F.binary_cross_entropy_with_logits(
                logits,
                target,
                weight=weight,
                reduction='sum',
            )
            / batch
        )


def _weighted_mean_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (element * weights).sum() / weights.sum().clamp_min(1.0)


def pu_bce_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    images: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(det_logits.device.type, enabled=False):
        logits, target = _binary_target(det_logits.float(), coords, mask)
        weights = pu_loss_weights(images, target, positive_threshold=0.5)
        return _weighted_mean_bce(logits, target, weights)


def pu_heatmap_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    images: torch.Tensor,
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
        weights = pu_loss_weights(images, target)
        return _weighted_mean_bce(logits, target, weights)


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
    images: torch.Tensor | None = None,
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
            det_logits,
            coords,
            mask,
            heatmap_sigma=heatmap_sigma,
            heatmap_target=heatmap_target,
            neg_weight=neg_weight,
        )
    if kind in {'pu_bce', 'pu_heatmap'}:
        if images is None:
            raise ValueError(f'{kind} requires images for PU weighting')
        if kind == 'pu_bce':
            return pu_bce_loss(det_logits, coords, mask, images)
        return pu_heatmap_loss(
            det_logits,
            coords,
            mask,
            images,
            heatmap_sigma=heatmap_sigma,
            heatmap_target=heatmap_target,
        )
    raise ValueError(f'Unknown det_loss {kind!r}; expected one of {DET_LOSSES}')
