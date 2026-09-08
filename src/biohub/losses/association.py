import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

EDGE_LOSSES = ('focal_softmax', 'ce_softmax', 'asl_softmax')


def compute_gt_transition_matrix(
    gt_ids_t: np.ndarray,
    gt_ids_t1: np.ndarray,
    edge_attrs: pl.DataFrame,
) -> torch.Tensor:
    t_to_row = {nid: i for i, nid in enumerate(gt_ids_t)}
    t1_to_col = {nid: i for i, nid in enumerate(gt_ids_t1)}

    matrix = torch.zeros(len(gt_ids_t), len(gt_ids_t1), dtype=torch.float32)
    for source_id, target_id in zip(edge_attrs['source_id'], edge_attrs['target_id']):
        if source_id in t_to_row and target_id in t1_to_col:
            matrix[t_to_row[source_id], t1_to_col[target_id]] = 1.0

    return matrix


def _active_mask(target: torch.Tensor) -> torch.Tensor:
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    return active_rows.unsqueeze(1) | active_cols.unsqueeze(0)


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
) -> torch.Tensor:
    mask = _active_mask(target)
    if not mask.any():
        return logits.sum() * 0

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** focal_gamma) * bce

    weight = torch.ones_like(loss)
    weight[target.sum(dim=1) > 1] = div_weight

    return (loss * weight)[mask].mean()


def ce_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    div_weight: float = 1.0,
) -> torch.Tensor:
    active_cols = target.sum(dim=0) > 0
    if not active_cols.any():
        return logits.sum() * 0
    logp = F.log_softmax(logits, dim=0)
    nll = -(target * logp).sum(dim=0)
    weight = torch.ones_like(nll)
    div_cols = (target > 0.5) & (target.sum(dim=1, keepdim=True) > 1)
    weight = torch.where(div_cols.any(dim=0), nll.new_full((), div_weight), weight)
    return (nll * weight)[active_cols].mean()


def asl_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
) -> torch.Tensor:
    mask = _active_mask(target)
    if not mask.any():
        return logits.sum() * 0
    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    pos_term = ((1 - probs) ** 1.0) * bce
    neg_term = (probs**focal_gamma) * bce
    loss = torch.where(target > 0.5, pos_term, neg_term)
    weight = torch.ones_like(loss)
    weight[target.sum(dim=1) > 1] = div_weight
    return (loss * weight)[mask].mean()


def association_loss(
    kind: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
) -> torch.Tensor:
    # BCE(probabilities) is forbidden inside CUDA autocast. Keep probability
    # reductions in FP32 even when the encoder/attention run in BF16/FP16.
    with torch.autocast(logits.device.type, enabled=False):
        return _association_loss_fp32(
            kind, logits.float(), target.float(), focal_gamma=focal_gamma, div_weight=div_weight
        )


def _association_loss_fp32(kind, logits, target, *, focal_gamma, div_weight):
    if kind == 'focal_softmax':
        return compute_loss(logits, target, focal_gamma=focal_gamma, div_weight=div_weight)
    if kind == 'ce_softmax':
        return ce_softmax_loss(logits, target, div_weight=div_weight)
    if kind == 'asl_softmax':
        return asl_softmax_loss(logits, target, focal_gamma=focal_gamma, div_weight=div_weight)
    raise ValueError(f'Unknown edge_loss {kind!r}; expected one of {EDGE_LOSSES}')


def _gate_logits(
    logits: torch.Tensor,
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    gate_distance: float,
) -> torch.Tensor:
    if gate_distance <= 0 or logits.numel() == 0:
        return logits
    dists = torch.cdist(coords_src, coords_tgt)
    keep = dists <= gate_distance
    if logits.shape[1] > 0:
        nearest = dists.argmin(dim=0)
        keep[nearest, torch.arange(logits.shape[1], device=logits.device)] = True
    return logits.masked_fill(~keep, -1.0e4)


def compute_batch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_t: torch.Tensor,
    mask_t1: torch.Tensor,
    *,
    kind: str = 'focal_softmax',
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
    coords_src: torch.Tensor | None = None,
    coords_tgt: torch.Tensor | None = None,
    gate_distance: float = 0.0,
) -> torch.Tensor:
    B = logits.shape[0]
    losses = []
    source_counts = mask_t.sum(dim=1).tolist()
    target_counts = mask_t1.sum(dim=1).tolist()
    for b in range(B):
        nt = int(source_counts[b])
        nt1 = int(target_counts[b])
        pair = logits[b, :nt, :nt1]
        if coords_src is not None and coords_tgt is not None:
            pair = _gate_logits(pair, coords_src[b, :nt], coords_tgt[b, :nt1], gate_distance)
        losses.append(
            association_loss(
                kind,
                pair,
                target[b, :nt, :nt1],
                focal_gamma=focal_gamma,
                div_weight=div_weight,
            )
        )
    return torch.stack(losses).mean()


def evaluate_pair(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_threshold: float = 0.5,
) -> tuple[float, int, int]:
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0

    loss = compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > edge_threshold).float()

    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()

    return loss, int(correct), int(total)


def pair_event_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_threshold: float = 0.5,
) -> tuple[int, int, int, int, int, int]:
    if logits.numel() == 0 or target.numel() == 0:
        return 0, 0, 0, 0, 0, 0
    probs = torch.softmax(logits, dim=0)
    pred_edge = probs > edge_threshold
    gt_edge = target > 0.5
    edge_tp = int((pred_edge & gt_edge).sum().item())
    edge_fp = int((pred_edge & ~gt_edge).sum().item())
    edge_fn = int((gt_edge & ~pred_edge).sum().item())
    pred_div = pred_edge.sum(dim=1) > 1
    gt_div = gt_edge.sum(dim=1) > 1
    division_tp = int((pred_div & gt_div).sum().item())
    division_fp = int((pred_div & ~gt_div).sum().item())
    division_fn = int((gt_div & ~pred_div).sum().item())
    return edge_tp, edge_fp, edge_fn, division_tp, division_fp, division_fn
