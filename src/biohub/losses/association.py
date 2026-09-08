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
    mask = _active_mask(target).to(dtype=logits.dtype)
    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t).clamp_min(1e-8) ** focal_gamma) * bce
    weight = torch.ones_like(loss)
    weight[target.sum(dim=1) > 1] = div_weight
    return (loss * weight * mask).sum() / mask.sum().clamp(min=1)


def ce_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    div_weight: float = 1.0,
) -> torch.Tensor:
    col_w = (target.sum(dim=0) > 0).to(dtype=logits.dtype)
    logp = F.log_softmax(logits, dim=0)
    nll = -(target * logp).sum(dim=0)
    weight = torch.ones_like(nll)
    div_cols = (target > 0.5) & (target.sum(dim=1, keepdim=True) > 1)
    weight = torch.where(div_cols.any(dim=0), nll.new_full((), div_weight), weight)
    return (nll * weight * col_w).sum() / col_w.sum().clamp(min=1)


def asl_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
) -> torch.Tensor:
    mask = _active_mask(target).to(dtype=logits.dtype)
    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    pos_term = ((1 - probs) ** 1.0) * bce
    neg_term = (probs.clamp_min(1e-8) ** focal_gamma) * bce
    loss = torch.where(target > 0.5, pos_term, neg_term)
    weight = torch.ones_like(loss)
    weight[target.sum(dim=1) > 1] = div_weight
    return (loss * weight * mask).sum() / mask.sum().clamp(min=1)


def association_loss(
    kind: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    div_weight: float = 1.0,
) -> torch.Tensor:
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


def _active_mask_batched(target: torch.Tensor) -> torch.Tensor:
    active_rows = target.sum(dim=2) > 0
    active_cols = target.sum(dim=1) > 0
    return active_rows.unsqueeze(-1) | active_cols.unsqueeze(-2)


def _prefix_ok(counts: torch.Tensor, length: int) -> torch.Tensor:
    return torch.arange(length, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)


def _gate_logits_batched(
    logits: torch.Tensor,
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    src_ok: torch.Tensor,
    tgt_ok: torch.Tensor,
    gate_distance: float,
) -> torch.Tensor:
    if gate_distance <= 0 or logits.numel() == 0:
        return logits
    dists = torch.cdist(coords_src.float(), coords_tgt.float())
    invalid = (~src_ok).unsqueeze(-1) | (~tgt_ok).unsqueeze(-2)
    dists = dists.masked_fill(invalid, 1.0e6)
    keep = dists <= gate_distance
    if logits.shape[-1] > 0:
        nearest = dists.argmin(dim=1)
        batch_idx = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
        tgt_idx = torch.arange(logits.shape[-1], device=logits.device)
        keep[batch_idx, nearest, tgt_idx] = True
    keep = keep & src_ok.unsqueeze(-1) & tgt_ok.unsqueeze(-2)
    return logits.masked_fill(~keep, -1.0e4)


def _focal_batched(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_ok: torch.Tensor,
    *,
    focal_gamma: float,
    div_weight: float,
) -> torch.Tensor:
    mask = (_active_mask_batched(target) & pair_ok).to(dtype=logits.dtype)
    probs = torch.softmax(logits, dim=1)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t).clamp_min(1e-8) ** focal_gamma) * bce
    weight = torch.ones_like(loss)
    weight = weight.masked_fill((target.sum(dim=2) > 1).unsqueeze(-1), div_weight)
    denom = mask.sum(dim=(1, 2)).clamp(min=1)
    return (loss * weight * mask).sum(dim=(1, 2)) / denom


def _ce_batched(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_ok: torch.Tensor,
    *,
    div_weight: float,
) -> torch.Tensor:
    col_w = ((target.sum(dim=1) > 0) & pair_ok.any(dim=1)).to(dtype=logits.dtype)
    logp = F.log_softmax(logits, dim=1)
    nll = -(target * logp).sum(dim=1)
    weight = torch.ones_like(nll)
    div_cols = (target > 0.5) & (target.sum(dim=2, keepdim=True) > 1)
    weight = torch.where(div_cols.any(dim=1), nll.new_full((), div_weight), weight)
    denom = col_w.sum(dim=1).clamp(min=1)
    return (nll * weight * col_w).sum(dim=1) / denom


def _asl_batched(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_ok: torch.Tensor,
    *,
    focal_gamma: float,
    div_weight: float,
) -> torch.Tensor:
    mask = (_active_mask_batched(target) & pair_ok).to(dtype=logits.dtype)
    probs = torch.softmax(logits, dim=1)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    pos_term = ((1 - probs) ** 1.0) * bce
    neg_term = (probs.clamp_min(1e-8) ** focal_gamma) * bce
    loss = torch.where(target > 0.5, pos_term, neg_term)
    weight = torch.ones_like(loss)
    weight = weight.masked_fill((target.sum(dim=2) > 1).unsqueeze(-1), div_weight)
    denom = mask.sum(dim=(1, 2)).clamp(min=1)
    return (loss * weight * mask).sum(dim=(1, 2)) / denom


def _association_loss_batched_fp32(
    kind: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_ok: torch.Tensor,
    *,
    focal_gamma: float,
    div_weight: float,
) -> torch.Tensor:
    if kind == 'focal_softmax':
        return _focal_batched(
            logits, target, pair_ok, focal_gamma=focal_gamma, div_weight=div_weight
        )
    if kind == 'ce_softmax':
        return _ce_batched(logits, target, pair_ok, div_weight=div_weight)
    if kind == 'asl_softmax':
        return _asl_batched(logits, target, pair_ok, focal_gamma=focal_gamma, div_weight=div_weight)
    raise ValueError(f'Unknown edge_loss {kind!r}; expected one of {EDGE_LOSSES}')


def _count_mask(
    counts: list[int] | torch.Tensor | None,
    fallback: torch.Tensor,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if counts is None:
        return fallback.bool()
    values = torch.as_tensor(counts, device=device, dtype=torch.long)
    return _prefix_ok(values, length)


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
    source_counts: list[int] | torch.Tensor | None = None,
    target_counts: list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    n_src = logits.shape[1]
    n_tgt = logits.shape[2]
    src_ok = _count_mask(source_counts, mask_t, n_src, logits.device)
    tgt_ok = _count_mask(target_counts, mask_t1, n_tgt, logits.device)
    pair_ok = src_ok.unsqueeze(-1) & tgt_ok.unsqueeze(-2)
    gated = logits.masked_fill(~pair_ok, -1.0e4)
    if coords_src is not None and coords_tgt is not None:
        gated = _gate_logits_batched(gated, coords_src, coords_tgt, src_ok, tgt_ok, gate_distance)
    masked_target = target.masked_fill(~pair_ok, 0)
    with torch.autocast(logits.device.type, enabled=False):
        per = _association_loss_batched_fp32(
            kind,
            gated.float(),
            masked_target.float(),
            pair_ok,
            focal_gamma=focal_gamma,
            div_weight=div_weight,
        )
    return per.mean()


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


def evaluate_pairs_batched(logits, target, src_mask, tgt_mask, edge_threshold=0.5):
    """Sum existing window-proxy statistics on device; no dense matrices sent to CPU.

    Columns: loss sum, correct, total, edge TP/FP/FN, division TP/FP/FN.
    This preserves pair_event_counts semantics; it is NOT the official graph scorer.
    """
    if logits.numel() == 0:
        return torch.zeros(9, device=logits.device, dtype=torch.float64)
    valid = src_mask.unsqueeze(-1) & tgt_mask.unsqueeze(1)
    safe = logits.float().masked_fill(~src_mask.unsqueeze(-1), -float('inf'))
    safe = torch.where(src_mask.any(1)[:, None, None], safe, 0.0)
    probs = safe.softmax(dim=1).masked_fill(~valid, 0.0)
    target = target.float().masked_fill(~valid, 0.0)
    active = _active_mask_batched(target) & valid
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = (
        (
            ((1 - p_t).clamp_min(1e-8).square() * bce * active).sum((1, 2))
            / active.sum((1, 2)).clamp_min(1)
        )
        .double()
        .sum()
    )
    pred = (probs > edge_threshold) & valid
    gt = (target > 0.5) & valid
    pred_div = pred.sum(-1) > 1
    gt_div = gt.sum(-1) > 1
    return torch.stack(
        [
            loss,
            ((pred == target) & active).sum(),
            active.sum(),
            (pred & gt).sum(),
            (pred & ~gt).sum(),
            (gt & ~pred).sum(),
            (pred_div & gt_div).sum(),
            (pred_div & ~gt_div).sum(),
            (gt_div & ~pred_div).sum(),
        ]
    )


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
