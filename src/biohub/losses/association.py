import numpy as np
import polars as pl
import torch
import torch.nn.functional as F


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


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction='none')
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** 2) * bce

    div_rows = target.sum(dim=1) > 1
    weight = torch.ones_like(loss)
    weight[div_rows] = 1.0

    return (loss * weight)[mask].mean()


def compute_batch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_t: torch.Tensor,
    mask_t1: torch.Tensor,
) -> torch.Tensor:
    B = logits.shape[0]
    losses = []
    for b in range(B):
        nt = mask_t[b].sum().item()
        nt1 = mask_t1[b].sum().item()
        losses.append(compute_loss(logits[b, :nt, :nt1], target[b, :nt, :nt1]))
    return torch.stack(losses).mean()


def evaluate_pair(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, int, int]:
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0

    loss = compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()

    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()

    return loss, int(correct), int(total)
