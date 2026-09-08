import torch
from scipy.optimize import linear_sum_assignment


def greedy_assign(dists: torch.Tensor, max_distance: float) -> torch.Tensor:
    n_det, n_gt = dists.shape
    matched = torch.full((n_det,), -1, dtype=torch.long, device=dists.device)
    if n_det == 0 or n_gt == 0:
        return matched
    min_d, min_i = dists.min(dim=1)
    order = min_d.argsort()
    # Exactly the old nearest-only greedy policy, including argsort tie order.
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n_det, device=dists.device)
    winner = torch.full((n_gt,), n_det, dtype=torch.long, device=dists.device)
    winner.scatter_reduce_(0, min_i, rank, reduce='amin', include_self=True)
    valid = (rank == winner[min_i]) & (min_d <= max_distance)
    return torch.where(valid, min_i, matched)


def hungarian_assign(dists: torch.Tensor, max_distance: float) -> torch.Tensor:
    n_det, n_gt = dists.shape
    matched = torch.full((n_det,), -1, dtype=torch.long, device=dists.device)
    if n_det == 0 or n_gt == 0:
        return matched
    costs = dists.detach().float().cpu().numpy()
    rows, cols = linear_sum_assignment(costs)
    valid = costs[rows, cols] <= max_distance
    rows = torch.as_tensor(rows[valid], device=dists.device)
    cols = torch.as_tensor(cols[valid], device=dists.device)
    matched[rows] = cols
    return matched


def sinkhorn_coupling(
    dists: torch.Tensor,
    *,
    tau: float,
    iters: int,
) -> torch.Tensor:
    if dists.numel() == 0:
        return dists.new_zeros(dists.shape)
    log_kernel = -dists / max(float(tau), 1e-6)
    for _ in range(max(int(iters), 1)):
        log_kernel = log_kernel - torch.logsumexp(log_kernel, dim=1, keepdim=True)
        log_kernel = log_kernel - torch.logsumexp(log_kernel, dim=0, keepdim=True)
    return log_kernel.exp()


def match_peaks(
    dists: torch.Tensor,
    max_distance: float,
    *,
    kind: str,
    tau: float = 0.1,
    iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_det, n_gt = dists.shape
    if kind == 'greedy':
        matched = greedy_assign(dists, max_distance)
        coupling = torch.zeros(n_det, n_gt, device=dists.device, dtype=dists.dtype)
        valid = matched >= 0
        if valid.any():
            coupling[torch.arange(n_det, device=dists.device)[valid], matched[valid]] = 1.0
        return matched, coupling
    if kind == 'hungarian':
        matched = hungarian_assign(dists, max_distance)
        coupling = torch.zeros(n_det, n_gt, device=dists.device, dtype=dists.dtype)
        valid = matched >= 0
        if valid.any():
            coupling[torch.arange(n_det, device=dists.device)[valid], matched[valid]] = 1.0
        return matched, coupling
    if kind == 'sinkhorn':
        far = dists > max_distance
        gated = dists.masked_fill(far, dists.new_tensor(1e4))
        coupling = sinkhorn_coupling(gated, tau=tau, iters=iters)
        coupling = coupling.masked_fill(far, 0.0)
        # Rectangular Sinkhorn normalizes columns to one; rows may exceed one.
        # Sub-probability rows keep transported BCE targets within [0, 1].
        coupling = coupling / coupling.sum(dim=1, keepdim=True).clamp(min=1.0)
        cost = (-coupling).masked_fill(far, 1e3)
        matched = hungarian_assign(cost, 0.0)
        valid = matched >= 0
        if valid.any():
            chosen = matched.clamp(min=0)
            too_far = dists[torch.arange(n_det, device=dists.device), chosen] > max_distance
            matched = matched.masked_fill(valid & too_far, -1)
        return matched, coupling
    raise ValueError(f'Unknown match_assign {kind!r}')
