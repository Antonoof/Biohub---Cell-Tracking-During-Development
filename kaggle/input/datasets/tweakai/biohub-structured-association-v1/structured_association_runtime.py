"""Inference-only structured A+B association corrector.

The runtime consumes the same top-16, <=20 um A+B candidate graph used during
training.  It builds 77 local + multi-frame features without images or GT,
scores complete source-child and target-parent competitor sets, averages the
five whole-video fold heads, and returns probabilities for the shared ILP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


FEATURE_NAMES = [
    "base_logit", "prob_a", "prob_b", "prob_fused", "prob_disagreement",
    "prob_min", "prob_max", "distance_um", "delta_z_um", "delta_y_um",
    "delta_x_um", "abs_dz_um", "abs_dy_um", "abs_dx_um", "det_src",
    "det_tgt", "det_a_src", "det_b_src", "det_a_tgt", "det_b_tgt",
    "det_disagree_src", "det_disagree_tgt", "source_rank", "target_rank",
    "source_count", "target_count", "source_best_prob", "target_best_prob",
    "source_best_margin", "target_best_margin", "edge_from_source_best",
    "edge_from_target_best", "density_src_15um", "density_tgt_15um",
    "z_boundary_src", "z_boundary_tgt", "frame_fraction", "frozen_transition",
    "prev_valid", "prev_prob", "prev_prob_a", "prev_prob_b", "prev_distance",
    "prev2_valid", "prev2_prob", "prev2_distance", "future_valid", "future_prob",
    "future_prob_a", "future_prob_b", "future_distance", "future2_valid",
    "future2_prob", "future2_distance", "prev_motion_residual",
    "future_motion_residual", "prev_current_cosine", "current_future_cosine",
    "prev_current_speed_logratio", "current_future_speed_logratio",
    "prev2_prev_cosine", "future_future2_cosine", "three_edge_prob_min",
    "three_edge_prob_mean", "three_edge_prob_geomean", "previous_prob_delta",
    "future_prob_delta", "prev_det", "next_det", "prev_det_delta",
    "next_det_delta", "prev_density", "next_density", "prev_member_disagreement",
    "future_member_disagreement", "context_both_sides", "context_depth",
]


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)


def _logit(x):
    x = np.asarray(x, np.float64).clip(1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _grouped_rank(ids, prob, n_nodes):
    order = np.lexsort((-prob, ids))
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    counts = ends - starts
    unique = sid[starts]
    rank_sorted = np.arange(len(order)) - np.repeat(starts, counts) + 1
    rank = np.empty(len(order), np.float32)
    rank[order] = rank_sorted.astype(np.float32)
    best_edge = np.full(n_nodes, -1, np.int64)
    best_edge[unique] = order[starts]
    best_prob = np.zeros(n_nodes, np.float32)
    best_prob[unique] = prob[order[starts]]
    second_prob = np.zeros(n_nodes, np.float32)
    has_second = counts > 1
    second_prob[unique[has_second]] = prob[order[starts[has_second] + 1]]
    count = np.zeros(n_nodes, np.float32)
    count[unique] = counts.astype(np.float32)
    return rank, best_edge, best_prob, second_prob, count


def _safe_take(values, index, valid, fill=0.0):
    out = np.full(len(index), fill, np.float32)
    if valid.any():
        out[valid] = values[index[valid]].astype(np.float32)
    return out


def _vector_norm(x):
    return np.linalg.norm(x, axis=1).astype(np.float32)


def _cosine(a, b, valid):
    out = np.zeros(len(a), np.float32)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = valid & (den > 1e-6)
    out[ok] = (a[ok] * b[ok]).sum(1) / den[ok]
    return out


def _log_speed_ratio(a, b, valid):
    out = np.zeros(len(a), np.float32)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = valid & (na > 1e-5) & (nb > 1e-5)
    out[ok] = np.log((nb[ok] + 1e-3) / (na[ok] + 1e-3)).clip(-4, 4)
    return out


def _node_density(pos, offsets, radius=15.0):
    density = np.zeros(len(pos), np.float32)
    for t in range(len(offsets) - 1):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        if hi > lo:
            density[lo:hi] = cKDTree(pos[lo:hi]).query_ball_point(
                pos[lo:hi], radius, return_length=True,
            ).astype(np.float32) - 1.0
    return density


def build_features(
    coords, frame_offsets, fused_det_prob, member_det_prob, frozen_sources,
    image_shape, voxel_scale_um, source, target, fused_edge_prob,
    member_edge_prob, edge_distance_um,
):
    coords = np.asarray(coords)
    offsets = np.asarray(frame_offsets)
    det = np.asarray(fused_det_prob, np.float32)
    member_det = np.asarray(member_det_prob, np.float32)
    frozen_sources = {int(x) for x in np.asarray(frozen_sources).ravel()}
    image_shape = np.asarray(image_shape)
    scale = np.asarray(voxel_scale_um, np.float32)
    source = np.asarray(source, np.int64)
    target = np.asarray(target, np.int64)
    prob = np.asarray(fused_edge_prob, np.float32).clip(1e-6, 1 - 1e-6)
    member_prob = np.asarray(member_edge_prob, np.float32)
    distance = np.asarray(edge_distance_um, np.float32)
    if member_prob.ndim != 2 or member_prob.shape[1] < 2:
        raise ValueError(f"expected two A+B edge probabilities, got {member_prob.shape}")
    if member_det.ndim != 2 or member_det.shape[1] < 2:
        raise ValueError(f"expected two A+B detection probabilities, got {member_det.shape}")

    n_nodes = len(coords)
    pos = coords[:, 1:].astype(np.float32) * scale[None, :]
    density = _node_density(pos, offsets)
    out_rank, best_out, best_out_p, second_out_p, out_count = _grouped_rank(source, prob, n_nodes)
    in_rank, best_in, best_in_p, second_in_p, in_count = _grouped_rank(target, prob, n_nodes)

    es, et, p = source, target, prob
    delta = pos[et] - pos[es]
    frame = coords[es, 0].astype(np.int32)
    zden = max(float(image_shape[1] - 1), 1.0)
    z_src = coords[es, 1].astype(np.float32) / zden
    z_tgt = coords[et, 1].astype(np.float32) / zden

    prev_e = best_in[es]
    prev_valid = prev_e >= 0
    prev_node = np.full(len(es), -1, np.int64)
    prev_node[prev_valid] = source[prev_e[prev_valid]]
    prev2_e = np.full(len(es), -1, np.int64)
    prev2_e[prev_valid] = best_in[prev_node[prev_valid]]
    prev2_valid = prev2_e >= 0
    prev2_node = np.full(len(es), -1, np.int64)
    prev2_node[prev2_valid] = source[prev2_e[prev2_valid]]

    future_e = best_out[et]
    future_valid = future_e >= 0
    next_node = np.full(len(es), -1, np.int64)
    next_node[future_valid] = target[future_e[future_valid]]
    future2_e = np.full(len(es), -1, np.int64)
    future2_e[future_valid] = best_out[next_node[future_valid]]
    future2_valid = future2_e >= 0
    next2_node = np.full(len(es), -1, np.int64)
    next2_node[future2_valid] = target[future2_e[future2_valid]]

    dprev = np.zeros_like(delta)
    dprev[prev_valid] = pos[es[prev_valid]] - pos[prev_node[prev_valid]]
    dprev2 = np.zeros_like(delta)
    dprev2[prev2_valid] = pos[prev_node[prev2_valid]] - pos[prev2_node[prev2_valid]]
    dfuture = np.zeros_like(delta)
    dfuture[future_valid] = pos[next_node[future_valid]] - pos[et[future_valid]]
    dfuture2 = np.zeros_like(delta)
    dfuture2[future2_valid] = pos[next2_node[future2_valid]] - pos[next_node[future2_valid]]

    prev_p = _safe_take(prob, prev_e, prev_valid)
    prev2_p = _safe_take(prob, prev2_e, prev2_valid)
    future_p = _safe_take(prob, future_e, future_valid)
    future2_p = _safe_take(prob, future2_e, future2_valid)
    prev_pa = _safe_take(member_prob[:, 0], prev_e, prev_valid)
    prev_pb = _safe_take(member_prob[:, 1], prev_e, prev_valid)
    future_pa = _safe_take(member_prob[:, 0], future_e, future_valid)
    future_pb = _safe_take(member_prob[:, 1], future_e, future_valid)
    prev_dist = _safe_take(distance, prev_e, prev_valid)
    prev2_dist = _safe_take(distance, prev2_e, prev2_valid)
    future_dist = _safe_take(distance, future_e, future_valid)
    future2_dist = _safe_take(distance, future2_e, future2_valid)
    prev_det = _safe_take(det, prev_node, prev_valid)
    next_det = _safe_take(det, next_node, future_valid)
    prev_density = _safe_take(density, prev_node, prev_valid)
    next_density = _safe_take(density, next_node, future_valid)

    both = prev_valid & future_valid
    context_depth = (prev_valid.astype(np.float32) + prev2_valid.astype(np.float32)
                     + future_valid.astype(np.float32) + future2_valid.astype(np.float32))
    path_min = np.minimum(np.minimum(prev_p, p), future_p)
    path_mean = (prev_p + p + future_p) / 3.0
    path_geo = np.cbrt(np.maximum(prev_p * p * future_p, 0.0)).astype(np.float32)

    cols = [
        np.log(p / (1 - p)), member_prob[:, 0], member_prob[:, 1], p,
        np.abs(member_prob[:, 0] - member_prob[:, 1]),
        np.minimum(member_prob[:, 0], member_prob[:, 1]),
        np.maximum(member_prob[:, 0], member_prob[:, 1]), distance,
        delta[:, 0], delta[:, 1], delta[:, 2], np.abs(delta[:, 0]),
        np.abs(delta[:, 1]), np.abs(delta[:, 2]), det[es], det[et],
        member_det[es, 0], member_det[es, 1], member_det[et, 0], member_det[et, 1],
        np.abs(member_det[es, 0] - member_det[es, 1]),
        np.abs(member_det[et, 0] - member_det[et, 1]), out_rank, in_rank,
        out_count[es], in_count[et], best_out_p[es], best_in_p[et],
        best_out_p[es] - second_out_p[es], best_in_p[et] - second_in_p[et],
        p - best_out_p[es], p - best_in_p[et], density[es], density[et],
        np.minimum(z_src, 1 - z_src), np.minimum(z_tgt, 1 - z_tgt),
        frame.astype(np.float32) / max(float(image_shape[0] - 1), 1.0),
        np.asarray([float(int(t) in frozen_sources) for t in frame], np.float32),
        prev_valid.astype(np.float32), prev_p, prev_pa, prev_pb, prev_dist,
        prev2_valid.astype(np.float32), prev2_p, prev2_dist,
        future_valid.astype(np.float32), future_p, future_pa, future_pb, future_dist,
        future2_valid.astype(np.float32), future2_p, future2_dist,
        _vector_norm(delta - dprev), _vector_norm(dfuture - delta),
        _cosine(dprev, delta, prev_valid), _cosine(delta, dfuture, future_valid),
        _log_speed_ratio(dprev, delta, prev_valid),
        _log_speed_ratio(delta, dfuture, future_valid),
        _cosine(dprev2, dprev, prev2_valid), _cosine(dfuture, dfuture2, future2_valid),
        path_min, path_mean, path_geo, p - prev_p, future_p - p,
        prev_det, next_det, det[es] - prev_det, next_det - det[et],
        prev_density, next_density, np.abs(prev_pa - prev_pb),
        np.abs(future_pa - future_pb), both.astype(np.float32), context_depth,
    ]
    x = np.column_stack(cols).astype(np.float32)
    if x.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((x.shape, len(FEATURE_NAMES)))
    return np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)


@dataclass
class Groups:
    edge_index: np.ndarray
    mask: np.ndarray
    kind: np.ndarray


def _direction_groups(node_ids, base_prob, max_candidates, kind):
    order = np.argsort(node_ids, kind="stable")
    sorted_ids = node_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    rows, kinds = [], []
    for start, end in zip(starts, ends):
        group = order[start:end]
        ranked = group[np.argsort(-base_prob[group], kind="stable")]
        rows.append(ranked[:max_candidates].astype(np.int32))
        kinds.append(kind)
    return rows, kinds


def build_groups(source, target, base_prob, source_k=16, target_k=32):
    sr, sk = _direction_groups(np.asarray(source), base_prob, source_k, 0)
    tr, tk = _direction_groups(np.asarray(target), base_prob, target_k, 1)
    rows, kinds = sr + tr, sk + tk
    width = max(source_k, target_k)
    edge_index = np.full((len(rows), width), -1, np.int32)
    mask = np.zeros((len(rows), width), bool)
    for i, item in enumerate(rows):
        edge_index[i, :len(item)] = item
        mask[i, :len(item)] = True
    return Groups(edge_index, mask, np.asarray(kinds, np.uint8))


class SetwiseAssociationHead(nn.Module):
    def __init__(self, feature_count, hidden, mean, std, residual_cap, pointwise_cap):
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32))
        self.residual_cap = float(residual_cap)
        self.pointwise_cap = float(pointwise_cap)
        self.edge_encoder = nn.Sequential(
            nn.Linear(feature_count, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.kind_embedding = nn.Embedding(2, hidden)
        self.residual = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
        self.pointwise_residual = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    def encode(self, x):
        return self.edge_encoder(((x - self.feature_mean) / self.feature_std).clamp(-8, 8))

    def forward(self, x, mask, kind):
        edge_h = self.encode(x)
        pointwise = x[..., 0] + self.pointwise_cap * torch.tanh(
            self.pointwise_residual(edge_h).squeeze(-1)
        )
        h = edge_h + self.kind_embedding(kind)[:, None, :]
        m = mask[..., None]
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        maxv = h.masked_fill(~m, torch.finfo(h.dtype).min).amax(1)
        context = torch.cat([h, mean[:, None, :].expand_as(h), maxv[:, None, :].expand_as(h)], -1)
        delta = self.residual_cap * torch.tanh(self.residual(context).squeeze(-1))
        delta -= (delta * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        return (pointwise + delta).masked_fill(~mask, -1e9)


def load_models(root, device="cpu"):
    root = Path(root)
    models = []
    for path in sorted(root.glob("fold_*_model.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if list(ck["feature_names"]) != FEATURE_NAMES:
            raise ValueError(f"feature schema mismatch in {path}")
        state = ck["state_dict"]
        model = SetwiseAssociationHead(
            len(FEATURE_NAMES), int(ck["hidden"]),
            state["feature_mean"].numpy(), state["feature_std"].numpy(),
            float(ck.get("residual_cap", 1.5)), float(ck.get("pointwise_cap", 4.0)),
        )
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"no fold_*_model.pt under {root}")
    return models


@torch.no_grad()
def _predict_one(model, x, groups, batch_size, device):
    score_sum = np.zeros(len(x), np.float64)
    score_count = np.zeros(len(x), np.uint8)
    for start in range(0, len(groups.kind), batch_size):
        batch = np.arange(start, min(start + batch_size, len(groups.kind)))
        idx = groups.edge_index[batch]
        mask = groups.mask[batch]
        safe = idx.copy()
        safe[~mask] = 0
        xb = torch.from_numpy(x[safe]).to(device)
        mb = torch.from_numpy(mask).to(device)
        kb = torch.from_numpy(groups.kind[batch].astype(np.int64)).to(device)
        logits = model(xb, mb, kb).cpu().numpy()
        np.add.at(score_sum, idx[mask], logits[mask])
        np.add.at(score_count, idx[mask], 1)
    raw = x[:, 0].astype(np.float64).copy()
    valid = score_count > 0
    raw[valid] = score_sum[valid] / score_count[valid]
    return raw.astype(np.float32)


def apply(models, x, source, target, base_prob, strength=1.0, delta_clip=2.0,
          batch_size=2048, device="cpu"):
    groups = build_groups(source, target, base_prob, 16, 32)
    logits = np.mean([
        _predict_one(model, x, groups, batch_size, device) for model in models
    ], axis=0)
    base_logit = _logit(base_prob)
    delta = np.clip(logits - base_logit, -delta_clip, delta_clip)
    probability = _sigmoid(base_logit + float(strength) * delta)
    return probability, {
        "candidate_edges": int(len(probability)),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "delta_positive_frac": float((delta > 0).mean()),
        "models": int(len(models)),
        "strength": float(strength),
        "delta_clip": float(delta_clip),
    }
