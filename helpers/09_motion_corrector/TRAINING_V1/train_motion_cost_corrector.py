#!/usr/bin/env python3
"""Learn a bounded residual cost for the V8 sequential motion relinker.

Supports multiple architectures (mlp / realmlp / tabm / …), feature transforms,
and training knobs for honest GKF5 sweeps. Checkpoint schema stays compatible:
model state_dict + mean/std + features + residual_scale (+ arch metadata).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import finetune_edge_head_on_proposals as ft  # noqa: E402

FEATURES = [
    "base_cost",
    "raw_dist",
    "registered_dist",
    "motion_dist",
    "abs_dz",
    "abs_dy",
    "abs_dx",
    "abs_reg_dz",
    "abs_reg_dy",
    "abs_reg_dx",
    "velocity_z",
    "velocity_y",
    "velocity_x",
    "velocity_mag",
    "shift_z",
    "shift_y",
    "shift_x",
    "shift_mag",
    "det_src",
    "det_tgt",
    "det_disagree_src",
    "det_disagree_tgt",
    "density_src",
    "density_tgt",
    "z_boundary_src",
    "z_boundary_tgt",
    "has_predecessor",
    "frozen",
]
# Available in cache build but not in submitted GEFF → drop for serve parity.
RUNTIME_DROP = {"det_src", "det_tgt", "det_disagree_src", "det_disagree_tgt", "frozen"}

# Indices into runtime-safe feature list (after RUNTIME_DROP).
# Order matches FEATURES with drops removed.
RUNTIME_SAFE = [n for n in FEATURES if n not in RUNTIME_DROP]
DIST_NAMES = {"base_cost", "raw_dist", "registered_dist", "motion_dist", "velocity_mag", "shift_mag"}
ABS_DELTA = {"abs_dz", "abs_dy", "abs_dx", "abs_reg_dz", "abs_reg_dy", "abs_reg_dx"}


def argspec():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/train"))
    p.add_argument("--proposals", type=Path, default=Path("data/ab_proposals_export/biohub_ab_proposals"))
    p.add_argument("--splits", type=Path, default=Path(__file__).resolve().parent / "splits_ensembleB.json")
    p.add_argument("--cache", type=Path, default=Path("data/motion_cost_cache"))
    p.add_argument("--output", type=Path, default=Path("artifacts/motion_cost_corrector"))
    p.add_argument("--tight", type=float, default=6.2)
    p.add_argument("--relaxed", type=float, default=9.5)
    p.add_argument("--velocity-weight", type=float, default=0.52)
    p.add_argument("--residual-scale", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--negative-ratio", type=int, default=20)
    p.add_argument("--seed", type=int, default=2028)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--parent-mode",
        choices=["proposal_nn", "gt"],
        default="proposal_nn",
        help="Velocity parents: proposal_nn=serve-matched (default), gt=teacher-forced leak",
    )
    p.add_argument("--fold", type=int, default=0, help="Fold index when --splits is a list of folds (honest GKF5)")
    # Sweep knobs
    p.add_argument(
        "--arch",
        default="mlp",
        choices=[
            "mlp",
            "mlp_wide",
            "mlp_deep",
            "resmlp",
            "realmlp",
            "realmlp_wide",
            "tabm",
            "tabm_wide",
            "gated",
            "se_mlp",
            "highway",
        ],
    )
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--tabm-k", type=int, default=8, help="TabM ensemble size")
    p.add_argument("--hidden", type=int, default=0, help="Override hidden size (0=arch default)")
    p.add_argument(
        "--feat-mode",
        default="raw",
        choices=["raw", "log_dist", "quad", "interact", "log_interact", "full"],
        help="Feature transforms applied after RUNTIME_DROP (serve-reproducible)",
    )
    p.add_argument("--loss", default="focal", choices=["focal", "bce", "focal_soft", "asymmetric"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--logit-bias", type=float, default=2.5)
    p.add_argument("--learn-logit-bias", action="store_true")
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "plateau"])
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--exp-name", default="", help="Stored in checkpoint/metrics for sweeps")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature transforms (from runtime-safe columns only)
# ---------------------------------------------------------------------------


def transform_features(x: np.ndarray, names: list[str], mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "raw":
        return x.astype(np.float32), list(names)

    idx = {n: i for i, n in enumerate(names)}
    cols = [x]
    out_names = list(names)

    def col(n):
        return x[:, idx[n] : idx[n] + 1]

    if mode in ("log_dist", "log_interact", "full"):
        for n in names:
            if n in DIST_NAMES or n in ABS_DELTA:
                cols.append(np.log1p(np.maximum(col(n), 0.0)))
                out_names.append(f"log1p_{n}")

    if mode in ("quad", "full"):
        for n in ("base_cost", "motion_dist", "registered_dist", "raw_dist", "velocity_mag"):
            if n in idx:
                cols.append(col(n) ** 2)
                out_names.append(f"sq_{n}")

    if mode in ("interact", "log_interact", "full"):
        pairs = [
            ("base_cost", "has_predecessor"),
            ("motion_dist", "has_predecessor"),
            ("registered_dist", "density_src"),
            ("motion_dist", "density_tgt"),
            ("velocity_mag", "shift_mag"),
            ("base_cost", "z_boundary_src"),
            ("raw_dist", "registered_dist"),
            ("motion_dist", "registered_dist"),
            ("density_src", "density_tgt"),
            ("abs_dz", "abs_reg_dz"),
        ]
        for a, b in pairs:
            if a in idx and b in idx:
                cols.append(col(a) * col(b))
                out_names.append(f"{a}*{b}")

    return np.concatenate(cols, axis=1).astype(np.float32), out_names


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def _zero_last_linear(m: nn.Module) -> None:
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class MLP(nn.Module):
    def __init__(self, n: int, widths: list[int], dropout: float = 0.05):
        super().__init__()
        layers: list[nn.Module] = []
        d = n
        for w in widths:
            layers += [nn.Linear(d, w), nn.SiLU(), nn.Dropout(dropout)]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        _zero_last_linear(self.net)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h


class ResMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 96, depth: int = 3, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Linear(n, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.blocks(F.silu(self.in_proj(x)))).squeeze(-1)


class RealMLP(nn.Module):
    """RealMLP-inspired: LayerNorm stem, residual MLP blocks, SiLU, mild dropout."""

    def __init__(self, n: int, hidden: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class TabM(nn.Module):
    """TabM-mini: K parallel MLPs, mean prediction (BatchEnsemble-free)."""

    def __init__(self, n: int, hidden: int = 64, depth: int = 2, k: int = 8, dropout: float = 0.05):
        super().__init__()
        self.k = k
        self.adapters = nn.ModuleList([MLP(n, [hidden] * depth + [hidden // 2], dropout) for _ in range(k)])

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.adapters], dim=0)
        if self.training:
            # train on mean; slight noise via random subset average for diversity
            return preds.mean(0)
        return preds.mean(0)


class GatedMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 96, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(n)
        self.u = nn.Linear(n, hidden)
        self.v = nn.Linear(n, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.norm(x)
        h = F.silu(self.u(x)) * torch.sigmoid(self.v(x))
        return self.out(self.drop(h)).squeeze(-1)


class SEMLP(nn.Module):
    """Feature squeeze-excitation then MLP."""

    def __init__(self, n: int, hidden: int = 96, dropout: float = 0.05):
        super().__init__()
        r = max(4, n // 4)
        self.se = nn.Sequential(nn.Linear(n, r), nn.SiLU(), nn.Linear(r, n), nn.Sigmoid())
        self.mlp = MLP(n, [hidden, hidden // 2], dropout)

    def forward(self, x):
        return self.mlp(x * self.se(x))


class HighwayMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 96, layers: int = 3, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Linear(n, hidden)
        self.H = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.T = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        for t in self.T:
            nn.init.constant_(t.bias, -2.0)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = F.silu(self.in_proj(x))
        for H, T in zip(self.H, self.T):
            t = torch.sigmoid(T(h))
            h = self.drop(F.silu(H(h)) * t + h * (1 - t))
        return self.out(h).squeeze(-1)


def build_model(arch: str, n_in: int, args) -> nn.Module:
    h = args.hidden
    d = args.dropout
    if arch == "mlp":
        return MLP(n_in, [h or 64, 32], d)
    if arch == "mlp_wide":
        return MLP(n_in, [h or 128, 64], d)
    if arch == "mlp_deep":
        return MLP(n_in, [h or 96, 96, 64, 32], d)
    if arch == "resmlp":
        return ResMLP(n_in, hidden=h or 96, depth=3, dropout=d)
    if arch == "realmlp":
        return RealMLP(n_in, hidden=h or 128, depth=4, dropout=max(d, 0.08))
    if arch == "realmlp_wide":
        return RealMLP(n_in, hidden=h or 256, depth=5, dropout=max(d, 0.1))
    if arch == "tabm":
        return TabM(n_in, hidden=h or 64, depth=2, k=args.tabm_k, dropout=d)
    if arch == "tabm_wide":
        return TabM(n_in, hidden=h or 96, depth=3, k=max(args.tabm_k, 12), dropout=d)
    if arch == "gated":
        return GatedMLP(n_in, hidden=h or 96, dropout=d)
    if arch == "se_mlp":
        return SEMLP(n_in, hidden=h or 96, dropout=d)
    if arch == "highway":
        return HighwayMLP(n_in, hidden=h or 96, layers=3, dropout=d)
    raise ValueError(arch)


# ---------------------------------------------------------------------------
# Cache / scoring (unchanged geometry)
# ---------------------------------------------------------------------------


def shifts_for_video(v):
    shifts = []
    scale = np.asarray(v.voxel_scale_um, np.float64)
    for t in range(v.image_shape_raw[0] - 1):
        a0, a1 = int(v.offsets[t]), int(v.offsets[t + 1])
        b0, b1 = int(v.offsets[t + 1]), int(v.offsets[t + 2])
        a = v.coords[a0:a1, 1:].astype(np.float64) * scale
        b = v.coords[b0:b1, 1:].astype(np.float64) * scale
        if t in v.frozen_sources or not len(a) or not len(b):
            shifts.append(np.zeros(3))
            continue
        ta, tb = cKDTree(a), cKDTree(b)
        dab, jab = tb.query(a, k=1)
        dba, jba = ta.query(b, k=1)
        ia = np.arange(len(a))
        good = (dab <= 10) & (jba[jab] == ia)
        shifts.append(np.median(b[jab[good]] - a[good], axis=0) if good.sum() >= 5 else np.zeros(3))
    return shifts


def video_rows(v, group_start, train, args):
    scale = np.asarray(v.voxel_scale_um, np.float64)
    shifts = shifts_for_video(v)
    parent_mode = getattr(args, "parent_mode", "proposal_nn")
    if parent_mode == "gt":
        parent = {int(d): int(s) for s, d in v.gt_edges}
    else:
        parent = {}
    rows = []
    group_meta = []
    gid = group_start
    for t in range(v.image_shape_raw[0] - 1):
        a0, a1 = int(v.offsets[t]), int(v.offsets[t + 1])
        b0, b1 = int(v.offsets[t + 1]), int(v.offsets[t + 2])
        n0, n1 = a1 - a0, b1 - b0
        group_meta.append((gid, n0, n1, len(v.edges_by_t.get(t, ()))))
        if not n0 or not n1:
            gid += 1
            continue
        c0 = v.coords[a0:a1, 1:].astype(np.float64) * scale
        c1 = v.coords[b0:b1, 1:].astype(np.float64) * scale
        m0 = v.matches[a0:a1]
        m1 = v.matches[b0:b1]
        shift = np.asarray(shifts[t])
        prev_shift = np.asarray(shifts[t - 1] if t else np.zeros(3))
        reg_delta = c1[None, :, :] - c0[:, None, :] - shift
        reg = np.linalg.norm(reg_delta, axis=2)
        candidate = reg <= args.relaxed
        row_active = np.fromiter((int(x) in v.outgoing for x in m0), bool, n0)
        col_active = np.fromiter((int(x) in v.incoming for x in m1), bool, n1)
        sup = row_active[:, None] | col_active[None, :]
        label = np.zeros((n0, n1), bool)
        right = {int(g): j for j, g in enumerate(m1) if g >= 0}
        for i, g in enumerate(m0):
            if g >= 0:
                for d in v.children.get(int(g), ()):
                    if d in right:
                        label[i, right[d]] = True
        vel = np.zeros((n0, 3))
        has = np.zeros(n0)
        if parent_mode == "gt":
            prev_lookup = {}
            if t:
                p0, p1 = int(v.offsets[t - 1]), int(v.offsets[t])
                for k, g in enumerate(v.matches[p0:p1]):
                    if g >= 0:
                        prev_lookup[int(g)] = v.coords[p0 + k, 1:].astype(np.float64) * scale
            for i, g in enumerate(m0):
                pg = parent.get(int(g), -1) if g >= 0 else -1
                if pg in prev_lookup:
                    vel[i] = c0[i] - prev_lookup[pg] - prev_shift
                    has[i] = 1
        elif t:
            p0, p1 = int(v.offsets[t - 1]), int(v.offsets[t])
            prev = v.coords[p0:p1, 1:].astype(np.float64) * scale
            if len(prev):
                tree = cKDTree(prev)
                dist, jix = tree.query(c0 - shift, k=1)
                if np.ndim(dist) == 0:
                    dist = np.asarray([dist])
                    jix = np.asarray([jix])
                ok = dist <= args.relaxed
                vel[ok] = c0[ok] - prev[jix[ok]] - prev_shift
                has[ok] = 1
        predicted = c0 + shift + args.velocity_weight * vel
        motion_delta = c1[None, :, :] - predicted[:, None, :]
        motion = np.linalg.norm(motion_delta, axis=2)
        raw_delta = c1[None, :, :] - c0[:, None, :]
        raw = np.linalg.norm(raw_delta, axis=2)
        base_cost = motion + 0.05 * reg
        dens0 = (cdist(c0, c0) <= 15).sum(1) - 1
        dens1 = (cdist(c1, c1) <= 15).sum(1) - 1
        zmax = max((v.image_shape_raw[1] - 1) * scale[0], 1)
        zb0 = np.minimum(c0[:, 0] / zmax, 1 - c0[:, 0] / zmax)
        zb1 = np.minimum(c1[:, 0] / zmax, 1 - c1[:, 0] / zmax)
        ii, jj = np.nonzero(candidate if not train else (candidate & sup))
        if train and len(ii):
            pos = np.flatnonzero(label[ii, jj])
            neg = np.flatnonzero(~label[ii, jj])
            nk = min(len(neg), max(64, max(1, len(pos)) * args.negative_ratio))
            if nk < len(neg):
                neg = neg[np.argpartition(base_cost[ii[neg], jj[neg]], nk - 1)[:nk]]
            take = np.concatenate([pos, neg])
            ii, jj = ii[take], jj[take]
        if len(ii):
            md0 = v.member_det_prob[a0:a1]
            md1 = v.member_det_prob[b0:b1]
            f = np.column_stack(
                [
                    base_cost[ii, jj],
                    raw[ii, jj],
                    reg[ii, jj],
                    motion[ii, jj],
                    np.abs(raw_delta[ii, jj, 0]),
                    np.abs(raw_delta[ii, jj, 1]),
                    np.abs(raw_delta[ii, jj, 2]),
                    np.abs(reg_delta[ii, jj, 0]),
                    np.abs(reg_delta[ii, jj, 1]),
                    np.abs(reg_delta[ii, jj, 2]),
                    vel[ii, 0],
                    vel[ii, 1],
                    vel[ii, 2],
                    np.linalg.norm(vel[ii], axis=1),
                    np.full(len(ii), shift[0]),
                    np.full(len(ii), shift[1]),
                    np.full(len(ii), shift[2]),
                    np.full(len(ii), np.linalg.norm(shift)),
                    v.fused_det_prob[a0:a1][ii],
                    v.fused_det_prob[b0:b1][jj],
                    np.abs(md0[ii, 0] - md0[ii, 1]),
                    np.abs(md1[jj, 0] - md1[jj, 1]),
                    dens0[ii],
                    dens1[jj],
                    zb0[ii],
                    zb1[jj],
                    has[ii],
                    np.full(len(ii), float(t in v.frozen_sources)),
                ]
            ).astype(np.float32)
            rows.append(
                (
                    f,
                    label[ii, jj].astype(np.uint8),
                    sup[ii, jj].astype(np.uint8),
                    np.full(len(ii), gid, np.int32),
                    ii.astype(np.int32),
                    jj.astype(np.int32),
                    reg[ii, jj].astype(np.float32),
                )
            )
        gid += 1
    return rows, group_meta, gid


def make_cache(stems, name, args):
    allrows = []
    meta = []
    gid = 0
    for stem in tqdm(stems[: args.max_videos or None], desc=f"cache {name}"):
        v = ft.load_proposal_video(args.data, args.proposals, stem, 7.0)
        r, m, gid = video_rows(v, gid, name == "train", args)
        allrows.extend(r)
        meta.extend(m)
    arrays = [np.concatenate([r[i] for r in allrows]) for i in range(7)]
    gm = np.asarray(meta, np.int32)
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache / f"{name}.npz",
        features=arrays[0],
        labels=arrays[1],
        supervision=arrays[2],
        groups=arrays[3],
        src=arrays[4],
        tgt=arrays[5],
        registered=arrays[6],
        group_meta=gm,
        feature_names=np.asarray(FEATURES),
    )
    print(name, len(arrays[1]), "positive", int(arrays[1].sum()), "groups", len(gm))


def assignments(cost, reg, groups, src, tgt, meta, args):
    selected = np.zeros(len(cost), bool)
    for gid, n0, n1, _ in meta:
        idx = np.flatnonzero(groups == gid)
        if not len(idx) or not n0 or not n1:
            continue
        used0 = set()
        used1 = set()
        for gate in (args.tight, args.relaxed):
            q = idx[(reg[idx] <= gate) & ~np.isin(src[idx], list(used0)) & ~np.isin(tgt[idx], list(used1))]
            if not len(q):
                continue
            mat = np.full((n0, n1), 1e6)
            mat[src[q], tgt[q]] = cost[q]
            ri, ci = linear_sum_assignment(mat)
            for a, b in zip(ri, ci):
                if mat[a, b] >= 1e6:
                    continue
                hit = q[(src[q] == a) & (tgt[q] == b)]
                if len(hit):
                    selected[hit[0]] = True
                    used0.add(int(a))
                    used1.add(int(b))
    return selected


def score(selected, y, sup, groups, meta):
    tp = int((selected & (y > 0)).sum())
    fp = int((selected & (y == 0) & (sup > 0)).sum())
    hit = np.bincount(groups[selected & (y > 0)], minlength=len(meta))
    fn = int(np.maximum(meta[:, 3] - hit, 0).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "jaccard": tp / max(tp + fp + fn, 1),
    }


@torch.no_grad()
def residuals(model, x, mean, std, args):
    out = []
    model.eval()
    for i in range(0, len(x), args.batch_size):
        xb = torch.from_numpy((x[i : i + args.batch_size] - mean) / std).to(args.device)
        raw = model(xb)
        out.append((args.residual_scale * torch.tanh(raw / args.residual_scale)).cpu().numpy())
    return np.concatenate(out)


def compute_loss(logit, yb, args):
    if args.loss == "bce":
        w = torch.where(yb > 0.5, torch.full_like(yb, args.pos_weight), torch.ones_like(yb))
        return F.binary_cross_entropy_with_logits(logit, yb, weight=w)
    if args.loss == "asymmetric":
        # penalize FN more via pos_weight on BCE, mild focal
        bce = F.binary_cross_entropy_with_logits(logit, yb, reduction="none")
        p = torch.sigmoid(logit)
        pt = p * yb + (1 - p) * (1 - yb)
        w = torch.where(yb > 0.5, torch.full_like(yb, args.pos_weight), torch.ones_like(yb))
        return (((1 - pt) ** args.focal_gamma) * bce * w).mean()
    # focal / focal_soft
    bce = F.binary_cross_entropy_with_logits(logit, yb, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * yb + (1 - p) * (1 - yb)
    gamma = args.focal_gamma if args.loss == "focal" else 1.0
    w = torch.where(yb > 0.5, torch.full_like(yb, args.pos_weight), torch.ones_like(yb))
    return (((1 - pt) ** gamma) * bce * w).mean()


def main():
    args = argspec()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    tr, va = ft.load_split(args.splits, args.fold)
    print(
        f"Motion fold={args.fold} arch={args.arch} feat={args.feat_mode} loss={args.loss} "
        f"parent_mode={args.parent_mode} train={len(tr)} val={len(va)} exp={args.exp_name or '-'}",
        flush=True,
    )
    if args.rebuild_cache or not (args.cache / "train.npz").exists():
        make_cache(tr, "train", args)
        make_cache(va, "val", args)
    a = np.load(args.cache / "train.npz")
    v = np.load(args.cache / "val.npz")
    names = [str(q) for q in a["feature_names"]]
    keep = np.asarray([n not in RUNTIME_DROP for n in names])
    runtime_features = [n for n, k in zip(names, keep) if k]
    x0 = a["features"][:, keep].astype(np.float32)
    xv0 = v["features"][:, keep].astype(np.float32)
    x, feat_names = transform_features(x0, runtime_features, args.feat_mode)
    xv, _ = transform_features(xv0, runtime_features, args.feat_mode)
    y = a["labels"].astype(np.float32)
    yv = v["labels"]
    sv = v["supervision"]
    gv = v["groups"]
    src = v["src"]
    tgt = v["tgt"]
    reg = v["registered"]
    meta = v["group_meta"]
    # base_cost is always column 0 of raw runtime features (before transform)
    base_cost_val = xv0[:, 0]
    mean = x.mean(0)
    std = x.std(0).clip(1e-4)
    print("Features", len(feat_names), "mode", args.feat_mode, flush=True)

    model = build_model(args.arch, x.shape[1], args).to(args.device)
    logit_bias = nn.Parameter(torch.tensor([args.logit_bias], device=args.device)) if args.learn_logit_bias else None
    params = list(model.parameters()) + ([logit_bias] if logit_bias is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    elif args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    args.output.mkdir(parents=True, exist_ok=True)
    base_sel = assignments(base_cost_val, reg, gv, src, tgt, meta, args)
    baseline = score(base_sel, yv, sv, gv, meta)
    best = baseline["jaccard"]
    print("MOTION BASELINE", baseline, flush=True)

    ckpt_meta = {
        "arch": args.arch,
        "feat_mode": args.feat_mode,
        "loss": args.loss,
        "tabm_k": args.tabm_k,
        "dropout": args.dropout,
        "hidden": args.hidden,
        "logit_bias": args.logit_bias,
        "learn_logit_bias": args.learn_logit_bias,
        "exp_name": args.exp_name,
        "tight": args.tight,
        "relaxed": args.relaxed,
        "velocity_weight": args.velocity_weight,
        "parent_mode": args.parent_mode,
        "fold": args.fold,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "features": feat_names,
            "runtime_base_features": runtime_features,
            "residual_scale": args.residual_scale,
            "metrics": baseline,
            "meta": ckpt_meta,
            "logit_bias": float(logit_bias.detach().cpu()) if logit_bias is not None else args.logit_bias,
        },
        args.output / "motion_corrector_best.pt",
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    stale = 0
    hist = []
    mt = torch.from_numpy(mean).to(args.device)
    st = torch.from_numpy(std).to(args.device)

    for ep in range(args.epochs):
        if args.warmup_epochs and ep < args.warmup_epochs:
            warm = (ep + 1) / args.warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = args.lr * warm
        model.train()
        ls = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            res_raw = model((xb - mt) / st)
            res = args.residual_scale * torch.tanh(res_raw / args.residual_scale)
            bias = logit_bias if logit_bias is not None else args.logit_bias
            # xb[:,0] after transform may not be base_cost if mode adds cols first — base is always first col of raw;
            # we keep raw base_cost as first column in all transform modes.
            logit = bias - xb[:, 0] + res
            loss = compute_loss(logit, yb, args)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ls.append(float(loss.detach()))

        corr = residuals(model, xv, mean, std, args)
        sel = assignments(base_cost_val - corr, reg, gv, src, tgt, meta, args)
        m = score(sel, yv, sv, gv, meta)
        row = {"epoch": ep, "loss": float(np.mean(ls)), "seconds": time.time() - t0, **m}
        hist.append(row)
        (args.output / "metrics.json").write_text(json.dumps(hist, indent=2))
        print("Epoch", ep, row, flush=True)

        if m["jaccard"] > best:
            best = m["jaccard"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "features": feat_names,
                    "runtime_base_features": runtime_features,
                    "residual_scale": args.residual_scale,
                    "metrics": m,
                    "meta": ckpt_meta,
                    "logit_bias": float(logit_bias.detach().cpu()) if logit_bias is not None else args.logit_bias,
                },
                args.output / "motion_corrector_best.pt",
            )
            print(" NEW MOTION BEST", best, flush=True)
        else:
            stale += 1

        if sched is not None:
            if args.scheduler == "plateau":
                sched.step(m["jaccard"])
            elif ep >= args.warmup_epochs:
                sched.step()

        if stale >= args.patience:
            print("Early stopping", flush=True)
            break

    summary = {
        "best_jaccard": best,
        "baseline_jaccard": baseline["jaccard"],
        "delta": best - baseline["jaccard"],
        "arch": args.arch,
        "feat_mode": args.feat_mode,
        "loss": args.loss,
        "residual_scale": args.residual_scale,
        "lr": args.lr,
        "exp_name": args.exp_name,
        "fold": args.fold,
        "epochs_ran": len(hist),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done", best, args.output / "motion_corrector_best.pt", flush=True)


if __name__ == "__main__":
    main()
