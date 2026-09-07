#!/usr/bin/env python3

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset

from biohub.features.motion import MOTION_TRAIN_FEATURES as FEATURES
from biohub.features.motion import RUNTIME_DROP
from biohub.infer.config import GraphConfig
from biohub.models.motion import MotionResidual
from biohub.modules.graph.motion import motion_relink_edges
from biohub.paths import PROJECT_ROOT
from biohub.train import motion_cache as ft
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.utils.parallel import ordered_process_map
from biohub.utils.seed import dataloader_generator, seed_everything
from biohub.utils.yaml_config import load_yaml

CACHE_GENERATOR_VERSION = 'proposal_sha256_v1'


def argspec():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data/train'))
    p.add_argument(
        '--proposals', type=Path, default=Path('data/ab_proposals_export/biohub_ab_proposals')
    )
    p.add_argument('--splits', type=Path, default=Path('data/splits_ensembleB.json'))
    p.add_argument('--cache', type=Path, default=Path('data/motion_cost_cache'))
    p.add_argument('--output', type=Path, default=Path('runs/weights/motion_cost_corrector'))
    p.add_argument('--tight', type=float, default=6.2)
    p.add_argument('--relaxed', type=float, default=9.5)
    p.add_argument('--velocity-weight', type=float, default=0.52)
    p.add_argument('--residual-scale', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--negative-ratio', type=int, default=20)
    p.add_argument('--seed', type=int, default=2028)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


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
    parent = {int(d): int(s) for s, d in v.gt_edges}
    rows = []
    group_meta = []
    gid = group_start
    pred_prev = None
    for t in range(v.image_shape_raw[0] - 1):
        a0, a1 = int(v.offsets[t]), int(v.offsets[t + 1])
        b0, b1 = int(v.offsets[t + 1]), int(v.offsets[t + 2])
        n0, n1 = a1 - a0, b1 - b0
        group_meta.append((gid, n0, n1, len(v.edges_by_t.get(t, ()))))
        if not n0 or not n1:
            pred_prev = None
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
        if train:
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
        elif pred_prev is not None and len(pred_prev) == n0:
            valid = np.isfinite(pred_prev).all(axis=1)
            vel[valid] = c0[valid] - pred_prev[valid] - prev_shift
            has[valid] = 1
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
        if not train:
            pred_prev = np.full((n1, 3), np.nan)
            if n0 and n1 and candidate.any():
                cost = np.full((n0, n1), 1e9)
                cost[candidate] = base_cost[candidate]
                row_ind, col_ind = linear_sum_assignment(cost)
                for source_i, target_j in zip(row_ind, col_ind):
                    if candidate[source_i, target_j]:
                        pred_prev[target_j] = c0[source_i]
        gid += 1
    return rows, group_meta, gid


def rollout_graph(video) -> dict:
    nodes_by_id = {}
    for index in range(len(video.coords)):
        t, z, y, x = (float(value) for value in video.coords[index])
        nodes_by_id[int(index)] = {'t': int(t), 'z': z, 'y': y, 'x': x}
    return {
        'nodes_by_id': nodes_by_id,
        'matches': np.asarray(video.matches),
        'gt_edges': [(int(src), int(dst)) for src, dst in video.gt_edges],
    }


def _motion_video_cache(item):
    stem, name, args = item
    video = ft.load_proposal_video(args.data, args.proposals, stem, 7.0)
    rows, meta, _gid = video_rows(video, 0, name == 'train', args)
    payload = rollout_graph(video) if name != 'train' else None
    return rows, meta, payload


def make_cache(stems, name, args):
    limited = stems[: args.max_videos or None]
    results = ordered_process_map(_motion_video_cache, [(stem, name, args) for stem in limited])
    allrows = []
    meta = []
    gid = 0
    rollouts = []
    for rows, group_meta, payload in results:
        for item in group_meta:
            meta.append((int(item[0]) + gid, item[1], item[2], item[3]))
        for row in rows:
            allrows.append((row[0], row[1], row[2], row[3] + gid, row[4], row[5], row[6]))
        gid += len(group_meta)
        if payload is not None:
            rollouts.append(payload)
    arrays = [np.concatenate([r[i] for r in allrows]) for i in range(7)]
    gm = np.asarray(meta, np.int32)
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache / f'{name}.npz',
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
    print(name, len(arrays[1]), 'positive', int(arrays[1].sum()), 'groups', len(gm))
    manifest_path = args.cache / f'{name}_manifest.json'
    manifest_path.write_text(
        json.dumps(cache_manifest(limited, name, args), indent=2, sort_keys=True) + '\n'
    )
    if name != 'train':
        (args.cache / f'{name}_rollout.pkl').write_bytes(pickle.dumps(rollouts))


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ''
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def cache_manifest(stems, name, args) -> dict:
    proposals = Path(args.proposals)
    return {
        'name': name,
        'stems': [str(stem) for stem in stems],
        'proposals': str(proposals.resolve()),
        'data': str(Path(args.data).resolve()),
        'tight': float(args.tight),
        'relaxed': float(args.relaxed),
        'velocity_weight': float(args.velocity_weight),
        'negative_ratio': int(args.negative_ratio),
        'feature_names': list(FEATURES),
        'history': 'gt_parent' if name == 'train' else 'hungarian_base_cost',
        'generator_version': CACHE_GENERATOR_VERSION,
        'proposal_sha256': {str(stem): _file_sha256(proposals / f'{stem}.npz') for stem in stems},
    }


def assert_cache_manifest(stems, name, args) -> None:
    expected = cache_manifest(stems, name, args)
    path = Path(args.cache) / f'{name}_manifest.json'
    if not path.is_file():
        raise RuntimeError(f'Motion cache manifest missing: {path}; pass --rebuild-cache')
    actual = json.loads(path.read_text())
    if actual != expected:
        raise RuntimeError(
            f'Motion cache fingerprint mismatch for {path.name}; pass --rebuild-cache'
        )
    if name != 'train' and not (Path(args.cache) / f'{name}_rollout.pkl').is_file():
        raise RuntimeError(
            f'Motion rollout cache missing: {name}_rollout.pkl; pass --rebuild-cache'
        )


def serving_motion_runtime(corrector=None, *, config_path: Path | None = None):
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / 'configs' / 'infer.yaml'
    raw = load_yaml(path)
    graph = GraphConfig.model_validate(raw['graph'])
    voxel = tuple(float(value) for value in raw['voxel_scale_um'])
    strength = float(raw.get('models', {}).get('motion_corrector_strength', 1.0))
    axes = np.asarray(
        graph.relink_velocity_weight_axes or [graph.relink_velocity_weight] * 3,
        dtype=float,
    )
    bundle = None
    if corrector is not None:
        source = corrector['model']
        replica = MotionResidual(int(source.net[0].in_features))
        replica.load_state_dict(
            {key: value.detach().cpu() for key, value in source.state_dict().items()}
        )
        replica.eval()
        bundle = {
            'model': replica,
            'mean': torch.as_tensor(corrector['mean'], dtype=torch.float32).cpu(),
            'std': torch.as_tensor(corrector['std'], dtype=torch.float32).cpu(),
            'residual_scale': float(corrector['residual_scale']),
        }
    return SimpleNamespace(
        motion_relink=True,
        relink_max_frame_nodes=int(graph.relink_max_frame_nodes),
        relink_velocity_axes=axes,
        relink_motion_steps=int(graph.relink_motion_steps),
        relink_frame_registration=bool(graph.relink_frame_registration),
        relink_relaxed_um=float(graph.relink_relaxed_um),
        relink_registration_weight=float(graph.relink_registration_weight),
        relink_corrector_one_step=bool(graph.relink_corrector_uses_one_step),
        z_extent_um=float(graph.motion_feature_z_extent_um),
        relink_raw_weight=float(graph.relink_raw_distance_weight),
        relink_learned_bonus=float(graph.relink_learned_bonus),
        motion_corrector=bundle,
        motion_corrector_strength=strength,
        relink_tight_um=float(graph.relink_tight_um),
        relink_max_match_cost=float(graph.relink_max_match_cost),
        relink_cap_includes_learned=bool(graph.relink_cost_cap_includes_learned),
        relink_orphan_prior=bool(graph.relink_orphan_prior),
        relink_orphan_base_um=float(graph.relink_orphan_base_um),
        relink_orphan_floor=float(graph.relink_orphan_floor),
        relink_orphan_scale=float(graph.relink_orphan_scale),
        relink_joint_assignment=bool(graph.relink_joint_assignment),
        relink_tight_bonus_um=float(graph.relink_tight_bonus_um),
        scale=np.asarray(voxel, dtype=np.float64),
        voxel_scale_um=voxel,
    )


def score_motion_rollout(payloads, corrector=None, *, config_path: Path | None = None) -> dict:
    upgrade = serving_motion_runtime(corrector, config_path=config_path)
    tp = fp = fn = 0
    for payload in payloads:
        stats: dict = defaultdict(int)
        selected = motion_relink_edges(upgrade, payload['nodes_by_id'], stats)
        pred = set()
        matches = np.asarray(payload['matches'])
        for edge in selected:
            src = int(matches[int(edge['source_id'])])
            tgt = int(matches[int(edge['target_id'])])
            if src >= 0 and tgt >= 0:
                pred.add((src, tgt))
        gt = {(int(src), int(dst)) for src, dst in payload['gt_edges']}
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'jaccard': tp / max(tp + fp + fn, 1),
    }


def assignments(cost, reg, groups, src, tgt, meta, args):
    selected = np.zeros(len(cost), bool)
    for gid, n0, n1, _ in meta:
        idx = np.flatnonzero(groups == gid)
        if not len(idx) or not n0 or not n1:
            continue
        used0 = set()
        used1 = set()
        for gate in (args.tight, args.relaxed):
            q = idx[
                (reg[idx] <= gate)
                & ~np.isin(src[idx], list(used0))
                & ~np.isin(tgt[idx], list(used1))
            ]
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
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'jaccard': tp / max(tp + fp + fn, 1),
    }


@torch.no_grad()
def residuals(model, x, mean, std, args):
    out = []
    model.eval()
    for i in range(0, len(x), args.batch_size):
        out.append(
            (
                args.residual_scale
                * torch.tanh(
                    model(
                        torch.from_numpy((x[i : i + args.batch_size] - mean) / std).to(args.device)
                    )
                    / args.residual_scale
                )
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(out)


def main():
    args = argspec()
    seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    args.device = str(torch.device(args.device if torch.cuda.is_available() else 'cpu'))
    tr, va = ft.load_split(args.splits, 0)
    train_stems = tr[: args.max_videos or None]
    val_stems = va[: args.max_videos or None]
    if args.rebuild_cache or not (args.cache / 'train.npz').exists():
        make_cache(tr, 'train', args)
        make_cache(va, 'val', args)
    else:
        assert_cache_manifest(train_stems, 'train', args)
        assert_cache_manifest(val_stems, 'val', args)
    a = np.load(args.cache / 'train.npz')
    v = np.load(args.cache / 'val.npz')
    rollout_path = args.cache / 'val_rollout.pkl'
    if not rollout_path.is_file():
        raise RuntimeError(f'Motion rollout cache missing: {rollout_path}; pass --rebuild-cache')
    rollout_payloads = pickle.loads(rollout_path.read_bytes())
    names = [str(q) for q in a['feature_names']]
    keep = np.asarray([n not in RUNTIME_DROP for n in names])
    runtime_features = [n for n, k in zip(names, keep) if k]
    x = a['features'][:, keep].astype(np.float32)
    y = a['labels'].astype(np.float32)
    xv = v['features'][:, keep].astype(np.float32)
    yv = v['labels']
    sv = v['supervision']
    gv = v['groups']
    src = v['src']
    tgt = v['tgt']
    reg = v['registered']
    meta = v['group_meta']
    mean = x.mean(0)
    std = x.std(0).clip(1e-4)
    print('Runtime-safe features', len(runtime_features), runtime_features)
    model = MotionResidual(x.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    writer = open_writer(args.output)
    base_sel = assignments(xv[:, 0], reg, gv, src, tgt, meta, args)
    diagnostic = score(base_sel, yv, sv, gv, meta)
    baseline_corrector = {
        'model': model,
        'mean': mean,
        'std': std,
        'residual_scale': args.residual_scale,
    }
    baseline = score_motion_rollout(rollout_payloads, baseline_corrector)
    best = baseline['jaccard']
    print('MOTION DIAGNOSTIC', diagnostic)
    print('MOTION ROLLOUT BASELINE', baseline)
    torch.save(
        {
            'model': model.state_dict(),
            'mean': mean,
            'std': std,
            'features': runtime_features,
            'residual_scale': args.residual_scale,
            'metrics': baseline,
            'diagnostic': diagnostic,
        },
        args.output / 'motion_corrector_best.pt',
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        generator=dataloader_generator(args.seed),
    )
    stale = 0
    hist = []
    mt = torch.from_numpy(mean).to(args.device)
    st = torch.from_numpy(std).to(args.device)
    for ep in range(args.epochs):
        model.train()
        ls = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            res = args.residual_scale * torch.tanh(model((xb - mt) / st) / args.residual_scale)
            logit = 2.5 - xb[:, 0] + res
            bce = F.binary_cross_entropy_with_logits(logit, yb, reduction='none')
            p = torch.sigmoid(logit)
            pt = p * yb + (1 - p) * (1 - yb)
            loss = (((1 - pt) ** 2) * bce).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            ls.append(float(loss.detach()))
        corr = residuals(model, xv, mean, std, args)
        selected = assignments(xv[:, 0] - corr, reg, gv, src, tgt, meta, args)
        diagnostic = score(selected, yv, sv, gv, meta)
        rollout = score_motion_rollout(
            rollout_payloads,
            {
                'model': model,
                'mean': mean,
                'std': std,
                'residual_scale': args.residual_scale,
            },
        )
        row = {
            'epoch': ep,
            'loss': float(np.mean(ls)),
            'seconds': time.time() - t0,
            'diagnostic_jaccard': diagnostic['jaccard'],
            **rollout,
        }
        hist.append(row)
        log_scalars(
            writer,
            ep,
            {
                'train/loss': row['loss'],
                'val/jaccard': row['jaccard'],
                'val/diagnostic_jaccard': diagnostic['jaccard'],
                'val/precision': row.get('precision', 0.0),
                'val/recall': row.get('recall', 0.0),
            },
        )
        (args.output / 'metrics.json').write_text(json.dumps(hist, indent=2))
        print('Epoch', ep, row)
        if row['jaccard'] > best:
            best = row['jaccard']
            stale = 0
            torch.save(
                {
                    'model': model.state_dict(),
                    'mean': mean,
                    'std': std,
                    'features': runtime_features,
                    'residual_scale': args.residual_scale,
                    'metrics': rollout,
                    'diagnostic': diagnostic,
                },
                args.output / 'motion_corrector_best.pt',
            )
            print(' NEW MOTION BEST', best)
        else:
            stale += 1
        if stale >= args.patience:
            print('Early stopping')
            break
    writer.close()
    print('Done', best, args.output / 'motion_corrector_best.pt')


if __name__ == '__main__':
    main()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)
