import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from biohub.data.motion import (
    ProposalWindowDataset,
    load_proposal_video,
)
from biohub.models.temporal_unet import TemporalUNet3D
from biohub.train import detector as base
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument(
        '--proposals', type=Path, required=True, help='Extracted biohub_ab_proposals directory'
    )
    p.add_argument('--splits', type=Path, default=Path('data/splits_ensembleB.json'))
    p.add_argument('--fold', type=int, default=0)
    p.add_argument(
        '--init-weights', type=Path, required=True, help='Existing model B edge_predictor_best.pth'
    )
    p.add_argument('--init-config', type=Path, default=None)
    p.add_argument(
        '--anchor-weights', type=Path, required=True, help='Frozen model A edge_predictor_best.pth'
    )
    p.add_argument('--anchor-config', type=Path, default=None)
    p.add_argument(
        '--output', type=Path, default=Path('runs/weights/unet_transformer_b2_proposals/split_1')
    )
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps-per-epoch', type=int, default=750)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--min-lr', type=float, default=2e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--match-um', type=float, default=7.0)
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--division-positive-weight', type=float, default=1.0)
    p.add_argument('--anchor-weight', type=float, default=1.0)
    p.add_argument('--trainable-weight', type=float, default=1.0)
    p.add_argument('--edge-threshold', type=float, default=0.54)
    p.add_argument('--soft-jaccard-weight', type=float, default=0.001)
    p.add_argument('--anchor-error-weight', type=float, default=1.5)
    p.add_argument('--disagreement-weight', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--max-nodes', type=int, default=2800)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=('fp16', 'bf16', 'off'), default='fp16')
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path, fold: int):
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        obj = obj[fold]
    elif 'folds' in obj:
        obj = obj['folds'][fold]
    elif str(fold) in obj:
        obj = obj[str(fold)]
    train = [Path(x).stem for x in obj['train']]
    val = [Path(x).stem for x in obj.get('test', obj.get('val', []))]
    if not train or not val:
        raise ValueError('Split must contain nonempty train and test/val lists')
    return train, val


def collate(batch):
    B = len(batch)
    m0 = max(len(x['coords0']) for x in batch)
    m1 = max(len(x['coords1']) for x in batch)
    coords0 = torch.zeros(B, m0, 3)
    coords1 = torch.zeros(B, m1, 3)
    mask0 = torch.zeros(B, m0, dtype=torch.bool)
    mask1 = torch.zeros(B, m1, dtype=torch.bool)
    target = torch.zeros(B, m0, m1)
    supervision = torch.zeros(B, m0, m1, dtype=torch.bool)
    det0 = torch.zeros(B, m0)
    det1 = torch.zeros(B, m1)
    member_det0 = torch.zeros(B, m0, 2)
    member_det1 = torch.zeros(B, m1, 2)
    for b, x in enumerate(batch):
        n0, n1 = len(x['coords0']), len(x['coords1'])
        coords0[b, :n0] = x['coords0']
        coords1[b, :n1] = x['coords1']
        mask0[b, :n0] = True
        mask1[b, :n1] = True
        target[b, :n0, :n1] = x['target']
        supervision[b, :n0, :n1] = x['supervision']
        det0[b, :n0] = x['det0']
        det1[b, :n1] = x['det1']
        member_det0[b, :n0] = x['member_det0']
        member_det1[b, :n1] = x['member_det1']
    return {
        'imgs': torch.stack([x['imgs'] for x in batch]),
        'coords0': coords0,
        'coords1': coords1,
        'mask0': mask0,
        'mask1': mask1,
        'target': target,
        'supervision': supervision,
        'det0': det0,
        'det1': det1,
        'member_det0': member_det0,
        'member_det1': member_det1,
        'downsample': torch.stack([x['downsample'] for x in batch]),
        'voxel_scale': torch.stack([x['voxel_scale'] for x in batch]),
        'image_shape': torch.stack([x['image_shape'] for x in batch]),
        'gt_edges_total': torch.stack([x['gt_edges_total'] for x in batch]),
        'frozen': torch.stack([x['frozen'] for x in batch]),
        'video_idx': torch.stack([x['video_idx'] for x in batch]),
        'frame': torch.stack([x['frame'] for x in batch]),
    }


def forward_logits(model, batch, device):
    imgs = batch['imgs'].to(device, dtype=torch.float32, non_blocking=True)
    c0 = batch['coords0'].to(device, non_blocking=True)
    c1 = batch['coords1'].to(device, non_blocking=True)
    m0 = batch['mask0'].to(device, non_blocking=True)
    m1 = batch['mask1'].to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        features, _ = model.encode(imgs.float())
        f0 = model.index_features(features[:, 0], c0, m0)
        f1 = model.index_features(features[:, 1], c1, m1)
    B, M0 = c0.shape[:2]
    M1 = c1.shape[1]
    t0 = torch.zeros(B, M0, 1, device=device)
    t1 = torch.ones(B, M1, 1, device=device)
    shape = tuple(int(x) for x in batch['image_shape'][0].tolist())
    p0 = base.pos_embed_torch(torch.cat([t0, c0], -1), (2, *shape[1:]))
    p1 = base.pos_embed_torch(torch.cat([t1, c1], -1), (2, *shape[1:]))
    ds = batch['downsample'][:, None, :].to(device)
    return model.predict_edges(f0, f1, c0 * ds, c1 * ds, p0, p1, m0, m1)


def fused_edge_loss(logits_a, logits_b, target, supervision, mask0, mask1, args):
    focal_losses, jaccard_losses = [], []
    denom = args.anchor_weight + args.trainable_weight
    if denom <= 0:
        raise ValueError('Fusion weights must sum to a positive value')
    for b in range(len(logits_b)):
        n0, n1 = int(mask0[b].sum()), int(mask1[b].sum())
        la = logits_a[b, :n0, :n1].float()
        lb = logits_b[b, :n0, :n1].float()
        pa = torch.softmax(la, dim=0)
        pb = torch.softmax(lb, dim=0)
        prob = torch.softmax(
            (args.anchor_weight * la + args.trainable_weight * lb) / denom,
            dim=0,
        ).clamp(1e-7, 1 - 1e-7)
        y = target[b, :n0, :n1]
        sup = supervision[b, :n0, :n1]
        if not sup.any():
            continue
        y = y.float()
        bce = -(y * prob.log() + (1 - y) * torch.log1p(-prob))
        pt = prob * y + (1 - prob) * (1 - y)
        weight = torch.ones_like(y)
        div_rows = y.sum(1) > 1
        weight[div_rows] *= args.division_positive_weight
        weight *= 1.0 + args.anchor_error_weight * (pa - y).abs().detach()
        weight *= 1.0 + args.disagreement_weight * (pa - pb).abs().detach()
        focal_losses.append(((((1 - pt) ** args.focal_gamma) * bce * weight)[sup]).mean())
        ps, ys = prob[sup], y[sup]
        inter = (ps * ys).sum()
        union = (ps + ys - ps * ys).sum()
        jaccard_losses.append(1.0 - (inter + 1e-6) / (union + 1e-6))
    if not focal_losses:
        zero = logits_b.sum() * 0
        return zero, zero.detach(), zero.detach()
    focal = torch.stack(focal_losses).mean()
    soft_j = torch.stack(jaccard_losses).mean()
    return focal + args.soft_jaccard_weight * soft_j, focal.detach(), soft_j.detach()


@torch.no_grad()
def evaluate(anchor, model, loader, device, amp_dtype, args):
    anchor.eval()
    model.eval()
    model.unet.eval()
    model.detect_head.eval()
    thresholds = tuple(sorted(set((0.4, 0.5, args.edge_threshold, 0.6))))
    counts = {t: [0, 0, 0] for t in thresholds}
    losses = []
    focals = []
    soft_js = []
    denom = args.anchor_weight + args.trainable_weight
    for batch in loader:
        target = batch['target'].to(device)
        sup = batch['supervision'].to(device)
        m0 = batch['mask0'].to(device)
        m1 = batch['mask1'].to(device)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            logits_a = forward_logits(anchor, batch, device)
            logits_b = forward_logits(model, batch, device)
            loss, focal, soft_j = fused_edge_loss(logits_a, logits_b, target, sup, m0, m1, args)
        losses.append(float(loss))
        focals.append(float(focal))
        soft_js.append(float(soft_j))
        for b in range(len(logits_b)):
            n0, n1 = int(m0[b].sum()), int(m1[b].sum())
            y = target[b, :n0, :n1].bool()
            s = sup[b, :n0, :n1]
            la = logits_a[b, :n0, :n1].float()
            lb = logits_b[b, :n0, :n1].float()
            prob = torch.softmax(
                (args.anchor_weight * la + args.trainable_weight * lb) / denom, dim=0
            )
            for th in thresholds:
                pred = prob >= th
                c = counts[th]
                tp = int((pred & y & s).sum())
                fp = int((pred & ~y & s).sum())
                c[0] += tp
                c[1] += fp
                c[2] += max(0, int(batch['gt_edges_total'][b]) - tp)
    rows = {}
    for th, (tp, fp, fn) in counts.items():
        rows[th] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'jaccard': tp / max(tp + fp + fn, 1),
        }
    grid_best = max(rows, key=lambda t: rows[t]['jaccard'])
    primary = min(rows, key=lambda t: abs(t - args.edge_threshold))
    return {
        'loss': float(np.mean(losses)),
        'focal': float(np.mean(focals)),
        'soft_jaccard_loss': float(np.mean(soft_js)),
        'best_threshold': primary,
        'best': rows[primary],
        'grid_best_threshold': grid_best,
        'grid_best': rows[grid_best],
        'thresholds': rows,
    }


def build_model(config, weights, device):
    unet = TemporalUNet3D(
        in_channels=1, out_channels=config['unet_out_channels'], layers=config['unet_layers']
    )
    model = base.UNetNodeTransformer(unet, config['unet_out_channels'], 4 * base.POS_EMBED_DIM).to(
        device
    )
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    for p in model.unet.parameters():
        p.requires_grad = False
    for p in model.detect_head.parameters():
        p.requires_grad = False
    return model


def main():
    args = parse_args()
    seed_all(args.seed)
    train_stems, val_stems = load_split(args.splits, args.fold)
    print(f'Loading proposals: {len(train_stems)} train / {len(val_stems)} val')
    train_v = [
        load_proposal_video(args.data, args.proposals, s, args.match_um)
        for s in tqdm(train_stems, desc='train metadata')
    ]
    val_v = [
        load_proposal_video(args.data, args.proposals, s, args.match_um)
        for s in tqdm(val_stems, desc='val metadata')
    ]
    train_ds = ProposalWindowDataset(
        train_v, args.max_nodes, True, args.steps_per_epoch, args.batch_size, args.seed
    )
    val_ds = ProposalWindowDataset(
        val_v, args.max_nodes, False, None, args.batch_size, args.seed + 1
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    default = {
        'unet_out_channels': 32,
        'unet_layers': [32, 64, 128],
        'downsample': [1, 4, 4],
        'window_size': 2,
        'pool_kernel_um': 5.0,
    }
    cfg_path = args.init_config or args.init_weights.parent / 'config.json'
    config = {**default, **(json.loads(cfg_path.read_text()) if cfg_path.exists() else {})}
    anchor_cfg_path = args.anchor_config or args.anchor_weights.parent / 'config.json'
    anchor_config = {
        **default,
        **(json.loads(anchor_cfg_path.read_text()) if anchor_cfg_path.exists() else {}),
    }
    if any(
        anchor_config[k] != config[k]
        for k in ('unet_out_channels', 'unet_layers', 'window_size', 'downsample')
    ):
        raise ValueError('Anchor A and trainable B architectures/configs must match')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    anchor = build_model(anchor_config, args.anchor_weights, device)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    model = build_model(config, args.init_weights, device)
    params = [p for p in model.transformer.parameters() if p.requires_grad]
    print(
        f'Frozen anchor A + trainable B transformer parameters: {sum(p.numel() for p in params):,}'
    )
    print(
        f'Fusion A:B={args.anchor_weight:g}:{args.trainable_weight:g}; '
        f'selection threshold={args.edge_threshold:g}'
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp == 'fp16')
    amp = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'off': None}[args.amp]
    start = 0
    best = -1.0
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = ck['epoch'] + 1
        best = ck['best_jaccard']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'training_args.json').write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    if args.dry_run:
        batch = next(iter(train_loader))
        with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
            with torch.no_grad():
                logits_a = forward_logits(anchor, batch, device)
            logits_b = forward_logits(model, batch, device)
            loss, focal, soft_j = fused_edge_loss(
                logits_a,
                logits_b,
                batch['target'].to(device),
                batch['supervision'].to(device),
                batch['mask0'].to(device),
                batch['mask1'].to(device),
                args,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        print(
            'dry-run',
            tuple(logits_b.shape),
            'loss',
            float(loss.detach()),
            'focal',
            float(focal),
            'softJ',
            float(soft_j),
            'backward_ok',
            True,
            'grad_norm',
            grad_norm,
        )
        return

    baseline = evaluate(anchor, model, val_loader, device, amp, args)
    best = (
        baseline['best']['jaccard'] if not args.resume else max(best, baseline['best']['jaccard'])
    )
    base_row = {'epoch': -1, 'train_loss': None, 'lr': 0.0, 'seconds': 0.0, **baseline}
    history = [base_row]
    (args.output / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))
    state = {k.replace('unet.module.', 'unet.', 1): v for k, v in model.state_dict().items()}
    torch.save(state, args.output / 'edge_predictor_best.pth')
    torch.save(state, args.output / 'edge_predictor_baseline.pth')
    print(
        f'BASELINE A+B: J={best:.4f} P={baseline["best"]["precision"]:.4f} '
        f'R={baseline["best"]["recall"]:.4f} th={args.edge_threshold:g} '
        f'grid_best={baseline["grid_best_threshold"]}',
        flush=True,
    )
    stale = 0
    writer = open_writer(args.output)
    for epoch in range(start, args.epochs):
        ratio = args.min_lr / args.lr
        factor = ratio + (1 - ratio) * 0.5 * (
            1 + math.cos(math.pi * epoch / max(args.epochs - 1, 1))
        )
        for g in opt.param_groups:
            g['lr'] = args.lr * factor
        model.train()
        model.unet.eval()
        model.detect_head.eval()
        anchor.eval()
        losses = []
        focals = []
        soft_js = []
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            target = batch['target'].to(device)
            sup = batch['supervision'].to(device)
            m0 = batch['mask0'].to(device)
            m1 = batch['mask1'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
                with torch.no_grad():
                    logits_a = forward_logits(anchor, batch, device)
                logits_b = forward_logits(model, batch, device)
                loss, focal, soft_j = fused_edge_loss(logits_a, logits_b, target, sup, m0, m1, args)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            focals.append(float(focal))
            soft_js.append(float(soft_j))
            if step % 25 == 0:
                print(
                    f' epoch {epoch:02d} step {step:04d}/{len(train_loader)} '
                    f'loss={np.mean(losses[-25:]):.6f} '
                    f'focal={np.mean(focals[-25:]):.6f} '
                    f'softJ={np.mean(soft_js[-25:]):.4f}',
                    flush=True,
                )
        metrics = evaluate(anchor, model, val_loader, device, amp, args)
        j = metrics['best']['jaccard']
        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'train_focal': float(np.mean(focals)),
            'train_soft_jaccard_loss': float(np.mean(soft_js)),
            'lr': opt.param_groups[0]['lr'],
            'seconds': time.time() - t0,
            **metrics,
        }
        history.append(row)
        log_scalars(
            writer,
            epoch,
            {
                'train/loss': row['train_loss'],
                'val/loss': metrics['loss'],
                'val/jaccard': j,
                'val/precision': metrics['best']['precision'],
                'val/recall': metrics['best']['recall'],
                'train/lr': row['lr'],
            },
        )
        (args.output / 'metrics.json').write_text(json.dumps(history, indent=2))
        state = {k.replace('unet.module.', 'unet.', 1): v for k, v in model.state_dict().items()}
        ck = {
            'epoch': epoch,
            'model': state,
            'optimizer': opt.state_dict(),
            'best_jaccard': max(best, j),
            'metrics': metrics,
        }
        torch.save(ck, args.output / 'last_training_state.pt')
        torch.save(state, args.output / 'edge_predictor_last.pth')
        print(
            f'Epoch {epoch:02d}: train={row["train_loss"]:.6f} val={metrics["loss"]:.6f} '
            f'J@{args.edge_threshold:g}={j:.4f} P={metrics["best"]["precision"]:.4f} '
            f'R={metrics["best"]["recall"]:.4f} grid_best={metrics["grid_best_threshold"]} '
            f'time={row["seconds"] / 60:.1f}m',
            flush=True,
        )
        if j > best:
            best = j
            stale = 0
            torch.save(ck, args.output / 'best_training_state.pt')
            torch.save(state, args.output / 'edge_predictor_best.pth')
            print(f' NEW FUSED BEST {best:.4f}', flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(
                    f'Early stopping after {stale} epochs without fused-J improvement', flush=True
                )
                break
    writer.close()
    print(
        'Done; best fused proposal edge Jaccard',
        best,
        '->',
        args.output / 'edge_predictor_best.pth',
    )


if __name__ == '__main__':
    main()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)
