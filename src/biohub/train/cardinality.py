#!/usr/bin/env python3

import argparse
import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import torch

from biohub.data.decoder import VideoRows, load_graph_coordinates
from biohub.data.decoder import load_video as load_decoder_video
from biohub.data.tracker import load_video
from biohub.models.option_head import OptionHead
from biohub.train import decoder as trainer_mod
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.validation.cv import EMBRYOS, movie_group_kfold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--raw-data', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--full-population-cache', type=Path, required=True)
    p.add_argument('--decoder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--mine-epoch', type=int, default=30)
    p.add_argument('--negative-ratio', type=int, default=12)
    p.add_argument('--hard-negative-ratio', type=int, default=24)
    p.add_argument(
        '--max-pairs',
        type=int,
        default=128,
        help='Stable geometry-ordered per-source option cap, shared by train and inference.',
    )
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--hidden-source', type=int, default=8)
    p.add_argument('--hidden-pair', type=int, default=12)
    p.add_argument('--learning-rate', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=2037)
    p.add_argument('--deterministic', action='store_true')
    return p.parse_args()


@dataclass(frozen=True)
class SourceRef:
    video: int
    source: int


@dataclass
class PreparedVideo:
    video: VideoRows
    pair_features: np.ndarray
    pair_rows: list[np.ndarray]


@dataclass
class Normalizer:
    source_mean: np.ndarray
    source_scale: np.ndarray
    pair_mean: np.ndarray
    pair_scale: np.ndarray

    def source(self, value: np.ndarray) -> np.ndarray:
        return np.clip((value - self.source_mean) / self.source_scale, -10.0, 10.0)

    def pair(self, value: np.ndarray) -> np.ndarray:
        return np.clip((value - self.pair_mean) / self.pair_scale, -10.0, 10.0)


def evidence_root(stem: str, roots: tuple[Path, ...]) -> Path:
    matches = [root for root in roots if (root / f'{stem}.npz').exists()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one Model-C evidence file for {stem}, found {len(matches)}')
    return matches[0]


def exact_source_labels(video, cache: dict, graph_coordinates: dict, truth) -> np.ndarray:
    source_t = np.asarray(
        [graph_coordinates.get(int(node), (-1, None))[0] for node in video.source_node],
        np.int32,
    )
    divider_time = {
        int(event['event_id']): int(truth.times[truth.row_of_id[int(event['gt_id'])]])
        for event in cache['event_rows']
    }
    result = np.zeros(len(video.source_y), np.int8)
    for event, frame in divider_time.items():
        result[(video.source_event == event) & (source_t == frame)] = 1
    return result


def load_partition(names: list[str], args, trainer, raw_loader) -> list[VideoRows]:
    event_cache = args.root / 'division_official_event_cache_public914_model_c_latent_v1_parts'
    graph_dir = args.root / 'division_candidate_audit_v1' / 'pre_safe_graphs'
    c_roots = (
        args.root / 'model_c_native_evidence_train175',
        args.root / 'model_c_native_evidence_held20',
        args.root / 'model_c_native_evidence_practice4',
    )
    load_args = SimpleNamespace(
        event_cache=event_cache,
        full_population_cache=args.full_population_cache,
        graph_dir=graph_dir,
        public_primary_evidence=args.root / 'public_primary_native_evidence_all199_v1',
        public_secondary_evidence=args.root / 'public_secondary_native_evidence_all199_v1',
        ctc_evidence=None,
    )
    result = []
    exact_rows = original_rows = events = exact_missing = 0
    for index, stem in enumerate(names, 1):
        cache = torch.load(event_cache / f'{stem}.pt', map_location='cpu', weights_only=False)
        video = load_decoder_video(stem, load_args, evidence_root(stem, c_roots))
        graph_coordinates = load_graph_coordinates(graph_dir / f'{stem}.geff')
        truth = raw_loader(args.raw_data, stem).graph
        labels = exact_source_labels(video, cache, graph_coordinates, truth)
        original_rows += int(np.count_nonzero(video.source_y == 1))
        exact_rows += int(labels.sum())
        events += int(video.event_total)
        for event in cache['event_rows']:
            if not np.any((video.source_event == int(event['event_id'])) & (labels == 1)):
                exact_missing += 1
        result.append(dataclasses.replace(video, source_y=labels))
        if index % 25 == 0 or index == len(names):
            print(f'  loaded {index}/{len(names)}', flush=True)
    print(
        f'  exact labels: events={events} original_positive_rows={original_rows} '
        f'exact_positive_rows={exact_rows} events_without_exact_source={exact_missing}',
        flush=True,
    )
    return result


def rows_by_source(owner: np.ndarray, n_sources: int) -> list[np.ndarray]:
    result = [np.empty(0, np.int64) for _ in range(n_sources)]
    order = np.argsort(owner, kind='stable')
    if not len(order):
        return result
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        result[int(sorted_owner[left])] = order[left:right]
    return result


def prepare(
    videos: list[VideoRows],
    use_public: bool = False,
    mode: str | None = None,
) -> list[PreparedVideo]:
    if mode is None:
        mode = 'combined' if use_public else 'c_v2'
    result = []
    for video in videos:
        blocks = [video.pair_x]
        if mode in ('c_v2', 'combined'):
            blocks.append(video.c_pair_x)
        if mode in ('p1p2_v2', 'combined') or use_public:
            blocks.append(video.public_pair_x)
        pair_features = np.nan_to_num(
            np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        result.append(
            PreparedVideo(
                video=video,
                pair_features=pair_features,
                pair_rows=rows_by_source(video.pair_owner, len(video.source_x)),
            )
        )
    return result


def source_refs(videos: list[PreparedVideo]) -> tuple[list[SourceRef], list[SourceRef]]:
    positive, negative = [], []
    for vindex, prepared in enumerate(videos):
        video = prepared.video
        for source, rows in enumerate(prepared.pair_rows):
            if not len(rows):
                continue
            ref = SourceRef(vindex, source)
            if video.source_y[source] == 1:
                if np.any(video.pair_y[rows] == 1):
                    positive.append(ref)
            else:
                negative.append(ref)
    return positive, negative


def selected_pair_rows(
    prepared: PreparedVideo, source: int, max_pairs: int, training: bool
) -> np.ndarray:
    rows = prepared.pair_rows[source]
    if len(rows) <= max_pairs:
        return rows
    selected = rows[:max_pairs]
    if training and prepared.video.source_y[source] == 1:
        positive = rows[prepared.video.pair_y[rows] == 1]
        selected = np.unique(np.r_[selected, positive])
    return selected


def fit_normalizer(videos: list[PreparedVideo]) -> Normalizer:
    source = np.concatenate([v.video.source_x for v in videos], axis=0).astype(np.float64)
    pair = np.concatenate([v.pair_features for v in videos], axis=0).astype(np.float64)
    source_mean = source.mean(axis=0)
    pair_mean = pair.mean(axis=0)
    source_scale = source.std(axis=0)
    pair_scale = pair.std(axis=0)
    source_scale[source_scale < 1e-4] = 1.0
    pair_scale[pair_scale < 1e-4] = 1.0
    return Normalizer(
        source_mean.astype(np.float32),
        source_scale.astype(np.float32),
        pair_mean.astype(np.float32),
        pair_scale.astype(np.float32),
    )


def collate(
    refs: list[SourceRef],
    videos: list[PreparedVideo],
    normalizer: Normalizer,
    max_pairs: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
    chosen = [selected_pair_rows(videos[r.video], r.source, max_pairs, training) for r in refs]
    width = max(len(rows) for rows in chosen)
    source_x = np.stack([normalizer.source(videos[r.video].video.source_x[r.source]) for r in refs])
    pair_dim = videos[refs[0].video].pair_features.shape[1]
    pair_x = np.zeros((len(refs), width, pair_dim), np.float32)
    mask = np.zeros((len(refs), width), bool)
    positive_mask = np.zeros((len(refs), width), bool)
    is_division = np.zeros(len(refs), bool)
    for row, (ref, pair_rows) in enumerate(zip(refs, chosen)):
        prepared = videos[ref.video]
        pair_x[row, : len(pair_rows)] = normalizer.pair(prepared.pair_features[pair_rows])
        mask[row, : len(pair_rows)] = True
        positive_mask[row, : len(pair_rows)] = prepared.video.pair_y[pair_rows] == 1
        is_division[row] = prepared.video.source_y[ref.source] == 1
    return (
        torch.from_numpy(source_x),
        torch.from_numpy(pair_x),
        torch.from_numpy(mask),
        torch.from_numpy(positive_mask),
        torch.from_numpy(is_division),
        chosen,
    )


def grouped_loss(
    continue_logit: torch.Tensor,
    pair_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    is_division: torch.Tensor,
) -> torch.Tensor:
    denominator = torch.logsumexp(torch.cat([continue_logit[:, None], pair_logits], dim=1), dim=1)
    target = continue_logit.clone()
    if bool(is_division.any()):
        positive_logits = pair_logits.masked_fill(~positive_mask, -1e9)
        target[is_division] = torch.logsumexp(positive_logits[is_division], dim=1)
    return (denominator - target).mean()


@torch.no_grad()
def score_refs(
    model: OptionHead,
    refs: list[SourceRef],
    videos: list[PreparedVideo],
    normalizer: Normalizer,
    args,
) -> np.ndarray:
    model.eval()
    result = np.zeros(len(refs), np.float32)
    for left in range(0, len(refs), args.batch_size):
        batch_refs = refs[left : left + args.batch_size]
        sx, px, mask, _, _, _ = collate(batch_refs, videos, normalizer, args.max_pairs, False)
        device = next(model.parameters()).device
        sx, px, mask = sx.to(device), px.to(device), mask.to(device)
        cont, pair = model(sx, px, mask)
        pair_total = torch.logsumexp(pair, dim=1)
        result[left : left + len(batch_refs)] = torch.sigmoid(pair_total - cont).cpu().numpy()
    return result


def fit_model(videos: list[PreparedVideo], args, seed: int) -> tuple[OptionHead, Normalizer, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    positive, negative = source_refs(videos)
    if not positive or not negative:
        raise RuntimeError(
            f'Invalid training references: positive={len(positive)} negative={len(negative)}'
        )
    normalizer = fit_normalizer(videos)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptionHead(
        videos[0].video.source_x.shape[1],
        videos[0].pair_features.shape[1],
        args.hidden_source,
        args.hidden_pair,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng: Any = np.random.default_rng(seed)
    hard_pool: list[SourceRef] = []
    history = []
    for epoch in range(args.epochs):
        n_negative = min(len(negative), len(positive) * args.negative_ratio)
        if hard_pool and epoch >= args.mine_epoch:
            n_hard = min(len(hard_pool), int(round(n_negative * 0.75)))
            n_random = n_negative - n_hard
            sampled = list(rng.choice(hard_pool, n_hard, replace=False)) + list(
                rng.choice(negative, n_random, replace=False)
            )
        else:
            sampled = list(rng.choice(negative, n_negative, replace=False))
        epoch_refs = positive + sampled
        rng.shuffle(epoch_refs)
        model.train()
        losses = []
        for left in range(0, len(epoch_refs), args.batch_size):
            batch_refs = epoch_refs[left : left + args.batch_size]
            sx, px, mask, positive_mask, is_division, _ = collate(
                batch_refs, videos, normalizer, args.max_pairs, True
            )
            sx, px, mask, positive_mask, is_division = (
                sx.to(device),
                px.to(device),
                mask.to(device),
                positive_mask.to(device),
                is_division.to(device),
            )
            cont, pair = model(sx, px, mask)
            loss = grouped_loss(cont, pair, positive_mask, is_division)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch + 1 == args.mine_epoch:
            scores = score_refs(model, negative, videos, normalizer, args)
            count = min(len(negative), len(positive) * args.hard_negative_ratio)
            order = np.argsort(scores)[-count:]
            hard_pool = [negative[int(index)] for index in order]
            print(f'    mined {len(hard_pool)} safe hard continuations', flush=True)
        row = {'epoch': epoch, 'loss': float(np.mean(losses))}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.mine_epoch:
            print(f'    epoch {epoch:02d}: loss={row["loss"]:.5f}', flush=True)
    details = {
        'positive_sources': len(positive),
        'confirmed_continuation_sources': len(negative),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'history': history,
    }
    return model, normalizer, details


@torch.no_grad()
def score_videos(model, normalizer, videos: list[PreparedVideo], args):
    model.eval()
    scored = []
    for prepared in videos:
        video = prepared.video
        scores = np.zeros(len(video.source_x), np.float32)
        best = np.full(len(video.source_x), -1, np.int64)
        refs = [SourceRef(0, source) for source, rows in enumerate(prepared.pair_rows) if len(rows)]
        for left in range(0, len(refs), args.batch_size):
            batch_refs = refs[left : left + args.batch_size]
            sx, px, mask, _, _, chosen = collate(
                batch_refs, [prepared], normalizer, args.max_pairs, False
            )
            device = next(model.parameters()).device
            sx, px, mask = sx.to(device), px.to(device), mask.to(device)
            cont, pair = model(sx, px, mask)
            probability = torch.sigmoid(torch.logsumexp(pair, dim=1) - cont).cpu().numpy()
            pair_choice = torch.argmax(pair, dim=1).cpu().numpy()
            for offset, ref in enumerate(batch_refs):
                scores[ref.source] = probability[offset]
                best[ref.source] = int(chosen[offset][int(pair_choice[offset])])
        pair_label = np.zeros(len(video.source_x), np.int8)
        valid = best >= 0
        pair_label[valid] = video.pair_y[best[valid]]
        scored.append((video, scores, best, pair_label))
    return scored


def save_head(path: Path, model: OptionHead, normalizer: Normalizer, config: dict) -> None:
    torch.save(
        {
            'state_dict': model.state_dict(),
            'source_mean': normalizer.source_mean,
            'source_scale': normalizer.source_scale,
            'pair_mean': normalizer.pair_mean,
            'pair_scale': normalizer.pair_scale,
            'config': config,
        },
        path,
    )


def train_family(
    label: str,
    raw_train: list[VideoRows],
    raw_held: list[VideoRows],
    args,
    trainer,
    use_public: bool = False,
    mode: str | None = None,
):
    train = prepare(raw_train, use_public=use_public, mode=mode)
    held = prepare(raw_held, use_public=use_public, mode=mode)
    groups = np.asarray([video.stem for video in raw_train])
    oof = []
    fold_details = []
    writer = open_writer(args.output)
    for fold, (fit_index, val_index) in enumerate(movie_group_kfold(groups, args.folds)):
        fit = [train[int(index)] for index in fit_index]
        val = [train[int(index)] for index in val_index]
        print(f'{label} fold {fold}: fit={len(fit)} val={len(val)}', flush=True)
        model, normalizer, details = fit_model(fit, args, args.seed + fold)
        oof.extend(score_videos(model, normalizer, val, args))
        fold_details.append(details)
        log_scalars(
            writer,
            fold,
            {f'{label}/fit_videos': len(fit), f'{label}/val_videos': len(val)},
        )
    frozen = trainer.threshold_sweep(oof)
    print(f'{label} GROUPED OOF {json.dumps(frozen)}', flush=True)
    final_model, final_normalizer, final_details = fit_model(train, args, args.seed + 100)
    held_scored = score_videos(final_model, final_normalizer, held, args)
    held_frozen = trainer.metric(held_scored, frozen['threshold'])
    held_diagnostic = trainer.threshold_sweep(held_scored)
    by_embryo = {
        embryo: trainer.metric(
            [row for row in held_scored if row[0].embryo == embryo], frozen['threshold']
        )
        for embryo in EMBRYOS
    }
    print(f'{label} HELD FROZEN {json.dumps(held_frozen)}', flush=True)
    log_scalars(
        writer,
        0,
        {
            f'{label}/oof_jaccard': frozen['jaccard'],
            f'{label}/held_jaccard': held_frozen['jaccard'],
        },
    )
    writer.close()
    return {
        'label': label,
        'use_public_p1_p2_evidence': use_public,
        'oof_frozen': frozen,
        'held_frozen': held_frozen,
        'held_diagnostic_best_do_not_deploy': held_diagnostic,
        'held_by_embryo': by_embryo,
        'fold_details': fold_details,
        'final_details': final_details,
        'model': final_model,
        'normalizer': final_normalizer,
    }


def deployed_baseline(raw_held: list[VideoRows], args, trainer) -> dict:
    pair = joblib.load(args.decoder / 'pair_model.joblib')
    source = joblib.load(args.decoder / 'source_model.joblib')
    scored = trainer.score_videos(raw_held, pair, source, use_c=True)
    return {
        'frozen_0_96': trainer.metric(scored, 0.96),
        'held_diagnostic_best_do_not_deploy': trainer.threshold_sweep(scored),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = trainer_mod
    split = json.loads(args.split.read_text())
    print('Loading exact-frame train-175', flush=True)
    train = load_partition(list(map(str, split['train'])), args, trainer, load_video)
    print('Loading untouched held-20', flush=True)
    held = load_partition(list(map(str, split['held'])), args, trainer, load_video)
    baseline = deployed_baseline(held, args, trainer)
    print(f'EXACT-LABEL DEPLOYED BASELINE {json.dumps(baseline)}', flush=True)
    control = train_family('CARDINALITY C+V2', train, held, args, trainer, use_public=False)
    public = train_family('CARDINALITY C+V2+P1P2', train, held, args, trainer, use_public=True)

    promoted = bool(
        public['oof_frozen']['jaccard'] > control['oof_frozen']['jaccard']
        and public['held_frozen']['jaccard'] > control['held_frozen']['jaccard']
        and public['held_frozen']['jaccard'] > baseline['frozen_0_96']['jaccard']
    )
    chosen = public if promoted else control
    config = {
        'source_dim': int(chosen['normalizer'].source_mean.shape[0]),
        'pair_dim': int(chosen['normalizer'].pair_mean.shape[0]),
        'hidden_source': args.hidden_source,
        'hidden_pair': args.hidden_pair,
        'max_pairs': args.max_pairs,
        'threshold': chosen['oof_frozen']['threshold'],
        'use_public_p1_p2_evidence': chosen['use_public_p1_p2_evidence'],
    }
    save_head(
        args.output / 'source_cardinality_head.pt', chosen['model'], chosen['normalizer'], config
    )
    for family in (control, public):
        family.pop('model')
        family.pop('normalizer')
    summary = {
        'version': 'public934-source-cardinality-head-v2',
        'contract': {
            'options': (
                'one CONTINUE plus all retained DIVIDE(daughter_a,daughter_b) pairs per source'
            ),
            'loss': (
                'source-local grouped softmax; multiple correct pairs use positive-set logsumexp'
            ),
            'division_time_label': (
                'exact GT parent frame only; nearby matched rows are not positives'
            ),
            'unknown_policy': 'unannotated sources receive zero supervised loss',
            'null_policy': 'no NULL daughter class exists',
            'threshold_policy': 'frozen from grouped train-video OOF before held-20',
        },
        'deployed_decoder_exact_label_baseline': baseline,
        'control': control,
        'public_p1_p2_variant': public,
        'public_variant_promoted': promoted,
        'saved_family': chosen['label'],
        'saved_config': config,
        'practice_evaluated': False,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'DONE output={args.output} public_variant_promoted={promoted}', flush=True)


if __name__ == '__main__':
    main()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)
