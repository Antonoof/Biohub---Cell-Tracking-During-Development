#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main

BIO = Path('data')
DIRECT_FEATURES = [
    'p1',
    'p2',
    'pmax',
    'pmean',
    'present',
    'winner_count',
    'rank_min',
    'distance',
]
LEAF_FEATURES = DIRECT_FEATURES + [
    'nearest_final_same_frame_um',
    'density10',
    'is_missing_target',
]
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--population',
        type=Path,
        default=BIO / 'native_endpoint_candidate_graft_v2',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=BIO / 'native_endpoint_candidate_graft_screen_v1',
    )
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def fold_for(dataset: str, folds: int) -> int:
    digest = hashlib.sha1(dataset.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little') % folds


def matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def make_model(
    seed: int,
    positives: int,
    negatives: int,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    max_leaf_nodes: int = 15,
    l2_regularization: float = 2.0,
) -> HistGradientBoostingClassifier:
    minimum = max(4, min(16, (positives + negatives) // 20))
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=minimum,
        l2_regularization=l2_regularization,
        random_state=seed,
    )


def threshold_rows(frame: pd.DataFrame, score: np.ndarray) -> list[dict[str, object]]:
    known = frame.label_known.to_numpy(bool)
    positive = frame.label_positive.fillna(False).to_numpy(bool)
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        selected = score >= threshold
        tp = int(np.sum(selected & known & positive))
        fp = int(np.sum(selected & known & ~positive))
        known_positive = int(np.sum(known & positive))
        rows.append(
            {
                'threshold': threshold,
                'selected_all': int(selected.sum()),
                'selected_videos': int(frame.loc[selected, 'dataset'].nunique()),
                'known_tp': tp,
                'known_fp': fp,
                'known_precision': tp / max(tp + fp, 1),
                'known_recall': tp / max(known_positive, 1),
            }
        )
    return rows


def crossfit(
    frame: pd.DataFrame, features: list[str], folds: int, seed: int
) -> tuple[np.ndarray, dict]:
    frame = frame.copy()
    frame['fold'] = frame.dataset.map(lambda value: fold_for(str(value), folds))
    known = frame.label_known.to_numpy(bool)
    labels = frame.label_positive.fillna(False).to_numpy(np.int8)
    score = np.zeros(len(frame), np.float32)
    fold_rows = []
    for fold in range(folds):
        fit = known & (frame.fold.to_numpy() != fold)
        validation = frame.fold.to_numpy() == fold
        y = labels[fit]
        if len(np.unique(y)) < 2:
            raise RuntimeError(f'Fold {fold} training labels contain one class: {np.bincount(y)}')
        model = make_model(seed + fold, int(y.sum()), int((1 - y).sum()))
        model.fit(matrix(frame.loc[fit], features), y)
        score[validation] = model.predict_proba(matrix(frame.loc[validation], features))[:, 1]
        fold_rows.append(
            {
                'fold': fold,
                'fit_known': int(fit.sum()),
                'fit_positive': int(y.sum()),
                'validation_rows': int(validation.sum()),
                'validation_known': int((validation & known).sum()),
            }
        )
    known_score = score[known]
    known_label = labels[known]
    summary = {
        'rows': len(frame),
        'known': int(known.sum()),
        'positive': int(known_label.sum()),
        'negative': int(len(known_label) - known_label.sum()),
        'roc_auc': float(roc_auc_score(known_label, known_score)),
        'average_precision': float(average_precision_score(known_label, known_score)),
        'folds': fold_rows,
        'thresholds': threshold_rows(frame, score),
    }
    return score, summary


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite: {args.output}')
    args.output.mkdir(parents=True)
    writer = open_writer(args.output)
    direct = pd.read_parquet(args.population / 'direct_edges.parquet')
    leaf = pd.read_parquet(args.population / 'endpoint_leaves.parquet')
    leaf['is_missing_target'] = (leaf.transaction_type == 'missing_target').astype(np.float32)
    direct_score, direct_summary = crossfit(direct, DIRECT_FEATURES, args.folds, args.seed)
    leaf_score, leaf_summary = crossfit(leaf, LEAF_FEATURES, args.folds, args.seed + 100)
    direct['oof_score'] = direct_score
    leaf['oof_score'] = leaf_score
    direct.to_parquet(args.output / 'direct_edges_scored.parquet', index=False)
    leaf.to_parquet(args.output / 'endpoint_leaves_scored.parquet', index=False)
    summary = {
        'version': 'native-endpoint-candidate-graft-screen-v1',
        'contract': 'label-free candidate construction; known-only grouped-video OOF labels',
        'direct': direct_summary,
        'leaf': leaf_summary,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, value in (('direct', direct_summary), ('leaf', leaf_summary)):
        print(f'{name}: AUC={value["roc_auc"]:.5f} AP={value["average_precision"]:.5f}')
        print(pd.DataFrame(value['thresholds']).to_string(index=False))
        log_scalars(
            writer,
            0,
            {
                f'{name}/roc_auc': value['roc_auc'],
                f'{name}/average_precision': value['average_precision'],
            },
        )
    writer.close()
    print(f'Wrote {args.output}')


def screen_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)


def fit_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--population',
        type=Path,
        default=BIO / 'native_endpoint_candidate_graft_v2/direct_edges.parquet',
    )
    parser.add_argument(
        '--oof-report',
        type=Path,
        default=BIO / 'native_endpoint_candidate_graft_screen_v1/summary.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=BIO / 'candidategraft_direct_v1',
    )
    parser.add_argument('--threshold', type=float, default=0.90)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-iter', type=int, default=250)
    parser.add_argument('--max-leaf-nodes', type=int, default=15)
    parser.add_argument('--l2', type=float, default=2.0)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def fit_main() -> None:
    args = fit_parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite: {args.output}')
    args.output.mkdir(parents=True)
    frame = pd.read_parquet(args.population)
    known = frame.label_known.to_numpy(bool)
    labels = frame.label_positive.fillna(False).to_numpy(np.int8)
    fit = frame.loc[known].copy()
    y = labels[known]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError('Known fitting rows must contain both classes')
    model = make_model(
        args.seed,
        positives,
        negatives,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2,
    )
    model.fit(matrix(fit, DIRECT_FEATURES), y)
    joblib.dump(model, args.output / 'candidategraft_direct.joblib', compress=3)
    writer = open_writer(args.output)
    log_scalars(
        writer,
        0,
        {
            'fit/rows': len(fit),
            'fit/positive': positives,
            'fit/negative': negatives,
            'fit/threshold': args.threshold,
        },
    )
    writer.close()
    oof = json.loads(args.oof_report.read_text())
    report = {
        'version': 'candidategraft-direct-v1',
        'contract': (
            'label-free full native P1/P2 candidate population; final fit uses '
            'annotated known rows only; unannotated rows are never negatives'
        ),
        'features': DIRECT_FEATURES,
        'threshold': float(args.threshold),
        'fit_rows': int(len(fit)),
        'fit_positive': positives,
        'fit_negative': negatives,
        'population_rows': int(len(frame)),
        'oof': oof['direct'],
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    fit_main()


def train_from_config(cfg: dict) -> None:
    if 'oof_report' in cfg:
        run_argparse_main(fit_main, cfg)
        return
    run_argparse_main(main, cfg)
