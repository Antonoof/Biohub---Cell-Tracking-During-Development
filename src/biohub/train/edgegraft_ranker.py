#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from biohub.train import edgegraft as base
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.utils.seed import seed_everything


def make_model(args, fold: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.iterations,
        max_leaf_nodes=args.leaves,
        l2_regularization=args.l2,
        min_samples_leaf=args.min_samples_leaf,
        max_bins=args.max_bins,
        random_state=args.seed + fold,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('data/edgegraft_current_labels_v2'))
    parser.add_argument('--output', type=Path, default=Path('data/edgegraft_current_ranker_v3'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--leaves', type=int, default=31)
    parser.add_argument('--learning-rate', type=float, default=0.06)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260819)
    parser.add_argument('--min-samples-leaf', type=int, default=40)
    parser.add_argument('--max-bins', type=int, default=255)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite: {args.output}')
    args.output.mkdir(parents=True)

    frame = base.load_labels(args.labels)
    features = base.feature_names(frame)
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y = frame.y.to_numpy(np.int8)
    weights = base.target_weights(frame)
    oof = np.full(len(frame), np.nan, np.float32)
    reports = []
    writer = open_writer(args.output)
    for fold in range(args.folds):
        train = frame.video_fold.to_numpy() != fold
        held = ~train
        model = make_model(args, fold)
        model.fit(x[train], y[train], sample_weight=weights[train])
        joblib.dump(model, args.output / f'fold_{fold}.joblib', compress=3)
        oof[held] = model.predict_proba(x[held])[:, 1]
        report = base.target_metrics(frame.loc[held], oof[held])
        report['fold'] = fold
        reports.append(report)
        log_scalars(
            writer,
            fold,
            {
                'fold/model_accuracy': report.get('model_accuracy', 0.0),
                'fold/baseline_accuracy': report.get('baseline_accuracy', 0.0),
                'fold/net_correct': report.get('net_correct', 0.0),
                'fold/recovered': report.get('recovered', 0.0),
                'fold/lost': report.get('lost', 0.0),
            },
        )
        print(f'fold {fold}: {json.dumps(report, sort_keys=True)}', flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError('OOF scoring incomplete')
    columns = ['dataset', 'video_fold', 'component', 'source', 'target', 'y', 'is_current_parent']
    frame[columns].assign(oof_score=oof).to_parquet(args.output / 'oof_scores.parquet', index=False)
    report = {
        'version': 'edgegraft-current-ranker-v3',
        'features': features,
        'folds': reports,
        'overall': base.target_metrics(frame, oof),
        'model_format': 'sklearn HistGradientBoostingClassifier mechanism-test artifact',
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    overall = report['overall']
    log_scalars(
        writer,
        0,
        {
            'overall/model_accuracy': overall.get('model_accuracy', 0.0),
            'overall/baseline_accuracy': overall.get('baseline_accuracy', 0.0),
            'overall/net_correct': overall.get('net_correct', 0.0),
        },
    )
    writer.close()
    print(json.dumps(report['overall'], indent=2), flush=True)


if __name__ == '__main__':
    main()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)
