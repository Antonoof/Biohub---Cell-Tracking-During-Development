import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.validation.cv import movie_group_kfold

RANK = [
    'source_frame_percentile',
    'source_frame_log_margin',
    'v2_frame_percentile',
    'v2_frame_log_margin',
    'agreement4',
    'agreement5',
    'time_norm',
]
TOP = [
    'steal_distance_delta_um',
    'alternate_minus_current_um',
    'incumbent_edge_prob',
    'alternate_parent_um',
    'steal_distance_ratio',
    'incumbent_back_depth',
    'predicted_midpoint_error_um',
]


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[*TOP, *RANK]].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=324459)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--min-samples-leaf', type=int, default=20)
    parser.add_argument('--max-features', type=float, default=0.75)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def make_model(args, fold: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight='balanced',
        n_jobs=args.n_jobs,
        random_state=args.seed + fold,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite: {args.output}')
    args.output.mkdir(parents=True)
    full = pd.read_parquet(args.bank)
    train = full.loc[full.panel.eq('train175')].copy()
    known = train.loc[train.source_label.notna()].copy().reset_index(drop=True)
    known['target'] = (known.source_label.eq(1) & known.best_pair_label.eq(1)).astype(np.int8)
    x_known = matrix(known)
    y = known.target.to_numpy(np.int8)
    groups = known.dataset.astype(str).to_numpy()
    splits = movie_group_kfold(groups, args.folds)
    keep = [
        'dataset',
        'panel',
        'source',
        'a',
        'b',
        'current_child',
        'source_label',
        'best_pair_label',
        'source_score',
        'v2_source_score_raw',
        'agreement4',
        'agreement5',
        'source_time',
    ]
    parts: list[pd.DataFrame] = []
    models: list[ExtraTreesClassifier] = []
    fold_rows: list[dict[str, int]] = []
    writer = open_writer(args.output)
    for fold, (fit, held) in enumerate(splits):
        model = make_model(args, fold)
        model.fit(x_known[fit], y[fit])
        models.append(model)
        held_videos = set(known.iloc[held].dataset.astype(str).unique())
        local = train.loc[train.dataset.astype(str).isin(held_videos), keep].copy()
        local['score'] = model.predict_proba(matrix(train.loc[local.index]))[:, 1]
        local['fold'] = np.int8(fold)
        parts.append(local)
        fold_rows.append(
            {
                'fold': fold,
                'fit_known_rows': int(len(fit)),
                'held_known_rows': int(len(held)),
                'held_videos': int(len(held_videos)),
                'full_rows_scored': int(len(local)),
            }
        )
        log_scalars(
            writer,
            fold,
            {
                'fold/fit_known_rows': len(fit),
                'fold/held_known_rows': len(held),
                'fold/held_videos': len(held_videos),
                'fold/full_rows_scored': len(local),
            },
        )
    covered_videos = {str(value) for part in parts for value in part.dataset.unique()}
    all_videos = set(train.dataset.astype(str).unique())
    unlabeled_only_videos = sorted(all_videos - covered_videos)
    if unlabeled_only_videos:
        local = train.loc[train.dataset.astype(str).isin(unlabeled_only_videos), keep].copy()
        x_local = matrix(train.loc[local.index])
        local['score'] = np.mean([model.predict_proba(x_local)[:, 1] for model in models], axis=0)
        local['fold'] = np.int8(-1)
        parts.append(local)
    scored = pd.concat(parts, ignore_index=True)
    scored.to_parquet(args.output / 'all_source_scores.parquet', index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / 'folds.csv', index=False)
    payload = {
        'version': 'ownership-open-recall-oof-v3',
        'rows': int(len(scored)),
        'videos': int(scored.dataset.nunique()),
        'folds': fold_rows,
    }
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    log_scalars(writer, 0, {'summary/rows': payload['rows'], 'summary/videos': payload['videos']})
    writer.close()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)


if __name__ == '__main__':
    main()
