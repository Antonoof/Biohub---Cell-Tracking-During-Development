import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

from biohub.modules.ownership import FEATURES
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main
from biohub.utils.seed import seed_everything
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


if list(FEATURES) != [*TOP, *RANK]:
    raise RuntimeError('ownership FEATURES must stay TOP then RANK')


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


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
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def choose_threshold(
    scored: pd.DataFrame, override: float | None
) -> tuple[float, list[dict[str, float | int]]]:
    frame = scored.copy()
    frame['known_target'] = frame.source_label.eq(1) & frame.best_pair_label.eq(1)
    known_mask = frame.source_label.notna().to_numpy(bool)
    target = frame.known_target.to_numpy(bool)
    unknown_mask = ~known_mask
    scores = frame.score.to_numpy(np.float64)
    positive_scores = np.sort(frame.loc[frame.known_target, 'score'].to_numpy(np.float64))
    thresholds: set[float] = {0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10}
    if len(positive_scores):
        for recall in (0.50, 0.65, 0.75, 0.85, 0.95, 1.00):
            needed = max(1, int(np.ceil(recall * len(positive_scores))))
            thresholds.add(float(positive_scores[-needed]))
    records: list[dict[str, float | int]] = []
    for threshold in sorted(thresholds, reverse=True):
        selected = scores >= threshold
        tp = int(np.sum(selected & target))
        fp = int(np.sum(selected & known_mask & ~target))
        fn = int(np.sum(target & ~selected))
        records.append(
            {
                'threshold': threshold,
                'selected_sources': int(selected.sum()),
                'selected_videos': int(frame.loc[selected, 'dataset'].nunique()),
                'known_tp': tp,
                'known_fp': fp,
                'known_fn': fn,
                'known_recall': tp / max(tp + fn, 1),
                'known_precision': tp / max(tp + fp, 1),
                'known_jaccard': tp / max(tp + fp + fn, 1),
                'unknown_selected': int(np.sum(selected & unknown_mask)),
            }
        )
    if override is not None:
        return float(override), records
    sweep = pd.DataFrame(records).sort_values(
        ['known_jaccard', 'threshold'],
        ascending=[False, False],
    )
    return float(sweep.iloc[0]['threshold']), records


def export_serving_artifact(
    output: Path,
    models: list[ExtraTreesClassifier],
    scored: pd.DataFrame,
    fold_rows: list[dict[str, int]],
    args: argparse.Namespace,
) -> dict:
    model_dir = output / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_records = []
    for fold, model in enumerate(models):
        model_path = model_dir / f'ownership_fold_{fold}.joblib'
        joblib.dump(model, model_path, compress=3)
        record = dict(fold_rows[fold])
        record['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        fold_records.append(record)
    threshold, sweep_rows = choose_threshold(scored, args.threshold)
    pd.DataFrame(sweep_rows).to_csv(output / 'threshold_sweep.csv', index=False)
    spec = {
        'version': 'full-population-ownership-v1',
        'status': 'candidate',
        'features': list(FEATURES),
        'threshold': threshold,
        'folds': int(len(models)),
        'model': {
            'type': 'sklearn ExtraTreesClassifier',
            'n_estimators': int(args.n_estimators),
            'max_depth': int(args.max_depth),
            'min_samples_leaf': int(args.min_samples_leaf),
            'max_features': float(args.max_features),
            'class_weight': 'balanced',
            'seed_base': int(args.seed),
        },
        'serving': {
            'fit': 'known-only grouped-video folds on train175',
            'unknown_sources_used_as_negatives': False,
            'hidden_model_route': 'sha256(dataset) modulo 5; one unseen fold model',
            'winner': 'highest candidate per source above frozen OOF threshold',
        },
        'versions': {
            'python': sys.version,
            'sklearn': sklearn.__version__,
            'joblib': joblib.__version__,
        },
        'fold_records': fold_records,
    }
    (output / 'deploy_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


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
    seed_everything(int(args.seed), deterministic=bool(args.deterministic))
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
    spec = export_serving_artifact(args.output, models, scored, fold_rows, args)
    payload = {
        'version': 'ownership-open-recall-oof-v3',
        'rows': int(len(scored)),
        'videos': int(scored.dataset.nunique()),
        'folds': fold_rows,
        'threshold': spec['threshold'],
    }
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    log_scalars(
        writer,
        0,
        {
            'summary/rows': payload['rows'],
            'summary/videos': payload['videos'],
            'summary/threshold': spec['threshold'],
        },
    )
    writer.close()


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)


if __name__ == '__main__':
    main()
