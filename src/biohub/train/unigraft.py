import json
from pathlib import Path

from biohub.data.tracker import load_video
from biohub.train import decoder as trainer_mod
from biohub.train.cardinality import (
    deployed_baseline,
    load_partition,
    parse_args,
    save_head,
    train_family,
)
from biohub.train.tensorboard import log_scalars, open_writer
from biohub.utils.cli import run_argparse_main


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    writer = open_writer(args.output)
    trainer = trainer_mod
    split = json.loads(args.split.read_text())
    train = load_partition(list(map(str, split['train'])), args, trainer, load_video)
    held = load_partition(list(map(str, split['held'])), args, trainer, load_video)
    families = {
        'c_v2': train_family('UG C+V2', train, held, args, trainer, mode='c_v2'),
        'p1p2_v2': train_family('UG P1P2+V2', train, held, args, trainer, mode='p1p2_v2'),
        'combined': train_family('UG COMBINED', train, held, args, trainer, mode='combined'),
    }
    requested = families['p1p2_v2']
    treatment_config = {
        'source_dim': int(requested['normalizer'].source_mean.shape[0]),
        'pair_dim': int(requested['normalizer'].pair_mean.shape[0]),
        'hidden_source': args.hidden_source,
        'hidden_pair': args.hidden_pair,
        'max_pairs': args.max_pairs,
        'threshold': requested['oof_frozen']['threshold'],
        'evidence_mode': 'v2_geometry_plus_p1p2_without_model_c',
    }
    save_head(
        Path(args.output) / 'p1p2_only_source_cardinality_head.pt',
        requested['model'],
        requested['normalizer'],
        treatment_config,
    )
    for family in families.values():
        family.pop('model')
        family.pop('normalizer')
    summary = {
        'version': 'public934-p1p2-only-cardinality-head-v1',
        'families': families,
        'treatment_config': treatment_config,
        'deployed_baseline': deployed_baseline(held, args, trainer),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    for name, family in families.items():
        log_scalars(
            writer,
            0,
            {f'{name}/oof_jaccard': family['oof_frozen']['jaccard']},
        )
    writer.close()
    print(json.dumps(summary, indent=2, default=str), flush=True)


def train_from_config(cfg: dict) -> None:
    run_argparse_main(main, cfg)


if __name__ == '__main__':
    main()
