# File index

## Root

- `README.md` - entry point and reproduction sequence
- `TRAINING_AND_GRAPH_PROVENANCE.md` - training population and graph lineage
- `VALIDATION.md` - validation definitions and recorded results
- `SHA256SUMS.csv` - generated integrity manifest

## Documentation

- `docs/biohub_production_957_full_technical_architecture.pdf`
- `docs/EXPERIMENTS.md`

## Production

- `production/fork-of-fork-of-division-focused.ipynb` - authoritative inference
  implementation

## Helper source directories

- `helpers/01_p1_p2_base/` - UNet/transformer trainer, P1 artifact manifest,
  P2 snapshot, P2 split, and P2 training configuration
- `helpers/02_model_c/` - Model C division trainers, deployment specs,
  configuration, runtime, and training summary
- `helpers/03_source_cardinality/` - exact source-cardinality trainer,
  extraction/replay scripts, runtime, deploy spec, and validation summaries
- `helpers/04_unigraft_p1p2/` - independent P1/P2 head trainer and training
  summary
- `helpers/05_live_v2/` - runtime-only bundle construction, packaging,
  validation, manifest, and runtime
- `helpers/06_full_population_ownership/` - OOF fitter, packager, deployment
  spec, manifest, and runtime
- `helpers/07_edgegraft/` - parent-ranker trainer, transaction-gate trainer,
  packager, reports, manifest, and all deployed runtime modules
- `helpers/08_candidategraft/` - direct classifier trainer, package builder,
  replay scripts, report, manifest, and runtime
- `helpers/09_motion_corrector/` - exact V1 trainer, split metadata, checkpoint
  inspection, validation metrics, and provenance README
- `helpers/10_deepcenter/` - detector trainer, launch script, packager,
  configuration, split, gate summary, and artifact manifest

All copied source files are preserved byte-for-byte. The friendly filenames in
this index are implemented as directory placement; the original filenames are
retained unless a shorter handoff name is required to distinguish its role.

