# Training guide

This guide lists the canonical trainer for each learned component. Commands are
templates and assume the portable layout in `DATA_LAYOUT.md`. Run them from the
repository root.

The retained P1/P2 source uses a `src/` layout without packaging metadata. Add
it to the import path once per shell before running the commands below:

```bash
export PYTHONPATH="$PWD/helpers/01_p1_p2_base/shared_repo/src:$PYTHONPATH"
```

PowerShell equivalent:

```powershell
$env:PYTHONPATH = "$(Resolve-Path helpers/01_p1_p2_base/shared_repo/src);$env:PYTHONPATH"
```

## 1. P1/P2 image and association models

```bash
python helpers/01_p1_p2_base/shared_repo/scripts/train_unet_transformer.py --help
```

P1 and P2 share the same architecture and trainer. Their saved configurations,
split information where available, and P2 history are under
`helpers/01_p1_p2_base/`.

## 2. Model C division evidence

```bash
python helpers/02_model_c/train_division_pair_model.py \
  --repo external/bio_track_repo \
  --data data/train \
  --division-audit data/division_candidate_audit_v1 \
  --cache data/division_training_cache_v1.npz \
  --output artifacts/model_c_division_pair
```

Fit the combined Model C plus V2 decoder after its event and evidence caches
have been materialized:

```bash
python helpers/02_model_c/train_model_c_v2_event_decoder.py \
  --event-cache data/division_official_event_cache_v2_parts \
  --full-population-cache data/division_v3_full_population_cache \
  --graph-dir data/division_candidate_audit_v1/pre_safe_graphs \
  --split helpers/02_model_c/division_balanced_175_20_split.json \
  --output artifacts/model_c_v2_event_decoder
```

## 3. UniGRAFT source cardinality

```bash
python helpers/03_source_cardinality/train_source_cardinality_head.py \
  --trainer helpers/02_model_c/train_model_c_v2_event_decoder.py \
  --root data/public914_backbone_matched_v1 \
  --raw-data data/train \
  --split helpers/03_source_cardinality/division_balanced_175_20_split.json \
  --full-population-cache data/division_v3_full_population_cache \
  --decoder artifacts/model_c_v2_event_decoder \
  --output artifacts/source_cardinality
```

The head performs source-local choice among CONTINUE and DIVIDE options. Five
folds are grouped by movie. Unknown sources receive no supervised loss.

## 4. Independent P1/P2 UniGRAFT branch

```bash
cd helpers/04_unigraft_p1p2
python train_p1p2_only_cardinality_head.py
cd ../..
```

This trainer keeps the source/pair population fixed while comparing Model C,
native P1/P2, and combined evidence families. Thresholds are frozen from
grouped OOF predictions before held-movie evaluation.

## 5. Full-population ownership

```bash
python helpers/06_full_population_ownership/build_ug12_ownership_geometry_bank_v2.py
python helpers/06_full_population_ownership/score_and_fit_ownership_oof.py \
  --bank data/ug12_ownership_geometry_bank_v2/ownership_geometry.parquet \
  --output artifacts/ownership_oof
```

Only graph-matched known rows enter the ExtraTrees fit. The full unknown
population is scored for serving but is not negative supervision.

## 6. EdgeGRAFT

EdgeGRAFT has two fitted stages: a parent ranker and a transaction gate. Run
from its training directory because the preserved scripts import companion
modules there.

```bash
cd helpers/07_edgegraft/training
python train_edgegraft_current_ranker_v3.py \
  --labels ../../../data/edgegraft_current_labels_v2 \
  --output ../../../artifacts/edgegraft_parent_ranker
python train_edgegraft_v3_metric_transaction_gate.py \
  --decisions ../../../data/edgegraft_v2_rich_oof_decisions \
  --baseline ../../../data/ug23_boundary_oof_exact_v1/final_candidate \
  --data ../../../data/train \
  --output ../../../artifacts/edgegraft_transaction_gate
cd ../../..
```

The transaction gate is fitted only on non-neutral labeled replacements.
Existing fork neighborhoods are protected at serving time.

## 7. CandidateGRAFT

```bash
cd helpers/08_candidategraft/training
python train_candidategraft_direct_v1.py \
  --population ../../../data/native_endpoint_candidate_graft_v2/direct_edges.parquet \
  --oof-report ../../../data/native_endpoint_candidate_graft_screen_v1/summary.json \
  --output ../../../artifacts/candidategraft_direct
cd ../../..
```

CandidateGRAFT is add-only: it links an unmatched source to an unmatched target
and never creates a fork.

## 8. Motion Corrector

```bash
cd helpers/09_motion_corrector/TRAINING_V1
python train_motion_cost_corrector.py \
  --data ../../../data/train \
  --proposals ../../../data/ab_proposals_export/biohub_ab_proposals \
  --splits splits_ensembleB.json \
  --cache ../../../data/motion_cost_cache \
  --output ../../../artifacts/motion_cost_corrector \
  --rebuild-cache
cd ../../..
```

The model learns a bounded residual added to the geometric assignment cost. It
does not directly emit graph edges.

## 9. DeepCenter

```bash
python helpers/10_deepcenter/train_full_frame_center_detector.py \
  --data-dir data/train \
  --output-dir artifacts/deepcenter
```

DeepCenter is a center-likelihood confirmation model. Production does not use
it as an unrestricted node generator.

## Validation after training

For every learned helper:

1. preserve complete movie grouping;
2. write OOF predictions before threshold selection;
3. freeze the threshold and feature order;
4. evaluate once on held movies;
5. replay the complete graph at the correct serving position;
6. validate topology and record artifact hashes.

See `VALIDATION.md` for retained measurements and
`TRAINING_AND_GRAPH_PROVENANCE.md` for the exact graph identity of each model.

