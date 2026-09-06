# Training and graph provenance

This file answers two questions for every production attachment:

1. How was the learned component trained or fitted?
2. What graph, proposal population, or image population supplied its training
   rows?

## 1. P1 inference support package

Dataset: `pilkwang/biohub-tracking-support-pack-50ep-v1`

Role: primary 3D node detector and adjacent-frame association model. It also
ships the inference repository used by the notebook.

Training code: `helpers/01_p1_p2_base/train_unet_transformer.py`.

Training population: Biohub fluorescence volumes and GEFF tracking labels. The
model is trained directly from images and labeled temporal associations; it is
not trained on a post-processed production graph.

Architecture record: UNet channels `32/64/128`, output channels `32`, downsample
`1/4/4`, node-transformer window `2`, and `5.0 um` pooling.

Provenance caution: the current package name contains `50ep`, while the local
manifest that reproduces its exact checkpoint hash identifies the preserved
checkpoint as a 400-epoch snapshot package. The production notebook resolves
the checkpoint by SHA-256, so the hash is authoritative. The repackaged P1
record does not retain a complete train/validation movie split.

Serving graph: P1 supplies the primary detection and association population
from which the production ILP graph is built.

## 2. P2 independent temporal package

Dataset: `pilkwang/biohub-temporal-unet3d-seed314159-v1`

Role: independent seed of the P1 architecture. It contributes detection fusion,
low-margin continuation consensus, bidirectional evidence, and downstream
division/ownership features.

Training code: `helpers/01_p1_p2_base/train_unet_transformer.py`.

Training population: all 199 Biohub training movies according to the preserved
split manifest. Base and effective seed are both `314159`. The saved snapshot
ran through epoch 400; the selected best checkpoint is epoch 381.

Serving graph: P2 does not own a separate final graph. Its evidence is fused
into the P1 candidate population before the ILP and is retained for UniGRAFT,
EdgeGRAFT, and CandidateGRAFT.

## 3. Model C combined division package

Dataset: `tweakai/biohub-model-c-combined-primary-v1`

Role: independent morphology and association evidence for division candidates,
plus the division GBM and combined Model C/V2 decoder.

Training code:

- `helpers/02_model_c/train_division_pair_model.py`
- `helpers/02_model_c/train_model_c_v2_event_decoder.py`
- `helpers/02_model_c/train_unet_transformer.py` for the neural checkpoint
  family

Pair/source GBM population: 199 movies, 1,095,856 candidate pairs with 479
positive pairs, and 127,398 source rows with 436 positive sources. Pair
probabilities used to train the source model were generated with grouped
five-fold video cross-fitting. Unknown rows were not used as negatives.

Combined decoder population: Division V2 geometry augmented with 24 native
Model C evidence features. The recorded split is 175 training movies and 20
held movies. Pair input width is 102 and source input width is 144. The four
practice movies were not used for training or threshold selection.

Serving graph: Model C has zero ownership of the P1/P2 base detections and
ordinary continuation graph. It maps native candidates to the current P1/P2
graph and supplies division evidence to UG1.

## 4. Source-cardinality V2 package

Dataset: `tweakai/biohub-public934-source-cardinality-v2`

Role: makes a grouped source decision between one CONTINUE option and retained
DIVIDE(daughter A, daughter B) options.

Training code:

- `helpers/03_source_cardinality/train_source_cardinality_head.py`
- `helpers/03_source_cardinality/extract_held_population.py`
- `helpers/03_source_cardinality/replay_held_graph.py`

Training graph: the graph-matched P1/P2 `.934` division-candidate substrate,
using 78 V2 geometry features, 24 Model C evidence features, 24 P1 features,
and 24 P2 features per pair. Source width is 41, pair width is 150, and at most
128 pairs are kept in stable geometry order.

Training protocol: five grouped movie folds, source-local softmax, exact parent
frame for positive division labels, confirmed continuations for CONTINUE, and
zero supervised loss for unknown sources. There is no NULL daughter class. The
threshold `0.40` was frozen from grouped OOF before held-20 evaluation.

Serving graph: UG1 applies atomic tube-NMS and daughter ownership transactions
to the post-gap graph.

## 5. Independent P1/P2 UniGRAFT package

Dataset: `tweakai/biohub-unigraft-p1p2-independent-v1`

Role: UG2, a division/cardinality branch trained without Model C so native
P1/P2 evidence can add complementary events.

Training code: `helpers/04_unigraft_p1p2/train_p1p2_only_cardinality_head.py`.

Training graph: the same exact-frame 175/20 source and pair population used by
the production cardinality V2 study. Inputs are 41 source features and 126 pair
features. Unknown sources are excluded, five folds are grouped by movie, the
pair cap is 128, and the frozen threshold is `0.51`.

Serving graph: UG2 runs after UG1 and merges only compatible atomic division
transactions.

## 6. Live V2 global bundle

Dataset: `tweakai/biohub-live-v2-global-bundle-v2`

Role: deterministic post-UG1/UG2 missed-division transaction logic.

Training code: none. The package contains no model weights and was not fitted.

Construction and validation code:

- `helpers/05_live_v2/build_live_v2_global_bundle.py`
- `helpers/05_live_v2/package_live_v2_global_bundle.py`
- `helpers/05_live_v2/validate_live_v2_global_bundle.py`
- `helpers/05_live_v2/live_v2_global_bundle_runtime.py`

Serving graph: reuses the already materialized 128-pair V2 population after
UG1/UG2 and before full-population ownership and EdgeGRAFT. It performs no
additional image pass.

## 7. Full-population ownership package

Dataset: `tweakai/biohub-ownership-exact-v2`

Role: learned source-level ownership arbitration across every eligible source.

Training and packaging code:

- `helpers/06_full_population_ownership/score_and_fit_ownership_oof.py`
- `helpers/06_full_population_ownership/package_full_population_ownership.py`
- `helpers/06_full_population_ownership/ownership_runtime.py`

Training graph: train-175 candidate rows after UG1, UG2, and the Live V2
specialist, before EdgeGRAFT. The candidate population contains 2,089,875 rows.
Only known rows enter the ExtraTrees fit; unknown rows are scored later but are
not negatives.

Training protocol: five GroupKFold splits by movie. Each fold is an
ExtraTreesClassifier with 400 trees, maximum depth 3, minimum leaf size 20,
balanced classes, and fold-specific seeds beginning at 324459. The frozen
threshold is `0.9470820974745776`.

Serving graph: hidden movies are routed deterministically to one fold model by
dataset hash. The highest candidate per source above threshold may commit an
ownership transaction. There is no source-population cap.

## 8. EdgeGRAFT V3 full-population package

Dataset: `tweakai/biohub-edgegraft-v3-full-population-v1`

Role: post-division continuation parent repair. It compares the incumbent
parent with alternatives and may atomically replace one one-to-one ownership
edge while protecting fork neighborhoods.

Training code:

- `helpers/07_edgegraft/train_edgegraft_parent_ranker.py`
- `helpers/07_edgegraft/train_edgegraft_transaction_gate.py`
- `helpers/07_edgegraft/package_edgegraft_v3.py`

Training graph: the artifact record names the exact substrate as the `.951`
UG2/UG3 boundary-track-rescue graph, using 175 movies. Historical stage names
are retained here because they identify the saved graph bank. Candidate
materialization produced 13,667,730 conflict rows and 97,484 target decisions.

Training protocol: a grouped five-fold HistGradientBoosting parent ranker, then
a second metric-aware gate trained only on non-neutral replacement
transactions. Metric-neutral or unmatched transactions were excluded rather
than labeled negative.

Serving graph: production runs EdgeGRAFT after the complete current division
ownership stack and before destructive cleanup. It reads the raw graph and
saved P1/P2 evidence and commits replacements atomically.

## 9. CandidateGRAFT direct package

Dataset: `tweakai/biohub-candidategraft-direct-v1`

Role: add-only one-frame continuation recovery when the source has no child and
the target has no parent.

Training code:

- `helpers/08_candidategraft/train_candidategraft_direct.py`
- `helpers/08_candidategraft/build_candidategraft_package.py`
- `helpers/08_candidategraft/replay_final_fit.py`
- `helpers/08_candidategraft/replay_held20.py`

Training graph: the `.952` production graph and its native P1/P2 candidate
population. There are 29,231 population rows across 175 videos. The fit uses
348 known rows: 304 positive and 44 negative. Unannotated rows are excluded.

Training protocol: five grouped movie folds, eight features, and an OOF-frozen
threshold of `0.90`. The eight features are P1, P2, maximum probability, mean
probability, evidence presence, winner count, best rank, and distance.

Serving graph: first after EdgeGRAFT and before component pruning, then again
on the final retained-node population. It never deletes an edge or creates a
fork.

## 10. Motion Corrector V1 package

Dataset: `tweakai/biohub-motion-corrector-v1`

Role: learned bounded residual added to the geometric continuation assignment
cost before the Hungarian solve.

Training code: `helpers/09_motion_corrector/train_motion_cost_corrector.py`.

Training graph: historical A+B fused proposal population, not the current
P1/P2 production graph. The trainer read 199 proposal files and generated
geometric continuation candidates. Split: 190 training movies and 9 validation
movies. The four practice movies were on the historical training side.

Population: 354,255 training candidate pairs and 540,613 validation pairs.
Candidate radius was 9.5 um. Hard negatives were mined per transition. The MLP
is `23 -> 64 -> 32 -> 1` with SiLU and dropout 0.05. Training used AdamW,
learning rate 0.002, batch 8192, focal-weighted BCE, maximum 40 epochs, and
patience 7.

Serving graph: registered, z-damped P1/P2 motion candidates. Production keeps
the original one-step feature contract for the MLP while geometric prediction
uses three linked steps. This is a known train/serve graph difference and is
documented rather than hidden.

## 11. DeepCenter center-prior package

Dataset: `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`

Role: targeted image confirmation for long synthetic gap nodes and
wider-geometry daughter candidates.

Training code:

- `helpers/10_deepcenter/train_full_frame_center_detector.py`
- `helpers/10_deepcenter/run_full_frame_center_training.sh`
- `helpers/10_deepcenter/build_full_frame_center_pack.py`

Training population: raw Biohub volumes and sparse GEFF cell centroids. It is
not trained on a tracking graph. Centroids are rendered as Gaussian center
targets. Dark background is supervised as negative, bright unlabeled voxels
receive low weight, and labeled centers receive positive weight 12. Training
uses a 90/10 movie split, seed 2026, XY pooling factor 4, random flips,
batch size 8, and learning rate 0.001.

Serving graph: no free node generation. The frozen best checkpoint confirms
only a targeted graph hypothesis at score `0.25`. The notebook expects the
epoch-2 best checkpoint by validation loss and verifies its SHA-256.

## Notebook-native helpers

The exact runtime implementations are in
`production/fork-of-fork-of-division-focused.ipynb`.

- P1/P2 fusion and harmonic association support
- sub-voxel peak refinement
- ILP configuration and raw graph sanitation
- registered z-damped motion assignment and single-parent repair
- one-frame gap recovery and synthetic midpoint refinement
- wider-geometry division pass
- retention guard and guarded ownership restoration
- isolated-node pruning and minimum-track filter
- boundary rescue
- normal and dense node budgets
- line-fit coordinate smoothing
- final ownership/topology validation
- streaming scheduler and atomic shard writer

These are deterministic or optimizer-based runtime stages. They have no model
training graph beyond the graph state named in the production flow.

