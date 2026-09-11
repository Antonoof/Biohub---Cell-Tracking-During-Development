# Model C → Division V2 Integration Loss Audit

Date: 2026-07-23

## Question

Model C recovered both annotated daughter edges for 9 of 17 division events on
20 videos held out from Model C training. Why did the combined V2 + Model C
decoder report only 4 of 17 held events, and why did the final four-practice
graph recover no new official division?

## Checkpoint integrity

The held `9/17` result is real and reproducible, but checkpoint names had become
ambiguous:

| Checkpoint | SHA-256 | Held division-pair result |
|---|---|---:|
| `best_division_pair.pth` | `3a4cfd682cfea77ec225af2f0e910a752dbf5b43ebe57cc40b0adc9fc4214cda` | **9/17** |
| `best_division_pair_epoch067.pth` | `ae7db2e50e6a5bba57aa0d8b238142723b081bdc14193fe36d195cbb9ed2660f` | **8/17** |

The combined-primary Kaggle artifact contains the correct `3a4cfd...` checkpoint.
Checkpoint identity should therefore be recorded by hash rather than the
historical “epoch 67” label.

## What `9/17` measures

This is direct Model C inference on the untouched held-20 split. A division is
counted only when:

1. Model C detects the annotated parent;
2. both annotated daughters are detected; and
3. both correct parent-to-daughter probabilities exceed the edge threshold.

It is a raw next-frame daughter-pair recall measurement. It is not an official
component-level graph metric.

## Exact loss trace

All nine recovered Model C events were traced through the saved V2 candidate
cache and the deployed combined decoder. No model was retrained.

| Integration outcome | Events |
|---|---:|
| Exact daughter pair exists in V2 candidates | 9/9 |
| Survives combined decoder | **4/9** |
| Rejected by source threshold `0.96` | **4/9** |
| Decoder changes to wrong daughter pair | **1/9** |
| Lost by tube NMS | 0/9 |
| Lost by daughter collision locking | 0/9 |

The four surviving events are exactly the combined decoder's four held true
positives. The reduction from `9/17` to `4/17` therefore occurs inside the
learned decoder, before final graph insertion.

### Per-event trace

| Dataset | t | Model C weaker-edge probability | Decoder score | Outcome |
|---|---:|---:|---:|---|
| `44b6_267148e4` | 3 | 0.5685 | 0.0054 | source threshold |
| `44b6_7a302da0` | 90 | 0.9641 | 0.9952 | survives |
| `44b6_c50204e0` | 28 | 0.9867 | 0.9833 | survives |
| `44b6_c50204e0` | 65 | 0.7941 | 0.4882 | source threshold |
| `44b6_d5e7d891` | 12 | 0.2253 after graph mapping | 0.0061 | wrong pair |
| `44b6_d5e7d891` | 47 | 0.6921 | 0.0039 | source threshold |
| `6bba_062c8d37` | 89 | 0.9996 | 0.9920 | survives |
| `6bba_57b7cc1e` | 23 | 0.9999 | 0.9935 | survives |
| `6bba_f8ffd5e7` | 10 | 0.6785 | 0.0037 | source threshold |

## Why a direct bypass is unsafe

A transparent Model-C-only rescue was also tested on the held candidate cache:
choose one pair per source, require both edges to be target winners, rank by the
weaker edge probability, then apply tube NMS and target locking.

| Model C edge threshold | TP | FP | FN | Jaccard proxy |
|---:|---:|---:|---:|---:|
| 0.50 | 7 | 234 | 10 | 0.0279 |
| 0.70 | 5 | 127 | 12 | 0.0347 |
| 0.90 | 4 | 32 | 13 | 0.0816 |
| 0.98 (held-best diagnostic only) | 3 | 2 | 14 | 0.1579 |

Therefore, lowering the combined threshold or blindly accepting Model C forks
would recover some true divisions but create an unacceptable continuation-fork
flood.

## Root integration problem

Model C was trained on the 175 videos used to train the combined decoder. Model
C evidence on those 175 videos is therefore in-sample, while its evidence on
the held 20 and hidden test is out-of-sample. Grouped folds in the decoder do
not remove this train/serve skew because the underlying Model C feature
generator had already seen every decoder-training video.

This explains how Model C can recover a held pair correctly while the learned
source decoder assigns it a score near zero.

There is a second, more fundamental integration error: Model C is attached only
after A+B graph construction and post-processing. It is allowed to score V2's
surviving parent/pair candidates, but it is not allowed to contribute its own
detections or restore an A+B proposal removed earlier.

On the untouched practice division at `6bba_05db0fb1`, `t=24`:

* Model C recovered both correct daughter edges (`0.9624`, `0.8765`);
* the parent and daughters all mapped to raw A+B proposals within
  `0.0`, `2.298`, and `1.625` µm;
* after A+B post-processing, the nearest surviving node for one daughter was
  `8.125` µm away;
* therefore the late V2 candidate graph could not express the correct pair.

The 89 practice forks selected by the combined decoder came from other
surviving V2 candidates. The one correct native Model C triplet was structurally
unavailable before the `0.96` source threshold was applied.

## Late-stage daughter-only repair test

The proposed low-risk repair was tested before changing the runtime:

1. freeze every V2 parent/no-parent decision;
2. preserve the exact number of selected forks;
3. allow Model C to change only the daughter pair when both Model C edges are
   target winners.

This did **not** pass.

| Model C rerank threshold | Held metric | Good / bad pair changes | Practice metric |
|---:|---:|---:|---:|
| Frozen V2 baseline | 0.7391 | — | 0.5000 |
| 0.50 | 0.6000 | 0 / 2 | 0.2000 |
| 0.70 | 0.6667 | 0 / 1 | 0.5000 |
| 0.90 | 0.6667 | 0 / 1 | 0.5000 |
| 0.98 | 0.7391 | 0 / 0 | 0.5000 |

Model C made no correct daughter substitution at any tested threshold. At
thresholds low enough to act, it replaced correct V2 pairs with incorrect
pairs. At `0.98`, it was harmless only because it made no changes.

This result rejects only **late-stage daughter reranking on the already-pruned
V2 graph**. It does not reject Model C as an early detection/proposal ensemble.

## Correct next architecture

The current late-stage Model C integration should not replace production V2.
The production-safe system remains frozen A+B + Division V2 until an early-fused
version passes validation.

A true Model C ensemble must:

1. run Model C before destructive A+B node pruning;
2. map each Model C parent-and-two-daughters triplet jointly into the raw A+B
   proposal population;
3. protect or restore mapped raw A+B proposal nodes needed by a high-confidence
   Model C triplet;
4. carry the complete triplet through ILP/post-processing as an atomic candidate;
5. use sparse-safe parent precision gating and one-parent-per-child constraints;
6. generate genuinely out-of-fold Model C outputs before training any learned
   parent arbitration gate;
7. require improvement under final graph insertion and the official component
   metric.

Producing genuinely out-of-fold Model C predictions requires fold-specific
Model C checkpoints. Until those exist, a newly trained Model C parent gate
cannot be claimed leakage-safe.

## Early triplet-preservation transaction test

The missing-node hypothesis was tested directly on 2026-07-23.  This was a
single-event transaction test, not a fitted selector:

1. read Model C's saved recovered t=24 parent-and-two-daughter proposal;
2. map the parent and surviving daughter into the final graph;
3. restore the missing raw A+B daughter proposal at t+1;
4. attach it to the nearest safe parentless t+2 continuation;
5. add the complete two-child fork atomically; and
6. score the four complete graphs with the host's patched component evaluator.

The transaction used no GT coordinates or identities to choose nodes.  GT was
used only by the final evaluator.

Structural checks:

| Item | Result |
|---|---:|
| Parent mapping | 2.958 um |
| Existing daughter mapping | 1.675 um |
| Restored daughter to parentless continuation | 5.281 um |
| Model C daughter edge probabilities | 0.9624 / 0.8765 |
| Parent out-degree after atomic transaction | 2 |

### Frozen V2 graph

| Metric | Baseline | Early triplet |
|---|---:|---:|
| Patched division TP / FP / FN | 1 / 2 / 2 | **2 / 2 / 1** |
| Patched division Jaccard | 0.2000 | **0.4000** |
| Edge Jaccard | 0.93258 | **0.93348** |
| Adjusted edge Jaccard | 0.92939 | **0.93029** |
| Combined local score | 0.94939 | **0.97029** |

### Saved `.931` combined-notebook output

The exact four submission shards from
`ointly-trained-decoder-results.zip` were converted back to GEFF and tested
independently.

| Metric | Saved `.931` output | + early triplet |
|---|---:|---:|
| Patched division TP / FP / FN | 0 / 0 / 3 | **1 / 0 / 2** |
| Patched division Jaccard | 0.0000 | **0.3333** |
| Edge Jaccard | 0.93518 | **0.93608** |
| Adjusted edge Jaccard | 0.93192 | **0.93282** |
| Combined local score | 0.93192 | **0.96616** |

This proves that the early atomic graph transaction is component-correct and
does not require an adjusted-edge sacrifice for the known complementary Model C
event.  It does **not** yet validate a population-wide acceptance rule.  The
next gate is to apply the same transaction constraints to all Model C proposals
accepted without GT and measure TP/FP plus adjusted edge on the untouched
practice graphs and genuinely held videos.

Reproduction script:

`scripts/test_model_c_early_triplet_preservation.py`

Generated outputs:

`data/model_c_early_triplet_t24_test`

`data/model_c_combined_931_local`

## Full-population mapping and coupled-rescue audit

The population-wide follow-up was completed on 2026-07-24 against an exact
replay of the submitted `.931` combined-primary architecture.

### Mapping finding

The diagnostic full-population exporter originally used a frame-wide greedy
one-to-one map from native Model C detections to A+B proposals. That map
dropped or diverted at least one member of four of the nine Model C-recovered
held events. Independent nearest same-frame mapping reproduced the expected
A+B parent and both daughters for all nine events.

This was a **diagnostic exporter bug, not the submitted `.931` decoder bug**.
The submitted combined-primary runtime already maps each V2 source and pair
independently to native Model C evidence. Its trace confirms that the missed
`44b6_c50204e0, t=65` pair existed in the V2 candidate set and was chosen by
the pair decoder, but its source score was only `0.4882`, below the frozen
`0.96` parent threshold.

### Generic rewire sweep: rejected

Independent mapping made the missed `t=65` event visible to the generic
one-child rewire at a raw Model C pair threshold of `0.70`. It also exposed
too many unrelated high-confidence rewires. Every tested threshold below
`0.98` reduced the exact held-20 score; `0.98` and above were no-ops.

Exact combined-primary held-20 baseline:

| Metric | Value |
|---|---:|
| Division TP / FP / FN | 4 / 4 / 13 |
| Division Jaccard | 0.19048 |
| Adjusted edge Jaccard | 0.87054 |
| Combined local score | 0.88958 |

### Combined-ranker-coupled rescue: rejected

A narrower experimental rescue retained the trained combined pair ranker and
allowed a below-threshold parent only when:

- the combined decoder's best V2 pair was used;
- both Model C daughters were present;
- both were target winners;
- minimum daughter probability was at least `0.75`;
- minimum parent margin was at least `0.70`; and
- source score was at least `0.25`.

It recovered the missed `t=65` event on its one-video gate:

| Metric | Frozen `.931` | Coupled rescue |
|---|---:|---:|
| Division TP / FP / FN | 1 / 1 / 1 | **2 / 1 / 0** |
| Division Jaccard | 0.3333 | **0.6667** |
| Exact graph score | about 0.8099 | **0.82595** |

However, the required four-video safety panel failed:

| Metric | Frozen `.931` | Coupled rescue |
|---|---:|---:|
| Division TP / FP / FN | 2 / 3 / 5 | 3 / 7 / 4 |
| Division Jaccard | 0.2000 | 0.2143 |
| Adjusted edge Jaccard | **0.69559** | 0.69257 |
| Exact graph score | **0.71559** | 0.71400 |

The rescue gained one true division but added four scored false forks and
reduced the exact panel score by about `0.00159`. It therefore failed
promotion. The submitted `.931` notebook and artifact remain unchanged.

Reproduction outputs:

`data/model_c_combined_primary_rewire_indmap6_held20_sweep`

`data/model_c_combined_rescue_v2_smoke_c502`

`data/model_c_combined_rescue_v2_panel4`

## V2 / combined / NULL arbiter: promoted by exact graph replay

A true per-lineage arbitration experiment was completed on 2026-07-24.
Unlike the submitted `.931` combined-primary decoder, this model preserved
both proposals and learned one correctness score for each option:

1. keep frozen Division V2's source and daughter pair;
2. use the V2+Model-C combined source and daughter pair; or
3. abstain (`NULL`).

The experiment used the original saved fold-clean V2 OOF source scores for
all 195 non-practice videos. Model-C combined scores were regenerated with
grouped OOF models on the 175 training videos. Unknown sources were not used
as negatives. The arbiter threshold was frozen from grouped OOF predictions
before the untouched held-20 evaluation.

| Selector | Held-20 TP / FP / FN | Division Jaccard |
|---|---:|---:|
| Frozen V2, original OOF | **16 / 7 / 1** | **0.66667** |
| `.931` combined-primary | 4 / 4 / 13 | 0.19048 |
| V2 / combined / NULL arbiter | 10 / 3 / 7 | 0.50000 |

The arbiter recovered substantially more divisions than the submitted
combined-primary selector and therefore **passed the relevant held-20 proxy
gate** (`0.50000 > 0.19048`). Frozen V2's `0.66667` is a useful diagnostic
reference, but it is not the production baseline the arbiter must beat.

The arbiter was consequently advanced to exact patched component-metric
replay. A one-video smoke test on `44b6_c50204e0` regressed, so it was not
used as the final decision by itself. A paired four-video panel then improved
the complete metric:

| Four-video exact graph metric | `.931` combined-primary | Arbiter |
|---|---:|---:|
| Division TP / FP / FN | 2 / 3 / 5 | **3 / 2 / 4** |
| Division Jaccard | 0.20000 | **0.33333** |
| Adjusted edge Jaccard | **0.69559** | 0.69390 |
| Combined score | 0.71559 | **0.72724** |

The full held-20 paired replay confirmed that this was a transferable
component-level improvement:

| Held-20 exact patched graph metric | `.931` combined-primary | Arbiter |
|---|---:|---:|
| Division TP / FP / FN | 4 / 4 / 13 | **9 / 4 / 8** |
| Division Jaccard | 0.19048 | **0.42857** |
| Edge Jaccard | 0.87243 | **0.87282** |
| Adjusted edge Jaccard | **0.87054** | 0.87042 |
| Combined score | 0.88958 | **0.91328** |

The exact held-20 score improved by `+0.02369`. Adjusted-edge Jaccard changed
by only `-0.00012`, while five additional true division components were
recovered without increasing division false positives. This passes the
relevant promotion gate against the submitted `.931` decoder. Frozen V2
remains a diagnostic reference, not the production baseline that the arbiter
must beat.

Training script:

`scripts/train_model_c_v2_arbiter.py`

Generated outputs:

`data/model_c_v2_arbiter_v1`

`data/model_c_v2_arbiter_exact_smoke_c502`

Exact paired summaries:

`data/model_c_v2_arbiter_exact_smoke_c502/exact_metric_held20/summary.json`

`data/model_c_combined_primary_held20_exact/exact_metric_held20/summary.json`

## Reproducibility

Audit script:

`scripts/trace_model_c_recovered_events.py`

Generated outputs:

`data/model_c_v2_nine_event_trace/summary.json`

`data/model_c_v2_nine_event_trace/trace.json`

`data/model_c_v2_nine_event_trace/trace.csv`
