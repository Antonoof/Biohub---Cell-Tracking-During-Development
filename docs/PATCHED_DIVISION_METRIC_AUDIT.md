# Official Division Metric Patch Audit

Last updated: 2026-07-18

## Why this audit exists

The competition hosts patched the division metric after a component-based
exploit was identified. The old evaluator could credit a fork when its weakly
connected component overlapped the expected daughter components, even when the
predicted fork did not contain the required local directed parent-to-daughter
topology.

The patched evaluator requires a local parent anchor, two distinct directed
daughter branches, downstream evidence for both daughters, and branches that
remain unique rather than merging. Cross-component and malformed forks are
false positives. Predicted forks are still nodes with out-degree at least two,
and final credit is assigned with maximum-cardinality bipartite matching.

All division measurements recorded before 2026-07-18 are retained as historical
results, but they are no longer valid promotion gates. The exact patched graph
evaluator is now the only authoritative local division gate.

## Reproducibility

- Official repository checkout:
  `data/kaggle-cell-tracking-competition-patched`
- Evaluated commit:
  `075fc5f5a52d11077f9dc2b074644618f26939e2`
- Division patch commit:
  `aa65e90`
- Local diagnostic wrapper:
  `scripts/audit_patched_division_metric.py`

The wrapper imports the official patched implementation and adds only TP/FP
fork IDs and rejection-reason diagnostics. It does not reimplement or alter the
metric.

## Exact four-practice comparison

| System | Metric version | Adjusted edge | Division TP/FP/FN | Division Jaccard | Combined score |
|---|---|---:|---:|---:|---:|
| Optimized V1 (`0.920` submission artifact) | Pre-patch | 0.9307 | 1 / 2 / 2 | 0.2000 | 0.9507 |
| Optimized V1 | **Patched** | 0.9307 | **0 / 3 / 3** | **0.0000** | **0.9307** |
| Official-Event V2 (`0.921` submission artifact) | Pre-patch | 0.9294 | 2 / 0 / 1 | 0.6667 | 0.9961 |
| Official-Event V2 | **Patched** | 0.9294 | **1 / 2 / 2** | **0.2000** | **0.9494** |

The patch does not change the adjusted-edge term. It changes which predicted
forks receive division credit. V2 remains structurally better than V1 on the
four held practice clips, but the old `2 TP / 0 FP / 1 FN` claim was optimistic
under the patched definition.

## Exact 24-video division-rich panel

The saved final V2 graphs in
`data/v3_v2_final_state_replay_positive24/v2_final_shards` were
converted to GEFF and evaluated with both versions of the official package.

| Metric version | Adjusted edge | Division TP/FP/FN | Division Jaccard | Combined score |
|---|---:|---:|---:|---:|
| Pre-patch | 0.8659 | 56 / 10 / 7 | 0.7671 | 0.9426 |
| **Patched** | 0.8659 | **46 / 21 / 17** | **0.5476** | **0.9207** |

This larger panel is the most useful conclusion. The patch exposes a systematic
gap, but V2 still recovers 46 valid local directed division events. It is not a
metric-only artifact and remains the strongest division helper currently
available.

The patched false positives are primarily evaluable continuation parents whose
two predicted branches remain locally unique but do not match a true division.
They are not dominated by malformed or cross-component forks. Simple distance
thresholds do not cleanly separate them from true positives, so a geometry-only
veto is not justified.

## Decision and next experiment

1. **Freeze V2** as the current production reference until Monday's official
   rescore is available.
2. **Stop using pre-patch division Jaccard for promotion.** Historical numbers
   remain useful only for reproduction.
3. **Do not integrate multi-scale DoG node rescue.** Its candidate flood and
   tiny exact-metric ceiling do not address the patched division failure.
4. Build a **patch-aware fork selector** on frozen V2 graphs:
   - positive labels: exact patched TP forks;
   - safe negatives: exact patched evaluable FP/continuation forks;
   - unknown or unannotated forks: zero supervised loss;
   - grouped whole-video validation with the four practice clips excluded;
   - one fork per parent lineage and one parent per daughter;
   - exact patched graph metric as the promotion gate.
5. Start with a keep/drop selector for existing V2 forks. This directly targets
   the 21 patched false positives without creating a new detection flood. Only
   after pruning transfers should the model attempt recovery of the 17 misses.

The immediate target is not the obsolete `0.970` leaderboard. The target is to
beat frozen V2 under the patched evaluator without reducing adjusted-edge
quality. The post-rescore leaderboard will determine the real competitive gap.

## Patch-aware keep/drop selector screen

A conservative V3 screen was completed before authorizing a full 195-video
replay. It used the 24 saved final V2 graphs and supervised only the exact
patched outcomes:

- 46 patched TP forks: positive/keep;
- 21 patched FP forks: safe negative/drop candidate;
- 438 other forks: unknown, never used as negatives.

Whole-video grouped outer folds and train-only inner threshold selection were
used. Three model families were tested: regularized logistic regression,
shallow Extra Trees, and shallow histogram GBM.

The first feature set contained final-graph geometry, density, motion
continuity, branch survival, and local topology. The second added the complete
V2 runtime source and pair features, V1 source score, V2 gate score, and pair
probability aggregates. The third added patched-metric-aligned signals:
multi-frame sister separation, midpoint conservation, daughter speed symmetry,
raw A+B downstream edge-probability support, and branch persistence.

| Screen | Best held-video AUC | Patched TP/FP/FN | Patched Jaccard | Decision |
|---|---:|---:|---:|---|
| Frozen V2 | - | 46 / 21 / 17 | **0.5476** | Control |
| Graph-only selector | 0.481 | 44 / 19 / 19 | 0.5366 | Reject |
| V2 runtime-feature selector | 0.568 | 46 / 21 / 17 | 0.5476 | Reject; no FP removed |
| Runtime + downstream branch selector | 0.529 | 45 / 21 / 18 | 0.5357 | Reject |

The runtime-feature model's deployment threshold would also have removed 417
of 438 unknown/unannotated forks. That is unacceptable under sparse annotation
and confirms that the apparent in-sample separation is not safe.

**Decision:** do not package a patch-aware keep/drop model and do not run the
full 195-video V2 replay for this feature family. The patched false forks are
not separable from true forks using current single-transition features, final
graph geometry, or the tested downstream association summaries. V2 stays
frozen and unchanged.

Reproducibility:

- Trainer/screen: `scripts/train_patch_aware_fork_selector.py`
- Graph-only output: `data/patch_aware_fork_selector_screen_v1`
- V2 runtime-feature output: `data/patch_aware_fork_selector_screen_v2`
- Downstream-branch output: `data/patch_aware_fork_selector_screen_v3`
