# Biohub experiment registry

Last updated: 2026-09-07

This file is the concise source of truth for production status, completed
leaderboard tests, and closed local experiments. Local validation and hidden
leaderboard results are reported separately. Displayed Kaggle scores are rounded
to three decimals.

## Current production

| Field | Value |
|---|---|
| Status | **PRODUCTION** |
| Hidden score | **0.957** |
| Notebook | `historical\fork-of-fork-of-division-focused.ipynb` |
| Notebook SHA-256 | `0C141C83CE67FB24A4C30FCC65B06856331F2AA17119BAC8F36502FB337F7EC2` |
| Previous production score | 0.954 |
| Displayed improvement | **+0.003** |
| Production change | Wider-geometry division pass before Multi-UniGRAFT |

### Production execution order

1. Verify dependencies, manifests, model artifacts, and checkpoint hashes.
2. Run P1/P2 detection and association inference.
3. Fuse detection logits with P2 weight `0.475`, retain at least `90%` of the
   P1 candidate population, apply low-margin P2 association consensus, and add
   forward/reverse harmonic support.
4. Build the initial graph with the production ILP.
5. Sanitize edges and run registered, z-damped multi-step motion relinking.
6. Repair one-frame gaps with native-node reuse and DeepCenter confirmation for
   long synthetic gaps.
7. Run the wider-geometry division pass with the following active limits:
   parent/daughter radius `9.0 um`, daughter-pair radius `14.0 um`,
   existing-child radius `10.0 um`, sister symmetry `0.60`, mutual-nearest
   orphan ownership, `t+2` divergence `2.25 um`, and DeepCenter threshold `0.25`.
8. Run the complete Multi-UniGRAFT stack: Model C division GBM, source
   cardinality, independent P1/P2 UG1/UG2 evidence, live V2, and
   full-population ownership.
9. Run EdgeGRAFT V3 once on the full retained population.
10. Run CandidateGRAFT before destructive pruning.
11. Remove isolated nodes, filter short components, rescue valid boundary
    tracks, enforce the node budget, and apply line-fit coordinate smoothing.
12. Run CandidateGRAFT again on the final retained-node population.
13. Validate ownership, frame order, node degrees, shard completeness, and
    submission structure before writing `submission.csv`.

## Pending experiments

### Production `.957` edge-feature TTA — **PENDING / UNTESTED**

| Field | Planned value |
|---|---|
| Parent | Frozen `historical\fork-of-fork-of-division-focused.ipynb` (`0.957`) |
| Scope | One isolated inference change; no graph-helper or threshold changes |
| Change | Inverse-transform and average the primary model's encoder feature maps across the eight detection TTA views before edge prediction |
| Extra detector forwards | None; reuse the TTA forwards already executed for heatmap averaging |
| Rationale | The production path averages transformed detection heatmaps but currently discards augmented encoder features used by the edge predictor |
| Status | Future path only; no notebook build, local score, or hidden submission yet |

The first test must use primary-model edge-feature TTA only. Symmetric P1/P2
feature TTA is a separate follow-up and must not be stacked unless the isolated
primary test survives validation. No score improvement is claimed for this
entry.

Any future entry without a completed result must be labeled **PENDING** until it
finishes or is explicitly closed.

## Hidden leaderboard score chronology

The table records the strongest production line in promotion order. Deltas are
displayed-score changes relative to the immediately preceding production anchor.

| Milestone | Hidden score | Delta | Final status |
|---|---:|---:|---|
| Single-model reference family | 0.893 | - | Superseded |
| True A+B pre-ILP ensemble | 0.899 | +0.006 | Superseded |
| Minimum track length 6 | 0.901 | +0.002 | Retained concept |
| Edge-threshold plateau | 0.903 | +0.002 | Superseded |
| Line-fit smoothing | 0.904 | +0.001 | Retained component |
| Combined tuned post-processing | 0.905 | +0.001 | Superseded |
| Adaptive low-density filtering | 0.906 | +0.001 | Superseded |
| Registration-aware graph | 0.907 | +0.001 | Superseded |
| Learned division GBM | 0.920 | +0.013 | Retained component |
| Official-event parent gate | 0.921 | +0.001 | Superseded |
| Model C primary division decoder | 0.931 | +0.010 | Retained component |
| P1/P2 backbone plus production decoder | 0.934 | +0.003 | Superseded |
| Source-cardinality division system | 0.944 | +0.010 | Retained component |
| Frame-retention guard | 0.945 | +0.001 | Retained component |
| Improved P1/P2 backbone/finalization | 0.947 | +0.002 | Superseded |
| Frame-retention guard on improved backbone | 0.948 | +0.001 | Retained component |
| Guard-aware P2 ownership | 0.949 | +0.001 | Retained component |
| Independent P1/P2 Multi-UniGRAFT | 0.951 | +0.002 | Retained component |
| V17 division-focused pipeline | 0.952 | +0.001 | Superseded |
| Division-focused production pipeline | 0.953 | +0.001 | Superseded |
| BaseGraph-off, z-damped production pipeline | 0.954 | +0.001 | Superseded |
| Wider-geometry division before Multi-UniGRAFT | **0.957** | **+0.003** | **Current production** |

## Completed recent Kaggle tests

| Experiment | Parent | Hidden result | Displayed delta | Decision |
|---|---:|---:|---:|---|
| Upstream BaseGraph repair | 0.953 | 0.952 | -0.001 | Rejected |
| Reciprocal-P1 GraphGRAFT / endpoint ownership | 0.953 | 0.949 | -0.004 | Rejected |
| Broad DeepCenter second-daughter recovery | 0.953 | 0.866 | -0.087 | Rejected |
| DeepCenter motion-balance gate | 0.953 | 0.897 | -0.056 | Rejected |
| BODY/nnU-Net ownership integration | 0.954 | 0.954 | 0.000 | Completed; neutral |
| Dense-movie node budget `0.98` | 0.954 | 0.954 | 0.000 | Retained in the 0.954 parent |
| Full-strength pre-ILP NodeSelector | 0.954 | 0.950 | -0.004 | Rejected |
| Quarter-quota pre-ILP NodeSelector | 0.954 | 0.953 | -0.001 | Rejected |
| P2 detection-fusion weight `0.80` | 0.954 | 0.952 | -0.002 | Rejected |
| Same-video GT-shielded metric diagnostic | 0.954 | 0.954 | 0.000 | Completed diagnostic; no production change |
| Wider-geometry division pass V2 | 0.954 | **0.957** | **+0.003** | **Promoted** |

### Metric-diagnostic result

The metric diagnostic required a hidden dataset stem to have a same-named
released training GEFF containing `estimated_number_of_nodes`. Without that
exact match, its pruning target equals the current node count and the change is
a no-op. The completed hidden score was `0.954`, equal to its parent. It is not
a production component.

## Closed local-only experiments

These experiments have no outstanding run. They are closed and are not pending.

| Experiment | Completed evidence | Final status |
|---|---|---|
| ILP-preserving unmatched motion fill | Four-video adjusted edge delta `-0.000553`; one additional FP | Closed; not promoted |
| Phenotype-routed helper isolation V2 | Four-video adjusted edge delta `+0.001781`; no recorded hidden result | Closed; not promoted |
| Quarter-quota NodeSelector with `0.90` movie gate | Local preview regressed relative to the selected control | Closed; not submitted |
| Wider-geometry division pass V1 | Full-175 proxy delta `-0.013079`; superseded by the stricter V2 pass | Superseded |
| CORE/BODY replacement architecture | Detection work completed; no production-quality tracking graph | Closed and paused |
| Minimal gap-smoothing review candidate | Static build only; no hidden submission | Closed; not promoted |

## Validated production components

| Component | Best isolated hidden evidence | Production status |
|---|---|---|
| Learned division GBM | `0.907 -> 0.920` | Active |
| Model C primary division decoder | `0.921 -> 0.931` | Active |
| Source-cardinality division system | `0.934 -> 0.944` | Active |
| Frame-retention guard | `0.944 -> 0.945`, then `0.947 -> 0.948` | Active |
| Guard-aware P2 ownership | `0.948 -> 0.949` | Active |
| Independent P1/P2 Multi-UniGRAFT | `0.949 -> 0.951` | Active |
| Division-focused V17 and later ownership stack | `0.951 -> 0.952 -> 0.953` | Active lineage |
| BaseGraph-off, z-damped graph | Reached `0.954` | Active lineage |
| Wider-geometry division pass V2 | `0.954 -> 0.957` | Active production addition |

CandidateGRAFT, EdgeGRAFT V3, gap recovery, DeepCenter gap confirmation,
boundary rescue, dense node budgeting, and line-fit smoothing remain active in
the production notebook. Their individual hidden contribution is not assigned
unless an isolated hidden ablation exists.

## Completed rejected methods

These results are retained to prevent repeated work.

| Method | Evidence | Decision |
|---|---|---|
| Practice-supervised division specialist | Hidden `0.945 -> 0.940` | Do not repeat |
| Held-panel-selected cardinality A+B | Hidden `0.944 -> 0.941` | Do not repeat |
| Nested-validated cardinality A+B | Hidden `0.944 -> 0.939` | Do not repeat |
| Upstream BaseGraph repair | Hidden `0.953 -> 0.952` | Do not repeat unchanged |
| Reciprocal GraphGRAFT | Hidden `0.953 -> 0.949` | Do not repeat unchanged |
| Broad DeepCenter fork recovery | Hidden `0.953 -> 0.866` | Permanently rejected |
| DeepCenter motion-balance gate | Hidden `0.953 -> 0.897` | Rejected |
| Full-strength pre-ILP NodeSelector | Hidden `0.954 -> 0.950` | Rejected |
| Quarter-quota pre-ILP NodeSelector | Hidden `0.954 -> 0.953` | Rejected |
| P2 fusion weight `0.80` | Hidden `0.954 -> 0.952` | Rejected as a global setting |

## Reporting rules

1. A production promotion requires a completed hidden result that exceeds the
   current production score.
2. Local, held-out, OOF, and four-video results must be labeled as local evidence.
3. A displayed tie is recorded as neutral unless an exact hidden score is known.
4. A run without a result is labeled **PENDING**. When abandoned, it is changed
   to **CLOSED** with the reason.
5. Scores measured on different graph populations are never combined as one
   production delta.
6. Current production changes are documented in execution order in the notebook
   header and summarized here after the hidden result is known.

## Archive

The complete pre-cleanup 3,197-line ledger is preserved at:

`artifact_governance/archive/experiment_logs/EXPERIMENTS_FULL_PRE_CLEANUP_20260905.md`

It is historical reference only. This file is the active registry.
