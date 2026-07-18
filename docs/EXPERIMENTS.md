# Biohub experiment registry and team decision log

Last updated: 2026-07-18

This is the team-facing source of truth for what has been tried, what actually
improved the submission, what failed, and what remains weak. Its purpose is to
prevent repeated experiments and to keep local proxy gains separate from real
Kaggle evidence.

The current rollback-safe production notebook is:

`C:\Kaggle\division-gbm-w-updated-model.ipynb`

The current best hidden Kaggle score is **0.921**.

## Evidence and decision labels

| Label | Meaning | Trust level |
|---|---|---|
| **LB** | Hidden Kaggle score from a completed submission | Final promotion evidence |
| **GRAPH-4** | Exact official graph evaluator on all four practice clips | Strong local gate, but only three annotated practice divisions |
| **OOF-GRAPH** | Whole-video grouped out-of-fold graph replay | Strong generalization evidence |
| **OOF-ROW** | Grouped held-video candidate or row metric | Screening only |
| **PROXY** | Adjusted-edge or division proxy | Directional; never sufficient alone |
| **SMOKE** | Runtime, loading, schema, or tiny-sample check | Correctness only, not accuracy |

Decisions are **KEEP**, **ACTIVE**, **REJECT**, or **INCONCLUSIVE**.

## Current production system

| Stage | Current choice | Status |
|---|---|---|
| Detector/linker | True pre-ILP A+B ensemble, weights 1:1 | **KEEP** |
| Detection / shared-edge thresholds | `0.99 / 0.54` | **KEEP** |
| ILP weights | edge `-1.0`, appearance `0.1`, disappearance `0.1`, division `1.5` | **KEEP** |
| Motion relink | tight `6.0 um`, relaxed `9.5 um`, velocity `0.52`, learned bonus `0.78` | **KEEP** |
| Learned motion residual | enabled, strength `1.0` | **KEEP**; hidden score rounded unchanged at 0.907 |
| Gap close | `5.4 um` | **KEEP** |
| Gap2 | enabled; total `9.2 um`, step `3.9 um`, fraction `0.0032`, absolute `140` | **KEEP** |
| Line fit | weight `0.78`, window `3` | **KEEP** |
| Short-track filter | adaptive; standard `7`, low-density `9` | **KEEP** |
| Safe division geometry | `4.7 / 6.85 / 7.45 um`, frame cap `0.0072`, global cap `0.00375` | **KEEP** |
| Registration | registration-aware geometry validation and line-fit stabilization | **KEEP** |
| Division helper | Official-Event Parent Gate V2, four streaming CPU workers | **KEEP** |
| Anytime safety | atomic A+B shard first; helper replaces only a completed video | **KEEP** |

The four-worker streaming implementation preserved the division result while
cutting the hidden run to about six hours. Runtime coverage is no longer the
primary bottleneck.

## Hidden Kaggle score chronology

| Milestone | Score | Delta | Decision / lesson |
|---|---:|---:|---|
| Public single-model/reference family | 0.893 | - | Historical starting point |
| True A+B pre-ILP ensemble | 0.899 | +0.006 | **KEEP**: real diversity gain |
| A+B plus short-track minimum 6 | 0.901 | +0.002 | Short-track pruning was useful |
| Edge threshold `0.52` | 0.903 | +0.002 | Useful; `0.54` matched it |
| Edge threshold `0.54` | 0.903 | +0.002 | **KEEP**: chosen plateau center |
| Line-fit weight `0.80` | 0.904 | +0.001 | Useful; later settled at `0.78` |
| Combined tuned post-processing | 0.905 | +0.001 | Gap/safe-division/GAP2 bundle transferred |
| Adaptive low-density minimum track length `9` | 0.906 | +0.001 | **KEEP** concept |
| Registration/adjusted-proxy V8 | 0.907 | +0.001 | **KEEP** |
| Learned motion-cost notebook | 0.907 | 0.000 rounded | Kept because local graphs improved without hidden regression |
| Learned Division GBM V1 | 0.920 | +0.013 | **KEEP**: largest honest gain |
| Optimized four-worker V1 | 0.920 | 0.000 | **KEEP**: same accuracy, much better runtime |
| Official-Event Parent Gate V2 | **0.921** | **+0.001** | **KEEP**: current best production system |

## Parameter and post-processing ledger

### Confirmed useful

| Experiment | Evidence | Result | Decision |
|---|---|---|---|
| True A+B fusion before one ILP | LB | 0.893-class reference to 0.899 | **KEEP** |
| Minimum track length 6 | LB | 0.899 to 0.901 | Superseded by adaptive filter |
| Edge threshold `0.52` and `0.54` | LB | both 0.903 | **KEEP 0.54** |
| Line-fit weight `0.80` | LB | 0.903 to 0.904 | **KEEP 0.78-0.80 region** |
| Gap close `5.4 um` | local ablation + combined LB | selected into 0.905 candidate | **KEEP** |
| Motion relaxed gate `9.5 um` | local ablation + combined LB | better than `9.0` | **KEEP** |
| Safe-division precision bundle | local ablation + combined LB | selected into 0.905 candidate | **KEEP** |
| GAP2 `9.2 / 3.9 um` | local ablation + combined LB | selected into 0.905 candidate | **KEEP** |
| Adaptive low-density pruning | LB | low-density 9 reached 0.906 | **KEEP** |
| Registration-aware correction | LB | reached 0.907 | **KEEP** |
| Learned motion cost | OOF/proxy + LB | motion J `0.8817 -> 0.8910`; no hidden regression | **KEEP** |

### Neutral or too small to justify another unchanged run

| Experiment | Result | Decision |
|---|---|---|
| Node refinement + safe-division precision alone | remained 0.899 | **REJECT unchanged** |
| GAP2 disabled | remained 0.901 | Current conservative GAP2 retained |
| Dense selective prune | remained 0.906 | **REJECT unchanged** |
| `OUTPUT_EDGE_MAX_UM=13.5` | no useful local movement | **REJECT unchanged** |
| `GAP2_MAX_LINKS_ABS=120` | no useful local movement | **REJECT unchanged** |
| `GAP_CLOSE_REUSE_UM=3.2` ablation | no useful movement | Keep current value; do not repeat |
| `GAP_CLOSE_MAX_ADDED_FRAC=0.045` | no useful movement | Keep current `0.052` |
| `SAFE_DIV_GLOBAL_FRAC_CAP=0.0035` | no useful movement | Keep `0.00375` |
| Detector TTA8 smoke | did not earn graph promotion | **REJECT as a blind global add-on** |

### Regressions

| Experiment | Evidence/result | Decision |
|---|---|---|
| Global edge threshold `0.40` | LB 0.899 from 0.901 | **REJECT** |
| Edge threshold `0.55` | local regression | **REJECT**; plateau is `0.52-0.54` |
| Per-member A `0.99`, B `0.999` calibration inside fused ensemble | LB 0.900 from 0.901 | **REJECT** |
| Detector threshold `0.985` + line fit `0.85` | no LB gain in earlier pipeline | **REJECT unchanged** |
| Detector threshold `0.992` | local candidate not promoted | **REJECT unchanged** |
| Gap close `5.2 um` | local regression | **REJECT** |
| Motion relaxed `9.0 um` | local regression | **REJECT** |
| Motion velocity weight `0.58` | local regression | **REJECT** |
| Motion velocity weight `0.48` | no improvement | **REJECT unchanged** |
| Learned bonus `0.72` | local regression | **REJECT** |
| Line-fit weight `0.75` or `0.85` | worse than `0.78-0.80` | **REJECT** |
| Line-fit window `4` | local regression | **REJECT**; keep `3` |
| Three-tier short-track `10/13/7` | LB 0.905 vs 0.906 | **REJECT** |
| Adaptive `9/10/7` | LB 0.905 vs 0.906 | **REJECT** |
| ILP division weight `1.25` | local candidate not better | Keep `1.5` |

### Diagnostic and superseded explorations

| Exploration | What it established | Decision |
|---|---|---|
| Symmetric Z-boundary disagreement pruning | A-only and B-only disagreement was not confined to one clean boundary band; hard deletion had no promotion evidence | **REJECT unchanged**; revisit only with an outcome-labelled boundary audit |
| Registration V1-V8 ladder | Local-flow, duplicate-aware, pre-ILP, full-edge, subpixel, bidirectional, and geometry-validation variants were screened | Intermediate versions are superseded; retain the V8 logic that reached 0.907 |
| Public 0.893 safe-division/node-refinement bundle | Supplied useful settings, but the complete bundle alone stayed at 0.899 | Individual retained settings are already represented above; do not rerun the original bundle |
| Practice-GT strict precision calculation | Sparse GT made raw node/edge precision look near zero on dense clips | Use sparse recall and official adjusted metrics; never prune from the debug precision columns |
| Frozen-frame audit and runtime skipping | Byte-identical transitions exist and should not be learned as motion | Keep frozen-transition awareness where implemented; it is not a standalone score claim |
| Public multi-scale DoG rule-based result | Detection scaling improved that weaker method by about +0.040; global division edges hurt it | **INCONCLUSIVE** for A+B. Test only as selective missing-node rescue |
| Public TTA/metric-exploit notebooks | Demonstrated a scoring vulnerability rather than usable lineage tracking | Excluded from production and from the honest score chronology |

## Base-model and ensemble experiments

| Experiment | Best evidence | Decision and lesson |
|---|---|---|
| Model B solo | LB 0.867; threshold `0.997` raised it to 0.873 | Weak alone but useful for diversity; solo calibration does not transfer to fused A+B |
| A+B equal weights | LB 0.899, later 0.921 with helpers | **KEEP** |
| A+B weight variants | 1:1 and 1:0.33 both initially 0.899 | No clear weight gain; keep 1:1 |
| Three-model A+B+C, partially trained C | LB 0.895 vs 0.899 | **REJECT**. Diversity did not overcome weaker probabilities |
| Fully trained C | training composite 0.9497 | **INCONCLUSIVE** as a clean graph comparison |
| Larger A2 `[48,96,192]`, output 48 | A2+A1 LB 0.897 vs A+B 0.901 | **REJECT** |
| A2 pretrain + Biohub fine-tune | best fine-tune composite 0.0174 at epoch 5 | Plateaued; **REJECT unchanged** |
| Proposal-aware B2 edge-head fine-tune | fused proposal J `0.8897 -> 0.8904` | Negligible; **REJECT** |
| Residual edge corrector | offline J `0.8904 -> 0.9005` | Full strength overpruned; soft version below reference. **REJECT** |
| Learned motion-cost corrector | held J `0.8817 -> 0.8910`; LB stayed 0.907 | **KEEP** as a non-regressing residual helper |
| Public third model added to A+B | sparse local edge metrics worsened | **REJECT** |
| nnUNet segmentation tracker | LB about 0.693 with DoG, 0.668 without | Architecture/inference mismatch; **REJECT** |
| Cellpose SAM V2 detector | one clip looked good, another produced 0-21 centers where hundreds were expected | Severe domain failure; **REJECT** |
| Fine-tuned Trackastra | local no-division graph proxy about 0.8785 vs A+B about 0.9254; slow after A+B | **REJECT as production path** |
| HOCT with local-ellipsoid adapter | pair AP `0.01178 -> 0.00416`; top-1 `0.4083 -> 0.3550` | Adapter rejected; watershed/mask adapter remains **INCONCLUSIVE** |

## Association/linking helper experiments

| Experiment | Evidence | Decision |
|---|---|---|
| Multi-frame contextual linker | OOF row `0.8580 -> 0.8591`; independent graph panel `-0.00043` | **REJECT**. Small row gain did not survive ILP |
| Structured Association V1, strength 1.0 | OOF-GRAPH +0.0251 on 12 videos, 10/12 up | Strong screening result |
| Structured Association V1, full integrated strength 1.0 | GRAPH-4 adjusted edge +0.00420, but division `0.6667 -> 0.3333`; combined proxy -0.0291 | **REJECT full strength** |
| Structured Association V1, strength 0.50 | OOF-GRAPH +0.0162, 9/12 up | **ACTIVE** full integrated run. Promote only if division and combined score do not regress |

Structured Association is the best current attempt at correcting the A+B core,
but it proves that better edge totals can still damage the official component
outcome. The graph metric, not edge-row Jaccard, is the gate.

## Division program summary

The detailed division record is in [DIVISION_GBM_V1_V2.md](DIVISION_GBM_V1_V2.md).
The reusable data inventory is in [DIVISION_DATA_CATALOG.md](DIVISION_DATA_CATALOG.md).

### What was proven

- Stable A+B 199-video audit: division TP/FP/FN `6 / 315 / 145`, Jaccard
  `0.01288`.
- A broad `14/20 um` parent/daughter-pair generator recovers `142/151` GT
  divisions (`94.0%`).
- A+B already contains sufficient detected nodes for `146/151` events
  (`96.7%`).
- The main division problem is **event selection and graph transaction**, not
  basic node availability.
- Division V1 raised the hidden score `0.907 -> 0.920`.
- Official-Event V2 reached held-four `2 TP / 0 FP / 1 unreachable FN`, division
  Jaccard `0.6667`, and raised hidden score to `0.921`.

### Division paths tried and not promoted

| Path | Result / failure mode |
|---|---|
| Simple safe divisions and geometry-only rules | Very low baseline division Jaccard; insufficient parent selection |
| Parent classifiers with absolute thresholds | Threshold/calibration collapse across embryos |
| Daughter-pair ranker alone | Pair ranking improved, but parent gate remained the bottleneck |
| Lifetime-max tracklet aggregation | Length-biased extreme-value artifacts selected wrong interior frames |
| Listwise NULL/rank selector prototypes | Better ranks but did not clear honest validation gates |
| CE-pretrained appearance transfer | Domain/temporal-localization mismatch; no reliable gain |
| Biohub appearance-only encoders | Static appearance did not beat dynamic/geometry features |
| Pseudo-labeling | Sparse-safe precision and held-GT gain were insufficient for promotion |
| V2.1 full-population gate | OOF Jaccard about 0.2408, but exact held graph replay failed; **REJECT** |
| Reduced-feature temporal Stage A | Worse than V2.1 localization; **REJECT** |
| Outcome-aligned synthetic `t+1` bridge V3 | Only about 20 annotated sources eligible; no broad promotion path established |
| Trackastra division integration | Too slow and weaker graph behavior |

### Division rules that must remain true

- Never label sparse-GT unknown cells or transitions as negatives.
- Exclude the four practice clips from every fit, calibration, and threshold
  decision used to evaluate those clips.
- Train on the same event/action that runtime inserts. Exact-daughter labels are
  not interchangeable with official component-success labels.
- Validate with the official graph transaction. Parent AP, pair AP, or candidate
  Jaccard cannot promote a division model by itself.
- Preserve one parent per child, one fork per parent/tube, and atomic fallback.

## Current weakest parts, ranked

### 1. Division selection and component-safe insertion

This remains the highest-value specialized weakness. V2 is a real improvement,
but it added only `+0.001` beyond V1 on hidden data. The broad generator has a
high oracle ceiling, yet selectors repeatedly fail through sparse supervision,
cross-embryo calibration, or graph-component side effects.

Needed improvement: an event-outcome selector trained and validated on the
actual final graph transaction, with sparse-safe supervision and whole-video
held folds. Do not return to exact-next-frame daughter classification alone.

### 2. Core A+B association and topology selection

The 199-video audit shows meaningful non-division headroom:

- proposal recovery for annotated next-frame links: `125,221 / 126,756`
  (`98.8%`);
- final edge TP/FP/FN: `122,415 / 6,581 / 6,468`;
- aggregate edge Jaccard: `0.9037`;
- 32/199 videos below adjusted edge `0.80`;
- 81/199 videos below adjusted edge `0.90`;
- the worst 40 videos hold about `60.4%` of residual association errors.

The proposals usually contain the answer. Selecting globally consistent
topology is weaker than proposal coverage. Structured Association is aimed at
this weakness, but must be softened enough not to break division components.

### 3. Detection completeness in hard videos

The old full audit had aggregate node recall about `0.9779`, and division-node
coverage is `96.7%`, so detection is not the primary division bottleneck.
However, hard videos still contain missing nodes and fragmented components. A
public rule-based study found multi-scale DoG was its largest lever
(`0.786 -> 0.826`), whereas global gap closing and division edges did not help.

The credible use here is **selective rescue**, not replacing A+B: propose a DoG
node only near an A+B endpoint/gap, require temporal support, and cap additions.
This idea is **INCONCLUSIVE** and has not earned integration.

### 4. Cross-embryo generalization

Most hard division misses and several weak graph cases are in `6bba`. Models
that look excellent on `44b6` frequently lose precision or recall on `6bba`.
Every new helper must report both embryo families and use whole-video folds.

### 5. Metric/component coupling

Improving edge Jaccard alone can reduce the final score by breaking a division
component, changing track identities, or altering the node-penalty term. The
full-strength structured association run is the clearest example. Future work
must report edge, adjusted edge, division, and combined official outcomes.

## Work that should not be repeated

1. Do not rerun global edge thresholds `0.40` or `0.55`, gap close `5.2`, motion
   relaxed `9.0`, line-fit `0.75/0.85`, or line-fit window `4` unchanged.
2. Do not use a solo Model B threshold as a per-member threshold inside the
   fused pre-ILP ensemble. Shared detections make A+B one fused model for tuning.
3. Do not infer production quality from training accuracy, recall, candidate
   AP, or row-level Jaccard.
4. Do not treat sparse unannotated predictions as false positives in training
   or local scoring.
5. Do not refit thresholds on the four practice clips or separately by embryo.
6. Do not deploy a helper solely because it improves aggregate edge counts;
   check official division components and adjusted-node penalty.
7. Do not rerun A+B proposal or edge exports already listed below.
8. Do not replace A+B globally with Cellpose, nnUNet, Trackastra, or the tested
   HOCT ellipsoid adapter without a materially different validation path.
9. Do not submit metric-exploit/degenerate-division notebooks as production
   tracking systems. They are excluded from this registry's score ladder.
10. Change one causal axis per test unless the combination is explicitly a
    confirmation run after individual ablations.

## Reusable assets: do not regenerate

| Asset | Contents |
|---|---|
| `/home/tweak/bio/ab_proposals_export/biohub_ab_proposals` | A+B proposals for all 199 videos |
| `/home/tweak/bio/ab_edge_probs_v16` | Per-model and fused A+B edge probabilities |
| `/home/tweak/bio/audit_907_v1` | Full 199-video stable-pipeline audit |
| `/home/tweak/bio/division_training_cache_v1.npz` | 127,398 sources and 1,095,856 pairs |
| `/home/tweak/bio/division_event_tubes_v14.npz` | Tube/node mapping aligned to V1 cache |
| `/home/tweak/bio/division_edge_features_v16` | 29 A+B alternative-edge features per pair |
| `/home/tweak/bio/division_v3_full_population_cache` | 4,995,373 full-population transitions |
| `/home/tweak/bio/division_gbm_deploy_v1_crossfit` | Corrected V1 teacher |
| `/home/tweak/bio/edge_corrector_cache` | Residual edge-corrector train/validation cache |
| `/home/tweak/bio/hoct_division_features_panel_v1` | Tested HOCT ellipsoid-adapter panel |

Before starting a large extraction, check [DIVISION_DATA_CATALOG.md](DIVISION_DATA_CATALOG.md)
and this table.

## Active and next experiments

1. **Finish Structured Association strength 0.50 integration test.** Promote
   only if adjusted edge improves and the V2 division result remains intact.
2. **If strength 0.50 harms division, use selective activation.** Apply the
   correction only to low-margin or high-ambiguity groups.
3. **Test selective multi-scale DoG node rescue.** Use cached weak-video
   endpoints/gaps, temporal support, and strict addition caps; do not replace
   A+B globally.
4. **Pursue a stronger base association model only with whole-graph training.**
   Pointwise proposal-aware edge fine-tuning has already plateaued.
5. **Division V3 only with official-event outcome labels.** Keep V2 frozen; do
   not reopen exact-daughter/localizer variants without new supervision.

## Detailed records

- [Learned Division V1/V2 and streaming deployment](DIVISION_GBM_V1_V2.md)
- [Division data catalog](DIVISION_DATA_CATALOG.md)
- [Structured Association V1](STRUCTURED_ASSOCIATION_V1.md)
- [Multi-frame contextual linker](MULTIFRAME_CONTEXT_LINKER_EXPERIMENT.md)
- [HOCT higher-order experiment](HOCT_EXPERIMENT.md)

## Required template for every new experiment

```text
Experiment ID:
Date:
Owner:
Parent notebook/checkpoint and SHA256:
Single causal change:
Hypothesis:
Training videos and excluded videos:
Validation type: LB / GRAPH-4 / OOF-GRAPH / OOF-ROW / PROXY / SMOKE
Frozen thresholds/settings:
Baseline metrics:
Candidate metrics:
Per-embryo result:
Runtime and storage:
Graph completeness/fallback count:
Decision: KEEP / ACTIVE / REJECT / INCONCLUSIVE
Reason:
Artifact paths:
```

No experiment is complete until its decision and reason are filled in. This is
the rule that prevents an old notebook from being mistaken for a better one.
