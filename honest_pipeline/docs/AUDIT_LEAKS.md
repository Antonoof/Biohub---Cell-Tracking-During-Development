# Leak & validation audit (William .957 pipeline)

Source tree: `william-duckworth-reproducible-training-pipeline`  
Data: `/data/projects/ryzhichkin/biohub/kaggle/input/competitions/biohub-cell-tracking-during-development/`  
Labeled movies: **199** (`44b6`×71 + `6bba`×128). Panel: **175 / 20 / 4**.

## Verdict

Downstream stages often use movie GroupKFold, but the **system is not honest**:
P2 (and incomplete P1 splits) poison every cascade graph, thresholds are often
fit on the same OOF used for headlines, and **embryo transfer is barely tested**
despite private test expecting unseen embryos.

## Stage table

| Stage | Role | Current split | Main leak / bias | Durable OOF? |
|---|---|---|---|---|
| P1 detector | centers + edges | organizer fold / incomplete manifest | non-embryo split; early-stop ≠ competition metric; peak gate on **logits** not probs | No |
| P2 detector | fusion seed | **train-on-all 199** (fake val overlap) | **hard cascade contamination** | No |
| ILP | base graph | none | cost weights selection bias | N/A |
| Motion corrector | assignment residual | 190/9; practice in train | **GT teacher forcing**; train/serve graph mismatch | No |
| Gap / sanitation / wider-geom | rules | none | rule radii tuned on dev; wider-geom uses t+2 (OK offline) | N/A |
| Model C pair/neural | division evidence | LOEO (neural) + GKF5 (probs) | sits on contaminated detectors | fold ckpts only |
| Model C V2 decoder | division select | 175 + GKF5; **held20 threshold path** | held used for calibration | metrics only |
| Source cardinality UG1 | CONTINUE/DIVIDE | GKF5 + OOF threshold | same-OOF threshold; stale `.934` substrate | summary only |
| UniGRAFT P1/P2 UG2 | cardinality | GKF5 | same | summary only |
| Live V2 | missed division rules | changed-subset of train | not all-175 / held transfer | N/A |
| Ownership | ExtraTrees arbiter | GKF5 | OOF threshold on same preds; good OOF bank otherwise | **Yes** parquet |
| EdgeGRAFT | parent repair | GKF5 on **stale `.951`** | stale substrate; future ctx features | **Yes** |
| CandidateGRAFT | continuation add | hash GKF5 then **final all-fit** | final-fit replay sold near OOF | screen parquet |
| DeepCenter | center confirm | embryo split → **train 1 embryo only** | anti-transfer; val-loss early stop | No |

## Cross-cutting leaks

1. **P2 train-on-all** contaminates every “OOF/held” downstream number.
2. **No single split contract** across stages → incomparable scores.
3. **Threshold on reported OOF** without nested inner folds.
4. **Stale graph banks** (`.934` / `.951` / `.952`) vs current stack.
5. **Motion teacher forcing** (GT parents in train, predicted at serve).
6. **Practice movies in motion train** while practice is “smoke only”.
7. **Exact train-substrate replay** mixed with true OOF claims.
8. Random edge-row CV correctly banned in `VALIDATION.md` — keep banned.

## Validation recommendation

**Outer: Leave-One-Embryo-Out** — honesty / promotion gate (private ≈ unseen embryo).  
**Inner: 5-fold GroupKFold-by-movie** — development, HPO, stacking OOF features.  
**held20** — frozen score-once E2E panel (never for threshold after peek).  
**practice4** — smoke only.  
**Deploy all-fit** — only after LOEO + held frozen; label artifacts `deploy_fit≠cv`.

Why not GKF-only: every movie fold still mixes both embryos → measures “new movie, same biology”, not private risk.  
Why not LOEO-only: 2 folds, halves data, underpowered for fitting (DeepCenter already shows one-embryo train failure mode).

## What `honest_pipeline/` enforces

- `splits/canonical_splits.json` — one membership table for all stages
- `biohub_cv` — LOEO / GKF / nested iterators, OOF parquet writers, run logs, inner-fold threshold policy
- No stage may use train-on-all inside a CV cascade
- Every learned stage writes `runs/<stage>/<id>/{config,env,metrics,summary,oof.parquet}`
