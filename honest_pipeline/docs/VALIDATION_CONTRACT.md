# Validation contract (all stages)

## Schemes (do not conflate)

| Name | What it measures | Use for |
|---|---|---|
| `loeo` | Unseen embryo | Promotion, global threshold/blend freeze |
| `gkf_movie` | Unseen movie, same embryos | Inner HPO, OOF features for stacking |
| `nested_loeo_gkf` | HPO inside outer embryo holdout | Config search without peeking outer val |
| `held20` | Frozen mixed-embryo panel | One-shot E2E after freeze |
| `practice4` | Smoke / wiring | Never metrics for decisions |
| `deploy_all` | Final serve weights | Not a CV claim |

## Required artifacts per learned stage

```text
runs/<stage>/<utc>_<name>/
  config.json
  env.json
  train.log
  metrics.jsonl
  summary.json          # must name scheme + population
  oof_<scheme>.parquet  # movie, fold_id, scheme, scores, labels when known
  oof_<scheme>.meta.json
  folds/                # optional per-fold checkpoints
```

## Threshold policy

1. Sweep thresholds on **inner** OOF folds only (`select_threshold_on_inner_oof`).
2. Freeze once. Report outer LOEO / held20 with that frozen value.
3. Never re-tune on held20 or practice4.
4. If a stage previously calibrated on held20 (Model C decoder), that path is **banned**.

## Cascade rule

Upstream detectors used inside any CV evaluation must themselves be **OOF**
for the evaluated movie (or LOEO for the evaluated embryo). Production
final-fit P1/P2 weights must not appear in honest cascade scores.

## Rebuild order (after detector OOF exists)

1. P1/P2 OOF banks  
2. ILP graphs from OOF evidence  
3. Motion (train on OOF/predicted parents, not GT)  
4. Division stack (Model C / cardinality)  
5. Ownership → EdgeGRAFT → CandidateGRAFT on OOF substrates  
6. DeepCenter LOEO 2-fold (both directions), not one-embryo train
