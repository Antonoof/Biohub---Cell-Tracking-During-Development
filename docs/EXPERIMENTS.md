# Experiment history

## Stable reference

- Public leaderboard: `0.907`
- Pipeline: true A+B logit ensemble, one ILP, adaptive short-track filtering,
  registration-aware motion relinking, gap recovery, and safe divisions.
- Local adjusted edge proxy: approximately `0.92542`.

## Learned motion-cost candidate

- Held-out sequential motion Jaccard: `0.88174 -> 0.89178`.
- Full four-clip adjusted proxy: `0.92542 -> 0.92707`.
- Full proxy edge changes: TP `+1`, FP `-3`, FN `-1`.

## Rejected paths

- Full-strength pre-ILP edge corrector: over-pruned the graph.
- Soft pre-ILP edge corrector: preserved nodes but scored below reference.
- Proposal-aware B-only and fused B fine-tuning: negligible improvement.
- Trackastra integration: excessive runtime and weaker local graph proxy.
