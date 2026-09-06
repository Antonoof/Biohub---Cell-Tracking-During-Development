# Biohub CandidateGRAFT Direct v1

Inference-only, add-only continuation recovery for the `.952` production graph.

- reuses P1/P2 native association evidence already exported in the GPU pass;
- adds only a one-frame edge from a childless source to a parentless target;
- never deletes or replaces an edge and never creates a fork;
- preserves all existing UniGRAFT division topology;
- trained only on annotated known rows; unannotated rows were never negatives;
- grouped-video OOF gate fixed at 0.90 before the final all-known refit.

No images, GT, video-specific decisions, or cached submission rows are included.
