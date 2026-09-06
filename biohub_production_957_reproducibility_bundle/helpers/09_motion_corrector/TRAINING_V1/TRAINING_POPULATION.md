# Historical V1 training population

The bundled `ab_proposals/` directory is the exact 199-file proposal population read by the original trainer.

Each NPZ contains:

- `coords`: detected node coordinates as `(t,z,y,x)`;
- `fused_det_prob`: equal-fusion A+B detection confidence;
- `member_det_prob`: the two member-model detection confidences;
- `frame_counts` and `frame_offsets`;
- `frozen_sources`;
- image shape, voxel scale, downsample, detection threshold, and pooling kernel metadata.

The proposal cache does not contain finalized tracking edges. The trainer builds source/target continuation candidates from this A+B node population, estimates mutual-nearest-neighbor frame shifts, applies registered-distance gating, matches proposal nodes to GT, and labels geometrically possible continuation pairs. Calling this the “A+B graph” is convenient shorthand, but technically it is the **A+B fused node/proposal population used to construct the supervised motion-assignment graph**.

The exact supervised matrices resulting from that construction are included under `training_cache/`.
