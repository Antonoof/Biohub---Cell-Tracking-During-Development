# Exact A+B proposal identity

The bundled `ab_proposals/manifest.json` confirms the historical population was created as follows:

- format: `biohub_ab_proposals_v1`;
- selected videos: 199;
- detection threshold: 0.99;
- pooling kernel: 5.0 um;
- equal weights: Model A 1.0 + Model B 1.0;
- window size: 2 for both members;
- downsample: `[1,4,4]` for both members;
- state: raw shared A+B detections **before edge prediction, ILP, or post-processing**.

Model A checkpoint SHA256:

`51b18e83ddd9c738c8a4cbee818a0901110ff6dd587b6162a26751a0173d4e99`

Model B checkpoint SHA256:

`ee79eec0d8888a3c0ed27185ecfcadf25a45be58aaaa39953a64698474850418`

Therefore the exact statement is: Motion V1 was trained on continuation candidates constructed from the historical equal-weight A+B raw shared-detection population. It was not trained on current P1/P2 detections, current P1/P2 continuation probabilities, or a current finalized production graph.
