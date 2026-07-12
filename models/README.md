# External model artifacts

Model files are not committed. The stable notebook expects Kaggle datasets
containing:

- A+B checkpoints: `biohub-ensemble-2modelv3`
- Learned motion cost: `biohub-motion-corrector-v1/motion_corrector_best.pt`
- Offline support pack: `biohub-tracking-support-pack-50ep-v1`

The learned motion checkpoint used by the current candidate has SHA-256:

```text
57B2FA4F0585C5915F46C2C99E9B48690B74654EA5652F19577963C474CF2AE5
```
