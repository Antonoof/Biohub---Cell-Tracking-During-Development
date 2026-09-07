# Biohub cell tracking

Train and infer the frozen production pipeline from YAML. Install with `uv`;
do not run helper scripts from `PYTHONPATH`. Neural trainers expect CUDA.

```bash
uv sync --group dev --group score --group infer
uv run python -m biohub.train.01_p1 --config configs/01_p1.yaml
uv run python -m biohub.infer.run --config configs/infer.yaml --movie 44b6_0113de3b
uv run python -m biohub.metrics.evaluate --config configs/infer.yaml --pred-source gt --panel smoke --require-complete
uv run ruff format src tests
uv run ruff check --fix src tests
uv run ty check
uv run pytest
uv run pytest -m slow
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python runs/direct_smoke/compare.py
```

`notebooks/infer.ipynb` loads `configs/infer.yaml`, runs detect and graph
upgrade through public package functions (`detect_job`, `predict_from_job`,
`GraphUpgrade.process_graph`), then writes `submission.csv` with the same
columns as the original production notebook:
`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.

Frozen serving weights stay in `kaggle/input/datasets/antonoof/all_files`.
Helper scripts in `biohub_production_957_reproducibility_bundle/` are the smoke
reference, not an import surface.

## Stages

Production infer order: detect → graph (cardinality / UniGRAFT / live_v2 /
ownership / EdgeGRAFT / CandidateGRAFT / motion / DeepCenter veto) → submission.

Numbered `python -m biohub.train.01_p1` modules are CLI shims only. The
training code lives in named modules (`biohub.train.detector`,
`biohub.train.division`, `biohub.train.edgegraft_ranker`, …). Every argparse
flag for a stage is in the matching YAML under `configs/` (and a short
`configs/smoke/` variant).

| Stage | What is trained | Train | Artifact | Infer | Depends on |
| --- | --- | --- | --- | --- | --- |
| 01 P1 | UNet + transformer detector | `python -m biohub.train.01_p1 --config configs/01_p1.yaml` | `runs/01_p1/.../edge_predictor_best.pth` | `python -m biohub.infer.01_detect` | data |
| 01 P2 | Same trainer, 199-train split, seed 314159 | `python -m biohub.train.01_p2 --config configs/01_p2.yaml` | `runs/01_p2/...` | fused in detect | 01 P1 data |
| 02 Model C | Detector trainer on Model C split | `python -m biohub.train.02_model_c --config configs/02_model_c.yaml` | Model C weights | native evidence in detect | 01 |
| 02 Division | Daughter-pair MLP | `python -m biohub.train.02_division --config configs/02_division.yaml` | `division_pair_model_best.pt` | graph division | 02 Model C |
| 02 Decoder | Model-C event decoder | `python -m biohub.train.02_division_decoder --config configs/02_division_decoder.yaml` | `pair_model.joblib` / `source_model.joblib` | wrapped in cardinality | 02 Division |
| 03 Cardinality | OptionHead CONTINUE vs DIVIDE | `python -m biohub.train.03_cardinality --config configs/03_cardinality.yaml` | `source_cardinality_head.pt` | graph UG1 | 02 |
| 04 UniGRAFT | P1/P2-only OptionHead | `python -m biohub.train.04_unigraft --config configs/04_unigraft.yaml` | `p1p2_only_source_cardinality_head.pt` | graph UG2 | 03 |
| 05 live_v2 | runtime only | — | bundle runtime | graph specialist | 04 |
| 06 Ownership | ExtraTrees OOF scores | `python -m biohub.train.06_ownership --config configs/06_ownership.yaml` | `all_source_scores.parquet` | graph ownership | 05 |
| 07 EdgeGRAFT ranker | HGB ranker folds | `python -m biohub.train.07_edgegraft_ranker --config configs/07_edgegraft_ranker.yaml` | `fold_*.joblib` | graph stage | detect |
| 07 EdgeGRAFT gate | HGB transaction gate | `python -m biohub.train.07_edgegraft_gate --config configs/07_edgegraft_gate.yaml` | `fold_*.joblib` | graph stage | ranker |
| 07 EdgeGRAFT oracle | component oracle audit | `python -m biohub.train.07_edgegraft_oracle --config configs/07_edgegraft_oracle.yaml` | `oracle.json` | — | gate |
| 08 CandidateGRAFT | OOF screen then direct-edge fit | screen: `configs/08_candidategraft_screen.yaml`; fit: `python -m biohub.train.08_candidategraft --config configs/08_candidategraft.yaml` | `candidategraft_direct.joblib` | graph | 07 |
| 09 Motion | residual cost MLP | `python -m biohub.train.09_motion --config configs/09_motion.yaml` | `motion_corrector_best.pt` | relink | detect |
| 09 Motion cache | fused edge-head finetune | `python -m biohub.train.09_motion_cache --config configs/09_motion_cache.yaml` | `edge_predictor_best.pth` | detect fusion | 01 P1 + 01 P2 |
| 10 DeepCenter | full-frame U-Net | `python -m biohub.train.10_deepcenter --config configs/10_deepcenter.yaml` | `best.pt` | gap/division veto | detect |

Val is the held split named in each YAML (`split:` / `val_fraction` / OOF folds).
Movie stages use 5-fold `GroupKFold` by video (`biohub.validation.cv.movie_group_kfold`).
Division uses a 2-fold embryo swap (`44b6↔6bba` via `embryo_two_fold`).
Smoke configs live in `configs/smoke/` (`configs/smoke/01_p1.yaml`, …): short
steps, `deterministic: true`, CUDA for neural stages (`device: cuda:0` /
`cpu: false`). `uv run pytest -m slow` is one-step helper vs package parity at
`atol=1e-6`. End-to-end helper vs package train smoke is
`runs/direct_smoke/compare.py` (GPU required). One-movie infer must match
`tests/fixtures/44b6_0113de3b_submission.csv`.

Trainers write TensorBoard scalars under `<output>/tensorboard`.

Scoring is `adj_edge_jaccard + 0.1 * division_jaccard` through
`evaluate → per_sample_metrics → summarise`. Never `evaluate_datasets().score`.
