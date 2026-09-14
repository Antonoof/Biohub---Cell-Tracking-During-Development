# Honest pipeline RUNBOOK — stage-h200-node2

Root: `/data/projects/ryzhichkin/biohub/honest_pipeline`  
Python: `/data/projects/ryzhichkin/biohub/.venv/bin/python`  
Data: `/data/projects/ryzhichkin/biohub/kaggle/input/competitions/biohub-cell-tracking-during-development/train`  
Free GPUs: **3,4,5,6,7** (0–1 busy)

Contract everywhere: **5-fold GroupKFold by movie on train175**, OOF+logs+weights per fold, thresholds only via **mean-over-folds on OOF** (never held20/practice). held20 = score-once after freeze.

```bash
export BIO=/data/projects/ryzhichkin/biohub
export HP=$BIO/honest_pipeline
export PY=$BIO/.venv/bin/python
export TRAIN=$BIO/kaggle/input/competitions/biohub-cell-tracking-during-development/train
export WILLIAM=$BIO/william-duckworth-reproducible-training-pipeline
cd $HP
$PY scripts/verify_splits.py
```

---

## Stage order & ETA

| # | Stage | Needs | Parallel | ETA (full) | Metrics to watch |
|---|---|---|---|---|---|
| 1 | **0_917 Exp203 detector** | public weights | GPU 6/7 | ~2 h / 199 | `runs/p1_candidate_compare/kaggle_train_all/classical_exp203` adj **0.870** |
| 2 | **P2 detector** | — | — | — | **dropped** (stack replaces P1+P2) |
| 3 | Assemble OOF index | P1/P2 fold weights | CPU | minutes | `assembled_*/summary.json` → `ready_for_oof_predict` |
| 4 | OOF predict + ILP graphs | P1/P2 OOF | multi-GPU | hours | graph Jaccard later |
| 5 | **DeepCenter** GKF5 | raw | 5 GPUs | **~2–6 h** / fold @30ep | `runs/11_deepcenter/*/weights/history.csv`, gate_summary.json |
| 6 | **Motion** GKF5 | proposals bank | 5 GPUs | **~1–3 h** / fold if cache exists | `runs/04_motion_corrector/*/weights/metrics.json` Jaccard |
| 7 | Model C decoder | evidence banks | CPU/GPU light | **~30–90 min** | `oof_*.parquet`, summary threshold OOF-only |
| 8 | Cardinality / UniGRAFT | graph+evidence | CPU | **~20–60 min** | OOF Jaccard + threshold_sweep |
| 9 | Ownership | geometry parquet | CPU | **~10–30 min** | `oof_gkf_movie.parquet`, threshold |
| 10 | EdgeGRAFT / CandidateGRAFT | labels on **OOF graphs** | CPU | **~30–90 min** | oof parquet; never stale `.951/.952` |

**Detector is 0_917 Exp203** (P1/P2 dropped). DeepCenter GKF5 already trained. Next: freeze gate + rebuild motion proposals from 0_917 graphs.

```bash
$PY $HP/scripts/freeze_0917_stack_oof.py
$PY $HP/scripts/export_exp203_proposals.py \
  --geff-dir $HP/runs/p1_candidate_compare/kaggle_train_all/classical_exp203 \
  --data-dir $TRAIN \
  --out-dir $BIO/data/exp203_0917_proposals
```

---

## 1) P1 — 5 folds on GPUs 3–7

> **H200 note:** first full launch crashed on Flash-SDPA (`CUDA error: invalid configuration argument`). Fixed in trainer (`math` SDPA only). Re-run after sync.

```bash
cd $HP
chmod +x scripts/launch_gkf5_parallel.sh

GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 FOLD_FLAG=--split TAG=p1_gkf5_ep50 \
STAGE_DIR=runs/_parallel \
./scripts/launch_gkf5_parallel.sh -- \
  $PY stages/01_p1_p2/launch_detector_train.py \
    --scheme gkf_movie --epochs 50 --tag p1_gkf5_ep50 --stage 01_p1_detector
```

Logs: `runs/_parallel/p1_gkf5_ep50/fold_*_gpu*.log`  
Weights: `runs/01_p1_detector/*p1_gkf5_ep50_split*/weights/.../edge_predictor_best.pth`

Assemble index after all 5 finish:

```bash
$PY scripts/assemble_detector_oof_index.py --stage 01_p1_detector --tag p1_gkf5_ep50 --scheme gkf_movie
```

Smoke (optional, 2 iters): already validated earlier.

---

## 2) P2 — same, independent seed/method (never alltrain)

```bash
GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 FOLD_FLAG=--split TAG=p2_gkf5_ep50 \
./scripts/launch_gkf5_parallel.sh -- \
  $PY stages/01_p1_p2/launch_detector_train.py \
    --scheme gkf_movie --epochs 50 --tag p2_gkf5_ep50 --stage 02_p2_detector \
    --method honest_02_p2_detector_gkf_movie_seed7
```

```bash
$PY scripts/assemble_detector_oof_index.py --stage 02_p2_detector --tag p2_gkf5_ep50 --scheme gkf_movie
```

---

## 3) DeepCenter — GKF5 (fixes one-embryo train)

```bash
# First sync patched train_full_frame_center_detector.py into WILLIAM (see sync note below)

GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 FOLD_FLAG=--fold TAG=dc_gkf5_ep30 \
./scripts/launch_gkf5_parallel.sh -- \
  $PY stages/11_deepcenter/launch_fold.py --scheme gkf_movie --epochs 30 --tag dc_gkf5_ep30
```

Metrics: `runs/11_deepcenter/*/weights/{history.csv,gate_summary.json,split_manifest.json}`  
Confirm `split_manifest.json` has `"scheme": "gkf_movie"` and both embryos in train.

---

## 4) Motion — only when proposals exist

```bash
# Set PROPOSALS to real bank when available, e.g.:
# export PROPOSALS=$BIO/data/ab_proposals_export/biohub_ab_proposals

GPUS=3,4,5,6,7 FOLDS=0,1,2,3,4 FOLD_FLAG=--fold TAG=motion_gkf5 \
./scripts/launch_gkf5_parallel.sh -- \
  $PY stages/04_motion/launch_fold.py \
    --tag motion_gkf5 --epochs 40 --proposals "$PROPOSALS" --parent-mode proposal_nn
```

`--parent-mode proposal_nn` = no GT teacher forcing (default).

---

## 5) Ownership (CPU, needs geometry bank)

```bash
$PY stages/08_ownership/launch_oof.py \
  --bank /path/to/ownership_geometry.parquet \
  --tag ownership_gkf5_v1
```

Look at: `runs/08_ownership/*/summary.json` (`threshold`, `mean_metric`), `oof_gkf_movie.parquet`.

---

## 6) Model C decoder (when evidence banks ready)

```bash
# From WILLIAM helper; output dir under honest runs:
OUT=$HP/runs/05_model_c/$(date -u +%Y%m%dT%H%M%SZ)_decoder
mkdir -p "$OUT"
$PY $WILLIAM/helpers/02_model_c/train_model_c_v2_event_decoder.py \
  --output "$OUT" \
  # ... pass evidence/cache/split args as in docs/TRAINING_GUIDE.md ...
```

Patched behavior:
- practice / promotion use **OOF-frozen** threshold only (held20 calibration banned)
- writes `oof_*.parquet`
- threshold = **mean-over-folds** Jaccard

---

## Where to look (every stage)

```
$HP/runs/<stage>/<utc>_<tag>/
  config.json env.json train.log summary.json
  weights/          # checkpoints
  oof_*.parquet     # when tabular / decoder
  folds/            # fold joblibs
```

Parallel launcher logs: `$HP/runs/_parallel/<TAG>/fold_*_gpu*.log`

Splits (shared): `$HP/splits/canonical_splits.json`, `dataset_splits_gkf5_train175.json`

---

## Sync note (run once after pulling code)

Patched files live in the git repo `helpers/`. Copy into WILLIAM on the node:

```bash
rsync -a $BIO/../  # or from laptop:
# From your Mac repo:
# rsync -az helpers/01_p1_p2_base/shared_repo/scripts/train_unet_transformer.py \
#   helpers/01_p1_p2_base/shared_repo/scripts/dataspec.py \
#   stage-h200-node2:$WILLIAM/helpers/01_p1_p2_base/shared_repo/scripts/
# similarly for model_c, deepcenter, motion, and honest_pipeline/
```

Recommended from Mac (in repo root):

```bash
rsync -az honest_pipeline/ stage-h200-node2:/data/projects/ryzhichkin/biohub/honest_pipeline/
rsync -az helpers/01_p1_p2_base/shared_repo/scripts/train_unet_transformer.py \
  helpers/01_p1_p2_base/shared_repo/scripts/dataspec.py \
  stage-h200-node2:$WILLIAM/helpers/01_p1_p2_base/shared_repo/scripts/
rsync -az helpers/02_model_c/train_model_c_v2_event_decoder.py \
  stage-h200-node2:$WILLIAM/helpers/02_model_c/
rsync -az helpers/10_deepcenter/train_full_frame_center_detector.py \
  stage-h200-node2:$WILLIAM/helpers/10_deepcenter/
rsync -az helpers/09_motion_corrector/TRAINING_V1/train_motion_cost_corrector.py \
  stage-h200-node2:$WILLIAM/helpers/09_motion_corrector/TRAINING_V1/
```

Also copy DeepCenter into honest launcher fallback path if WILLIAM missing the CLI flags — launcher prefers WILLIAM then local helpers.

---

## Honest metric rule

For each stage report **only**:
1. GKF5 OOF metric (mean ± std over 5 folds)  
2. Threshold/blend chosen by inner/mean-over-fold OOF  
3. Optional one-shot held20 **after** freeze (never for selection)

Do not compare to William’s contaminated P2-alltrain cascade numbers.
