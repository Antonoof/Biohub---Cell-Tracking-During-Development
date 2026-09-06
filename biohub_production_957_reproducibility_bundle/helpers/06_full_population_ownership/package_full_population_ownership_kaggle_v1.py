#!/usr/bin/env python3
"""Train, verify, and package the full-population ownership Kaggle candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

from benchmark_ownership_feature_groups_v3 import RANK, TOP


WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
BIO = Path("/home/tweak/bio")
BANK = BIO / "ug12_ownership_geometry_bank_v2/ownership_geometry.parquet"
FROZEN = BIO / "ownership_toprank_stage1_v1/summary.json"
EXPECTED = BIO / "ownership_full_population_oof_v1/selected_source_winners.parquet"
BASE_NOTEBOOK = WORKSPACE / "output/public-952-live-v2-global-bundle-v2.ipynb"
ARTIFACT = WORKSPACE / "artifacts/biohub-ownership-exact-v2"
OUTPUT_NOTEBOOK = WORKSPACE / "output/public-952-ownership-exact-v2.ipynb"
OUTPUT_ZIP = WORKSPACE / "output/biohub-ownership-exact-v2.zip"
RUNTIME_SOURCE = WORKSPACE / "scripts/ownership_full_population_runtime_v1.py"
FEATURES = [*TOP, *RANK]


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(np.float32)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_models() -> dict:
    if ARTIFACT.exists():
        shutil.rmtree(ARTIFACT)
    (ARTIFACT / "models").mkdir(parents=True)
    shutil.copy2(RUNTIME_SOURCE, ARTIFACT / RUNTIME_SOURCE.name)

    full = pd.read_parquet(BANK).reset_index(drop=True)
    train = full.loc[full.panel.eq("train175")].copy()
    known = train.loc[train.source_label.notna()].copy().reset_index(drop=True)
    known["target"] = (
        known.source_label.eq(1) & known.best_pair_label.eq(1)
    ).astype(np.int8)
    x_known = matrix(known)
    y = known.target.to_numpy(np.int8)
    groups = known.dataset.astype(str).to_numpy()
    threshold = float(json.loads(FROZEN.read_text())["threshold"])
    scored_parts = []
    records = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=5).split(x_known, y, groups)):
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=20,
            max_features=.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=324459 + fold,
        )
        model.fit(x_known[fit], y[fit])
        model_path = ARTIFACT / "models" / f"ownership_fold_{fold}.joblib"
        joblib.dump(model, model_path, compress=3)
        held_stems = set(map(str, known.iloc[held].dataset.unique()))
        local = train.loc[train.dataset.astype(str).isin(held_stems)].copy()
        local["score"] = model.predict_proba(matrix(local))[:, 1]
        local["fold"] = fold
        scored_parts.append(local)
        records.append({
            "fold": fold,
            "fit_known_rows": int(len(fit)),
            "held_known_rows": int(len(held)),
            "held_videos": int(len(held_stems)),
            "model_sha256": sha256(model_path),
        })

    scored = pd.concat(scored_parts, ignore_index=True)
    above = scored.loc[scored.score.ge(threshold)].sort_values(
        ["dataset", "source", "score"],
        ascending=[True, True, False],
        kind="stable",
    )
    winners = above.drop_duplicates(["dataset", "source"], keep="first")
    expected = pd.read_parquet(EXPECTED)
    key = ["dataset", "source", "a", "b"]
    actual_keys = winners[key].sort_values(key).reset_index(drop=True)
    expected_keys = expected[key].sort_values(key).reset_index(drop=True)
    if not actual_keys.equals(expected_keys):
        raise RuntimeError("Packaged five-fold models do not reproduce OOF winners")

    spec = {
        "version": "full-population-ownership-v1",
        "status": "candidate",
        "features": FEATURES,
        "threshold": threshold,
        "folds": 5,
        "model": {
            "type": "sklearn ExtraTreesClassifier",
            "n_estimators": 400,
            "max_depth": 3,
            "min_samples_leaf": 20,
            "max_features": 0.75,
            "class_weight": "balanced",
            "seed_base": 324459,
        },
        "serving": {
            "fit": "known-only grouped-video folds on train175",
            "unknown_sources_used_as_negatives": False,
            "hidden_model_route": "sha256(dataset) modulo 5; one unseen fold model",
            "winner": "highest candidate per source above frozen OOF threshold",
            "population_cap": None,
            "extra_image_or_gpu_pass": False,
            "order": "UG1 -> UG2 -> live V2 specialist -> ownership -> EdgeGRAFT V3 -> V17 cleanup",
        },
        "oof_parity": {
            "candidate_rows": int(len(train)),
            "selected_sources": int(len(winners)),
            "selected_videos": int(winners.dataset.nunique()),
            "expected_winner_keys_exact": True,
        },
        "versions": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "fold_records": records,
    }
    (ARTIFACT / "deploy_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (ARTIFACT / "manifest.json").write_text(json.dumps({
        "version": "biohub-ownership-exact-v2",
        "status": "candidate",
        "runtime": RUNTIME_SOURCE.name,
        "deploy_spec": "deploy_spec.json",
        "files": {
            path.relative_to(ARTIFACT).as_posix(): sha256(path)
            for path in sorted(ARTIFACT.rglob("*")) if path.is_file()
        },
    }, indent=2) + "\n")
    (ARTIFACT / "README.md").write_text(
        "# Biohub full-population ownership v1\n\n"
        "Attach this dataset to `public-952-ownership-exact-v2.ipynb`.\n"
        "It preserves all eligible hidden sources; sparse unknowns are not treated as negatives.\n"
    )
    return spec


def patch_notebook() -> None:
    notebook = nbformat.read(BASE_NOTEBOOK, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    cfg_cell = next(cell for cell in cells if '"live_v2_global_bundle": {' in cell.source)
    marker = "        # One full-population ownership repair after UG1/UG2, before cleanup."
    ownership_cfg = '''        # Broad source-ownership recovery. Unknown/unannotated sources are
        # never treated as negatives and there is no per-video source cap.
        "ownership_full_population": {
            "dataset": "tweakai/biohub-ownership-exact-v2",
            "dir": str(DATASETS_DIR / "tweakai" / "biohub-ownership-exact-v2"),
            "enabled": True,
        },
'''
    if marker not in cfg_cell.source:
        raise RuntimeError("Ownership CFG insertion marker not found")
    cfg_cell.source = cfg_cell.source.replace(marker, ownership_cfg + marker, 1)
    cfg_cell.source = cfg_cell.source.replace(
        '"experiment_tag": "biohub_public952_ug12_live_v2_global_bundle_v2"',
        '"experiment_tag": "biohub_public952_ownership_exact_v2"',
        1,
    )

    resolver_cell = next(cell for cell in cells if "# EdgeGRAFT V3 runtime" in cell.source)
    resolver_marker = "# EdgeGRAFT V3 runtime and frozen five-fold full-population gates."
    resolver = '''# Full-population ownership artifact. Resolve nested Kaggle uploads safely.
OWNERSHIP_FULL_DIR = Path(CFG["models"]["ownership_full_population"]["dir"])
if not (OWNERSHIP_FULL_DIR / "ownership_full_population_runtime_v1.py").is_file():
    candidates = sorted({
        path.parent
        for path in INPUT_DIR.rglob("ownership_full_population_runtime_v1.py")
        if (path.parent / "deploy_spec.json").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one full-population ownership artifact; "
            f"found {candidates}"
        )
    OWNERSHIP_FULL_DIR = candidates[0]
    CFG["models"]["ownership_full_population"]["dir"] = str(OWNERSHIP_FULL_DIR)

'''
    if resolver_marker not in resolver_cell.source:
        raise RuntimeError("Ownership resolver marker not found")
    resolver_cell.source = resolver_cell.source.replace(resolver_marker, resolver + resolver_marker, 1)

    post_cell = next(cell for cell in cells if "def init_division_runtime():" in cell.source)
    constants = '''LIVE_V2_BUNDLE_DIR = Path(_M["live_v2_global_bundle"]["dir"])
LIVE_V2_BUNDLE_ENABLED = bool(_M["live_v2_global_bundle"]["enabled"])
'''
    replacement_constants = constants + '''
OWNERSHIP_FULL_DIR = Path(_M["ownership_full_population"]["dir"])
OWNERSHIP_FULL_ENABLED = bool(_M["ownership_full_population"]["enabled"])
'''
    if constants not in post_cell.source:
        raise RuntimeError("Ownership postprocess constants marker not found")
    post_cell.source = post_cell.source.replace(constants, replacement_constants, 1)

    old_wrapper = '''    helper = _load_module("biohub_live_v2_global_bundle_runtime", helper_path)
    DIVISION_RUNTIME = helper.LiveV2GlobalBundleRuntime(ug12_runtime)
    return DIVISION_RUNTIME
'''
    new_wrapper = '''    helper = _load_module("biohub_live_v2_global_bundle_runtime", helper_path)
    live_runtime = helper.LiveV2GlobalBundleRuntime(ug12_runtime)
    if not OWNERSHIP_FULL_ENABLED:
        DIVISION_RUNTIME = live_runtime
        return DIVISION_RUNTIME
    ownership_path = OWNERSHIP_FULL_DIR / "ownership_full_population_runtime_v1.py"
    if not ownership_path.is_file():
        raise FileNotFoundError(f"Missing full-population ownership runtime: {ownership_path}")
    ownership = _load_module("biohub_full_population_ownership_runtime", ownership_path)
    DIVISION_RUNTIME = ownership.FullPopulationOwnershipRuntime(
        live_runtime, OWNERSHIP_FULL_DIR
    )
    return DIVISION_RUNTIME
'''
    if old_wrapper not in post_cell.source:
        raise RuntimeError("Ownership division-wrapper marker not found")
    post_cell.source = post_cell.source.replace(old_wrapper, new_wrapper, 1)

    notebook.cells[0].source = """# Multi-UniGRAFT `.952` + live V2 specialist + full-population ownership

All eligible source claims are scored; sparse unannotated sources are not false by default. Ownership runs atomically before EdgeGRAFT V3 and unchanged V17 cleanup.

## Exact all-175 host-patched comparison

| Stage | Division TP / FP / FN | Division J | Adjusted edge J | Composite proxy |
|---|---:|---:|---:|---:|
| Frozen UG1/UG2 control | 71 / 70 / 60 | 0.353234 | 0.906092 | 0.941415 |
| + live V2 specialist | 79 / 69 / 52 | 0.395000 | 0.907199 | 0.946699 |
| + full-population ownership | 89 / 72 / 42 | 0.438424 | 0.921017 | 0.964860 |

The ownership layer's exact marginal delta over the specialist is **+10 TP / +3 FP / -10 FN**, with **+0.018160** composite proxy. The complete stack's delta over frozen UG1/UG2 is **+18 TP / +2 FP / -18 FN**, with **+0.023444** composite proxy.

The earlier **11 TP / 0 known FP** result was a known-only forensic diagnostic before complete graph finalization; it is not presented as the final all-175 result. These are local production-matched diagnostics, not hidden-test scores.
"""
    notebook.metadata.setdefault("biohub", {})["full_population_ownership"] = {
        "version": "v1",
        "base": str(BASE_NOTEBOOK),
        "dataset_slug": "tweakai/biohub-ownership-exact-v2",
        "population_cap": None,
    }
    for index, cell in enumerate(cells):
        if cell.source.lstrip().startswith(("%%", "!")):
            continue
        compile(cell.source, f"cell_{index}", "exec")
    nbformat.write(notebook, OUTPUT_NOTEBOOK)


def make_zip() -> None:
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ARTIFACT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(ARTIFACT).as_posix())


def main() -> None:
    spec = train_models()
    patch_notebook()
    make_zip()
    payload = {
        "dataset_directory": str(ARTIFACT),
        "dataset_zip": str(OUTPUT_ZIP),
        "notebook": str(OUTPUT_NOTEBOOK),
        "notebook_sha256": sha256(OUTPUT_NOTEBOOK),
        "zip_sha256": sha256(OUTPUT_ZIP),
        "selected_oof_sources": spec["oof_parity"]["selected_sources"],
    }
    (WORKSPACE / "output/ownership-exact-package-v2.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
