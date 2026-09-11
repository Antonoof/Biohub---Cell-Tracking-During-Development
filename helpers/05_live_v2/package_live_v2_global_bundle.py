#!/usr/bin/env python3
"""Create the corrected Kaggle dataset package for the live V2 specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


WORKSPACE = Path(".")
VERSION = "biohub-live-v2-global-bundle-v2"
RUNTIME_NAME = "live_v2_global_bundle_runtime_v1.py"
SOURCE_RUNTIME = WORKSPACE / "scripts" / RUNTIME_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "output",
    )
    args = parser.parse_args()
    package = args.output_root / VERSION
    archive = args.output_root / f"{VERSION}.zip"
    if package.exists() or archive.exists():
        raise RuntimeError(f"Refusing to overwrite existing v2 package: {package} or {archive}")
    package.mkdir(parents=True)
    runtime = package / RUNTIME_NAME
    shutil.copy2(SOURCE_RUNTIME, runtime)

    manifest = {
        "version": VERSION,
        "status": "candidate",
        "runtime": RUNTIME_NAME,
        "runtime_sha256": sha256(runtime),
        "source_notebook": ".952-V17-division-focused.ipynb",
        "source_notebook_sha256": "7b6c70b6fa0993fd9f3d4bfcce185a06b2c94bb48e43ecabcd26c5276398c7b1",
        "graph_order": "UG1+UG2 -> specialist -> EdgeGRAFT V3 -> V17 cleanup",
        "placement": "after independent UG1/UG2 ownership and before EdgeGRAFT V3",
        "contains_model_weights": False,
        "contains_ground_truth": False,
        "contains_video_or_source_ids": False,
        "extra_image_or_gpu_pass": False,
        "fallback": "completed UG1/UG2 graph",
        "degree_contract": {"maximum_in_degree": 1, "maximum_out_degree": 2},
        "matched_current_v17_validation": {
            "scope": "all 42 structurally changed train videos",
            "division_tp_delta": 8,
            "division_fp_delta": 0,
            "division_fn_delta": -8,
            "changed_subset_score_delta": 0.009062701535055062,
            "absolute_all175_jaccard_claimed": False,
        },
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (package / "README.md").write_text(
        """# Biohub live V2 global division bundle v2

Corrected CPU-only missed-division specialist for the frozen `.952` V17
UG1/UG2 production notebook.

Inference order:

`UG1+UG2 -> specialist -> EdgeGRAFT V3 -> V17 cleanup`

The helper reuses the already materialized UG2 128-pair V2 population and
native Model-C/P1/P2 evidence. It performs no image inference or additional
GPU pass. It contains no GT, video/source decision table, or cached hidden
decisions. Existing forks are protected, changes are atomic, and a helper
failure returns the completed UG1/UG2 graph.

Matched current-order validation on all 42 structurally changed train videos:

- division: 22/36/31 -> 30/36/23;
- exact delta: +8 TP, 0 FP, -8 FN;
- adjusted edge: 0.891539 -> 0.891613 on the changed-video panel;
- composite proxy: +0.009063 on that changed-video panel.

No absolute all-175 V17 Jaccard is claimed. The older 0.455497 -> 0.502618
result belongs to a different boundary-finalized UG2+UG3 substrate and is not
the validation basis for this package.
""",
        encoding="utf-8",
    )

    checksums = {
        path.name: sha256(path)
        for path in sorted(package.iterdir())
        if path.is_file()
    }
    (package / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.iterdir()):
            if path.is_file():
                handle.write(path, arcname=f"{VERSION}/{path.name}")
    print(package)
    print(archive)


if __name__ == "__main__":
    main()
