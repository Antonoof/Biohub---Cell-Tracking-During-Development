#!/usr/bin/env python3
"""Offline release checks for the corrected live V2 Kaggle artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


WORKSPACE = Path(".")
SOURCE = Path("/mnt/c/Kaggle/.952-V17-division-focused.ipynb")
NOTEBOOK = WORKSPACE / "output/public-952-live-v2-global-bundle-v2.ipynb"
PACKAGE = WORKSPACE / "output/biohub-live-v2-global-bundle-v2"
ARCHIVE = WORKSPACE / "output/biohub-live-v2-global-bundle-v2.zip"
VERSION = "biohub-live-v2-global-bundle-v2"
RUNTIME = "live_v2_global_bundle_runtime_v1.py"
EXPECTED_SOURCE_SHA256 = "7b6c70b6fa0993fd9f3d4bfcce185a06b2c94bb48e43ecabcd26c5276398c7b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert sha256(SOURCE) == EXPECTED_SOURCE_SHA256, "production source changed"
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 9
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    joined = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert VERSION in joined
    assert "biohub-live-v2-global-bundle-v1" not in joined
    assert "UG3" not in joined
    assert joined.count("helper.LiveV2GlobalBundleRuntime(ug12_runtime)") == 1
    assert joined.count('"enabled": True,') >= 1
    metadata = notebook["metadata"]["live_v2_global_bundle"]
    assert metadata["version"] == "v2"
    assert metadata["graph_order"] == "UG1+UG2 -> specialist -> EdgeGRAFT V3 -> V17 cleanup"

    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == VERSION
    assert manifest["status"] == "candidate"
    assert manifest["source_notebook_sha256"] == EXPECTED_SOURCE_SHA256
    assert manifest["runtime_sha256"] == sha256(PACKAGE / RUNTIME)
    assert manifest["contains_ground_truth"] is False
    assert manifest["contains_video_or_source_ids"] is False
    assert manifest["extra_image_or_gpu_pass"] is False

    with zipfile.ZipFile(ARCHIVE) as handle:
        names = set(handle.namelist())
        expected = {
            f"{VERSION}/README.md",
            f"{VERSION}/SHA256SUMS.json",
            f"{VERSION}/manifest.json",
            f"{VERSION}/{RUNTIME}",
        }
        assert names == expected, (names, expected)
        with tempfile.TemporaryDirectory(prefix="live-v2-nested-") as directory:
            root = Path(directory) / "kaggle/input/datasets/tweakai/uploaded-slug/nested"
            root.mkdir(parents=True)
            handle.extractall(root)
            candidates = []
            for path in root.rglob("manifest.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("version") == VERSION
                    and payload.get("status") == "candidate"
                    and (path.parent / RUNTIME).is_file()
                ):
                    candidates.append(path.parent)
            assert len(candidates) == 1, candidates

    print(
        json.dumps(
            {
                "source_sha256": sha256(SOURCE),
                "notebook_sha256": sha256(NOTEBOOK),
                "archive_sha256": sha256(ARCHIVE),
                "compiled_code_cells": len(code_cells),
                "nested_artifact_candidates": 1,
                "ug3_references": 0,
                "specialist_wrappers": 1,
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
