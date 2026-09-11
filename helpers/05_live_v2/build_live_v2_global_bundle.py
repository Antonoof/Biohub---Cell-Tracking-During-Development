#!/usr/bin/env python3
"""Build the corrected .952 V17 UG1/UG2 + live V2 specialist notebook."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import build_public952_live_v2_global_bundle_v1 as legacy_builder


WORKSPACE = Path(".")
VERSION = "biohub-live-v2-global-bundle-v2"
RUNTIME = "live_v2_global_bundle_runtime_v1.py"


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def build(input_path: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="live-v2-notebook-") as directory:
        intermediate = Path(directory) / "v1.ipynb"
        legacy_builder.build(input_path, intermediate)
        notebook = json.loads(intermediate.read_text(encoding="utf-8"))

    cells = notebook["cells"]
    for cell in cells:
        text = source(cell)
        text = text.replace(
            "biohub-live-v2-global-bundle-v1",
            VERSION,
        )
        text = text.replace(
            "biohub_public952_ug12_live_v2_global_bundle_v1",
            "biohub_public952_ug12_live_v2_global_bundle_v2",
        )
        set_source(cell, text)

    set_source(
        cells[0],
        """# Multi-UniGRAFT `.952` V17 + global missed-division specialist

This candidate preserves the frozen `.952` P1/P2, UG1/UG2, boundary-rescue,
motion, EdgeGRAFT V3, and V17 cleanup flow. The new CPU-only specialist wraps
the completed independent UG1/UG2 runtime and runs exactly once before
EdgeGRAFT and cleanup. It reuses UG2's existing 128-pair V2 population and
native Model-C/P1/P2 evidence; it adds no image inference or GPU pass.

The runtime contains no GT labels, video/source decision table, or cached
hidden decisions. It protects existing forks and applies only atomic
parent-to-two-daughter transactions under in-degree <= 1 and out-degree <= 2.
Any specialist failure returns the completed UG1/UG2 graph unchanged; the
outer per-video baseline shard remains the submission-safe fallback.

Validation statement: on all 42 train videos structurally changed by the
specialist, both arms received identical EdgeGRAFT V3 and exact V17 cleanup.
The matched delta was +8 division TP, 0 FP, and -8 FN. No absolute all-175
V17 Jaccard is claimed because a complete matched V17 absolute control has not
been materialized.
""",
    )

    validation = source(cells[2])
    old = '''# Live global V2 bundle runtime. Kaggle may retain a nested ZIP directory.
LIVE_V2_BUNDLE_DIR = Path(CFG["models"]["live_v2_global_bundle"]["dir"])
if not (LIVE_V2_BUNDLE_DIR / "live_v2_global_bundle_runtime_v1.py").is_file():
    candidates = sorted({
        path.parent
        for path in INPUT_DIR.rglob("live_v2_global_bundle_runtime_v1.py")
        if (path.parent / "manifest.json").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one live V2 bundle artifact, found {candidates}")
    LIVE_V2_BUNDLE_DIR = candidates[0]
    CFG["models"]["live_v2_global_bundle"]["dir"] = str(LIVE_V2_BUNDLE_DIR)
for required in (
    LIVE_V2_BUNDLE_DIR / "live_v2_global_bundle_runtime_v1.py",
    LIVE_V2_BUNDLE_DIR / "manifest.json",
):
    if not required.is_file():
        raise FileNotFoundError(f"Incomplete live V2 bundle artifact: {required}")

'''
    new = f'''# Corrected live V2 bundle. Discovery is manifest-version scoped so the
# quarantined v1 dataset cannot be selected when both artifacts are attached.
LIVE_V2_BUNDLE_DIR = Path(CFG["models"]["live_v2_global_bundle"]["dir"])

def _valid_live_v2_bundle_v2(path):
    manifest_path = path / "manifest.json"
    runtime_path = path / "{RUNTIME}"
    if not manifest_path.is_file() or not runtime_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("version") == "{VERSION}" and manifest.get("status") == "candidate"

if not _valid_live_v2_bundle_v2(LIVE_V2_BUNDLE_DIR):
    candidates = sorted({{
        path.parent
        for path in INPUT_DIR.rglob("manifest.json")
        if _valid_live_v2_bundle_v2(path.parent)
    }})
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one corrected live V2 bundle v2 artifact; "
            f"found {{candidates}}"
        )
    LIVE_V2_BUNDLE_DIR = candidates[0]
    CFG["models"]["live_v2_global_bundle"]["dir"] = str(LIVE_V2_BUNDLE_DIR)

'''
    validation = replace_once(validation, old, new, "version-scoped artifact validation")
    set_source(cells[2], validation)

    notebook["metadata"]["live_v2_global_bundle"] = {
        "version": "v2",
        "dataset": f"tweakai/{VERSION}",
        "base": str(input_path),
        "base_sha256": "7b6c70b6fa0993fd9f3d4bfcce185a06b2c94bb48e43ecabcd26c5276398c7b1",
        "graph_order": "UG1+UG2 -> specialist -> EdgeGRAFT V3 -> V17 cleanup",
        "extra_v2_or_gpu_pass": False,
        "contains_gt_or_ids": False,
        "validation": "+8 TP / 0 FP / -8 FN on all 42 structurally changed train videos",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/c/Kaggle/.952-V17-division-focused.ipynb"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "output/public-952-live-v2-global-bundle-v2.ipynb",
    )
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
