#!/usr/bin/env python3
"""Package the independent P1/P2 UG2 branch on the current `.949` notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"historical\public-949-production-registration-step-v1.ipynb")
DEFAULT_OUTPUT = Path(r"historical\public-949-unigraft-p1p2-independent-v1.ipynb")
DEFAULT_STAGE = Path(r"historical\biohub-unigraft-p1p2-independent-v1")
DEFAULT_ZIP = Path(r"historical\biohub-unigraft-p1p2-independent-v1.zip")
DEFAULT_HEAD = (
    WORKSPACE
    / "artifacts"
    / "biohub-unigraft-p1p2-independent-v1"
    / "p1p2_only_source_cardinality_head.pt"
)
DEFAULT_SUMMARY = (
    WORKSPACE
    / "artifacts"
    / "biohub-unigraft-p1p2-independent-v1"
    / "training_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_text(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index].get("source", []))


def set_cell_text(notebook: dict, index: int, value: str) -> None:
    notebook["cells"][index]["source"] = value.splitlines(keepends=True)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def package_runtime(
    stage: Path,
    archive: Path,
    head: Path,
    summary: Path,
    source_hash: str,
) -> None:
    if not head.is_file() or not summary.is_file():
        raise FileNotFoundError(
            f"Missing packaged UG2 inputs: head={head} summary={summary}"
        )
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    copies = {
        WORKSPACE / "scripts" / "unigraft_p1p2_late_ensemble_runtime.py":
            "unigraft_p1p2_late_ensemble_runtime.py",
        WORKSPACE / "scripts" / "train_public934_p1p2_only_cardinality_head_v1.py":
            "train_public934_p1p2_only_cardinality_head_v1.py",
        WORKSPACE / "scripts" / "extract_public934_cardinality_held20_ug2.py":
            "extract_public934_cardinality_held20_ug2.py",
        head: "p1p2_only_source_cardinality_head.pt",
        summary: "training_summary.json",
    }
    for source, name in copies.items():
        shutil.copy2(source, stage / name)

    readme = """# Biohub UniGRAFT independent P1/P2 branch v1

This package adds an independent second division branch to the production
`.949` notebook. The production Model-C/V2/cardinality transaction (UG1)
finishes first. UG2 independently scores the same input graph with V2 geometry
plus native P1/P2 evidence, without Model C, then merges only non-conflicting
parent-to-two-daughter transactions.

Safety contract:

- no additional detector or backbone pass;
- source threshold 0.51 frozen from grouped-video OOF;
- maximum 128 daughter-pair options per source;
- UG1 and UG2 score independently;
- UG2 is applied after UG1 and cannot replace a protected UG1 fork;
- scoring or merge failure returns the completed UG1 graph unchanged.

Validation warning: the completed held-20 graph replay used a stale `.944`
graph/finalization substrate rather than the exact current `.949` substrate.
It improved its matched stale control, but it is not a clean `.949` promotion
measurement. Kaggle is the transfer test for this package.
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "version": "biohub-unigraft-p1p2-independent-v1",
        "source_notebook_sha256": source_hash,
        "architecture": {
            "UG1": "frozen production Model-C/V2/source-cardinality runtime",
            "UG2": "V2 geometry plus P1/P2 evidence; Model C excluded",
            "merge": "UG1 first, then non-conflicting atomic UG2 transactions",
            "extra_backbone_passes": 0,
            "threshold": 0.51,
            "max_pairs": 128,
        },
        "held20_replay": {
            "candidate_adjusted_edge_jaccard": 0.9052609477674817,
            "candidate_division": {"tp": 5, "fp": 4, "fn": 12},
            "candidate_division_jaccard": 0.23809523809523808,
            "candidate_composite": 0.9290704715770055,
            "matched_stale_control_composite": 0.9201797020071443,
            "delta_vs_matched_stale_control": 0.0088907695698612,
            "promotion_valid_against_current_949": False,
            "mismatch": [
                "UG1 graph substrate was saved .944 cardinality output",
                "finalizer used line-fit window 2 rather than 3",
                "finalizer omitted the current node-budget stage",
            ],
        },
        "files": {},
    }
    for path in sorted(stage.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if archive.exists():
        archive.unlink()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(stage.iterdir()):
            handle.write(path, path.name)


def patch_notebook(source: Path, output: Path) -> str:
    source_hash = sha256(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 9:
        raise RuntimeError("Expected the current nine-cell production notebook")

    config = cell_text(notebook, 1)
    config = replace_once(
        config,
        '    "experiment_tag": "biohub_registration_step_model_8",',
        '    "experiment_tag": "biohub_public949_unigraft_p1p2_independent_v1",',
        "experiment tag",
    )
    model_anchor = '        # Learned residual on the motion re-link assignment cost.\n'
    model_insert = '''        # Independent second division branch: V2 geometry + P1/P2 only.\n        "unigraft_p1p2": {\n            "dataset": "tweakai/biohub-unigraft-p1p2-independent-v1",\n            "dir": str(DATASETS_DIR / "tweakai" / "biohub-unigraft-p1p2-independent-v1"),\n            "enabled": True,\n        },\n        # Learned residual on the motion re-link assignment cost.\n'''
    config = replace_once(config, model_anchor, model_insert, "UG2 model config")
    set_cell_text(notebook, 1, config)

    post = cell_text(notebook, 5)
    constants_anchor = '''SOURCE_CARDINALITY_ENABLED = bool(_M["source_cardinality"]["enabled"])\n\nMOTION_CORRECTOR_ENABLED = bool(_M["motion_corrector"]["enabled"])\n'''
    constants_insert = '''SOURCE_CARDINALITY_ENABLED = bool(_M["source_cardinality"]["enabled"])\n\nUNIGRAFT_P1P2_DIR = Path(_M["unigraft_p1p2"]["dir"])\nUNIGRAFT_P1P2_ENABLED = bool(_M["unigraft_p1p2"]["enabled"])\n\nMOTION_CORRECTOR_ENABLED = bool(_M["motion_corrector"]["enabled"])\n'''
    post = replace_once(post, constants_anchor, constants_insert, "UG2 constants")
    runtime_anchor = '''    DIVISION_RUNTIME = _load_module(\n        "biohub_source_cardinality_graph_runtime",\n        SOURCE_CARDINALITY_DIR / "source_cardinality_graph_runtime.py",\n    ).SourceCardinalityGraphRuntime(\n        gbm, SOURCE_CARDINALITY_DIR, MODEL_C_DIR, legacy_fallback_runtime=legacy\n    )\n    return DIVISION_RUNTIME\n'''
    runtime_insert = '''    primary_runtime = _load_module(\n        "biohub_source_cardinality_graph_runtime",\n        SOURCE_CARDINALITY_DIR / "source_cardinality_graph_runtime.py",\n    ).SourceCardinalityGraphRuntime(\n        gbm, SOURCE_CARDINALITY_DIR, MODEL_C_DIR, legacy_fallback_runtime=legacy\n    )\n    if not UNIGRAFT_P1P2_ENABLED:\n        DIVISION_RUNTIME = primary_runtime\n        return DIVISION_RUNTIME\n\n    runtime_path = UNIGRAFT_P1P2_DIR / "unigraft_p1p2_late_ensemble_runtime.py"\n    head_path = UNIGRAFT_P1P2_DIR / "p1p2_only_source_cardinality_head.pt"\n    if not runtime_path.is_file() or not head_path.is_file():\n        raise FileNotFoundError(\n            f"Incomplete independent UG2 dataset: runtime={runtime_path} head={head_path}"\n        )\n    module = _load_module("biohub_unigraft_p1p2_late_runtime", runtime_path)\n    DIVISION_RUNTIME = module.IndependentP1P2LateEnsembleRuntime(\n        primary_runtime, UNIGRAFT_P1P2_DIR\n    )\n    return DIVISION_RUNTIME\n'''
    post = replace_once(post, runtime_anchor, runtime_insert, "UG2 runtime wrapper")
    set_cell_text(notebook, 5, post)

    notebook["cells"][0]["source"] = [
        "# Biohub production `.949` + independent P1/P2 UniGRAFT branch\n",
        "\n",
        "This candidate keeps the current `.949` P1/P2 detector, association path, "
        "registration-aware motion correction, gap recovery, Model-C/V2 decoder, "
        "source-cardinality head, cleanup, node budget, line-fit window 3, four-worker "
        "streaming, and per-video fallback.\n",
        "\n",
        "The only causal change is an independent second division branch. UG1 completes "
        "normally. UG2 scores V2 geometry plus native P1/P2 evidence without Model C, "
        "then adds only non-conflicting forks atomically. Any UG2 failure preserves UG1.\n",
        "\n",
        "Validation warning: the completed held-20 replay improved its matched stale "
        "control by `+0.00889`, but that replay used a `.944` graph/finalizer substrate. "
        "It is supporting evidence, not a clean `.949` promotion gate.\n",
    ]
    metadata = notebook.setdefault("metadata", {})
    metadata["title"] = "Biohub .949 + Independent P1/P2 UniGRAFT"
    metadata.setdefault("kaggle", {})["title"] = metadata["title"]
    metadata["unigraft_p1p2"] = {
        "source_sha256": source_hash,
        "UG1": "production Model-C/V2/cardinality",
        "UG2": "V2 plus P1/P2 only",
        "threshold": 0.51,
        "max_pairs": 128,
        "extra_backbone_passes": 0,
        "fallback": "completed UG1 graph per video",
        "held20_current_949_gate": "not available; stale-substrate mismatch disclosed",
    }

    joined = "\n".join(cell_text(notebook, index) for index in range(9))
    required = {
        "experiment": "biohub_public949_unigraft_p1p2_independent_v1",
        "runtime": "IndependentP1P2LateEnsembleRuntime",
        "head": "p1p2_only_source_cardinality_head.pt",
        "guarded P2 zero": '"guarded_secondary_edge_weight": 0.0',
        "registration": '"relink_frame_registration": True',
        "per-axis motion": '"relink_velocity_weight_axes": [0.0, 0.45, 0.47]',
        "node budget": '"node_budget": True',
        "line fit 3": '"linefit_window": 3',
        "four CPUs": '"cpu_workers": 4',
    }
    missing = [label for label, marker in required.items() if marker not in joined]
    if missing:
        raise RuntimeError(f"Generated notebook is missing production controls: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    rebuilt = json.loads(output.read_text(encoding="utf-8"))
    serialized = "\n".join(cell_text(rebuilt, index) for index in range(9))
    if "IndependentP1P2LateEnsembleRuntime" not in serialized:
        raise RuntimeError("Serialized notebook lost the UG2 runtime wrapper")
    return source_hash


def main() -> None:
    args = parse_args()
    source_hash = patch_notebook(args.source, args.output)
    package_runtime(args.stage, args.zip, args.head, args.summary, source_hash)
    print(f"Source: {args.source}")
    print(f"Source SHA-256: {source_hash}")
    print(f"Notebook: {args.output}")
    print(f"Notebook SHA-256: {sha256(args.output)}")
    print(f"Runtime dataset ZIP: {args.zip}")
    print(f"Runtime ZIP SHA-256: {sha256(args.zip)}")


if __name__ == "__main__":
    main()
