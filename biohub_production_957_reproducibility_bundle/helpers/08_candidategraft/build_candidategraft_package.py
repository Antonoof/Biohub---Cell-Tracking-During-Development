#!/usr/bin/env python3
"""Package CandidateGRAFT and attach it to the canonical .952 notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
DEFAULT_SOURCE = Path("/mnt/c/Kaggle/best.952-division-focused.ipynb")
DEFAULT_OUTPUT = Path("/mnt/c/Kaggle/best.952-candidategraft-direct-v1.ipynb")
DEFAULT_STAGE = WORKSPACE / "output/biohub-candidategraft-direct-v1"
DEFAULT_ZIP = WORKSPACE / "output/biohub-candidategraft-direct-v1.zip"
DEFAULT_MODEL = Path("/home/tweak/bio/candidategraft_direct_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_runtime(stage: Path, archive: Path, model: Path) -> dict:
    if stage.exists() or archive.exists():
        raise RuntimeError(f"Refusing to overwrite: {stage} / {archive}")
    stage.mkdir(parents=True)
    shutil.copy2(
        WORKSPACE / "scripts/candidategraft_direct_runtime_v1.py",
        stage / "candidategraft_direct_runtime_v1.py",
    )
    for name in ("candidategraft_direct.joblib", "report.json"):
        shutil.copy2(model / name, stage / name)
    readme = """# Biohub CandidateGRAFT Direct v1

Inference-only, add-only continuation recovery for the `.952` production graph.

- reuses P1/P2 native association evidence already exported in the GPU pass;
- adds only a one-frame edge from a childless source to a parentless target;
- never deletes or replaces an edge and never creates a fork;
- preserves all existing UniGRAFT division topology;
- trained only on annotated known rows; unannotated rows were never negatives;
- grouped-video OOF gate fixed at 0.90 before the final all-known refit.

No images, GT, video-specific decisions, or cached submission rows are included.
"""
    (stage / "README.md").write_text(readme)
    manifest = {
        "version": "biohub-candidategraft-direct-v1",
        "threshold": 0.90,
        "training_contract": "known-only labels; unknown rows excluded",
        "requires": ["P1 native evidence", "P2 native evidence", "fused_graph_node_id"],
        "exact_oof_control": 0.9722101726407437,
        "exact_oof_grouped_score": 0.9739415278548504,
        "exact_final_fit_score": 0.9741073155567402,
        "exact_final_fit_delta": 0.0018971429159965,
        "division_unchanged": {"tp": 96, "fp": 60, "fn": 35, "jaccard": 0.5026178010471204},
        "files": {},
    }
    for path in sorted(stage.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(stage.iterdir()):
            handle.write(path, path.name)
    return manifest


def find_cell(notebook: dict, marker: str) -> tuple[int, str]:
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if marker in source:
            matches.append((index, source))
    if len(matches) != 1:
        raise RuntimeError(f"{marker!r}: expected one cell, found {len(matches)}")
    return matches[0]


def build_notebook(source: Path, output: Path) -> None:
    notebook = json.loads(source.read_text())
    notebook["cells"][0]["source"] = (
        "# Multi-UniGRAFT `.952` + CandidateGRAFT direct-edge recovery\n\n"
        "This one-change candidate preserves the complete hidden-scored `.952` "
        "P1/P2 + UG1/UG2, live V2 ownership, full-population ownership, "
        "EdgeGRAFT V3, boundary rescue, DeepCenter gap veto, cleanup, four-worker "
        "streaming, and atomic fallback. CandidateGRAFT runs once on the final "
        "cleaned graph using the already-exported P1/P2 native association tables.\n\n"
        "CandidateGRAFT can only add a one-frame continuation between two retained "
        "nodes when the source has no child and the target has no parent. It never "
        "deletes or replaces an edge, never creates a fork, and preserves every "
        "existing UniGRAFT division. A runtime error leaves the already-written "
        "`.952` baseline shard untouched.\n\n"
        "Exact 175-video production-order control: score `0.972210`. Grouped-video "
        "OOF CandidateGRAFT: `0.973942` (`+0.001731`). Final all-known deployment "
        "fit replay: `0.974107` (`+0.001897`), with division fixed at "
        "`96 TP / 60 FP / 35 FN`, Jaccard `0.502618`.\n"
    ).splitlines(keepends=True)

    config_index, config = find_cell(notebook, '"edgegraft": {')
    config_anchor = '''        "edgegraft": {
            "dataset": "tweakai/biohub-edgegraft-v3-full-population-v1",
            "dir": str(DATASETS_DIR / "tweakai" / "biohub-edgegraft-v3-full-population-v1"),
            "enabled": True,
        },
'''
    config_insert = config_anchor + '''        # Add-only continuation recovery on the final cleaned graph.
        "candidategraft": {
            "dataset": "tweakai/biohub-candidategraft-direct-v1",
            "dir": str(DATASETS_DIR / "tweakai" / "biohub-candidategraft-direct-v1"),
            "enabled": True,
        },
'''
    config = replace_once(config, config_anchor, config_insert, "CFG CandidateGRAFT")
    config = replace_once(
        config,
        '"experiment_tag": "biohub_public952_ownership_exact_v2_deepcenter_gap_veto_v1"',
        '"experiment_tag": "biohub_public952_candidategraft_direct_v1"',
        "experiment tag",
    )
    notebook["cells"][config_index]["source"] = config.splitlines(keepends=True)

    artifact_index, artifact = find_cell(notebook, "# EdgeGRAFT V3 runtime and frozen")
    artifact_anchor = '''for family in ("ranker", "metric"):
    for fold in range(5):
        required = EDGEGRAFT_DIR / family / f"fold_{fold}.joblib"
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete EdgeGRAFT V3 artifact: {required}")

'''
    artifact_insert = artifact_anchor + '''# CandidateGRAFT final known-only gate and add-only runtime.
CANDIDATEGRAFT_DIR = Path(CFG["models"]["candidategraft"]["dir"])
if not (CANDIDATEGRAFT_DIR / "candidategraft_direct_runtime_v1.py").is_file():
    candidates = sorted({
        path.parent
        for path in INPUT_DIR.rglob("candidategraft_direct_runtime_v1.py")
        if (path.parent / "candidategraft_direct.joblib").is_file()
        and (path.parent / "report.json").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one CandidateGRAFT artifact, found {candidates}")
    CANDIDATEGRAFT_DIR = candidates[0]
    CFG["models"]["candidategraft"]["dir"] = str(CANDIDATEGRAFT_DIR)
for required in (
    CANDIDATEGRAFT_DIR / "candidategraft_direct_runtime_v1.py",
    CANDIDATEGRAFT_DIR / "candidategraft_direct.joblib",
    CANDIDATEGRAFT_DIR / "report.json",
):
    if not required.is_file():
        raise FileNotFoundError(f"Incomplete CandidateGRAFT artifact: {required}")

'''
    artifact = replace_once(
        artifact, artifact_anchor, artifact_insert, "artifact validation",
    )
    notebook["cells"][artifact_index]["source"] = artifact.splitlines(keepends=True)

    post_index, post = find_cell(notebook, "def filter_output_graph(")
    post = replace_once(
        post,
        '''EDGEGRAFT_DIR = Path(_M["edgegraft"]["dir"])
EDGEGRAFT_ENABLED = bool(_M["edgegraft"]["enabled"])
''',
        '''EDGEGRAFT_DIR = Path(_M["edgegraft"]["dir"])
EDGEGRAFT_ENABLED = bool(_M["edgegraft"]["enabled"])

CANDIDATEGRAFT_DIR = Path(_M["candidategraft"]["dir"])
CANDIDATEGRAFT_ENABLED = bool(_M["candidategraft"]["enabled"])
''',
        "postprocess constants",
    )
    post = replace_once(
        post,
        '''DIVISION_RUNTIME = None
EDGEGRAFT_RUNTIME = None
''',
        '''DIVISION_RUNTIME = None
EDGEGRAFT_RUNTIME = None
CANDIDATEGRAFT_RUNTIME = None
''',
        "runtime globals",
    )
    init_anchor = '''    EDGEGRAFT_RUNTIME = module.EdgeGraftV3Runtime(
        EDGEGRAFT_DIR / "ranker", EDGEGRAFT_DIR / "metric",
    )
    return EDGEGRAFT_RUNTIME


'''
    init_insert = init_anchor + '''def init_candidategraft_runtime():
    """Load the final known-only add-only gate once before worker fork."""
    global CANDIDATEGRAFT_RUNTIME
    if CANDIDATEGRAFT_RUNTIME is not None or not CANDIDATEGRAFT_ENABLED:
        return CANDIDATEGRAFT_RUNTIME
    module = _load_module(
        "biohub_candidategraft_direct_runtime",
        CANDIDATEGRAFT_DIR / "candidategraft_direct_runtime_v1.py",
    )
    CANDIDATEGRAFT_RUNTIME = module.CandidateGraftDirectRuntime(
        CANDIDATEGRAFT_DIR, EDGEGRAFT_DIR,
    )
    return CANDIDATEGRAFT_RUNTIME


'''
    post = replace_once(post, init_anchor, init_insert, "runtime initializer")
    final_anchor = '''    nodes_by_id, edges = enforce_node_budget(nodes_by_id, edges, stats, detected_nodes)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    validate_final_graph_contract(nodes_by_id, edges, stats)
'''
    final_insert = '''    nodes_by_id, edges = enforce_node_budget(nodes_by_id, edges, stats, detected_nodes)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)

    # Exact replay placement: after all destructive cleanup on the same final
    # retained-node population used by the 175-video promotion gate.
    if division_mode == "combined" and CANDIDATEGRAFT_ENABLED:
        if dataset is None:
            raise RuntimeError("CandidateGRAFT needs a dataset id")
        if CANDIDATEGRAFT_RUNTIME is None:
            init_candidategraft_runtime()
        candidate_started = time.monotonic()
        edges, candidate_stats = CANDIDATEGRAFT_RUNTIME.apply(
            dataset, raw_node_attrs, raw_edge_attrs, nodes_by_id, edges,
            _evidence(P1_EVIDENCE_DIR, dataset),
            _evidence(P2_EVIDENCE_DIR, dataset),
        )
        stats.update({
            f"candidategraft_{key}": value
            for key, value in candidate_stats.items()
        })
        stats["candidategraft_seconds"] = time.monotonic() - candidate_started
        stats["candidategraft_direct_add_only"] = 1
    else:
        stats["candidategraft_skipped_for_fallback"] = int(CANDIDATEGRAFT_ENABLED)
    validate_final_graph_contract(nodes_by_id, edges, stats)
'''
    post = replace_once(post, final_anchor, final_insert, "final graph insertion")
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)

    worker_index, worker = find_cell(notebook, "pp.init_edgegraft_runtime()")
    worker = replace_once(
        worker,
        "pp.init_edgegraft_runtime()\n",
        "pp.init_edgegraft_runtime()\npp.init_candidategraft_runtime()\n",
        "parent runtime initialization",
    )
    notebook["cells"][worker_index]["source"] = worker.splitlines(keepends=True)

    metadata = notebook.setdefault("metadata", {})
    metadata["title"] = "Biohub .952 + CandidateGRAFT Direct v1"
    metadata.setdefault("kaggle", {})["title"] = metadata["title"]
    metadata["candidategraft"] = {
        "version": "direct-v1",
        "gate": 0.90,
        "placement": "after final cleanup, before final contract validation",
        "exact_final_fit_delta": 0.0018971429159965,
        "fallback": "unchanged .952 baseline shard",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1))

    rebuilt = json.loads(output.read_text())
    full_text = "\n".join("".join(cell.get("source", [])) for cell in rebuilt["cells"])
    required = (
        "biohub-candidategraft-direct-v1",
        "CandidateGraftDirectRuntime",
        "pp.init_candidategraft_runtime()",
        "candidategraft_direct_add_only",
        "after all destructive cleanup",
    )
    missing = [marker for marker in required if marker not in full_text]
    if missing:
        raise RuntimeError(f"Built notebook is missing markers: {missing}")
    post_source = "".join(rebuilt["cells"][post_index]["source"])
    if post_source.startswith("%%writefile"):
        post_source = post_source.split("\n", 1)[1]
    compile(post_source, "postprocess.py", "exec")


def main() -> None:
    args = parse_args()
    manifest = package_runtime(args.stage, args.zip, args.model)
    build_notebook(args.source, args.output)
    print(json.dumps({
        "notebook": str(args.output),
        "dataset": str(args.stage),
        "dataset_zip": str(args.zip),
        "dataset_zip_sha256": sha256(args.zip),
        "manifest": manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
