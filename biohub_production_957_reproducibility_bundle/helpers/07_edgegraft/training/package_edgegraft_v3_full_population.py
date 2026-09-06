"""Package the exact-parity full-population EdgeGRAFT V3 Kaggle artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")
OUTPUT = WORKSPACE / "output" / "biohub-edgegraft-v3-full-population-v1"
ZIP_PATH = OUTPUT.with_suffix(".zip")
RANKER = Path("/home/tweak/bio/edgegraft_current_ranker_v3")
METRIC = Path("/home/tweak/bio/edgegraft_v3_metric_transaction_gate")

RUNTIME_FILES = [
    "edgegraft_component_runtime_v1.py",
    "edgegraft_v15_runtime_v1.py",
    "edgegraft_v3_deploy_runtime_v1.py",
    "edgegraft_v3_deploy_runtime_v2.py",
    "edgegraft_v3_deploy_runtime_v3.py",
    "edgegraft_v3_deploy_runtime_v4.py",
    "edgegraft_v3_deploy_runtime_v5.py",
]


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
OUTPUT.mkdir(parents=True)

for name in RUNTIME_FILES:
    copy_file(WORKSPACE / "scripts" / name, OUTPUT / name)

for family, source in (("ranker", RANKER), ("metric", METRIC)):
    copy_file(source / "report.json", OUTPUT / family / "report.json")
    for fold in range(5):
        copy_file(
            source / f"fold_{fold}.joblib",
            OUTPUT / family / f"fold_{fold}.joblib",
        )

manifest = {
    "artifact": "biohub-edgegraft-v3-full-population-v1",
    "runtime": "edgegraft_v3_deploy_runtime_v5.py",
    "runtime_class": "EdgeGraftV3Runtime",
    "policy": "V3 uncapped full-population transaction gate",
    "candidate_population": {
        "conflict_rows": 13_667_730,
        "target_decisions": 97_484,
        "videos": 175,
    },
    "exact_replay": {
        "baseline_adjusted_edge_jaccard": 0.9074678856411179,
        "candidate_adjusted_edge_jaccard": 0.9252279980329011,
        "gain": 0.01776011239178321,
        "oracle_available_gain": 0.05598368301050882,
        "oracle_ceiling_fraction": 0.31723730,
        "division_tp": 87,
        "division_fp": 60,
        "division_fn": 44,
        "division_jaccard": 0.45549738219895286,
    },
    "live_parity": [
        {
            "stem": "44b6_0c582fdc",
            "candidate_rows": 116_913,
            "decisions": 530,
            "applied": 60,
            "missing_edges": 0,
            "extra_edges": 0,
        },
        {
            "stem": "6bba_2540cd90",
            "candidate_rows": 4_540,
            "decisions": 6,
            "applied": 0,
            "missing_edges": 0,
            "extra_edges": 0,
        },
        {
            "stem": "6bba_767a1e17",
            "candidate_rows": 242_495,
            "decisions": 4_067,
            "selected": 3_048,
            "applied": 3_008,
            "missing_edges": 0,
            "extra_edges": 0,
        },
    ],
}

(OUTPUT / "README.md").write_text(
    "# EdgeGRAFT V3 full-population deployment artifact\n\n"
    "Frozen five-fold ranker and exact-metric transaction gate for the `.951` "
    "UG2/UG3 boundary-track-rescue graph. The runtime uses all native P1/P2 "
    "conflict rows, protects established forks, and applies atomic one-to-one "
    "continuation ownership replacements.\n",
    encoding="utf-8",
)

files = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
manifest["files"] = {
    str(path.relative_to(OUTPUT)): {
        "bytes": path.stat().st_size,
        "sha256": checksum(path),
    }
    for path in files
}
(OUTPUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT.parent))

print(f"artifact={OUTPUT}")
print(f"zip={ZIP_PATH}")
print(f"zip_bytes={ZIP_PATH.stat().st_size}")
