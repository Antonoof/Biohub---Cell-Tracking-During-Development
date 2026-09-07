import hashlib
from pathlib import Path

from biohub.paths import PROJECT_ROOT
from tests.parity.notebook import PRODUCTION_NOTEBOOK, PRODUCTION_NOTEBOOK_SHA256

BUNDLE_SRC = (
    PROJECT_ROOT
    / 'biohub_production_957_reproducibility_bundle'
    / 'helpers'
    / '01_p1_p2_base'
    / 'shared_repo'
    / 'src'
    / 'biohub_tracking'
)
EXPECTED_METRICS_SHA256 = '31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83'
EXPECTED_DIVISION_SHA256 = 'd1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_bundle_scorer_hashes() -> None:
    metrics_path = BUNDLE_SRC / 'metrics.py'
    division_path = BUNDLE_SRC / 'division_metrics.py'
    assert metrics_path.is_file()
    assert division_path.is_file()
    assert _sha256(metrics_path) == EXPECTED_METRICS_SHA256
    assert _sha256(division_path) == EXPECTED_DIVISION_SHA256
    assert PRODUCTION_NOTEBOOK.is_file()
    assert _sha256(PRODUCTION_NOTEBOOK) == PRODUCTION_NOTEBOOK_SHA256
