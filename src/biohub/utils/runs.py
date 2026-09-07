import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from biohub.paths import PROJECT_ROOT


def new_run_id(experiment: str) -> str:
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    return f'{stamp}_{experiment}'


def run_dir(run_id: str, runs_root: Path | None = None) -> Path:
    root = runs_root or (PROJECT_ROOT / 'runs')
    return Path(root) / run_id


def _dumpable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _dumpable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dumpable(item) for item in value]
    return value


def init_run(
    run_id: str,
    *,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    runs_root: Path | None = None,
) -> Path:
    path = run_dir(run_id, runs_root)
    (path / 'evaluation').mkdir(parents=True, exist_ok=True)
    (path / 'reports').mkdir(parents=True, exist_ok=True)
    if config is not None:
        (path / 'resolved_config.yaml').write_text(
            yaml.safe_dump(_dumpable(config), sort_keys=False)
        )
    manifest = {
        'run_id': run_id,
        'created_utc': datetime.now(UTC).isoformat(),
        'status': 'running',
        **(extra or {}),
    }
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    return path


def finish_run(path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest['status'] = status
    manifest['finished_utc'] = datetime.now(UTC).isoformat()
    if extra:
        manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
