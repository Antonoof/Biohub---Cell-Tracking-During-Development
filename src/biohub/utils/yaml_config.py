from pathlib import Path
from typing import Any

import yaml

from biohub.paths import PROJECT_ROOT


def resolve_path(value: Path | str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _is_pathish(value: str) -> bool:
    if value.startswith(('http://', 'https://')):
        return False
    suffixes = (
        '.json',
        '.yaml',
        '.yml',
        '.pt',
        '.pth',
        '.npz',
        '.joblib',
        '.csv',
        '.parquet',
        '.geff',
        '.zarr',
        '.txt',
    )
    return '/' in value or value.endswith(suffixes)


def _resolve_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    if isinstance(value, str) and _is_pathish(value):
        return resolve_path(value)
    return value


def load_yaml(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Config {config_path} must be a mapping, got {type(loaded)}')
    return _resolve_value(loaded)
