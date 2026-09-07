import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    env = os.environ.get('BIOHUB_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'src' / 'biohub').is_dir():
            return candidate
    raise FileNotFoundError(
        'Cannot locate the biohub project root. Set BIOHUB_ROOT or run from the repo.'
    )


PROJECT_ROOT = find_project_root()
