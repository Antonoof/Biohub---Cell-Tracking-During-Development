from pydantic import BaseModel, ConfigDict

from biohub.utils.yaml_config import load_yaml, resolve_path

__all__ = ['FrozenModel', 'load_yaml', 'resolve_path']


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
