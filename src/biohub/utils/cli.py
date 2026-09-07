import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from biohub.utils.yaml_config import load_yaml, resolve_path


def dict_to_argv(payload: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in payload.items():
        flag = '--' + str(key).replace('_', '-')
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, ','.join(str(item) for item in value)])
        else:
            text = str(value)
            if isinstance(value, str) and '/' in value:
                text = str(resolve_path(value))
            argv.extend([flag, text])
    return argv


def stage_main(train_fn: Callable[[dict[str, Any]], Any], argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    result = train_fn(load_yaml(args.config))
    if isinstance(result, int):
        return result
    return 0


def run_argparse_main(main_fn: Callable[[], Any], cfg: dict[str, Any]) -> Any:
    sys.argv = [sys.argv[0], *dict_to_argv(cfg)]
    return main_fn()
