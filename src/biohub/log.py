import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger('biohub')


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    LOGGER.handlers.clear()
    LOGGER.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / 'run.log')
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    LOGGER.propagate = False
    return LOGGER


def log_event(run_dir: Path, event: str, **payload: Any) -> None:
    record = {
        'ts': datetime.now(UTC).isoformat(),
        'event': event,
        **payload,
    }
    path = run_dir / 'events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, default=str) + '\n')
    LOGGER.info(
        '%s %s', event, {key: value for key, value in payload.items() if key != 'traceback'}
    )
