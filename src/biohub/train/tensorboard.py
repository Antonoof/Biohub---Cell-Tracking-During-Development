from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def open_writer(output: Path | str) -> SummaryWriter:
    log_dir = Path(output) / 'tensorboard'
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_scalars(writer: SummaryWriter | None, step: int, values: dict[str, Any]) -> None:
    if writer is None:
        return
    for name, value in values.items():
        if value is None:
            continue
        writer.add_scalar(name, float(value), step)
