"""Per-run logging: config, metrics, git-ish metadata, stdout tee."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_dir(runs_root: Path | str, stage: str, run_name: str | None = None) -> Path:
    runs_root = Path(runs_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or "run"
    path = runs_root / stage / f"{ts}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@dataclass
class RunLogger:
    run_dir: Path
    stage: str
    config: dict[str, Any] = field(default_factory=dict)
    _metrics: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hostname": platform.node(),
            "stage": self.stage,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["gpu_count"] = torch.cuda.device_count()
                env["gpu_names"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except Exception as exc:  # noqa: BLE001
            env["torch_error"] = str(exc)
        (self.run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")
        if self.config:
            self.write_config(self.config)
        self._log_path = self.run_dir / "train.log"
        self._log_path.touch()

    def write_config(self, config: dict[str, Any]) -> None:
        self.config = config
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with self._log_path.open("a") as f:
            f.write(line + "\n")

    def log_metric(self, name: str, value: float | int | dict, *, step: int | None = None) -> None:
        row: dict[str, Any] = {
            "name": name,
            "value": value,
            "wall_s": time.time() - self._t0,
        }
        if step is not None:
            row["step"] = step
        self._metrics.append(row)
        (self.run_dir / "metrics.jsonl").open("a").write(json.dumps(row, default=str) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        summary = {
            **summary,
            "stage": self.stage,
            "elapsed_s": time.time() - self._t0,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    def path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
