"""Run-scoped debug logging and timing helpers."""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aquinas_toolkit.cli.terminal import get_console


class RunDebugLogger:
    """Append-only logger for per-run diagnostics and timing details."""

    def __init__(self, path: Path, *, verbose: bool = False) -> None:
        self.path = path
        self.verbose = verbose
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        """Append one structured log line."""
        payload = {
            key: value
            for key, value in sorted(fields.items())
            if value is not None
        }
        tokens = [f"{key}={_stringify(value)}" for key, value in payload.items()]
        line = f"{_utc_timestamp()} event={event}"
        if tokens:
            line = f"{line} {' '.join(tokens)}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def timing(self, *, stage: str, phase: str, seconds: float, count: int | None = None) -> None:
        """Append one timing line."""
        self.log(
            "TIMING",
            stage=stage,
            phase=phase,
            seconds=f"{seconds:.6f}",
            count=count,
        )

    def exception(self, *, stage: str, error: BaseException) -> None:
        """Append exception details including traceback."""
        self.log("ERROR", stage=stage, error_type=type(error).__name__, message=str(error))
        trace = traceback.format_exc()
        for line in trace.splitlines():
            self.log("TRACE", stage=stage, line=line)

    def verbose_timing_summary(self, *, stage: str, timings: dict[str, float]) -> None:
        """Print a compact timing summary to console when verbose mode is enabled."""
        if not self.verbose:
            return
        console = get_console()
        console.print(f"[accent]Timing breakdown ({stage})[/]")
        for phase, seconds in sorted(timings.items(), key=lambda item: item[0]):
            console.print(f"  [key]{phase}[/] {seconds:.3f}s")


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("\n", "\\n")


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
