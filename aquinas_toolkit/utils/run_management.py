"""Run directory and metadata helpers for the AQUINAS pipeline CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RESULTS_DIR = Path("results")
LATEST_POINTER_NAME = "latest.json"
RUN_CONFIG_NAME = "config.yaml"
METADATA_NAME = "metadata.json"
STAGES_DIR_NAME = "stages"
STAGES = ("preprocess", "features", "train", "score")
STAGE_PREREQUISITES = {
    "preprocess": None,
    "features": "preprocess",
    "train": "features",
    "score": "train",
}
TERMINAL_STAGE_STATUSES = {"completed", "failed"}


class RunManagementError(RuntimeError):
    """Raised when run discovery or metadata validation fails."""


@dataclass(frozen=True)
class RunContext:
    """Resolved filesystem paths for a pipeline run."""

    run_id: str
    results_dir: Path
    run_dir: Path
    config_path: Path
    metadata_path: Path


def get_default_config_path() -> Path:
    """Return the repository-local default config path."""
    return Path.cwd() / DEFAULT_CONFIG_PATH


def get_results_dir(config_path: Path | None = None) -> Path:
    """Resolve the configured results directory from ``configs/default.yaml``."""
    resolved_config = config_path or get_default_config_path()
    if not resolved_config.exists():
        raise RunManagementError(
            f"Default config not found at {resolved_config}. Create {DEFAULT_CONFIG_PATH} first."
        )

    try:
        config_data = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RunManagementError(f"Could not parse config file {resolved_config}: {exc}") from exc

    output_config = config_data.get("output")
    if not isinstance(output_config, dict):
        return _resolve_path(DEFAULT_RESULTS_DIR)

    raw_results_dir = output_config.get("results_dir", str(DEFAULT_RESULTS_DIR))
    if not isinstance(raw_results_dir, str) or not raw_results_dir.strip():
        return _resolve_path(DEFAULT_RESULTS_DIR)

    return _resolve_path(Path(raw_results_dir))


def generate_run_id(results_dir: Path, now: datetime | None = None) -> str:
    """Generate a readable, sortable, Windows-safe run identifier."""
    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    base_run_id = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    candidate = base_run_id
    suffix = 2

    while (results_dir / candidate).exists():
        candidate = f"{base_run_id}-{suffix:02d}"
        suffix += 1

    return candidate


def create_run(name: str | None = None) -> RunContext:
    """Create a new immutable run directory and snapshot the active config."""
    results_dir = get_results_dir()
    default_config_path = get_default_config_path()

    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id(results_dir)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config_path = run_dir / RUN_CONFIG_NAME
    shutil.copy2(default_config_path, run_config_path)

    metadata_path = run_dir / METADATA_NAME
    metadata = {
        "run_id": run_id,
        "name": name or "unnamed",
        "created_at_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "stages": {stage: new_stage_state() for stage in STAGES},
    }
    write_metadata(run_dir, metadata)
    write_latest_pointer(results_dir, run_id)

    return RunContext(
        run_id=run_id,
        results_dir=results_dir,
        run_dir=run_dir,
        config_path=run_config_path,
        metadata_path=metadata_path,
    )


def resolve_run(run_id: str | None = None) -> RunContext:
    """Resolve an existing run either explicitly or via the latest pointer."""
    results_dir = get_results_dir()
    resolved_run_id = run_id or read_latest_pointer(results_dir)["run_id"]
    if not isinstance(resolved_run_id, str) or not resolved_run_id.strip():
        raise RunManagementError("Active run pointer is invalid: missing run_id.")

    resolved_run_id = resolved_run_id.strip()
    run_dir = results_dir / resolved_run_id
    if not run_dir.is_dir():
        if run_id is None:
            raise RunManagementError(
                f"Active run pointer points to missing run '{resolved_run_id}' in {results_dir}. "
                "Start a new run with `aquinas run` or `aquinas run preprocess`, "
                "or pass `--run-id`."
            )
        raise RunManagementError(f"Run '{resolved_run_id}' was not found in {results_dir}.")

    run_config_path = run_dir / RUN_CONFIG_NAME
    metadata_path = run_dir / METADATA_NAME
    if not run_config_path.is_file():
        raise RunManagementError(f"Run '{resolved_run_id}' is missing {RUN_CONFIG_NAME}.")
    if not metadata_path.is_file():
        raise RunManagementError(f"Run '{resolved_run_id}' is missing {METADATA_NAME}.")

    return RunContext(
        run_id=resolved_run_id,
        results_dir=results_dir,
        run_dir=run_dir,
        config_path=run_config_path,
        metadata_path=metadata_path,
    )


def read_latest_pointer(results_dir: Path) -> dict[str, Any]:
    """Read and validate ``latest.json`` from the results root."""
    latest_path = results_dir / LATEST_POINTER_NAME
    if not latest_path.is_file():
        raise RunManagementError(
            f"No active run pointer found at {latest_path}. "
            "Start a new run with `aquinas run` or `aquinas run preprocess`, or pass `--run-id`."
        )
    return read_json(latest_path, label="active run pointer")


def write_latest_pointer(results_dir: Path, run_id: str) -> None:
    """Atomically update ``latest.json`` to point at the selected run."""
    payload = {"run_id": run_id, "updated_at_utc": utc_timestamp()}
    write_json_atomic(results_dir / LATEST_POINTER_NAME, payload)


def read_metadata(run_dir: Path) -> dict[str, Any]:
    """Load a run metadata document."""
    return read_json(run_dir / METADATA_NAME, label="run metadata")


def write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    """Atomically persist run metadata."""
    write_json_atomic(run_dir / METADATA_NAME, metadata)


def stage_output_dir(run_dir: Path, stage: str) -> Path:
    """Return the stage output directory for a run."""
    ensure_valid_stage(stage)
    return run_dir / STAGES_DIR_NAME / stage


def ensure_stage_output_dir(run_dir: Path, stage: str) -> Path:
    """Create the stage directory only when the stage is actually executed."""
    output_dir = stage_output_dir(run_dir, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_stage_can_run(run_dir: Path, stage: str) -> None:
    """Validate prerequisites and current stage status before execution."""
    ensure_valid_stage(stage)
    metadata = read_metadata(run_dir)
    run_id = metadata.get("run_id", run_dir.name)
    stage_state = metadata["stages"][stage]
    stage_status = stage_state["status"]

    if stage_status == "completed":
        raise RunManagementError(
            f"Stage '{stage}' is already completed for run '{run_id}'. Create a new run instead."
        )
    if stage_status == "running":
        raise RunManagementError(
            f"Stage '{stage}' is already marked as running for run '{run_id}'."
        )

    prerequisite = STAGE_PREREQUISITES[stage]
    if prerequisite is None:
        return

    prerequisite_status = metadata["stages"][prerequisite]["status"]
    if prerequisite_status != "completed":
        raise RunManagementError(
            f"Stage '{stage}' requires completed '{prerequisite}' outputs for run '{run_id}'. "
            f"Current '{prerequisite}' status: '{prerequisite_status}'."
        )


def mark_stage_started(run_dir: Path, stage: str) -> None:
    """Mark a stage as running and reset any prior terminal state."""
    metadata = read_metadata(run_dir)
    metadata["stages"][stage] = {
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "completed_at_utc": None,
        "error": None,
    }
    write_metadata(run_dir, metadata)


def mark_stage_completed(run_dir: Path, stage: str) -> None:
    """Mark a stage as completed."""
    metadata = read_metadata(run_dir)
    stage_state = metadata["stages"][stage]
    if stage_state["status"] != "running":
        raise RunManagementError(
            f"Cannot mark stage '{stage}' as completed because it is not currently running."
        )

    metadata["stages"][stage] = {
        "status": "completed",
        "started_at_utc": stage_state["started_at_utc"],
        "completed_at_utc": utc_timestamp(),
        "error": None,
    }
    write_metadata(run_dir, metadata)


def mark_stage_failed(run_dir: Path, stage: str, error: str) -> None:
    """Mark a stage as failed."""
    metadata = read_metadata(run_dir)
    stage_state = metadata["stages"][stage]
    metadata["stages"][stage] = {
        "status": "failed",
        "started_at_utc": stage_state.get("started_at_utc"),
        "completed_at_utc": utc_timestamp(),
        "error": error,
    }
    write_metadata(run_dir, metadata)


def new_stage_state() -> dict[str, Any]:
    """Return the initial metadata payload for a stage."""
    return {
        "status": "not_started",
        "started_at_utc": None,
        "completed_at_utc": None,
        "error": None,
    }


def git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    return _git_command_output(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    """Return whether the current Git worktree has local modifications."""
    output = _git_command_output(["git", "status", "--short"])
    if output is None:
        return None
    return bool(output)


def utc_timestamp(now: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_valid_stage(stage: str) -> None:
    """Validate that a stage name is part of the pipeline contract."""
    if stage not in STAGES:
        raise RunManagementError(f"Unknown stage: {stage}")


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON document with a descriptive validation error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunManagementError(f"{label.capitalize()} file not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise RunManagementError(f"Could not parse {label} at {path}: {exc.msg}.") from exc

    if not isinstance(payload, dict):
        raise RunManagementError(f"{label.capitalize()} at {path} must contain a JSON object.")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically using a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _git_command_output(command: list[str]) -> str | None:
    """Run a Git command and return stripped stdout when successful."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path.cwd(),
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths against the current workspace root."""
    if path.is_absolute():
        return path
    return Path.cwd() / path
