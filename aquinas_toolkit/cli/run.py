"""``aquinas run [stage]`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.utils.run_management import (
    STAGES,
    RunContext,
    RunManagementError,
    create_run,
    ensure_stage_output_dir,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_started,
    resolve_run,
    validate_stage_can_run,
    write_latest_pointer,
)
from aquinas_toolkit.visualization import build_visualization_artifacts

STAGE_PACKAGE_DIRS = {
    "preprocess": "preprocessing",
    "features": "feature_extraction",
    "train": "training",
    "score": "scoring",
}

# Registry maps stage name -> callable that receives a RunContext.
# Add an entry here when a new stage is implemented; no other changes needed.
_STAGE_REGISTRY: dict[str, str] = {
    "preprocess": "aquinas_toolkit.preprocessing:run_preprocessing",
}


class StageNotImplementedError(RuntimeError):
    """Raised when a stage package has not been implemented yet."""


class RichRunArgumentParser(argparse.ArgumentParser):
    """Argument parser with Rich-rendered help and error output."""

    def print_help(self, file=None) -> None:  # noqa: ANN001
        terminal.print_run_help(STAGES)

    def error(self, message: str) -> None:
        terminal.print_error(message)
        terminal.print_run_help(STAGES)
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """Create the ``aquinas run`` argument parser."""
    parser = RichRunArgumentParser(
        prog="aquinas run",
        description="Run the full pipeline, or a single stage.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=STAGES,
        help=f"Stage to execute: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--name",
        help="Optional human-readable label stored in metadata when creating a new run.",
    )
    parser.add_argument(
        "--run-id",
        help="Existing run ID to resume for features, train, or score.",
    )
    return parser


def run() -> None:
    """Run the full pipeline, or a single stage if specified."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[2:])

    try:
        exit_code = run_command(stage=args.stage, name=args.name, run_id=args.run_id)
    except RunManagementError as exc:
        terminal.print_error(str(exc))
        sys.exit(1)

    if exit_code:
        sys.exit(exit_code)


def run_command(stage: str | None, name: str | None, run_id: str | None) -> int:
    """Execute the run command and return a process exit code."""
    creates_new_run = stage in {None, "preprocess"}

    if creates_new_run and run_id is not None:
        raise RunManagementError(
            "`--run-id` cannot be used with `aquinas run` or `aquinas run preprocess` "
            "because those commands always create a new run."
        )
    if not creates_new_run and name is not None:
        raise RunManagementError("`--name` can only be used when creating a new run.")

    if creates_new_run:
        run_context = create_run(name=name)
    else:
        run_context = resolve_run(run_id=run_id)
        if run_id is not None:
            write_latest_pointer(run_context.results_dir, run_context.run_id)

    terminal.print_run_summary(
        run_id=run_context.run_id,
        run_dir=run_context.run_dir,
        config_path=run_context.config_path,
        created_new=creates_new_run,
    )

    if stage is None:
        terminal.print_stage_status(
            "START",
            "pipeline",
            f"Running full pipeline for run {run_context.run_id} ({' -> '.join(STAGES)})",
        )
        stages_to_run = list(STAGES)
    else:
        stages_to_run = [stage]

    for current_stage in stages_to_run:
        try:
            _run_stage(current_stage, run_context)
        except StageNotImplementedError as exc:
            _refresh_visualization_bundle(run_context)
            _print_visualization_hint()
            terminal.print_stage_status("FAIL", current_stage, str(exc), stderr=True)
            return 1
        except RunManagementError as exc:
            _refresh_visualization_bundle(run_context)
            _print_visualization_hint()
            terminal.print_stage_status("FAIL", current_stage, str(exc), stderr=True)
            return 1
        except Exception as exc:  # pragma: no cover - defensive path
            _refresh_visualization_bundle(run_context)
            _print_visualization_hint()
            terminal.print_stage_status("FAIL", current_stage, str(exc), stderr=True)
            return 1

    _refresh_visualization_bundle(run_context)
    _print_visualization_hint()
    return 0


def _run_stage(stage: str, run_context: RunContext) -> None:
    """Run a single pipeline stage inside an existing run."""
    validate_stage_can_run(run_context.run_dir, stage)
    stage_dir = ensure_stage_output_dir(run_context.run_dir, stage)
    mark_stage_started(run_context.run_dir, stage)
    terminal.print_stage_status("START", stage, f"Run {run_context.run_id}")

    try:
        _execute_stage(stage, run_context)
    except Exception as exc:
        mark_stage_failed(run_context.run_dir, stage, str(exc))
        raise

    mark_stage_completed(run_context.run_dir, stage)
    terminal.print_stage_status("DONE", stage, f"Output: {stage_dir}")


def _execute_stage(stage: str, run_context: RunContext) -> None:
    """Dispatch stage execution to the registered stage implementation."""
    if stage not in _STAGE_REGISTRY:
        target = STAGE_PACKAGE_DIRS[stage]
        raise StageNotImplementedError(
            f"Not yet implemented. See aquinas_toolkit/{target}/ "
            f"(run {run_context.run_id}, config {run_context.config_path})."
        )

    import importlib

    module_path, func_name = _STAGE_REGISTRY[stage].split(":")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    fn(run_context)


def _refresh_visualization_bundle(run_context: RunContext) -> None:
    """Build or refresh the offline visualization bundle when data is available."""
    if not _visualization_inputs_available(run_context.config_path):
        return
    build_visualization_artifacts(run_context)


def _print_visualization_hint() -> None:
    """Print the post-run reminder about opening the viewer bundle."""
    terminal.print_stage_status("TIP", "viz", "Open the visualization with `aquinas viz open`.")


def _visualization_inputs_available(config_path: Path) -> bool:
    """Return whether the run config points to a locally available dataset tree."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        return False

    dataset_root_value = data_config.get("dataset_root", "AQUINAS_DATASET")
    dataset_root = Path(dataset_root_value)
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    if not dataset_root.is_dir():
        return False

    configured_sets = data_config.get("sets")
    if not isinstance(configured_sets, list) or not configured_sets:
        return False

    return all((dataset_root / set_name).is_dir() for set_name in configured_sets)
