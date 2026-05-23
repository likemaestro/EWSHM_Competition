"""``aquinas preprocess`` inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.preprocessing import (
    load_sensor_map,
    plot_nn_input_event,
    random_event_indices,
    summarize_nn_inputs,
)
from aquinas_toolkit.utils.run_management import RunManagementError, resolve_run, stage_output_dir


PREPROCESS_SUBCOMMANDS = ("quicklook",)


class RichPreprocessArgumentParser(argparse.ArgumentParser):
    """Argument parser with Rich-rendered help and error output."""

    def print_help(self, file=None) -> None:  # noqa: ANN001
        terminal.print_preprocess_help()

    def error(self, message: str) -> None:
        invalid_command = _extract_invalid_choice(message)
        if invalid_command is not None:
            suggestion = terminal.suggest_typo(invalid_command, PREPROCESS_SUBCOMMANDS)
            if suggestion is not None:
                terminal.get_console().print(
                    terminal.render_typo_hint(
                        command_name=invalid_command,
                        suggested_command=suggestion,
                    )
                )
            terminal.print_error(message)
            terminal.get_console().print(
                terminal.render_compact_choice_hint(
                    label="subcommands",
                    choices=PREPROCESS_SUBCOMMANDS,
                    help_command="aquinas preprocess --help",
                )
            )
            raise SystemExit(2)
        terminal.print_error(message)
        terminal.print_preprocess_help()
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """Create the ``aquinas preprocess`` argument parser."""
    parser = RichPreprocessArgumentParser(
        prog="aquinas preprocess",
        description="Inspect preprocess-stage artifacts.",
    )
    subparsers = parser.add_subparsers(dest="preprocess_command")

    quicklook_parser = subparsers.add_parser("quicklook", add_help=False)
    quicklook_parser.add_argument("--run-id", help="Existing run ID. Defaults to results/latest.json.")
    quicklook_parser.add_argument(
        "--event-index",
        type=int,
        help="Event row index to plot from the split NN input arrays.",
    )
    quicklook_parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Plot N deterministic random event rows.",
    )
    quicklook_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print NN input shapes and finite-value diagnostics.",
    )
    quicklook_parser.add_argument(
        "--sensor-map",
        action="store_true",
        help="Print included NN channel order from sensor_map.csv.",
    )
    quicklook_parser.add_argument(
        "--output",
        help="Output PNG path or directory. Defaults to nn_inputs/quicklook/.",
    )
    return parser


def _extract_invalid_choice(message: str) -> str | None:
    marker = "invalid choice: "
    if marker not in message:
        return None
    try:
        return message.split("'", 2)[1]
    except IndexError:
        return None


def run() -> None:
    """Run a preprocess inspection command."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[2:])

    if args.preprocess_command is None:
        terminal.print_preprocess_help()
        sys.exit(0)

    try:
        if args.preprocess_command == "quicklook":
            _run_quicklook(args)
            return
        raise RunManagementError(f"Unknown preprocess subcommand: {args.preprocess_command}")
    except (FileNotFoundError, IndexError, RunManagementError, ValueError) as exc:
        terminal.print_error(str(exc))
        sys.exit(1)


def _run_quicklook(args: argparse.Namespace) -> None:
    run_context = resolve_run(run_id=args.run_id)
    preprocess_dir = stage_output_dir(run_context.run_dir, "preprocess")

    if args.summary or _quicklook_has_no_action(args):
        terminal.get_console().print(json.dumps(summarize_nn_inputs(preprocess_dir), indent=2))

    if args.sensor_map:
        sensor_map = load_sensor_map(preprocess_dir)
        included = sensor_map.loc[sensor_map["include_flag"].astype(bool)].copy()
        columns = [
            "model_channel_id",
            "global_model_channel_index",
            "sensor_name",
            "sensor_type",
            "location",
            "axis_or_type",
        ]
        terminal.get_console().print(included.loc[:, columns].to_string(index=False))

    event_indices: list[int] = []
    if args.event_index is not None:
        event_indices.append(int(args.event_index))
    if args.random is not None:
        event_indices.extend(random_event_indices(preprocess_dir, count=int(args.random)))

    output = Path(args.output) if args.output else None
    for event_index in event_indices:
        path = plot_nn_input_event(preprocess_dir, event_index=event_index, output=output)
        terminal.print_stage_status("DONE", "preprocess", f"Quicklook plot: {path}")


def _quicklook_has_no_action(args: argparse.Namespace) -> bool:
    return (
        not args.summary
        and not args.sensor_map
        and args.event_index is None
        and args.random is None
    )
