"""``aquinas data`` command."""

from __future__ import annotations

import argparse
import sys

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.data_fetch import DatasetFetchError, fetch_dataset
from aquinas_toolkit.utils.dataset_config import (
    DatasetLayout,
    DatasetLayoutStatus,
    inspect_dataset_layout,
    load_dataset_layout,
)


class RichDataArgumentParser(argparse.ArgumentParser):
    """Argument parser with Rich-rendered help and error output."""

    def print_help(self, file=None) -> None:  # noqa: ANN001
        terminal.print_data_help()

    def error(self, message: str) -> None:
        terminal.print_error(message)
        terminal.print_data_help()
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = RichDataArgumentParser(
        prog="aquinas data",
        description="Download and manage the local AQUINAS dataset copy.",
    )
    subparsers = parser.add_subparsers(dest="data_command")

    fetch_parser = subparsers.add_parser("fetch", add_help=False)
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing dataset root if it already exists.",
    )
    fetch_parser.add_argument(
        "--assume-yes",
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="Skip overwrite confirmation prompts.",
    )
    fetch_parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep a copy of the downloaded ZIP next to the dataset root.",
    )

    subparsers.add_parser("status", add_help=False)
    subparsers.add_parser("verify", add_help=False)
    subparsers.add_parser("path", add_help=False)
    return parser


def run() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[2:])

    if args.data_command is None:
        terminal.print_data_help()
        sys.exit(0)

    layout = load_dataset_layout()
    if args.data_command == "fetch":
        try:
            _run_fetch(
                layout,
                force=bool(args.force),
                assume_yes=bool(args.assume_yes),
                keep_zip=bool(args.keep_zip),
            )
        except DatasetFetchError as exc:
            terminal.print_error(str(exc))
            sys.exit(1)
        return

    dataset_status = inspect_dataset_layout(layout)
    if args.data_command == "status":
        _run_status(dataset_status)
        return
    if args.data_command == "verify":
        _run_verify(dataset_status)
        return
    if args.data_command == "path":
        terminal.print_data_path(dataset_status.layout.dataset_root)
        return

    terminal.print_error(f"Unknown data subcommand: {args.data_command}")
    sys.exit(2)


def _run_fetch(layout: DatasetLayout, *, force: bool, assume_yes: bool, keep_zip: bool) -> None:
    terminal.print_stage_status("START", "data", f"Fetching dataset into {layout.dataset_root}")
    dataset_root = fetch_dataset(
        layout,
        force=force,
        assume_yes=assume_yes,
        keep_zip=keep_zip,
    )
    terminal.print_stage_status("DONE", "data", f"Dataset available at {dataset_root}")


def _run_status(dataset_status: DatasetLayoutStatus) -> None:
    terminal.print_data_status(dataset_status)
    if not dataset_status.dataset_is_complete:
        sys.exit(1)


def _run_verify(dataset_status: DatasetLayoutStatus) -> None:
    terminal.print_data_verify(dataset_status)
    if not dataset_status.dataset_is_complete:
        sys.exit(1)
