"""``aquinas viz`` command."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import webbrowser

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.utils.run_management import RunManagementError, resolve_run
from aquinas_toolkit.visualization import build_visualization_artifacts


class RichVizArgumentParser(argparse.ArgumentParser):
    """Argument parser with Rich-rendered help and error output."""

    def print_help(self, file=None) -> None:  # noqa: ANN001
        terminal.print_viz_help()

    def error(self, message: str) -> None:
        invalid_command = _extract_invalid_choice(message)
        if invalid_command is not None:
            choices = ("build", "open")
            suggestion = terminal.suggest_typo(invalid_command, choices)
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
                    choices=choices,
                    help_command="aquinas viz --help",
                )
            )
            raise SystemExit(2)
        terminal.print_error(message)
        terminal.print_viz_help()
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = RichVizArgumentParser(
        prog="aquinas viz",
        description="Build or open the offline AQUINAS bridge viewer.",
    )
    subparsers = parser.add_subparsers(dest="viz_command")

    build_parser = subparsers.add_parser("build", add_help=False)
    build_parser.add_argument("--run-id", help="Existing run ID. Defaults to results/latest.json.")
    build_parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        help="Restrict the bundle to one or more configured AQUINAS set folders.",
    )
    build_parser.add_argument(
        "--output",
        help="Optional output directory. Defaults to results/<run_id>/visualization.",
    )
    build_parser.add_argument(
        "--include-waveforms",
        action="store_true",
        help="Export capped waveform previews for the richest event groups.",
    )

    open_parser = subparsers.add_parser("open", add_help=False)
    open_parser.add_argument("--run-id", help="Existing run ID. Defaults to results/latest.json.")
    open_parser.add_argument(
        "--output",
        help="Optional bundle directory. Defaults to results/<run_id>/visualization.",
    )
    open_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local host interface for the temporary HTTP server. Default: 127.0.0.1.",
    )
    open_parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port for the temporary HTTP server. Default: auto-select.",
    )
    open_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local viewer server without opening a browser tab.",
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
    parser = build_parser()
    args = parser.parse_args(sys.argv[2:])

    if args.viz_command is None:
        terminal.print_viz_help()
        sys.exit(0)

    try:
        if args.viz_command == "build":
            _run_build(args)
        elif args.viz_command == "open":
            _run_open(args)
        else:  # pragma: no cover - argparse guards this
            raise RunManagementError(f"Unknown viz subcommand: {args.viz_command}")
    except RunManagementError as exc:
        terminal.print_error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        terminal.print_error(str(exc))
        sys.exit(1)


def _run_build(args: argparse.Namespace) -> None:
    run_context = resolve_run(run_id=args.run_id)
    output_dir = Path(args.output) if args.output else None
    result = build_visualization_artifacts(
        run_context,
        set_names=args.sets,
        output_dir=output_dir,
        include_waveforms=bool(args.include_waveforms),
    )
    terminal.print_viz_summary(
        run_id=result.run_id,
        output_dir=result.output_dir,
        manifest_path=result.manifest_path,
        index_path=result.index_path,
    )


def _run_open(args: argparse.Namespace) -> None:
    run_context = resolve_run(run_id=args.run_id)
    bundle_dir = Path(args.output) if args.output else (run_context.run_dir / "visualization")
    index_path = bundle_dir / "index.html"
    if not index_path.is_file():
        raise RunManagementError(
            f"Visualization bundle not found at {bundle_dir}. Run `aquinas viz build` first."
        )

    _serve_bundle(
        bundle_dir=bundle_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def _serve_bundle(
    *,
    bundle_dir: Path,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Serve the static viewer bundle over local HTTP until interrupted."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(bundle_dir))
    with ThreadingHTTPServer((host, port), handler) as server:
        actual_host, actual_port = server.server_address[:2]
        url = f"http://{actual_host}:{actual_port}/index.html"
        opened = webbrowser.open(url) if open_browser else False

        terminal.print_stage_status(
            "DONE" if opened else "WARN",
            "viz",
            f"Viewer URL: {url}",
            stderr=False,
        )
        terminal.print_stage_status(
            "START",
            "viz",
            "Serving visualization bundle over local HTTP. Press Ctrl+C to stop.",
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            terminal.print_stage_status("DONE", "viz", "Visualization server stopped.")
