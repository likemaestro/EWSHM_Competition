"""Shared Rich-based terminal rendering helpers for the AQUINAS CLI."""

from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, TextIO

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from aquinas_toolkit.utils.dataset_config import DatasetLayoutStatus

CLI_THEME = Theme(
    {
        "header": "bold bright_blue",
        "accent": "cyan",
        "key": "bold white",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
        "status_start": "bold bright_blue",
        "status_done": "bold green",
        "status_fail": "bold red",
        "status_tip": "bold yellow",
        "stage_set": "bold magenta",
        "stage_modal": "bold bright_magenta",
    }
)

_ACTIVE_PROGRESS: ContextVar[Progress | None] = ContextVar("_ACTIVE_PROGRESS", default=None)

class CLIView:
    """Simple wrapper that lets tests assert on the underlying plain text."""

    def __init__(self, renderable: Any, plain_text: str):
        self.renderable = renderable
        self.plain_text = plain_text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.renderable

    def __str__(self) -> str:
        return self.plain_text


def build_console(
    *,
    stderr: bool = False,
    force_terminal: bool | None = None,
    file: TextIO | None = None,
    record: bool = False,
) -> Console:
    """Build a console that honors ``NO_COLOR`` and optional test overrides."""
    no_color = bool(os.getenv("NO_COLOR"))
    stream = file or (sys.stderr if stderr else sys.stdout)
    terminal_override = False if no_color else force_terminal
    return Console(
        file=stream,
        stderr=stderr,
        force_terminal=terminal_override,
        record=record,
        theme=CLI_THEME,
        highlight=False,
        soft_wrap=True,
        no_color=no_color,
    )


@lru_cache(maxsize=8)
def _shared_console(stderr: bool, no_color: bool, stream_id: int) -> Console:
    stream = sys.stderr if stderr else sys.stdout
    return Console(
        file=stream,
        stderr=stderr,
        force_terminal=False if no_color else None,
        theme=CLI_THEME,
        highlight=False,
        soft_wrap=True,
        no_color=no_color,
    )


def get_console(*, stderr: bool = False) -> Console:
    """Return a shared console bound to the current stdout/stderr object."""
    no_color = bool(os.getenv("NO_COLOR"))
    stream = sys.stderr if stderr else sys.stdout
    return _shared_console(stderr, no_color, id(stream))


def build_progress(*, transient: bool = False) -> Progress:
    """Build a Rich progress display using the shared CLI console and theme."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
    )


def build_download_progress(*, transient: bool = False) -> Progress:
    """Build a download-oriented progress display with bytes, speed, and ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
        disable=not get_console().is_terminal,
    )


@contextmanager
def progress_context(*, transient: bool = False) -> Iterator[Progress]:
    """Yield the active shared progress display, creating one if needed."""
    active_progress = _ACTIVE_PROGRESS.get()
    if active_progress is not None:
        yield active_progress
        return

    progress = build_progress(transient=transient)
    with progress:
        token = _ACTIVE_PROGRESS.set(progress)
        try:
            yield progress
        finally:
            _ACTIVE_PROGRESS.reset(token)


def print_top_level_help() -> None:
    """Render the top-level CLI help view."""
    get_console().print(render_top_level_help())


def print_run_help(stages: tuple[str, ...] | list[str]) -> None:
    """Render the ``aquinas run`` help view."""
    get_console().print(render_run_help(stages))


def print_viz_help() -> None:
    """Render the ``aquinas viz`` help view."""
    get_console().print(render_viz_help())


def print_data_help() -> None:
    """Render the ``aquinas data`` help view."""
    get_console().print(render_data_help())


def print_data_status(dataset_status: DatasetLayoutStatus) -> None:
    """Render a human-readable dataset status summary."""
    get_console().print(render_data_status(dataset_status))


def print_data_verify(dataset_status: DatasetLayoutStatus) -> None:
    """Render a strict dataset verification summary."""
    get_console().print(render_data_verify(dataset_status))


def print_data_path(dataset_root: Path) -> None:
    """Print the resolved dataset root path for scripting."""
    get_console().print(str(dataset_root))


def print_version_text(version_text: str) -> None:
    """Render the CLI version line."""
    get_console().print(f"aquinas {version_text}")


def print_about(*, version_text: str) -> None:
    """Render toolkit metadata suitable for ``--about``."""
    get_console().print(render_about(version_text=version_text))


def print_error(message: str) -> None:
    """Render a styled error message to stderr."""
    get_console(stderr=True).print(_status_text("ERROR", None, message, "error"))


def print_warning(message: str) -> None:
    """Render a styled warning message."""
    get_console(stderr=True).print(_status_text("WARN", None, message, "warning"))


def print_run_summary(
    *,
    run_id: str,
    run_dir: Path,
    config_path: Path,
    created_new: bool,
) -> None:
    """Render a run summary panel for a created or resolved run."""
    title = "Created Run" if created_new else "Active Run"
    get_console().print(
        render_run_summary(
            title=title,
            run_id=run_id,
            run_dir=run_dir,
            config_path=config_path,
        )
    )


def print_stage_status(prefix: str, stage: str, message: str, *, stderr: bool = False) -> None:
    """Render a styled stage lifecycle line."""
    style = {
        "START": "status_start",
        "DONE": "status_done",
        "FAIL": "status_fail",
        "TIP": "status_tip",
    }.get(prefix, "accent")
    active_progress = _ACTIVE_PROGRESS.get()
    if active_progress is not None and not stderr:
        if prefix in {"START", "DONE", "FAIL"}:
            active_progress.console.print()
        active_progress.console.print(_status_text(prefix, stage, message, style))
        return
    get_console(stderr=stderr).print(_status_text(prefix, stage, message, style))


def print_info_summary(dataset_root: Path, set_count: int) -> None:
    """Render the dataset info header panel."""
    get_console().print(render_info_summary(dataset_root=dataset_root, set_count=set_count))


def print_info_table(rows: list[dict[str, Any]]) -> None:
    """Render the dataset info table."""
    get_console().print(render_info_table(rows))


def print_typo_hint(*, command_name: str, suggested_command: str) -> None:
    """Render a playful typo hint for near-miss top-level commands."""
    get_console().print(
        render_typo_hint(command_name=command_name, suggested_command=suggested_command)
    )


def print_compact_command_hint() -> None:
    """Render a short command summary after typo suggestions."""
    get_console().print(render_compact_command_hint())


def print_viz_summary(
    *,
    run_id: str,
    output_dir: Path,
    manifest_path: Path,
    index_path: Path,
) -> None:
    """Render the viewer artifact summary."""
    get_console().print(
        render_viz_summary(
            run_id=run_id,
            output_dir=output_dir,
            manifest_path=manifest_path,
            index_path=index_path,
        )
    )


def render_top_level_help() -> CLIView:
    """Build the top-level help renderable."""
    usage = Text("Usage: aquinas <command> [options]", style="key")
    commands = Table(
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        header_style="header",
        expand=True,
        show_lines=False,
    )
    commands.add_column("Command", style="key", no_wrap=True)
    commands.add_column("Description")
    commands.add_row("run [stage]", "Run the full pipeline or a single stage.")
    commands.add_row("info", "Show dataset summary (sensors, event counts, date ranges).")
    commands.add_row("data <subcommand>", "Download and manage the local dataset copy.")
    commands.add_row("viz <subcommand>", "Build or open the offline bridge viewer bundle.")
    commands.add_row("about / --about", "Show toolkit metadata and maintainers.")
    commands.add_row("version / --version", "Show installed CLI version.")
    commands.add_row("help", "Show this usage message.")

    notes = _notes_table(
        [
            ("Stages", "preprocess, features, train, score"),
            ("Data", "Use `aquinas data fetch` to bootstrap AQUINAS_DATASET."),
            ("Viewer", "Use `aquinas viz build` to package an offline visualization."),
            ("About", "Use `aquinas --about` to show toolkit metadata."),
            ("Version", "Use `aquinas --version` to show the installed version."),
            ("--name", "Use only when creating a new run."),
            ("--run-id", "Resume features, train, or score from an existing run."),
        ]
    )

    plain_text = (
        "Usage: aquinas <command> [options]\n\n"
        "Commands:\n"
        "  run [stage]   Run the analysis pipeline (all stages, or a specific one).\n"
        "  info          Show dataset summary (sensors, event counts, date ranges).\n"
        "  data <subcommand>  Download and manage the local dataset copy.\n"
        "  viz <subcommand>  Build or open the offline bridge viewer bundle.\n"
        "  about         Show toolkit metadata and maintainers.\n"
        "  version       Show installed CLI version.\n"
        "  help          Show this usage message.\n"
        "Notes:\n"
        "  Stages: preprocess, features, train, score\n"
        "  Data: Use `aquinas data fetch` to bootstrap AQUINAS_DATASET.\n"
        "  Viewer: Use `aquinas viz build` to package an offline visualization.\n"
        "  About: Use `aquinas --about` to show toolkit metadata.\n"
        "  Version: Use `aquinas --version` to show the installed version.\n"
        "  --name: Use only when creating a new run.\n"
        "  --run-id: Resume features, train, or score from an existing run."
    )

    return CLIView(
        Group(
            Panel.fit(
                Text("AQUINAS CLI", style="header"),
                title="Toolkit",
                border_style="accent",
                box=box.ROUNDED,
            ),
            usage,
            commands,
            Panel(notes, title="Usage Notes", border_style="accent", box=box.ROUNDED),
        ),
        plain_text,
    )


def render_run_help(stages: tuple[str, ...] | list[str]) -> CLIView:
    """Build the ``aquinas run`` help renderable."""
    usage = Text("Usage: aquinas run [stage] [--name NAME] [--run-id ID]", style="key")
    commands = Table(
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        header_style="header",
        expand=True,
        show_lines=False,
    )
    commands.add_column("Invocation", style="key", no_wrap=True)
    commands.add_column("Description")
    commands.add_row("aquinas run", "Create a new run and execute the full pipeline.")
    commands.add_row(
        "aquinas run preprocess [--name NAME]",
        "Create a new run and execute preprocessing only.",
    )
    commands.add_row(
        "aquinas run features [--run-id ID]",
        "Run feature extraction in an existing run.",
    )
    commands.add_row("aquinas run train [--run-id ID]", "Run model training in an existing run.")
    commands.add_row("aquinas run score [--run-id ID]", "Run scoring in an existing run.")

    options = _notes_table(
        [
            ("Stages", ", ".join(stages)),
            ("--name", "Optional label stored in metadata when creating a new run."),
            ("--run-id", "Explicit run to resume for features, train, or score."),
            ("--verbose", "Print detailed timing breakdowns to the console."),
            ("--help", "Show this help message."),
        ]
    )

    plain_text = (
        "Usage: aquinas run [stage] [--name NAME] [--run-id ID]\n\n"
        "Stages: "
        f"{', '.join(stages)}\n"
        "Options:\n"
        "  --name    Optional label stored in metadata when creating a new run.\n"
        "  --run-id  Explicit run to resume for features, train, or score.\n"
        "  --verbose Print detailed timing breakdowns to the console.\n"
        "  --help    Show this help message."
    )

    return CLIView(
        Group(
            Panel.fit(
                Text("AQUINAS RUN", style="header"),
                title="Pipeline Command",
                border_style="accent",
                box=box.ROUNDED,
            ),
            usage,
            commands,
            Panel(options, title="Options", border_style="accent", box=box.ROUNDED),
        ),
        plain_text,
    )


def render_viz_help() -> CLIView:
    """Build the ``aquinas viz`` help renderable."""
    usage = Text(
        "Usage: aquinas viz build [--run-id ID] [--set SET] [--output PATH] "
        "[--include-waveforms]\n"
        "       aquinas viz open [--run-id ID] [--output PATH] [--host HOST] [--port PORT]",
        style="key",
    )
    commands = Table(
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        header_style="header",
        expand=True,
        show_lines=False,
    )
    commands.add_column("Invocation", style="key", no_wrap=True)
    commands.add_column("Description")
    commands.add_row(
        "aquinas viz build [--run-id ID]",
        "Export visualization JSON plus a portable offline viewer bundle.",
    )
    commands.add_row(
        "aquinas viz open [--run-id ID]",
        "Serve the bundle over local HTTP and open it in the default browser.",
    )

    options = _notes_table(
        [
            ("--run-id", "Existing run to visualize. Defaults to results/latest.json."),
            ("--set", "Optional AQUINAS set filter. Repeat to include multiple sets."),
            ("--output", "Optional bundle directory override."),
            ("--include-waveforms", "Export capped waveform previews for selected event groups."),
            ("--host / --port", "Local HTTP server address for `aquinas viz open`."),
            ("--no-browser", "Start the local server without opening a browser tab."),
        ]
    )

    plain_text = (
        "Usage: aquinas viz build [--run-id ID] [--set SET] [--output PATH] "
        "[--include-waveforms]\n"
        "       aquinas viz open [--run-id ID] [--output PATH] [--host HOST] [--port PORT]\n\n"
        "Subcommands:\n"
        "  build  Export visualization JSON plus a portable offline viewer bundle.\n"
        "  open   Serve the bundle over local HTTP and open it in the default browser.\n"
        "Options:\n"
        "  --run-id            Existing run to visualize. Defaults to results/latest.json.\n"
        "  --set               Optional AQUINAS set filter. Repeat to include multiple sets.\n"
        "  --output            Optional bundle directory override.\n"
        "  --include-waveforms Export capped waveform previews for selected event groups.\n"
        "  --host / --port     Local HTTP server address for `aquinas viz open`.\n"
        "  --no-browser        Start the local server without opening a browser tab."
    )

    return CLIView(
        Group(
            Panel.fit(
                Text("AQUINAS VIZ", style="header"),
                title="Visualization Command",
                border_style="accent",
                box=box.ROUNDED,
            ),
            usage,
            commands,
            Panel(options, title="Options", border_style="accent", box=box.ROUNDED),
        ),
        plain_text,
    )


def render_data_help() -> CLIView:
    """Build the ``aquinas data`` help renderable."""
    usage = Text(
        "Usage: aquinas data fetch [--force] [--assume-yes] [--keep-zip]\n"
        "       aquinas data status\n"
        "       aquinas data verify\n"
        "       aquinas data path",
        style="key",
    )
    commands = Table(
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        header_style="header",
        expand=True,
        show_lines=False,
    )
    commands.add_column("Invocation", style="key", no_wrap=True)
    commands.add_column("Description")
    commands.add_row(
        "aquinas data fetch",
        "Download, verify (SHA256), and extract the static dataset archive.",
    )
    commands.add_row(
        "aquinas data status",
        "Show a human-readable summary of dataset readiness.",
    )
    commands.add_row(
        "aquinas data verify",
        "Validate that the configured dataset root is complete.",
    )
    commands.add_row(
        "aquinas data path",
        "Print the resolved dataset root path.",
    )

    options = _notes_table(
        [
            ("--force", "Replace an existing dataset root."),
            ("--assume-yes", "Skip overwrite confirmation prompts."),
            ("--yes", "Compatibility alias for `--assume-yes`."),
            ("--keep-zip", "Keep the downloaded ZIP next to the dataset root."),
        ]
    )

    plain_text = (
        "Usage: aquinas data fetch [--force] [--assume-yes] [--keep-zip]\n"
        "       aquinas data status\n"
        "       aquinas data verify\n"
        "       aquinas data path\n\n"
        "Subcommands:\n"
        "  fetch  Download, verify (SHA256), and extract the static dataset archive.\n"
        "  status Show a human-readable summary of dataset readiness.\n"
        "  verify Validate that the configured dataset root is complete.\n"
        "  path   Print the resolved dataset root path.\n"
        "Options:\n"
        "  --force        Replace an existing dataset root.\n"
        "  --assume-yes   Skip overwrite confirmation prompts.\n"
        "  --yes          Compatibility alias for `--assume-yes`.\n"
        "  --keep-zip     Keep the downloaded ZIP next to the dataset root."
    )

    return CLIView(
        Group(
            Panel.fit(
                Text("AQUINAS DATA", style="header"),
                title="Dataset Command",
                border_style="accent",
                box=box.ROUNDED,
            ),
            usage,
            commands,
            Panel(options, title="Options", border_style="accent", box=box.ROUNDED),
        ),
        plain_text,
    )


def render_data_status(dataset_status: DatasetLayoutStatus) -> CLIView:
    """Build a human-readable dataset readiness summary."""
    layout = dataset_status.layout
    present_count = len(layout.set_names) - len(dataset_status.missing_set_names)
    status_label = "complete" if dataset_status.dataset_is_complete else "incomplete"
    root_state = "stub" if dataset_status.dataset_root_is_stub else (
        "present" if dataset_status.dataset_root_exists else "missing"
    )
    missing_text = ", ".join(dataset_status.missing_set_names) if dataset_status.missing_set_names else "none"

    details = _detail_lines(
        [
            ("Dataset root", str(layout.dataset_root)),
            ("Configured sets", str(len(layout.set_names))),
            ("Present sets", str(present_count)),
            ("Missing sets", missing_text),
            ("Root state", root_state),
            ("Status", status_label),
        ]
    )
    plain_text = (
        f"Dataset root: {layout.dataset_root}\n"
        f"Configured sets: {len(layout.set_names)}\n"
        f"Present sets: {present_count}\n"
        f"Missing sets: {missing_text}\n"
        f"Root state: {root_state}\n"
        f"Status: {status_label}"
    )
    return CLIView(
        Panel(details, title="AQUINAS Data Status", border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def render_data_verify(dataset_status: DatasetLayoutStatus) -> CLIView:
    """Build a strict dataset verification summary."""
    layout = dataset_status.layout
    if dataset_status.dataset_is_complete:
        message = f"Dataset verification passed for {layout.dataset_root}"
        plain_text = f"OK: {message}"
        return CLIView(
            Panel(Text(message, style="success"), title="AQUINAS Data Verify", border_style="accent", box=box.ROUNDED),
            plain_text,
        )

    if dataset_status.dataset_root_is_stub:
        failure = f"Dataset root is only a stub bootstrap directory: {layout.dataset_root}"
    elif not dataset_status.dataset_root_exists:
        failure = f"Dataset root does not exist: {layout.dataset_root}"
    else:
        missing = ", ".join(dataset_status.missing_set_names)
        failure = f"Dataset is incomplete at {layout.dataset_root}. Missing set folders: {missing}"

    plain_text = f"FAIL: {failure}"
    return CLIView(
        Panel(Text(failure, style="error"), title="AQUINAS Data Verify", border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def render_run_summary(*, title: str, run_id: str, run_dir: Path, config_path: Path) -> CLIView:
    """Build the run summary panel renderable."""
    details = _detail_lines(
        [
            ("Run ID", run_id),
            ("Run directory", str(run_dir)),
            ("Config snapshot", str(config_path)),
        ]
    )

    plain_text = (
        f"{title}\n"
        f"Run ID: {run_id}\n"
        f"Run directory: {run_dir}\n"
        f"Config snapshot: {config_path}"
    )

    return CLIView(
        Panel(details, title=title, border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def render_info_summary(*, dataset_root: Path, set_count: int) -> CLIView:
    """Build the dataset summary header panel."""
    details = _detail_lines(
        [
            ("Dataset root", str(dataset_root)),
            ("Monthly sets", str(set_count)),
        ]
    )

    plain_text = f"Dataset root: {dataset_root}\nMonthly sets: {set_count}"

    return CLIView(
        Panel(details, title="AQUINAS Dataset", border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def render_info_table(rows: list[dict[str, Any]]) -> CLIView:
    """Build the dataset info table."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        header_style="header",
        expand=False,
        show_lines=False,
    )
    table.add_column("Set", style="key", no_wrap=True)
    table.add_column("Sensors", no_wrap=True)
    table.add_column("Events", no_wrap=True)
    table.add_column("Period")
    table.add_column("Status", no_wrap=True)

    plain_lines = ["Set | Sensors | Events | Period | Status"]
    for row in rows:
        status_style = "success"
        status_text = row["status"]
        row_style = None
        if row.get("level") == "warning":
            status_style = "warning"
            row_style = "warning"
        elif row.get("level") == "error":
            status_style = "error"
            row_style = "error"

        table.add_row(
            row["set_name"],
            row["sensors"],
            row["events"],
            row["period"],
            Text(status_text, style=status_style),
            style=row_style,
        )
        plain_lines.append(
            f"{row['set_name']} | {row['sensors']} | {row['events']} | "
            f"{row['period']} | {status_text}"
        )

    return CLIView(table, "\n".join(plain_lines))


def render_typo_hint(*, command_name: str, suggested_command: str) -> CLIView:
    """Build a compact typo-joke view for near-miss commands."""
    joke = _pick_typo_joke()
    suggestion = f"Did you mean `{suggested_command}`?"
    message = Group(
        Text(joke, style="warning"),
        Text(suggestion, style="muted"),
    )
    plain_text = f"{joke}\n{suggestion}"
    return CLIView(
        Panel(message, title=f"Close Enough: {command_name}", border_style="warning", box=box.ROUNDED),
        plain_text,
    )


def render_compact_command_hint() -> CLIView:
    """Build a compact fallback hint for typo-triggered unknown commands."""
    lines = Group(
        Text("Available commands: run, info, data, viz, about, version, help", style="key"),
        Text("Use `aquinas --help` for full usage.", style="muted"),
    )
    plain_text = (
        "Available commands: run, info, data, viz, about, version, help\n"
        "Use `aquinas --help` for full usage."
    )
    return CLIView(lines, plain_text)


def render_viz_summary(
    *,
    run_id: str,
    output_dir: Path,
    manifest_path: Path,
    index_path: Path,
) -> CLIView:
    """Build the visualization bundle summary renderable."""
    details = _detail_lines(
        [
            ("Run ID", run_id),
            ("Bundle directory", str(output_dir)),
            ("Manifest", str(manifest_path)),
            ("Viewer index", str(index_path)),
        ]
    )
    plain_text = (
        f"Run ID: {run_id}\n"
        f"Bundle directory: {output_dir}\n"
        f"Manifest: {manifest_path}\n"
        f"Viewer index: {index_path}"
    )
    return CLIView(
        Panel(details, title="Visualization Bundle", border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def render_about(*, version_text: str) -> CLIView:
    """Build the ``aquinas --about`` view."""
    maintainer_lines = Group(
        Text("Amir Zare Beiranvand"),
        Text("Liv Breivik"),
        Text("Mohsen Rezvani Alile"),
        Text("Murat Güven"),
        Text("Tommaso Panigati"),
        Text("Zhenkun Li"),
    )

    details = _detail_lines(
        [
            ("Name", "AQUINAS Toolkit"),
            ("Version", version_text),
            (
                "Purpose",
                "Unsupervised, data-driven structural health scoring for EWSHM 2026.",
            ),
        ]
    )
    details = Group(
        details,
        Text("Maintainers:", style="key"),
        maintainer_lines,
    )
    plain_text = (
        "Toolkit: AQUINAS Toolkit\n"
        f"Version: {version_text}\n"
        "Purpose: Unsupervised, data-driven structural health scoring for EWSHM 2026.\n"
        "Maintainers:\n"
        "  - Amir Zare Beiranvand\n"
        "  - Liv Breivik\n"
        "  - Mohsen Rezvani Alile\n"
        "  - Murat Güven\n"
        "  - Tommaso Panigati\n"
        "  - Zhenkun Li"
    )
    return CLIView(
        Panel(details, title="AQUINAS About", border_style="accent", box=box.ROUNDED),
        plain_text,
    )


def _notes_table(rows: list[tuple[str, str]]) -> Table:
    """Build a compact two-column key/value table."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="muted", no_wrap=True, ratio=1)
    table.add_column(style="key", ratio=4)
    for label, value in rows:
        table.add_row(label, value)
    return table


def _detail_lines(rows: list[tuple[str, str]]) -> Group:
    """Build a stack of key/value detail lines that stays readable when wrapped."""
    lines = []
    for label, value in rows:
        line = Text()
        line.append(f"{label}: ", style="key")
        line.append(value)
        lines.append(line)
    return Group(*lines)


def _status_text(prefix: str, label: str | None, message: str, style: str) -> Text:
    """Build a semantic status line."""
    text = Text()
    text.append(f"{prefix:<5}", style=style)
    if label:
        text.append(" ")
        text.append(label, style="key")
    text.append(" ")
    text.append(message)
    return text


def _pick_typo_joke() -> str:
    """Pick a global typo joke while avoiding immediate repeats."""
    global _LAST_TYPO_JOKE

    if len(_TYPO_JOKES) == 1:
        joke = _TYPO_JOKES[0]
        _LAST_TYPO_JOKE = joke
        return joke

    available = [joke for joke in _TYPO_JOKES if joke != _LAST_TYPO_JOKE]
    joke = _TYPO_RANDOM.choice(available or list(_TYPO_JOKES))
    _LAST_TYPO_JOKE = joke
    return joke


_TYPO_RANDOM = random.SystemRandom()
_LAST_TYPO_JOKE: str | None = None

_TYPO_JOKES: tuple[str, ...] = (
    "Identity theft is not a joke, Jim. Millions of commands suffer every year.",
    "Bears. Beets. Broken command.",
    "Did I stutter?",
    "I am not superstitious, but I am a little stitious about that typo.",
    "Well, well, well, how the turntables.",
    "No, God, please no.",
    "Dwight, you ignorant typo.",
    "That is not correct. That's what she said.",
    "Sometimes I'll start a command and I don't even know where it's going.",
    "Question. What kind of bear is best? Wrong question.",
    "Explain it to me like I'm five.",
    "I declare command bankruptcy.",
    "The worst thing about command prison was the dementors.",
    "Today, typing is the worst.",
    "Whenever I'm about to type something wrong, I think, 'Would an idiot do that?'",
    "You have no idea how high I can typo.",
    "I love inside jokes. I'd love to be part of one... with the right dataset.",
)
