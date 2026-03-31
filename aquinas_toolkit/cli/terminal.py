"""Shared Rich-based terminal rendering helpers for the AQUINAS CLI."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

CLI_THEME = Theme(
    {
        "header": "bold bright_blue",
        "accent": "cyan",
        "key": "bold white",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
    }
)


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


def print_top_level_help() -> None:
    """Render the top-level CLI help view."""
    get_console().print(render_top_level_help())


def print_run_help(stages: tuple[str, ...] | list[str]) -> None:
    """Render the ``aquinas run`` help view."""
    get_console().print(render_run_help(stages))


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
    style = {"START": "accent", "DONE": "success", "FAIL": "error"}.get(prefix, "accent")
    get_console(stderr=stderr).print(_status_text(prefix, stage, message, style))


def print_info_summary(dataset_root: Path, set_count: int) -> None:
    """Render the dataset info header panel."""
    get_console().print(render_info_summary(dataset_root=dataset_root, set_count=set_count))


def print_info_table(rows: list[dict[str, Any]]) -> None:
    """Render the dataset info table."""
    get_console().print(render_info_table(rows))


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
    commands.add_row("help", "Show this usage message.")

    notes = _notes_table(
        [
            ("Stages", "preprocess, features, train, score"),
            ("--name", "Use only when creating a new run."),
            ("--run-id", "Resume features, train, or score from an existing run."),
        ]
    )

    plain_text = (
        "Usage: aquinas <command> [options]\n\n"
        "Commands:\n"
        "  run [stage]   Run the analysis pipeline (all stages, or a specific one).\n"
        "  info          Show dataset summary (sensors, event counts, date ranges).\n"
        "  help          Show this usage message.\n"
        "Notes:\n"
        "  Stages: preprocess, features, train, score\n"
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
