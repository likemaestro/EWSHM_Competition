"""
Top-level CLI dispatcher for the AQUINAS toolkit.

Registered in ``pyproject.toml`` as the ``aquinas`` console script.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib import import_module
import sys

from aquinas_toolkit.cli import terminal


_TOP_LEVEL_COMMANDS = ("run", "info", "data", "viz", "about", "version", "help")


def main() -> None:
    """Dispatch the top-level ``aquinas`` command."""
    commands = {
        "run": "aquinas_toolkit.cli.run",
        "info": "aquinas_toolkit.cli.info",
        "data": "aquinas_toolkit.cli.data",
        "viz": "aquinas_toolkit.cli.viz",
    }

    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        terminal.print_top_level_help()
        return

    if args[0] in {"--version", "version"}:
        terminal.print_version_text(_resolve_cli_version())
        return

    if args[0] in {"--about", "about"}:
        terminal.print_about(version_text=_resolve_cli_version())
        return

    command_name = args[0]
    module_path = commands.get(command_name)
    if module_path is None:
        typo_guess = _suggest_top_level_command(command_name)
        if typo_guess is not None:
            terminal.print_typo_hint(command_name=command_name, suggested_command=typo_guess)
        terminal.print_error(f"Unknown command: {command_name}")
        terminal.print_compact_command_hint()
        sys.exit(2)

    handler = import_module(module_path).run
    handler()


def _resolve_cli_version() -> str:
    """Resolve the installed package version with a local fallback."""
    try:
        return package_version("aquinas-toolkit")
    except PackageNotFoundError:
        from aquinas_toolkit import __version__

        return __version__


def _suggest_top_level_command(command_name: str) -> str | None:
    """Return a close top-level command suggestion for obvious typos."""
    return terminal.suggest_typo(command_name, _TOP_LEVEL_COMMANDS)
