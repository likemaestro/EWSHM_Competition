"""
Top-level CLI dispatcher for the AQUINAS toolkit.

Registered in ``pyproject.toml`` as the ``aquinas`` console script.
"""

from importlib import import_module
import sys

from aquinas_toolkit.cli import terminal


def main() -> None:
    """Dispatch the top-level ``aquinas`` command."""
    commands = {
        "run": "aquinas_toolkit.cli.run",
        "info": "aquinas_toolkit.cli.info",
    }

    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        terminal.print_top_level_help()
        return

    command_name = args[0]
    module_path = commands.get(command_name)
    if module_path is None:
        terminal.print_error(f"Unknown command: {command_name}")
        terminal.print_top_level_help()
        sys.exit(2)

    handler = import_module(module_path).run
    handler()
