"""
Top-level CLI dispatcher for the AQUINAS toolkit.

Registered in ``pyproject.toml`` as the ``aquinas`` console script.
"""

from difflib import get_close_matches
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib import import_module
import sys

from aquinas_toolkit.cli import terminal


_TOP_LEVEL_COMMANDS = ("run", "info", "viz", "about", "version", "help")


def main() -> None:
    """Dispatch the top-level ``aquinas`` command."""
    commands = {
        "run": "aquinas_toolkit.cli.run",
        "info": "aquinas_toolkit.cli.info",
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
        if typo_guess is not None:
            terminal.print_compact_command_hint()
        else:
            terminal.print_top_level_help()
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
    if len(command_name) < 2:
        return None

    matches = get_close_matches(command_name, _TOP_LEVEL_COMMANDS, n=2, cutoff=0.55)
    if len(matches) != 1:
        return None

    candidate = matches[0]
    max_distance = 1 if len(candidate) <= 3 else 2
    if _is_single_adjacent_swap(command_name, candidate):
        return candidate
    if _edit_distance(command_name, candidate) > max_distance:
        return None
    return candidate


def _edit_distance(left: str, right: str) -> int:
    """Compute a small Levenshtein distance for typo matching."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current_row.append(
                min(
                    previous_row[right_index] + 1,
                    current_row[right_index - 1] + 1,
                    previous_row[right_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]


def _is_single_adjacent_swap(left: str, right: str) -> bool:
    """Return whether two strings differ by one adjacent transposition."""
    if len(left) != len(right) or len(left) < 2:
        return False

    differences = [index for index, (lchar, rchar) in enumerate(zip(left, right)) if lchar != rchar]
    if len(differences) != 2:
        return False

    first, second = differences
    return (
        second == first + 1
        and left[first] == right[second]
        and left[second] == right[first]
    )
