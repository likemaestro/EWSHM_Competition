"""``aquinas info`` command."""

import sys
from pathlib import Path

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.dataset_fetch import DatasetFetchError, fetch_dataset
from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.utils.dataset_config import (
    DatasetLayout,
    find_missing_set_names,
    load_dataset_layout,
)


def run() -> None:
    """Show a summary of the AQUINAS dataset."""
    layout = load_dataset_layout()
    if not _ensure_dataset_available(layout):
        sys.exit(1)

    dataset_root = layout.dataset_root
    set_dirs = [dataset_root / set_name for set_name in layout.set_names]

    terminal.print_info_summary(dataset_root=dataset_root.resolve(), set_count=len(set_dirs))
    rows: list[dict[str, str]] = []
    for set_dir in set_dirs:
        try:
            rows.append(_summarize_set(set_dir))
        except Exception as exc:
            rows.append(
                {
                    "set_name": set_dir.name,
                    "sensors": "-",
                    "events": "-",
                    "period": "-",
                    "status": _short_error_message(exc),
                    "level": "error",
                }
            )

    terminal.print_info_table(rows)


def _summarize_set(set_dir: Path) -> dict[str, str]:
    """Collect summary values for one AQUINAS set."""
    reader = AquinasReader(set_dir)
    sensors = reader.list_sensor_names()
    acc_count = sum(1 for sensor in sensors if "ACC" in sensor)
    str_count = sum(1 for sensor in sensors if "STR" in sensor)

    first_table = reader.load_index_table(sensors[0])
    event_count = len(first_table)

    start_col = reader.match_column(first_table, ["Start_Time", "start_time"])
    if start_col:
        times = first_table[start_col].sort_values()
        date_range = f"{times.iloc[0]} .. {times.iloc[-1]}"
    else:
        date_range = "unknown"

    return {
        "set_name": reader.set_name,
        "sensors": f"{len(sensors)} ({acc_count} ACC, {str_count} STR)",
        "events": f"~{event_count} per sensor",
        "period": date_range,
        "status": "ok",
        "level": "success",
    }


def _short_error_message(exc: Exception) -> str:
    """Return a compact table-friendly error summary."""
    message = str(exc)
    if "No TABLE_*.json found" in message:
        return "missing tables"
    return message.split(". ")[0]


def _ensure_dataset_available(layout: DatasetLayout) -> bool:
    missing_set_names = find_missing_set_names(layout)
    if not missing_set_names:
        return True

    missing_preview = ", ".join(missing_set_names[:3])
    if len(missing_set_names) > 3:
        missing_preview = f"{missing_preview}, +{len(missing_set_names) - 3} more"

    message = (
        f"Dataset is missing or incomplete at {layout.dataset_root}. "
        f"Missing set folders: {missing_preview}. "
        "Run `aquinas data fetch` (or `aquinas data fetch --force`) to bootstrap it."
    )

    if not (_is_interactive_terminal() and _confirm_fetch()):
        terminal.print_error(message)
        return False

    try:
        fetch_dataset(
            layout,
            force=False,
            assume_yes=False,
            keep_zip=False,
        )
    except DatasetFetchError as exc:
        terminal.print_error(str(exc))
        return False

    return not find_missing_set_names(layout)


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _confirm_fetch() -> bool:
    terminal.print_warning("Dataset is missing. Bootstrap from static archive source now?")
    answer = input("Fetch dataset now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}
