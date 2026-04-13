"""``aquinas info`` command."""

import sys
from pathlib import Path

from aquinas_toolkit.cli import terminal
from aquinas_toolkit.io import AquinasReader


def run() -> None:
    """Show a summary of the AQUINAS dataset."""
    dataset_root = Path("AQUINAS_DATASET")

    if not dataset_root.exists():
        terminal.print_error(
            f"Dataset folder not found at: {dataset_root.resolve()}. "
            "Place the AQUINAS dataset at AQUINAS_DATASET/ in the repo root."
        )
        sys.exit(1)

    set_dirs = sorted(dataset_root.glob("AQUINAS_SET*"))
    if not set_dirs:
        terminal.print_error(f"No AQUINAS_SET* folders found in {dataset_root}")
        sys.exit(1)

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
