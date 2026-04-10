"""Preprocessing stage orchestration and artifact writing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from aquinas_toolkit.cli.terminal import get_console
from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.preprocessing.alignment import AlignedEvent, SYNCHRO_PASSES, align_event_group
from aquinas_toolkit.preprocessing.core import (
    find_events,
    format_timestamp_utc,
    load_event_group,
    prepare_sensor_records,
)
from aquinas_toolkit.preprocessing.signals import SIGNAL_FILTER_METHODS, filter_loaded_event_group
from aquinas_toolkit.preprocessing.zeroing import ZEROING_METHODS, zero_loaded_event_group
from aquinas_toolkit.utils.run_management import RunContext, stage_output_dir

EVENT_MANIFEST_COLUMNS = [
    "event_id",
    "set_name",
    "deck",
    "start_time_utc",
    "end_time_utc",
    "active_sensor_count",
    "active_sensors",
    "excluded_sensor_count",
    "excluded_sensors",
    "excluded_sensor_reasons",
    "reference_sensor",
    "rows_before_alignment",
    "rows_after_alignment",
    "discarded",
    "discard_reason",
    "zeroing_method",
]

SENSOR_RECORD_COLUMNS = [
    "Record_UID",
    "File",
    "Start_Row",
    "End_Row",
    "Start_Time",
    "End_Time",
    "Duration",
    "Temperature",
    "sensor_name",
    "dataset",
    "set_name",
    "deck",
    "sensor_order",
    "start_time_utc",
    "end_time_utc",
    "raw_file",
    "start_row_1based",
    "end_row_1based",
    "event_id",
    "sensor_status",
    "exclusion_reason",
    "exclusion_source",
]

SENSOR_QC_REPORT_COLUMNS = [
    "set_name",
    "sensor_name",
    "event_count",
    "sensor_status",
    "exclusion_reason",
    "exclusion_source",
    "table_range_median",
    "table_range_nonzero_fraction",
    "table_mean_abs_median",
    "table_start_value_median",
    "table_end_value_median",
    "raw_range_spotcheck_median",
    "raw_to_table_range_ratio_spotcheck",
]


@dataclass(frozen=True)
class SensorExclusion:
    """Declarative sensor exclusion for specific dataset sets."""

    sensor_name: str
    sets: tuple[str, ...]
    reason: str
    source: str


@dataclass(frozen=True)
class PreprocessingSettings:
    """Runtime settings for the preprocess stage.

    Pipeline order: signal filtering → zeroing → alignment.
    """

    dataset_root: Path
    set_names: tuple[str, ...]
    sensor_exclusions: tuple[SensorExclusion, ...] = ()
    event_key_fields: tuple[str, ...] = ("deck", "Start_Time", "End_Time")
    # Signal filtering is the first conditioning step, applied to raw waveforms.
    signal_filter_method: str = "butterworth_bandpass"
    signal_filter_low_hz: float = 0.5
    signal_filter_high_hz: float = 20.0
    signal_filter_order: int = 4
    # Zeroing removes the per-sensor linear baseline after filtering.
    zeroing_method: str = "linear_endpoints"
    # Alignment synchronises timestamps across sensors after zeroing.
    alignment_method: str = "r_synchro"
    min_active_sensors_per_event: int = 1
    export_format: str = "csv.gz"
    partition_by: tuple[str, ...] = ("set_name", "deck")


def run_preprocessing(run_context: RunContext) -> None:
    """Execute preprocessing for every configured AQUINAS set."""
    settings = load_preprocessing_settings(run_context.config_path)
    preprocess_dir = stage_output_dir(run_context.run_dir, "preprocess")
    aligned_dir = preprocess_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    sensor_record_tables: list[pd.DataFrame] = []
    aligned_partitions: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    discard_reasons: Counter[str] = Counter()
    per_deck_total: Counter[str] = Counter()
    per_deck_retained: Counter[str] = Counter()
    exclusion_counts_by_set: Counter[str] = Counter()
    exclusion_counts_by_reason: Counter[str] = Counter()
    applied_sensor_names_by_set: dict[str, set[str]] = defaultdict(set)
    qc_reports: list[pd.DataFrame] = []

    console = get_console()
    set_names = list(settings.set_names)
    n_sets = len(set_names)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        for i, set_name in enumerate(set_names):
            progress.console.print(f"\n[accent]SET {i + 1}/{n_sets}[/]  [key]{set_name}[/]")

            load_task = progress.add_task("  Reading sensor records...", total=None)
            reader = AquinasReader(settings.dataset_root / set_name)
            sensor_records = prepare_sensor_records(reader)
            if sensor_records.empty:
                progress.remove_task(load_task)
                progress.console.print("  [warning]No records found, skipping.[/]")
                continue

            sensor_records = annotate_sensor_records(sensor_records, settings=settings, set_name=set_name)
            included_sensor_records, exclusion_log = apply_sensor_exclusions(sensor_records)
            sensor_record_tables.append(sensor_records.copy())
            qc_reports.append(build_sensor_qc_report(reader, sensor_records, settings=settings, set_name=set_name))
            progress.remove_task(load_task)

            for entry in exclusion_log:
                resolved_set_name = str(entry["set_name"])
                exclusion_counts_by_set[resolved_set_name] += int(entry["record_count"])
                exclusion_counts_by_reason[str(entry["exclusion_reason"])] += int(entry["record_count"])
                applied_sensor_names_by_set[resolved_set_name].add(str(entry["sensor_name"]))

            events = find_events(reader, records=included_sensor_records)
            event_task = progress.add_task("  Processing events", total=len(events))
            set_retained = 0

            for _, event_row in events.iterrows():
                per_deck_total[str(event_row["deck"])] += 1
                excluded_event_rows = sensor_records.loc[
                    (sensor_records["event_id"] == event_row["event_id"])
                    & (sensor_records["sensor_status"] == "excluded")
                ].copy()
                if int(event_row["active_sensor_count"]) < settings.min_active_sensors_per_event:
                    discard_reason = "insufficient_active_sensors"
                    discard_reasons[discard_reason] += 1
                    manifest_rows.append(
                        _build_manifest_row(
                            event_row,
                            excluded_event_rows=excluded_event_rows,
                            reference_sensor="",
                            rows_before_alignment=0,
                            rows_after_alignment=0,
                            discard_reason=discard_reason,
                            zeroing_method=settings.zeroing_method,
                        )
                    )
                    progress.advance(event_task)
                    continue

                loaded_event = load_event_group(reader, event_row, records=included_sensor_records)
                # Pipeline order: filter → zero → align
                filtered_event = filter_loaded_event_group(
                    loaded_event,
                    method=settings.signal_filter_method,
                    low_hz=settings.signal_filter_low_hz,
                    high_hz=settings.signal_filter_high_hz,
                    order=settings.signal_filter_order,
                )
                zeroed_event_group = zero_loaded_event_group(filtered_event, method=settings.zeroing_method)
                aligned_event = align_event_group(zeroed_event_group, method=settings.alignment_method)

                rows_before_alignment = 0
                if zeroed_event_group.waveforms:
                    rows_before_alignment = int(
                        len(zeroed_event_group.waveforms[aligned_event.reference_sensor][1])
                    )
                rows_after_alignment = int(aligned_event.alignment_diagnostics["rows_after_alignment"])
                if rows_after_alignment == 0:
                    discard_reason = "no_common_aligned_rows"
                    discard_reasons[discard_reason] += 1
                    manifest_rows.append(
                        _build_manifest_row(
                            event_row,
                            excluded_event_rows=excluded_event_rows,
                            reference_sensor=aligned_event.reference_sensor,
                            rows_before_alignment=rows_before_alignment,
                            rows_after_alignment=rows_after_alignment,
                            discard_reason=discard_reason,
                            zeroing_method=settings.zeroing_method,
                        )
                    )
                    progress.advance(event_task)
                    continue

                set_retained += 1
                per_deck_retained[aligned_event.deck] += 1
                manifest_rows.append(
                    _build_manifest_row(
                        event_row,
                        excluded_event_rows=excluded_event_rows,
                        reference_sensor=aligned_event.reference_sensor,
                        rows_before_alignment=rows_before_alignment,
                        rows_after_alignment=rows_after_alignment,
                        discard_reason=None,
                        zeroing_method=settings.zeroing_method,
                    )
                )
                aligned_partitions[(aligned_event.set_name, aligned_event.deck)].append(
                    aligned_event_to_long_frame(aligned_event)
                )
                progress.advance(event_task)

            progress.remove_task(event_task)
            set_discarded = len(events) - set_retained
            progress.console.print(
                f"  [success]done[/]  [key]{set_retained:,}[/] retained  "
                f"[muted]{set_discarded:,} discarded[/]"
            )

    _write_manifest(preprocess_dir / "event_manifest.csv", manifest_rows)
    _write_sensor_records(preprocess_dir / "sensor_records.csv", sensor_record_tables)
    _write_aligned_partitions(aligned_dir, aligned_partitions, settings.export_format)
    _write_sensor_qc_report(preprocess_dir / "sensor_qc_report.csv", qc_reports)
    _write_summary(
        preprocess_dir / "summary.json",
        settings=settings,
        manifest_rows=manifest_rows,
        discard_reasons=discard_reasons,
        per_deck_total=per_deck_total,
        per_deck_retained=per_deck_retained,
        exclusion_counts_by_set=exclusion_counts_by_set,
        exclusion_counts_by_reason=exclusion_counts_by_reason,
        applied_sensor_names_by_set=applied_sensor_names_by_set,
    )


def export_aligned_event(event: AlignedEvent, output_path: str | Path) -> Path:
    """Export one aligned event as a wide CSV artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_frame = event.aligned_waveform.copy()
    export_frame["timestamp_utc"] = export_frame["timestamp_utc"].map(format_timestamp_utc)
    export_frame.to_csv(path, index=False)
    return path


def aligned_event_to_long_frame(event: AlignedEvent) -> pd.DataFrame:
    """Convert an aligned event into the stage's long-form export shape."""
    long_frame = event.aligned_waveform.copy()
    long_frame["timestamp_utc"] = long_frame["timestamp_utc"].map(format_timestamp_utc)
    long_frame.insert(0, "sample_index", range(len(long_frame)))
    long_frame.insert(0, "event_id", event.event_id)
    return long_frame


def load_preprocessing_settings(config_path: Path) -> PreprocessingSettings:
    """Parse the snapped run config into stage settings."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = config.get("data") or {}
    preprocessing = config.get("preprocessing") or {}
    signal_filter = preprocessing.get("signal_filter") or {}
    alignment = preprocessing.get("alignment") or {}
    zeroing = preprocessing.get("zeroing") or {}
    filtering = preprocessing.get("filtering") or {}
    export = preprocessing.get("export") or {}
    event_grouping = preprocessing.get("event_grouping") or {}
    sensor_overrides = preprocessing.get("sensor_overrides") or {}
    sensor_exclusions = tuple(_parse_sensor_exclusion(entry) for entry in sensor_overrides.get("exclude", ()))

    dataset_root = data.get("dataset_root", "AQUINAS_DATASET")
    dataset_root_path = _resolve_repo_path(dataset_root)
    set_names = tuple(data.get("sets", ()))
    if not set_names:
        raise ValueError("Config must provide at least one dataset in data.sets.")

    signal_filter_method = str(signal_filter.get("method", "butterworth_bandpass"))
    if signal_filter_method not in SIGNAL_FILTER_METHODS:
        raise ValueError(
            f"Unsupported preprocessing.signal_filter.method: {signal_filter_method!r}. "
            f"Supported methods are {sorted(SIGNAL_FILTER_METHODS)}."
        )

    _validate_alignment_config(alignment)

    zeroing_method = str(zeroing.get("method", "linear_endpoints"))
    if zeroing_method not in ZEROING_METHODS:
        raise ValueError(
            f"Unsupported preprocessing.zeroing.method: {zeroing_method}. "
            f"Supported methods are {sorted(ZEROING_METHODS)}."
        )

    return PreprocessingSettings(
        dataset_root=dataset_root_path,
        set_names=set_names,
        sensor_exclusions=sensor_exclusions,
        event_key_fields=tuple(event_grouping.get("key_fields", ("deck", "Start_Time", "End_Time"))),
        signal_filter_method=signal_filter_method,
        signal_filter_low_hz=float(signal_filter.get("low_hz", 0.5)),
        signal_filter_high_hz=float(signal_filter.get("high_hz", 20.0)),
        signal_filter_order=int(signal_filter.get("order", 4)),
        zeroing_method=zeroing_method,
        alignment_method=str(alignment.get("method", "r_synchro")),
        min_active_sensors_per_event=int(filtering.get("min_active_sensors_per_event", 1)),
        export_format=str(export.get("format", "csv.gz")),
        partition_by=tuple(export.get("partition_by", ("set_name", "deck"))),
    )


def _validate_alignment_config(alignment: dict[str, Any]) -> None:
    allowed_keys = {"method"}
    stale_keys = sorted(set(alignment) - allowed_keys)
    if stale_keys:
        raise ValueError(
            "Legacy preprocessing.alignment keys are no longer supported: "
            + ", ".join(stale_keys)
        )

    method = str(alignment.get("method", "r_synchro"))
    if method != "r_synchro":
        # TODO: add additional alignment methods here
        raise ValueError(
            f"Unsupported preprocessing.alignment.method: {method}. Only 'r_synchro' is supported."
        )


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def annotate_sensor_records(
    sensor_records: pd.DataFrame,
    *,
    settings: PreprocessingSettings,
    set_name: str,
) -> pd.DataFrame:
    """Annotate sensor records with inclusion/exclusion status."""
    annotated = sensor_records.copy()
    annotated["sensor_status"] = "included"
    annotated["exclusion_reason"] = ""
    annotated["exclusion_source"] = ""

    for exclusion in settings.sensor_exclusions:
        if set_name not in exclusion.sets:
            continue
        mask = annotated["sensor_name"] == exclusion.sensor_name
        if not mask.any():
            continue
        annotated.loc[mask, "sensor_status"] = "excluded"
        annotated.loc[mask, "exclusion_reason"] = exclusion.reason
        annotated.loc[mask, "exclusion_source"] = exclusion.source

    return annotated


def apply_sensor_exclusions(sensor_records: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return included records and an exclusion log for summary reporting."""
    excluded = sensor_records.loc[sensor_records["sensor_status"] == "excluded"].copy()
    exclusion_log: list[dict[str, Any]] = []
    if not excluded.empty:
        grouped = (
            excluded.groupby(
                ["set_name", "sensor_name", "exclusion_reason", "exclusion_source"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "record_count"})
        )
        exclusion_log = grouped.to_dict("records")

    included = sensor_records.loc[sensor_records["sensor_status"] == "included"].copy()
    return included.reset_index(drop=True), exclusion_log


def build_sensor_qc_report(
    reader: AquinasReader,
    sensor_records: pd.DataFrame,
    *,
    settings: PreprocessingSettings,
    set_name: str,
    raw_spotcheck_samples: int = 3,
) -> pd.DataFrame:
    """Build a report-only per-sensor QC summary for one dataset set."""
    range_column = reader.match_column(sensor_records, ["Range"])
    mean_column = reader.match_column(sensor_records, ["Mean_Value"])
    start_value_column = reader.match_column(sensor_records, ["Start_Value"])
    end_value_column = reader.match_column(sensor_records, ["End_Value"])

    rows: list[dict[str, Any]] = []
    for sensor_name, sensor_group in sensor_records.groupby("sensor_name", sort=True):
        range_values = _numeric_series(sensor_group[range_column]) if range_column else pd.Series(dtype=float)
        mean_values = _numeric_series(sensor_group[mean_column]) if mean_column else pd.Series(dtype=float)
        start_values = _numeric_series(sensor_group[start_value_column]) if start_value_column else pd.Series(dtype=float)
        end_values = _numeric_series(sensor_group[end_value_column]) if end_value_column else pd.Series(dtype=float)

        status = str(sensor_group["sensor_status"].iloc[0])
        exclusion_reason = str(sensor_group["exclusion_reason"].iloc[0])
        exclusion_source = str(sensor_group["exclusion_source"].iloc[0])

        row = {
            "set_name": set_name,
            "sensor_name": sensor_name,
            "event_count": int(len(sensor_group)),
            "sensor_status": status,
            "exclusion_reason": exclusion_reason,
            "exclusion_source": exclusion_source,
            "table_range_median": _safe_median(range_values),
            "table_range_nonzero_fraction": float((range_values != 0).mean()) if not range_values.empty else float("nan"),
            "table_mean_abs_median": _safe_median(mean_values.abs()),
            "table_start_value_median": _safe_median(start_values),
            "table_end_value_median": _safe_median(end_values),
            "raw_range_spotcheck_median": float("nan"),
            "raw_to_table_range_ratio_spotcheck": float("nan"),
        }

        if status == "excluded":
            raw_ranges = []
            table_ranges = []
            for _, meta_row in sensor_group.head(raw_spotcheck_samples).iterrows():
                waveform = _load_waveform_slice(reader, meta_row)
                values = pd.to_numeric(waveform[sensor_name], errors="coerce").dropna()
                if values.empty:
                    continue
                raw_range = float(values.max() - values.min())
                raw_ranges.append(raw_range)
                if range_column is not None:
                    table_ranges.append(float(meta_row[range_column]))
            if raw_ranges:
                row["raw_range_spotcheck_median"] = _safe_median(pd.Series(raw_ranges, dtype=float))
                if table_ranges:
                    ratios = [
                        raw_range / table_range
                        for raw_range, table_range in zip(raw_ranges, table_ranges, strict=False)
                        if table_range != 0
                    ]
                    if ratios:
                        row["raw_to_table_range_ratio_spotcheck"] = _safe_median(
                            pd.Series(ratios, dtype=float)
                        )

        rows.append(row)

    return pd.DataFrame(rows)


def _parse_sensor_exclusion(entry: Any) -> SensorExclusion:
    if not isinstance(entry, dict):
        raise ValueError("Each preprocessing.sensor_overrides.exclude entry must be a mapping.")

    sensor_name = str(entry.get("sensor_name", "")).strip()
    if not sensor_name:
        raise ValueError("Sensor exclusion entries must define sensor_name.")

    sets = tuple(str(value) for value in entry.get("sets", ()) if str(value).strip())
    if not sets:
        raise ValueError(f"Sensor exclusion for '{sensor_name}' must define at least one set.")

    reason = str(entry.get("reason", "")).strip()
    source = str(entry.get("source", "")).strip()
    return SensorExclusion(sensor_name=sensor_name, sets=sets, reason=reason, source=source)


def _build_manifest_row(
    event_row: pd.Series,
    *,
    excluded_event_rows: pd.DataFrame,
    reference_sensor: str,
    rows_before_alignment: int,
    rows_after_alignment: int,
    discard_reason: str | None,
    zeroing_method: str,
) -> dict[str, Any]:
    excluded_sensors = sorted(set(excluded_event_rows["sensor_name"])) if not excluded_event_rows.empty else []
    excluded_reasons = (
        sorted(
            {
                str(reason)
                for reason in excluded_event_rows["exclusion_reason"].tolist()
                if str(reason).strip()
            }
        )
        if not excluded_event_rows.empty
        else []
    )
    return {
        "event_id": event_row["event_id"],
        "set_name": event_row["set_name"],
        "deck": event_row["deck"],
        "start_time_utc": format_timestamp_utc(event_row["start_time_utc"]),
        "end_time_utc": format_timestamp_utc(event_row["end_time_utc"]),
        "active_sensor_count": int(event_row["active_sensor_count"]),
        "active_sensors": ";".join(event_row["active_sensors"]),
        "excluded_sensor_count": len(excluded_sensors),
        "excluded_sensors": ";".join(excluded_sensors),
        "excluded_sensor_reasons": ";".join(excluded_reasons),
        "reference_sensor": reference_sensor,
        "rows_before_alignment": rows_before_alignment,
        "rows_after_alignment": rows_after_alignment,
        "discarded": discard_reason is not None,
        "discard_reason": discard_reason or "",
        "zeroing_method": zeroing_method,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=EVENT_MANIFEST_COLUMNS).to_csv(path, index=False)


def _write_sensor_records(path: Path, sensor_record_tables: list[pd.DataFrame]) -> None:
    if sensor_record_tables:
        sensor_records = pd.concat(sensor_record_tables, ignore_index=True)
        sensor_records = sensor_records.copy()
        sensor_records["start_time_utc"] = sensor_records["start_time_utc"].map(format_timestamp_utc)
        sensor_records["end_time_utc"] = sensor_records["end_time_utc"].map(format_timestamp_utc)
    else:
        sensor_records = pd.DataFrame(columns=SENSOR_RECORD_COLUMNS)
    sensor_records.to_csv(path, index=False)


def _write_sensor_qc_report(path: Path, qc_reports: list[pd.DataFrame]) -> None:
    if qc_reports:
        qc_report = pd.concat(qc_reports, ignore_index=True)
    else:
        qc_report = pd.DataFrame(columns=SENSOR_QC_REPORT_COLUMNS)
    qc_report.to_csv(path, index=False)


def _write_aligned_partitions(
    aligned_dir: Path,
    partitions: dict[tuple[str, str], list[pd.DataFrame]],
    export_format: str,
) -> None:
    suffix = ".csv.gz" if export_format == "csv.gz" else ".csv"
    for (set_name, deck), frames in partitions.items():
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(aligned_dir / f"{set_name}__{deck}{suffix}", index=False)


def _write_summary(
    path: Path,
    *,
    settings: PreprocessingSettings,
    manifest_rows: list[dict[str, Any]],
    discard_reasons: Counter[str],
    per_deck_total: Counter[str],
    per_deck_retained: Counter[str],
    exclusion_counts_by_set: Counter[str],
    exclusion_counts_by_reason: Counter[str],
    applied_sensor_names_by_set: dict[str, set[str]],
) -> None:
    total_events = len(manifest_rows)
    retained_events = sum(1 for row in manifest_rows if not row["discarded"])
    payload = {
        "total_events": total_events,
        "retained_events": retained_events,
        "discarded_events": total_events - retained_events,
        "discard_reasons": dict(discard_reasons),
        "per_deck_total": dict(per_deck_total),
        "per_deck_retained": dict(per_deck_retained),
        "signal_filter": {
            "method": settings.signal_filter_method,
            "low_hz": settings.signal_filter_low_hz,
            "high_hz": settings.signal_filter_high_hz,
            "order": settings.signal_filter_order,
            "stage": "before_zeroing",
        },
        "zeroing": {
            "method": settings.zeroing_method,
            "stage": "after_signal_filter_before_alignment",
        },
        "alignment": {
            "method": settings.alignment_method,
            "reference_policy": "first_selected",
            "passes": SYNCHRO_PASSES,
        },
        "event_grouping": {"key_fields": list(settings.event_key_fields)},
        "sensor_exclusions": {
            "configured": [asdict(exclusion) for exclusion in settings.sensor_exclusions],
            "applied_record_counts_by_set": dict(exclusion_counts_by_set),
            "applied_record_counts_by_reason": dict(exclusion_counts_by_reason),
            "applied_sensor_names_by_set": {
                set_name: sorted(sensor_names)
                for set_name, sensor_names in applied_sensor_names_by_set.items()
            },
        },
        "export": {
            "format": settings.export_format,
            "partition_by": list(settings.partition_by),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").dropna().astype(float)


def _safe_median(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.median())


def _load_waveform_slice(reader: AquinasReader, meta_row: pd.Series) -> pd.DataFrame:
    raw_df = reader.load_raw_file(str(meta_row["sensor_name"]), str(meta_row["raw_file"])).copy()
    waveform = raw_df.iloc[int(meta_row["start_row_1based"]) - 1 : int(meta_row["end_row_1based"])].copy()
    waveform = waveform.reset_index(drop=True)
    timestamp_col = reader.match_column(waveform, ["timestamp", "Timestamp"])
    sensor_name = str(meta_row["sensor_name"])
    if sensor_name in waveform.columns:
        value_col = sensor_name
    else:
        measure_columns = [column for column in waveform.columns if column != timestamp_col]
        if not measure_columns:
            raise KeyError(f"Raw waveform for sensor '{sensor_name}' does not contain sensor values.")
        value_col = measure_columns[0]
    result = waveform[[timestamp_col, value_col]].copy()
    return result.rename(columns={timestamp_col: "timestamp", value_col: sensor_name})
