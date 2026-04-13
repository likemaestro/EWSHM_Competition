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
    collapse_sensor_records,
    find_events,
    format_timestamp_utc,
    load_event_group,
    prepare_sensor_records,
)
from aquinas_toolkit.preprocessing.signals import SIGNAL_FILTER_METHODS, filter_loaded_event_group
from aquinas_toolkit.preprocessing.store import (
    ALIGNED_SAMPLE_COLUMNS,
    EVENT_SENSOR_COLUMNS,
    PreprocessStoreWriter,
    preprocess_store_path,
)
from aquinas_toolkit.preprocessing.zeroing import ZEROING_METHODS, zero_loaded_event_group
from aquinas_toolkit.utils.run_management import RunContext, stage_output_dir, write_stage_progress


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
    "table_row_index",
    "Record_UID",
    "File",
    "Start_Row",
    "End_Row",
    "Start_Time",
    "End_Time",
    "Duration",
    "Start_Value",
    "End_Value",
    "Diff_Value",
    "Min_Value",
    "Max_Value",
    "Mean_Value",
    "Range",
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

    Pipeline order: signal filtering -> zeroing -> alignment.
    """

    dataset_root: Path
    set_names: tuple[str, ...]
    sensor_exclusions: tuple[SensorExclusion, ...] = ()
    event_key_fields: tuple[str, ...] = ("deck", "Start_Time", "End_Time")
    signal_filter_method: str = "butterworth_bandpass"
    signal_filter_low_hz: float = 0.5
    signal_filter_high_hz: float = 20.0
    signal_filter_order: int = 4
    zeroing_method: str = "linear_endpoints"
    alignment_method: str = "r_synchro"
    min_active_sensors_per_event: int = 1
    storage_backend: str = "sqlite"
    aligned_export_enabled: bool = False
    aligned_export_format: str = "csv.gz"


def run_preprocessing(run_context: RunContext) -> None:
    """Execute preprocessing for every configured AQUINAS set."""
    settings = load_preprocessing_settings(run_context.config_path)
    preprocess_dir = stage_output_dir(run_context.run_dir, "preprocess")
    preprocess_db = preprocess_store_path(preprocess_dir)
    aligned_export_dir = preprocess_dir / "exports" / "aligned"
    if settings.aligned_export_enabled:
        aligned_export_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    discard_reasons: Counter[str] = Counter()
    per_deck_total: Counter[str] = Counter()
    per_deck_retained: Counter[str] = Counter()
    exclusion_counts_by_set: Counter[str] = Counter()
    exclusion_counts_by_reason: Counter[str] = Counter()
    applied_sensor_names_by_set: dict[str, set[str]] = defaultdict(set)
    preprocess_progress = {
        "current_set": None,
        "completed_sets": [],
        "written_partitions": [],
    }

    console = get_console()
    set_names = list(settings.set_names)
    store_writer = PreprocessStoreWriter(
        preprocess_db,
        run_id=run_context.run_id,
        settings_payload=_settings_payload(settings),
        set_names=set_names,
    )

    try:
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
            _run_preprocess_sets(
                run_context=run_context,
                settings=settings,
                set_names=set_names,
                store_writer=store_writer,
                progress=progress,
                manifest_rows=manifest_rows,
                discard_reasons=discard_reasons,
                per_deck_total=per_deck_total,
                per_deck_retained=per_deck_retained,
                exclusion_counts_by_set=exclusion_counts_by_set,
                exclusion_counts_by_reason=exclusion_counts_by_reason,
                applied_sensor_names_by_set=applied_sensor_names_by_set,
                preprocess_progress=preprocess_progress,
                aligned_export_dir=aligned_export_dir,
            )
    finally:
        store_writer.close()

    _write_summary(
        preprocess_dir / "summary.json",
        settings=settings,
        preprocess_db_path=preprocess_db,
        manifest_rows=manifest_rows,
        discard_reasons=discard_reasons,
        per_deck_total=per_deck_total,
        per_deck_retained=per_deck_retained,
        exclusion_counts_by_set=exclusion_counts_by_set,
        exclusion_counts_by_reason=exclusion_counts_by_reason,
        applied_sensor_names_by_set=applied_sensor_names_by_set,
    )


def _run_preprocess_sets(
    *,
    run_context: RunContext,
    settings: PreprocessingSettings,
    set_names: list[str],
    store_writer: PreprocessStoreWriter,
    progress: Progress,
    manifest_rows: list[dict[str, Any]],
    discard_reasons: Counter[str],
    per_deck_total: Counter[str],
    per_deck_retained: Counter[str],
    exclusion_counts_by_set: Counter[str],
    exclusion_counts_by_reason: Counter[str],
    applied_sensor_names_by_set: dict[str, set[str]],
    preprocess_progress: dict[str, Any],
    aligned_export_dir: Path,
) -> None:
    for index, set_name in enumerate(set_names, start=1):
        preprocess_progress["current_set"] = set_name
        _write_preprocess_progress(run_context, preprocess_progress)
        progress.console.print(f"\n[accent]SET {index}/{len(set_names)}[/]  [key]{set_name}[/]")

        load_task = progress.add_task("  Reading sensor records...", total=None)
        reader = AquinasReader(settings.dataset_root / set_name)
        sensor_records = prepare_sensor_records(reader)
        if sensor_records.empty:
            progress.remove_task(load_task)
            progress.console.print("  [warning]No records found, skipping.[/]")
            store_writer.write_set(
                sensor_records=pd.DataFrame(columns=SENSOR_RECORD_COLUMNS),
                qc_report=pd.DataFrame(columns=SENSOR_QC_REPORT_COLUMNS),
                events=pd.DataFrame(columns=EVENT_MANIFEST_COLUMNS),
                event_sensors=pd.DataFrame(columns=EVENT_SENSOR_COLUMNS),
                aligned_samples=pd.DataFrame(columns=ALIGNED_SAMPLE_COLUMNS),
            )
            _mark_set_progress_complete(
                preprocess_progress,
                set_name=set_name,
                written_partitions=[],
            )
            _write_preprocess_progress(run_context, preprocess_progress)
            continue

        sensor_records = annotate_sensor_records(sensor_records, settings=settings, set_name=set_name)
        included_sensor_records, exclusion_log = apply_sensor_exclusions(sensor_records)
        qc_report = build_sensor_qc_report(reader, sensor_records, settings=settings, set_name=set_name)
        progress.remove_task(load_task)

        for entry in exclusion_log:
            resolved_set_name = str(entry["set_name"])
            exclusion_counts_by_set[resolved_set_name] += int(entry["record_count"])
            exclusion_counts_by_reason[str(entry["exclusion_reason"])] += int(entry["record_count"])
            applied_sensor_names_by_set[resolved_set_name].add(str(entry["sensor_name"]))

        set_manifest_rows: list[dict[str, Any]] = []
        set_event_sensor_rows: list[dict[str, Any]] = []
        set_aligned_sample_rows: list[dict[str, Any]] = []
        set_aligned_partitions: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)

        events = find_events(reader, records=included_sensor_records)
        event_task = progress.add_task("  Processing events", total=len(events))
        set_retained = 0

        for _, event_row in events.iterrows():
            per_deck_total[str(event_row["deck"])] += 1
            event_sensor_records = sensor_records.loc[
                sensor_records["event_id"] == event_row["event_id"]
            ].copy()
            excluded_event_rows = event_sensor_records.loc[
                event_sensor_records["sensor_status"] == "excluded"
            ].copy()

            if int(event_row["active_sensor_count"]) < settings.min_active_sensors_per_event:
                discard_reason = "insufficient_active_sensors"
                discard_reasons[discard_reason] += 1
                manifest_row = _build_manifest_row(
                    event_row,
                    excluded_event_rows=excluded_event_rows,
                    reference_sensor="",
                    rows_before_alignment=0,
                    rows_after_alignment=0,
                    discard_reason=discard_reason,
                    zeroing_method=settings.zeroing_method,
                )
                manifest_rows.append(manifest_row)
                set_manifest_rows.append(manifest_row)
                set_event_sensor_rows.extend(
                    _build_event_sensor_rows(event_sensor_records, reference_sensor="")
                )
                progress.advance(event_task)
                continue

            loaded_event = load_event_group(reader, event_row, records=included_sensor_records)
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
                manifest_row = _build_manifest_row(
                    event_row,
                    excluded_event_rows=excluded_event_rows,
                    reference_sensor=aligned_event.reference_sensor,
                    rows_before_alignment=rows_before_alignment,
                    rows_after_alignment=rows_after_alignment,
                    discard_reason=discard_reason,
                    zeroing_method=settings.zeroing_method,
                )
                manifest_rows.append(manifest_row)
                set_manifest_rows.append(manifest_row)
                set_event_sensor_rows.extend(
                    _build_event_sensor_rows(
                        event_sensor_records,
                        reference_sensor=aligned_event.reference_sensor,
                    )
                )
                progress.advance(event_task)
                continue

            set_retained += 1
            per_deck_retained[aligned_event.deck] += 1
            manifest_row = _build_manifest_row(
                event_row,
                excluded_event_rows=excluded_event_rows,
                reference_sensor=aligned_event.reference_sensor,
                rows_before_alignment=rows_before_alignment,
                rows_after_alignment=rows_after_alignment,
                discard_reason=None,
                zeroing_method=settings.zeroing_method,
            )
            manifest_rows.append(manifest_row)
            set_manifest_rows.append(manifest_row)
            set_event_sensor_rows.extend(
                _build_event_sensor_rows(
                    event_sensor_records,
                    reference_sensor=aligned_event.reference_sensor,
                )
            )
            set_aligned_sample_rows.extend(aligned_event_to_sample_rows(aligned_event))
            set_aligned_partitions[(aligned_event.set_name, aligned_event.deck)].append(
                aligned_event_to_long_frame(aligned_event)
            )
            progress.advance(event_task)

        progress.remove_task(event_task)
        set_discarded = len(events) - set_retained
        progress.console.print(
            f"  [success]done[/]  [key]{set_retained:,}[/] retained  "
            f"[muted]{set_discarded:,} discarded[/]"
        )
        progress.console.print("  [accent]Writing preprocess store...[/]")
        write_task = progress.add_task("  Writing preprocess store...", total=None)
        # TODO: overlap this per-set canonical DB commit and optional aligned export
        # with next-set preprocessing via a bounded producer-consumer writer once
        # failure handling and metadata semantics stay deterministic.
        store_writer.write_set(
            sensor_records=sensor_records,
            qc_report=qc_report,
            events=pd.DataFrame(set_manifest_rows, columns=EVENT_MANIFEST_COLUMNS),
            event_sensors=pd.DataFrame(set_event_sensor_rows, columns=EVENT_SENSOR_COLUMNS),
            aligned_samples=pd.DataFrame(set_aligned_sample_rows, columns=ALIGNED_SAMPLE_COLUMNS),
        )
        written_partitions = _partition_labels_for_partitions(set_aligned_partitions)
        _mark_set_progress_complete(
            preprocess_progress,
            set_name=set_name,
            written_partitions=written_partitions,
        )
        _write_preprocess_progress(run_context, preprocess_progress)

        if settings.aligned_export_enabled:
            progress.console.print("  [accent]Writing aligned exports...[/]")
            _write_set_aligned_partitions(
                aligned_export_dir,
                partitions=set_aligned_partitions,
                export_format=settings.aligned_export_format,
            )

        progress.remove_task(write_task)


def export_aligned_event(event: AlignedEvent, output_path: str | Path) -> Path:
    """Export one aligned event as a wide CSV artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_frame = event.aligned_waveform.copy()
    export_frame["timestamp_utc"] = export_frame["timestamp_utc"].map(format_timestamp_utc)
    _write_dataframe_atomic(path, export_frame)
    return path


def aligned_event_to_long_frame(event: AlignedEvent) -> pd.DataFrame:
    """Convert an aligned event into the stage's wide per-event export shape."""
    long_frame = event.aligned_waveform.copy()
    long_frame["timestamp_utc"] = long_frame["timestamp_utc"].map(format_timestamp_utc)
    long_frame.insert(0, "sample_index", range(len(long_frame)))
    long_frame.insert(0, "event_id", event.event_id)
    return long_frame


def aligned_event_to_sample_rows(event: AlignedEvent) -> list[dict[str, Any]]:
    """Convert an aligned event into canonical long-form sample rows."""
    if event.aligned_waveform.empty:
        return []

    wide = event.aligned_waveform.copy()
    wide["timestamp_utc"] = wide["timestamp_utc"].map(format_timestamp_utc)
    wide.insert(0, "sample_index", range(len(wide)))
    wide.insert(0, "event_id", event.event_id)
    melted = wide.melt(
        id_vars=["event_id", "sample_index", "timestamp_utc"],
        var_name="sensor_name",
        value_name="value",
    )
    melted["set_name"] = event.set_name
    melted["deck"] = event.deck
    melted = melted.dropna(subset=["value"]).reset_index(drop=True)
    return melted[
        ["event_id", "set_name", "deck", "sensor_name", "sample_index", "timestamp_utc", "value"]
    ].to_dict("records")


def load_preprocessing_settings(config_path: Path) -> PreprocessingSettings:
    """Parse the snapped run config into stage settings."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = config.get("data") or {}
    preprocessing = config.get("preprocessing") or {}
    signal_filter = preprocessing.get("signal_filter") or {}
    alignment = preprocessing.get("alignment") or {}
    zeroing = preprocessing.get("zeroing") or {}
    filtering = preprocessing.get("filtering") or {}
    event_grouping = preprocessing.get("event_grouping") or {}
    sensor_overrides = preprocessing.get("sensor_overrides") or {}
    storage = preprocessing.get("storage") or {}
    exports = preprocessing.get("exports") or {}
    aligned_exports = exports.get("aligned_waveforms") or {}

    if "export" in preprocessing:
        raise ValueError(
            "Legacy preprocessing.export is no longer supported. "
            "Use preprocessing.storage and preprocessing.exports.aligned_waveforms instead."
        )

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

    storage_backend = str(storage.get("backend", "sqlite"))
    if storage_backend != "sqlite":
        raise ValueError(
            "Unsupported preprocessing.storage.backend: "
            f"{storage_backend!r}. Only 'sqlite' is supported."
        )

    aligned_export_enabled = bool(aligned_exports.get("enabled", False))
    aligned_export_format = str(aligned_exports.get("format", "csv.gz"))
    if aligned_export_format not in {"csv", "csv.gz"}:
        raise ValueError(
            "Unsupported preprocessing.exports.aligned_waveforms.format: "
            f"{aligned_export_format!r}. Supported formats are ['csv', 'csv.gz']."
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
        storage_backend=storage_backend,
        aligned_export_enabled=aligned_export_enabled,
        aligned_export_format=aligned_export_format,
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

    return pd.DataFrame(rows, columns=SENSOR_QC_REPORT_COLUMNS)


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
        "active_sensors": list(event_row["active_sensors"]),
        "excluded_sensor_count": len(excluded_sensors),
        "excluded_sensors": excluded_sensors,
        "excluded_sensor_reasons": excluded_reasons,
        "reference_sensor": reference_sensor,
        "rows_before_alignment": rows_before_alignment,
        "rows_after_alignment": rows_after_alignment,
        "discarded": discard_reason is not None,
        "discard_reason": discard_reason or "",
        "zeroing_method": zeroing_method,
    }


def _build_event_sensor_rows(
    event_sensor_records: pd.DataFrame,
    *,
    reference_sensor: str,
) -> list[dict[str, Any]]:
    if event_sensor_records.empty:
        return []

    collapsed = collapse_sensor_records(event_sensor_records)
    rows: list[dict[str, Any]] = []
    for _, row in collapsed.iterrows():
        rows.append(
            {
                "event_id": str(row["event_id"]),
                "set_name": str(row["set_name"]),
                "deck": str(row["deck"]),
                "sensor_name": str(row["sensor_name"]),
                "sensor_order": int(row["sensor_order"]),
                "sensor_status": str(row["sensor_status"]),
                "exclusion_reason": str(row["exclusion_reason"]),
                "exclusion_source": str(row["exclusion_source"]),
                "is_reference": int(str(row["sensor_name"]) == reference_sensor),
                "record_uid": _normalize_optional_scalar(row.get("Record_UID")),
                "raw_file": str(row["raw_file"]),
                "start_row_1based": int(row["start_row_1based"]),
                "end_row_1based": int(row["end_row_1based"]),
                "start_time_utc": format_timestamp_utc(row["start_time_utc"]),
                "end_time_utc": format_timestamp_utc(row["end_time_utc"]),
                "duration": _normalize_optional_scalar(row.get("Duration")),
                "temperature": _normalize_optional_scalar(row.get("Temperature")),
                "start_value": _normalize_optional_scalar(row.get("Start_Value")),
                "end_value": _normalize_optional_scalar(row.get("End_Value")),
                "diff_value": _normalize_optional_scalar(row.get("Diff_Value")),
                "min_value": _normalize_optional_scalar(row.get("Min_Value")),
                "max_value": _normalize_optional_scalar(row.get("Max_Value")),
                "mean_value": _normalize_optional_scalar(row.get("Mean_Value")),
                "range_value": _normalize_optional_scalar(row.get("Range")),
            }
        )
    return rows


def _write_set_aligned_partitions(
    aligned_dir: Path,
    partitions: dict[tuple[str, str], list[pd.DataFrame]],
    export_format: str,
) -> list[str]:
    suffix = ".csv.gz" if export_format == "csv.gz" else ".csv"
    written_partitions: list[str] = []
    for (set_name, deck), frames in partitions.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        deck_label = _deck_partition_label(deck)
        output_path = aligned_dir / f"{set_name}__{deck_label}{suffix}"
        _write_dataframe_atomic(output_path, combined)
        written_partitions.append(f"{set_name}__{deck_label}")
    return written_partitions


def _write_summary(
    path: Path,
    *,
    settings: PreprocessingSettings,
    preprocess_db_path: Path,
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
            "stage": "before_alignment",
        },
        "alignment": {
            "method": settings.alignment_method,
            "reference_policy": "first_selected",
            "passes": SYNCHRO_PASSES,
        },
        "event_grouping": {"key_fields": list(settings.event_key_fields)},
        "storage": {
            "backend": settings.storage_backend,
            "path": str(preprocess_db_path),
        },
        "exports": {
            "aligned_waveforms": {
                "enabled": settings.aligned_export_enabled,
                "format": settings.aligned_export_format,
                "path": str(path.parent / "exports" / "aligned"),
            }
        },
        "sensor_exclusions": {
            "configured": [asdict(exclusion) for exclusion in settings.sensor_exclusions],
            "applied_record_counts_by_set": dict(exclusion_counts_by_set),
            "applied_record_counts_by_reason": dict(exclusion_counts_by_reason),
            "applied_sensor_names_by_set": {
                set_name: sorted(sensor_names)
                for set_name, sensor_names in applied_sensor_names_by_set.items()
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_preprocess_progress(run_context: RunContext, progress_payload: dict[str, Any]) -> None:
    write_stage_progress(
        run_context.run_dir,
        "preprocess",
        {
            "current_set": progress_payload["current_set"],
            "completed_sets": list(progress_payload["completed_sets"]),
            "written_partitions": list(progress_payload["written_partitions"]),
        },
    )


def _mark_set_progress_complete(
    progress_payload: dict[str, Any],
    *,
    set_name: str,
    written_partitions: list[str],
) -> None:
    progress_payload["current_set"] = None
    progress_payload["completed_sets"].append(set_name)
    progress_payload["written_partitions"].extend(written_partitions)


def _write_dataframe_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    compression = "gzip" if path.suffix == ".gz" else None
    try:
        frame.to_csv(temp_path, index=False, compression=compression)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _deck_partition_label(deck: str) -> str:
    normalized = str(deck).strip().upper()
    if normalized in {"OLD", "NEW"}:
        return f"{normalized}_DECK"
    return normalized


def _partition_labels_for_partitions(
    partitions: dict[tuple[str, str], list[pd.DataFrame]],
) -> list[str]:
    labels = []
    for (set_name, deck), frames in partitions.items():
        if frames:
            labels.append(f"{set_name}__{_deck_partition_label(deck)}")
    return labels


def _settings_payload(settings: PreprocessingSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["dataset_root"] = str(settings.dataset_root)
    payload["set_names"] = list(settings.set_names)
    payload["sensor_exclusions"] = [asdict(exclusion) for exclusion in settings.sensor_exclusions]
    payload["event_key_fields"] = list(settings.event_key_fields)
    return payload


def _normalize_optional_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return value


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
