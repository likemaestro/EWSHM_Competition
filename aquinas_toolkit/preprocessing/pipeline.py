"""Preprocessing stage orchestration and artifact writing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.progress import Progress

from aquinas_toolkit.cli.terminal import progress_context
from aquinas_toolkit.io import AquinasReader, parse_sensor_name
from aquinas_toolkit.preprocessing.alignment import AlignedEvent, SYNCHRO_PASSES, align_event_group
from aquinas_toolkit.preprocessing.core import (
    LoadedEventGroup,
    collapse_sensor_records,
    find_events,
    format_timestamp_utc,
    load_event_group,
    prepare_sensor_records,
)
from aquinas_toolkit.preprocessing.signals import SIGNAL_FILTER_METHODS
from scipy.signal import butter
from aquinas_toolkit.preprocessing.store import (
    EVENT_SENSOR_COLUMNS,
    SENSOR_RECORD_COLUMNS,
    PreprocessStoreWriter,
    preprocess_store_path,
)
from aquinas_toolkit.preprocessing.zeroing import ZEROING_METHODS
from aquinas_toolkit.preprocessing.neural_inputs import (
    AccInputSettings,
    NeuralInputSettings,
    StrainInputSettings,
    build_neural_inputs,
)
from aquinas_toolkit.utils.dataset_paths import find_workspace_root
from aquinas_toolkit.utils.run_management import RunContext, stage_output_dir, write_stage_progress


# Flush aligned sample frames to SQLite every N retained events to keep
# per-SET peak memory bounded (~200 events × ~23K rows ≈ 4-5M rows max in RAM
# instead of ~250M rows for a full SET).
_ALIGNED_SAMPLE_FLUSH_EVENTS = 200

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

    Pipeline order: signal-specific filtering -> signal-specific zeroing -> alignment.
    """

    dataset_root: Path
    set_names: tuple[str, ...]
    sampling_rate_hz: float = 100.0
    sensor_exclusions: tuple[SensorExclusion, ...] = ()
    selected_decks: tuple[str, ...] = ()
    event_key_fields: tuple[str, ...] = ("deck", "Start_Time", "End_Time")
    strain_filter_method: str = "none"
    strain_zeroing_method: str = "linear_endpoints"
    acc_filter_method: str = "butterworth_bandpass"
    acc_filter_low_hz: float = 0.5
    acc_filter_high_hz: float = 20.0
    acc_filter_order: int = 4
    acc_zeroing_method: str = "linear_endpoints"
    alignment_method: str = "r_synchro"
    min_active_sensors_per_event: int = 1
    storage_backend: str = "sqlite"
    aligned_export_enabled: bool = False
    aligned_export_format: str = "csv.gz"
    neural_inputs: NeuralInputSettings = NeuralInputSettings()


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

    set_names = list(settings.set_names)
    store_writer = PreprocessStoreWriter(
        preprocess_db,
        run_id=run_context.run_id,
        settings_payload=_settings_payload(settings),
        set_names=set_names,
    )

    try:
        with progress_context(transient=False) as progress:
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
    build_neural_inputs(preprocess_dir, settings=settings.neural_inputs)


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
        progress.console.print(f"\n[stage_set]SET {index}/{len(set_names)}[/]  [key]{set_name}[/]")

        load_task = progress.add_task("  Reading sensor records...", total=None)
        reader = AquinasReader(settings.dataset_root / set_name)
        sensor_records = prepare_sensor_records(reader)
        if sensor_records.empty:
            progress.remove_task(load_task)
            progress.console.print("  [warning]No records found, skipping.[/]")
            store_writer.write_set(
                sensor_records=pd.DataFrame(columns=SENSOR_RECORD_COLUMNS),
                events=pd.DataFrame(columns=EVENT_MANIFEST_COLUMNS),
                event_sensors=pd.DataFrame(columns=EVENT_SENSOR_COLUMNS),
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
        progress.remove_task(load_task)

        # Write sensor metadata immediately so FK constraints on event_sensors
        # (sensor_name -> sensors) are satisfied during incremental flushes.
        store_writer.write_sensors_for_set(sensor_records)

        for entry in exclusion_log:
            resolved_set_name = str(entry["set_name"])
            exclusion_counts_by_set[resolved_set_name] += int(entry["record_count"])
            exclusion_counts_by_reason[str(entry["exclusion_reason"])] += int(entry["record_count"])
            applied_sensor_names_by_set[resolved_set_name].add(str(entry["sensor_name"]))

        set_manifest_rows: list[dict[str, Any]] = []          # ALL events for final summary
        set_event_sensor_rows: list[dict[str, Any]] = []       # ALL event-sensors (discarded go to write_set)
        set_aligned_sample_frames: list[pd.DataFrame] = []
        # Retained events and their event_sensor rows are flushed incrementally
        # alongside aligned_samples to satisfy the FK constraint (aligned_samples
        # references events).  Discarded events accumulate in set_manifest_rows /
        # set_event_sensor_rows and go into write_set at the end.
        flush_event_rows: list[dict[str, Any]] = []
        flush_event_sensor_rows: list[dict[str, Any]] = []
        set_aligned_partitions: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)

        events = find_events(reader, records=included_sensor_records)
        event_task = progress.add_task("  Processing events", total=len(events))
        set_retained = 0

        # Pre-group sensor records by event_id for O(1) lookup instead of
        # repeated .loc[] scans over the full DataFrame.
        sensor_records_by_event = dict(tuple(sensor_records.groupby("event_id")))
        included_records_by_event = dict(tuple(included_sensor_records.groupby("event_id")))

        # Separate fast-discard events from events that need heavy processing
        heavy_events = []
        for _, event_row in events.iterrows():
            per_deck_total[str(event_row["deck"])] += 1
            event_id = event_row["event_id"]
            event_sensor_records = sensor_records_by_event.get(event_id, pd.DataFrame())
            excluded_event_rows = (
                event_sensor_records.loc[
                    event_sensor_records["sensor_status"] == "excluded"
                ]
                if not event_sensor_records.empty
                else event_sensor_records
            )

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
                    zeroing_method=_zeroing_summary(settings),
                )
                manifest_rows.append(manifest_row)
                set_manifest_rows.append(manifest_row)
                set_event_sensor_rows.extend(
                    _build_event_sensor_rows(event_sensor_records, reference_sensor="")
                )
                progress.advance(event_task)
                continue

            heavy_events.append((
                event_row,
                event_sensor_records,
                excluded_event_rows,
                included_records_by_event.get(event_row["event_id"], pd.DataFrame()),
            ))

        # Pre-compute filter coefficients once (shared across all events)
        precomputed_sos = None
        if settings.acc_filter_method == "butterworth_bandpass":
            precomputed_sos = butter(
                settings.acc_filter_order,
                [settings.acc_filter_low_hz, settings.acc_filter_high_hz],
                btype="bandpass",
                fs=settings.sampling_rate_hz,
                output="sos",
            )

        # Process heavy events serially (GIL prevents thread parallelism
        # for the numpy/scipy-dominated inner loop).
        for event_row, event_sensor_records, excluded_event_rows, event_included_records in heavy_events:
            aligned_event, fused_event = _process_heavy_event(
                reader,
                event_row,
                event_included_records,
                settings,
                precomputed_sos,
            )

            rows_before_alignment = 0
            if fused_event.waveforms:
                rows_before_alignment = int(
                    len(fused_event.waveforms[aligned_event.reference_sensor][1])
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
                    zeroing_method=_zeroing_summary(settings),
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
                zeroing_method=_zeroing_summary(settings),
            )
            manifest_rows.append(manifest_row)
            set_manifest_rows.append(manifest_row)
            this_event_sensor_rows = _build_event_sensor_rows(
                event_sensor_records,
                reference_sensor=aligned_event.reference_sensor,
            )
            set_event_sensor_rows.extend(this_event_sensor_rows)
            # Track retained event rows for the incremental flush; they must be
            # written to the events table before waveform files (for metadata consistency).
            flush_event_rows.append(manifest_row)
            flush_event_sensor_rows.extend(this_event_sensor_rows)
            long_frame = aligned_event_to_long_frame(aligned_event)
            set_aligned_sample_frames.append(long_frame)
            set_aligned_partitions[(aligned_event.set_name, aligned_event.deck)].append(long_frame)
            progress.advance(event_task)

            # Flush to SQLite periodically — avoids accumulating the entire SET
            # (~250M rows) in memory before writing starts.
            if len(set_aligned_sample_frames) >= _ALIGNED_SAMPLE_FLUSH_EVENTS:
                store_writer.write_aligned_samples(
                    set_aligned_sample_frames,
                    events=pd.DataFrame(flush_event_rows, columns=EVENT_MANIFEST_COLUMNS),
                    event_sensors=pd.DataFrame(flush_event_sensor_rows, columns=EVENT_SENSOR_COLUMNS),
                )
                set_aligned_sample_frames.clear()
                flush_event_rows.clear()
                flush_event_sensor_rows.clear()

        progress.remove_task(event_task)
        # Flush any remaining retained events that didn't reach the threshold.
        if set_aligned_sample_frames:
            store_writer.write_aligned_samples(
                set_aligned_sample_frames,
                events=pd.DataFrame(flush_event_rows, columns=EVENT_MANIFEST_COLUMNS),
                event_sensors=pd.DataFrame(flush_event_sensor_rows, columns=EVENT_SENSOR_COLUMNS),
            )
            set_aligned_sample_frames.clear()
            flush_event_rows.clear()
            flush_event_sensor_rows.clear()
        set_discarded = len(events) - set_retained
        progress.console.print(
            f"  [success]done[/]  [key]{set_retained:,}[/] retained  "
            f"[muted]{set_discarded:,} discarded[/]"
        )
        progress.console.print("  [accent]Writing preprocess store...[/]")
        # write_set now only handles metadata and DISCARDED event rows (retained
        # events were already written incrementally with their aligned_samples).
        discarded_manifest_rows = [r for r in set_manifest_rows if r.get("discarded")]
        discarded_event_ids = {r["event_id"] for r in discarded_manifest_rows}
        discarded_event_sensor_rows = [
            r for r in set_event_sensor_rows if r["event_id"] in discarded_event_ids
        ]
        metadata_rows = (
            len(sensor_records) + len(discarded_manifest_rows)
            + len(discarded_event_sensor_rows)
        )
        write_task = progress.add_task(
            "  Committing metadata rows...",
            total=metadata_rows,
        )
        # TODO: overlap this per-set canonical DB commit and optional aligned export
        # with next-set preprocessing via a bounded producer-consumer writer once
        # failure handling and metadata semantics stay deterministic.
        store_writer.write_set(
            sensor_records=sensor_records,
            events=pd.DataFrame(discarded_manifest_rows, columns=EVENT_MANIFEST_COLUMNS),
            event_sensors=pd.DataFrame(discarded_event_sensor_rows, columns=EVENT_SENSOR_COLUMNS),
            on_progress=lambda n, _t=write_task: progress.advance(_t, n),
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
    ts = export_frame["timestamp_utc"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        export_frame["timestamp_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    else:
        export_frame["timestamp_utc"] = ts.map(format_timestamp_utc)
    _write_dataframe_atomic(path, export_frame)
    return path


def aligned_event_to_long_frame(event: AlignedEvent) -> pd.DataFrame:
    """Convert an aligned event into the stage's wide per-event export shape."""
    long_frame = event.aligned_waveform.copy()
    # Vectorized timestamp formatting instead of per-row .map(format_timestamp_utc)
    ts = long_frame["timestamp_utc"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        long_frame["timestamp_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    else:
        long_frame["timestamp_utc"] = long_frame["timestamp_utc"].map(format_timestamp_utc)
    long_frame.insert(0, "sample_index", range(len(long_frame)))
    long_frame.insert(0, "event_id", event.event_id)
    return long_frame


def _process_heavy_event(
    reader: AquinasReader,
    event_row: pd.Series,
    included_sensor_records: pd.DataFrame,
    settings: PreprocessingSettings,
    precomputed_sos: "np.ndarray | None" = None,
) -> tuple[AlignedEvent, LoadedEventGroup]:
    """Run load/filter/zero/align for one event.

    Fuses filter + zero into a single numpy pass per waveform to avoid
    redundant DataFrame copies and pd.Series ↔ numpy conversions.
    """
    from scipy.signal import sosfiltfilt

    from aquinas_toolkit.preprocessing.alignment import align_event_group
    from aquinas_toolkit.preprocessing.signals import _min_samples_for_sosfiltfilt

    loaded_event = load_event_group(reader, event_row, records=included_sensor_records)

    # Fused signal-specific filter + zero in pure numpy: one copy per waveform total.
    min_samples = _min_samples_for_sosfiltfilt(precomputed_sos) if precomputed_sos is not None else 0
    fused_waveforms: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for sensor_name, (meta, waveform) in loaded_event.waveforms.items():
        values = waveform[sensor_name].to_numpy(dtype=float, copy=True)
        np.nan_to_num(values, copy=False)
        parsed = parse_sensor_name(sensor_name)
        is_acc_z = parsed["quantity"] == "ACC" and parsed["axis"] == "Z"
        is_strain = parsed["quantity"] == "STR"

        if (
            is_acc_z
            and settings.acc_filter_method == "butterworth_bandpass"
            and precomputed_sos is not None
            and len(values) > min_samples
        ):
            values = sosfiltfilt(precomputed_sos, values)

        zeroing_method = "none"
        if is_strain:
            zeroing_method = settings.strain_zeroing_method
        elif is_acc_z:
            zeroing_method = settings.acc_zeroing_method
        if zeroing_method == "linear_endpoints" and len(values) > 1:
            baseline = np.linspace(values[0], values[-1], len(values))
            values = values - baseline
        elif zeroing_method == "linear_endpoints" and len(values) == 1:
            values = values - values[0]

        w = waveform.copy()
        w[sensor_name] = values
        fused_waveforms[sensor_name] = (meta, w)

    from dataclasses import replace as dc_replace

    fused_event = dc_replace(
        loaded_event,
        waveforms=fused_waveforms,
        zeroing_method=_zeroing_summary(settings),
    )

    aligned_event = align_event_group(fused_event, method=settings.alignment_method)
    return aligned_event, fused_event


def _zeroing_summary(settings: PreprocessingSettings) -> str:
    return f"strain:{settings.strain_zeroing_method};acc_z:{settings.acc_zeroing_method}"


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
    sensor_selection = preprocessing.get("sensor_selection") or {}
    strain_config = preprocessing.get("strain") or {}
    acc_config = preprocessing.get("acc") or {}
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

    sampling_rate_hz = float(preprocessing.get("sampling_rate_hz", 100.0))
    selected_decks = tuple(str(value).upper() for value in sensor_selection.get("decks", ()) if str(value).strip())
    strain_locations = tuple(
        str(value).upper()
        for value in strain_config.get("locations", ("INF", "SHE", "SUP"))
        if str(value).strip()
    )
    peak_window_half_samples = int(strain_config.get("peak_window_half_samples", 100))
    if peak_window_half_samples <= 0:
        raise ValueError("preprocessing.strain.peak_window_half_samples must be > 0.")
    acc_min_aligned_samples = int(acc_config.get("min_aligned_samples", 500))
    if acc_min_aligned_samples <= 0:
        raise ValueError("preprocessing.acc.min_aligned_samples must be > 0.")

    legacy_signal_filter_method = str(signal_filter.get("method", "butterworth_bandpass"))
    _validate_filter_method(legacy_signal_filter_method, "preprocessing.signal_filter.method")
    legacy_zeroing_method = str(zeroing.get("method", "linear_endpoints"))
    _validate_zeroing_method(legacy_zeroing_method, "preprocessing.zeroing.method")

    strain_filter = strain_config.get("filter") or {}
    strain_zeroing = strain_config.get("zeroing") or {}
    acc_filter = acc_config.get("filter") or {}
    acc_zeroing = acc_config.get("zeroing") or {}
    acc_frequency_transform = acc_config.get("frequency_transform") or {}

    strain_filter_method = str(strain_filter.get("method", "none"))
    _validate_filter_method(strain_filter_method, "preprocessing.strain.filter.method")
    if strain_filter_method != "none":
        raise ValueError("preprocessing.strain.filter.method must be 'none'.")
    strain_zeroing_method = str(strain_zeroing.get("method", legacy_zeroing_method))
    _validate_zeroing_method(strain_zeroing_method, "preprocessing.strain.zeroing.method")

    acc_filter_method = str(acc_filter.get("method", legacy_signal_filter_method))
    _validate_filter_method(acc_filter_method, "preprocessing.acc.filter.method")
    acc_filter_low_hz = float(acc_filter.get("low_hz", signal_filter.get("low_hz", 0.5)))
    acc_filter_high_hz = float(acc_filter.get("high_hz", signal_filter.get("high_hz", 20.0)))
    acc_filter_order = int(acc_filter.get("order", signal_filter.get("order", 4)))
    acc_zeroing_method = str(acc_zeroing.get("method", legacy_zeroing_method))
    _validate_zeroing_method(acc_zeroing_method, "preprocessing.acc.zeroing.method")

    _validate_alignment_config(alignment)

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
        sampling_rate_hz=sampling_rate_hz,
        sensor_exclusions=sensor_exclusions,
        selected_decks=selected_decks,
        event_key_fields=tuple(event_grouping.get("key_fields", ("deck", "Start_Time", "End_Time"))),
        strain_filter_method=strain_filter_method,
        strain_zeroing_method=strain_zeroing_method,
        acc_filter_method=acc_filter_method,
        acc_filter_low_hz=acc_filter_low_hz,
        acc_filter_high_hz=acc_filter_high_hz,
        acc_filter_order=acc_filter_order,
        acc_zeroing_method=acc_zeroing_method,
        alignment_method=str(alignment.get("method", "r_synchro")),
        min_active_sensors_per_event=int(filtering.get("min_active_sensors_per_event", 1)),
        storage_backend=storage_backend,
        aligned_export_enabled=aligned_export_enabled,
        aligned_export_format=aligned_export_format,
        neural_inputs=NeuralInputSettings(
            decks=selected_decks,
            sampling_rate_hz=sampling_rate_hz,
            strain=StrainInputSettings(
                peak_window_half_samples=peak_window_half_samples,
                locations=strain_locations,
            ),
            acc=AccInputSettings(
                min_aligned_samples=acc_min_aligned_samples,
                low_hz=float(acc_frequency_transform.get("low_hz", acc_filter_low_hz)),
                high_hz=float(acc_frequency_transform.get("high_hz", acc_filter_high_hz)),
            ),
        ),
    )


def _validate_filter_method(method: str, config_key: str) -> None:
    if method not in SIGNAL_FILTER_METHODS:
        raise ValueError(
            f"Unsupported {config_key}: {method!r}. Supported methods are {sorted(SIGNAL_FILTER_METHODS)}."
        )


def _validate_zeroing_method(method: str, config_key: str) -> None:
    if method not in ZEROING_METHODS:
        raise ValueError(
            f"Unsupported {config_key}: {method}. Supported methods are {sorted(ZEROING_METHODS)}."
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
    return find_workspace_root() / path


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

    if settings.selected_decks:
        deck_mask = annotated["deck"].astype(str).str.upper().isin(settings.selected_decks)
        annotated.loc[~deck_mask, "sensor_status"] = "excluded"
        annotated.loc[~deck_mask, "exclusion_reason"] = "deck not selected for preprocessing"
        annotated.loc[~deck_mask, "exclusion_source"] = "preprocessing.sensor_selection.decks"

    for index, row in annotated.iterrows():
        if annotated.at[index, "sensor_status"] == "excluded":
            continue
        parsed = parse_sensor_name(str(row["sensor_name"]))
        quantity = parsed["quantity"]
        axis = parsed["axis"]
        location = parsed["location"]
        is_strain = quantity == "STR" and str(location).upper() in settings.neural_inputs.strain.locations
        is_acc_z = quantity == "ACC" and axis == "Z"
        if not (is_strain or is_acc_z):
            annotated.at[index, "sensor_status"] = "excluded"
            annotated.at[index, "exclusion_reason"] = "channel not selected for neural preprocessing"
            annotated.at[index, "exclusion_source"] = "preprocessing sensor selection"

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
            "strain": {"method": settings.strain_filter_method},
            "acc_z": {
                "method": settings.acc_filter_method,
                "low_hz": settings.acc_filter_low_hz,
                "high_hz": settings.acc_filter_high_hz,
                "order": settings.acc_filter_order,
                "stage": "before_acc_zeroing",
            },
        },
        "zeroing": {
            "strain_method": settings.strain_zeroing_method,
            "acc_z_method": settings.acc_zeroing_method,
            "stage": "before_alignment",
        },
        "alignment": {
            "method": settings.alignment_method,
            "reference_policy": "first_selected",
            "passes": SYNCHRO_PASSES,
        },
        "event_grouping": {"key_fields": list(settings.event_key_fields)},
        "sampling_rate_hz": settings.sampling_rate_hz,
        "sensor_selection": {
            "decks": list(settings.selected_decks),
            "included_modalities": ["INF_STR", "SHE_STR", "SUP_STR", "ACC_Z"],
            "excluded_modalities": ["ACC_Y"],
        },
        "neural_inputs": {
            "path": str(path.parent / "neural_inputs.npy"),
            "report_dir": str(path.parent / "report"),
            "settings": asdict(settings.neural_inputs),
        },
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
