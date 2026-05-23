"""
Core event discovery and loading helpers for AQUINAS preprocessing.

This module implements the organizer-faithful preprocessing semantics
derived from ``AQUINAS_Explorer.R`` shared by François-Baptiste
Cartiaux (OSMOS Group) on April 9, 2026.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Iterable, Mapping

import pandas as pd

from aquinas_toolkit.io import AquinasReader


EVENT_REQUIRED_COLUMNS = ("File", "Start_Row", "End_Row", "Start_Time", "End_Time")
EVENT_GROUPING_METHODS = {"exact_window", "shared_start"}


@dataclass(frozen=True)
class LoadedEventGroup:
    """Loaded raw records for one grouped event."""

    event_id: str
    set_name: str
    deck: str
    start_time_utc: pd.Timestamp
    end_time_utc: pd.Timestamp
    sensor_records: pd.DataFrame
    waveforms: dict[str, tuple[pd.Series, pd.DataFrame]]
    zeroing_method: str = "none"


def derive_deck(sensor_name: str) -> str:
    """Return the deck token (``OLD`` or ``NEW``) from a sensor name."""
    return sensor_name.split("_", 1)[0]


def build_event_id(
    set_name: str,
    deck: str,
    start_time_utc: pd.Timestamp,
    end_time_utc: pd.Timestamp,
) -> str:
    """Build a deterministic event identifier."""
    return "__".join(
        [
            set_name,
            deck,
            _format_timestamp_token(start_time_utc),
            _format_timestamp_token(end_time_utc),
        ]
    )


def find_events(
    reader: AquinasReader,
    *,
    timestamp: str | pd.Timestamp | None = None,
    deck: str | None = None,
    sensor_pattern: str | None = None,
    records: pd.DataFrame | None = None,
    grouping_method: str = "exact_window",
) -> pd.DataFrame:
    """
    Return grouped events for one dataset folder.

    ``exact_window`` groups records by exact ``deck + Start_Time + End_Time``.
    ``shared_start`` groups records by ``deck + Start_Time`` and uses the maximum
    grouped ``End_Time`` as the event end. When ``timestamp`` is provided,
    matching uses strict organizer-style containment:
    ``Start_Time < timestamp < End_Time``.
    """
    prepared_records = records if records is not None else prepare_sensor_records(reader)
    filtered_records = filter_sensor_records(
        prepared_records,
        deck=deck,
        sensor_pattern=sensor_pattern,
    )
    filtered_records = assign_event_groups(filtered_records, method=grouping_method)

    grouped = group_sensor_records(filtered_records)
    if timestamp is not None:
        target_timestamp = parse_utc_timestamp(timestamp)
        grouped = grouped.loc[
            (grouped["start_time_utc"] < target_timestamp)
            & (grouped["end_time_utc"] > target_timestamp)
        ].copy()

    return grouped.reset_index(drop=True)


def load_event_group(
    reader: AquinasReader,
    event: str | Mapping[str, Any],
    *,
    sensor_names: Iterable[str] | None = None,
    records: pd.DataFrame | None = None,
) -> LoadedEventGroup:
    """Load widened waveform slices for one grouped event."""
    prepared_records = records if records is not None else prepare_sensor_records(reader)
    matched = _resolve_event_records(prepared_records, event)
    if sensor_names is None and not isinstance(event, str):
        event_sensors = event.get("active_sensors")
        if event_sensors is not None:
            sensor_names = event_sensors
    if sensor_names is not None:
        sensor_names_set = set(sensor_names)
        matched = matched.loc[matched["sensor_name"].isin(sensor_names_set)].copy()

    if matched.empty:
        raise ValueError("No sensor records matched the requested event.")

    collapsed = collapse_sensor_records(matched)
    waveforms: dict[str, tuple[pd.Series, pd.DataFrame]] = {}

    for _, row in collapsed.iterrows():
        sensor_name = str(row["sensor_name"])
        meta = row.copy()
        waveform = _load_waveform_from_record(reader, meta)
        waveforms[sensor_name] = (meta, waveform)

    first_row = collapsed.iloc[0]
    event_start_time = first_row.get("event_start_time_utc", first_row["start_time_utc"])
    event_end_time = first_row.get("event_end_time_utc", first_row["end_time_utc"])
    return LoadedEventGroup(
        event_id=str(first_row["event_id"]),
        set_name=str(first_row["set_name"]),
        deck=str(first_row["deck"]),
        start_time_utc=parse_utc_timestamp(event_start_time),
        end_time_utc=parse_utc_timestamp(event_end_time),
        sensor_records=collapsed,
        waveforms=waveforms,
        zeroing_method="none",
    )


def load_timestamp_query_frames(
    reader: AquinasReader,
    *,
    timestamp: str | pd.Timestamp,
    deck: str | None = None,
    sensor_pattern: str | None = None,
    records: pd.DataFrame | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Load organizer-style sensor slices for one timestamp query.

    The returned list preserves organizer sensor order and includes empty
    frames for selected sensors that do not contain the query timestamp.
    """
    prepared_records = records if records is not None else prepare_sensor_records(reader)
    filtered_records = filter_sensor_records(
        prepared_records,
        deck=deck,
        sensor_pattern=sensor_pattern,
    )
    selected_sensors = ordered_sensor_names(
        reader,
        deck=deck,
        sensor_pattern=sensor_pattern,
        records=filtered_records,
    )
    target_timestamp = parse_utc_timestamp(timestamp)

    frames: list[tuple[str, pd.DataFrame]] = []
    for sensor_name in selected_sensors:
        sensor_rows = filtered_records.loc[filtered_records["sensor_name"] == sensor_name].copy()
        matched = sensor_rows.loc[
            (sensor_rows["start_time_utc"] < target_timestamp)
            & (sensor_rows["end_time_utc"] > target_timestamp)
        ].copy()
        if matched.empty:
            frames.append((sensor_name, _empty_waveform_frame(sensor_name)))
            continue

        collapsed = collapse_sensor_records(matched)
        meta = collapsed.iloc[0].copy()
        frames.append((sensor_name, _load_waveform_from_record(reader, meta)))

    return frames


def prepare_sensor_records(reader: AquinasReader) -> pd.DataFrame:
    """Return all sensor records with organizer-order metadata."""
    records = reader.load_all_index_tables().copy()
    if records.empty:
        return records

    for required in EVENT_REQUIRED_COLUMNS:
        matched_column = reader.match_column(records, [required])
        if matched_column is None:
            raise KeyError(f"Index tables must contain '{required}' to preprocess events.")

    file_col = reader.match_column(records, ["File"])
    start_row_col = reader.match_column(records, ["Start_Row"])
    end_row_col = reader.match_column(records, ["End_Row"])
    start_time_col = reader.match_column(records, ["Start_Time"])
    end_time_col = reader.match_column(records, ["End_Time"])
    sensor_order_map = {
        sensor_name: index for index, sensor_name in enumerate(reader.list_sensor_names(), start=1)
    }

    records["set_name"] = records["dataset"]
    records["deck"] = records["sensor_name"].map(derive_deck)
    records["sensor_order"] = records["sensor_name"].map(sensor_order_map)
    records["table_row_index"] = range(len(records))
    records["start_time_utc"] = _parse_timestamps_fast(records[start_time_col])
    records["end_time_utc"] = _parse_timestamps_fast(records[end_time_col])
    records["raw_file"] = records[file_col]
    records["start_row_1based"] = records[start_row_col].map(
        lambda value: reader.to_int(value, "Start_Row")
    )
    records["end_row_1based"] = records[end_row_col].map(
        lambda value: reader.to_int(value, "End_Row")
    )
    return assign_event_groups(records, method="exact_window")


def assign_event_groups(records: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Assign deterministic grouped event IDs without changing raw record timing."""
    _validate_event_grouping_method(method)
    grouped_records = records.copy()
    if grouped_records.empty:
        for column in ("event_start_time_utc", "event_end_time_utc", "event_id"):
            if column not in grouped_records.columns:
                grouped_records[column] = pd.Series(dtype=object)
        return grouped_records

    grouped_records["event_start_time_utc"] = grouped_records["start_time_utc"]
    if method == "exact_window":
        grouped_records["event_end_time_utc"] = grouped_records["end_time_utc"]
    else:
        grouped_records["event_end_time_utc"] = grouped_records.groupby(
            ["set_name", "deck", "start_time_utc"], sort=False
        )["end_time_utc"].transform("max")

    grouped_records["event_id"] = [
        build_event_id(set_name, deck, start_time, end_time)
        for set_name, deck, start_time, end_time in zip(
            grouped_records["set_name"],
            grouped_records["deck"],
            grouped_records["event_start_time_utc"],
            grouped_records["event_end_time_utc"],
            strict=True,
        )
    ]
    return grouped_records


def filter_sensor_records(
    records: pd.DataFrame,
    *,
    deck: str | None = None,
    sensor_pattern: str | None = None,
) -> pd.DataFrame:
    """Apply deck and sensor filters while preserving organizer order."""
    filtered = records
    if deck is not None:
        filtered = filtered.loc[filtered["deck"] == deck].copy()
    if sensor_pattern is not None:
        mask = filtered["sensor_name"].map(
            lambda value: _sensor_matches_pattern(value, sensor_pattern)
        )
        filtered = filtered.loc[mask].copy()
    return filtered.reset_index(drop=True)


def ordered_sensor_names(
    reader: AquinasReader,
    *,
    deck: str | None = None,
    sensor_pattern: str | None = None,
    records: pd.DataFrame | None = None,
) -> list[str]:
    """Return organizer-order sensor names for the requested selection."""
    names = reader.list_sensor_names()
    if records is not None:
        available = set(records["sensor_name"].astype(str))
        names = [name for name in names if name in available]
    if deck is not None:
        names = [name for name in names if derive_deck(name) == deck]
    if sensor_pattern is not None:
        names = [name for name in names if _sensor_matches_pattern(name, sensor_pattern)]
    return names


def collapse_sensor_records(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple matched rows per sensor using organizer widening semantics."""
    if records.empty:
        return records.copy()

    ordered = records.sort_values(
        ["sensor_order", "start_time_utc", "start_row_1based"],
        kind="mergesort",
    )
    collapsed_rows: list[pd.Series] = []
    for _, sensor_rows in ordered.groupby("sensor_name", sort=False):
        raw_files = _ordered_unique(sensor_rows["raw_file"].astype(str).tolist())
        if len(raw_files) != 1:
            sensor_name = str(sensor_rows["sensor_name"].iloc[0])
            raise ValueError(
                f"Organizer-style loading requires exactly one raw file per sensor match, got {raw_files!r} "
                f"for sensor '{sensor_name}'."
            )

        first_row = sensor_rows.iloc[0]
        if len(sensor_rows) > 1:
            first_row = first_row.copy()
            first_row["start_row_1based"] = int(sensor_rows["start_row_1based"].min())
            first_row["end_row_1based"] = int(sensor_rows["end_row_1based"].max())
            first_row["start_time_utc"] = sensor_rows["start_time_utc"].min()
            first_row["end_time_utc"] = sensor_rows["end_time_utc"].max()
            first_row["raw_file"] = raw_files[0]
        collapsed_rows.append(first_row)

    return pd.DataFrame(collapsed_rows).reset_index(drop=True)


def group_sensor_records(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per grouped event with organizer-order coverage."""
    if records.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "set_name",
                "deck",
                "start_time_utc",
                "end_time_utc",
                "active_sensor_count",
                "active_sensors",
            ]
        )
    if "event_start_time_utc" not in records or "event_end_time_utc" not in records:
        records = assign_event_groups(records, method="exact_window")

    ordered = records.sort_values(
        ["set_name", "deck", "event_start_time_utc", "sensor_order", "start_row_1based"],
        kind="mergesort",
    )
    grouped = (
        ordered.groupby(
            ["event_id", "set_name", "deck", "event_start_time_utc", "event_end_time_utc"],
            as_index=False,
        )
        .agg(
            active_sensor_count=("sensor_name", "nunique"),
            active_sensors=("sensor_name", lambda values: _ordered_unique(values.tolist())),
        )
        .rename(
            columns={
                "event_start_time_utc": "start_time_utc",
                "event_end_time_utc": "end_time_utc",
            }
        )
    )
    return grouped.sort_values(["set_name", "deck", "start_time_utc"], kind="mergesort").reset_index(
        drop=True
    )


def parse_utc_timestamp(value: Any) -> pd.Timestamp:
    """Parse a scalar timestamp into a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def format_timestamp_utc(value: Any) -> str:
    """Return an ISO-8601 UTC string with millisecond precision."""
    timestamp = parse_utc_timestamp(value)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _format_timestamp_token(value: Any) -> str:
    timestamp = parse_utc_timestamp(value)
    if timestamp.microsecond:
        return timestamp.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"
    return timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_timestamps_fast(
    series: pd.Series, *, errors: str = "raise"
) -> pd.Series:
    """Parse timestamps using an explicit format with fallback to mixed.

    Always tries the explicit format first with ``errors='raise'`` so that
    a mismatch triggers the fallback rather than silently coercing values to
    NaT.  The caller-specified *errors* mode is only applied in the final
    parse so that ``errors='coerce'`` works correctly when individual values
    are truly unparseable (as opposed to using a different but valid format).
    """
    try:
        return pd.to_datetime(
            series, utc=True, format="%Y-%m-%d %H:%M:%S.%f", errors="raise"
        )
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, format="mixed", errors=errors)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _sensor_matches_pattern(sensor_name: str, sensor_pattern: str) -> bool:
    normalized_name = sensor_name.upper()
    normalized_pattern = sensor_pattern.upper()
    if any(token in normalized_pattern for token in "*?[]"):
        return fnmatch(normalized_name, normalized_pattern)
    return normalized_pattern in normalized_name


def _validate_event_grouping_method(method: str) -> None:
    if method not in EVENT_GROUPING_METHODS:
        supported = ", ".join(sorted(EVENT_GROUPING_METHODS))
        raise ValueError(f"Unsupported event grouping method {method!r}. Supported methods: {supported}.")


def _resolve_event_records(records: pd.DataFrame, event: str | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(event, str):
        matched = records.loc[records["event_id"] == event].copy()
    else:
        event_id = event.get("event_id")
        if event_id is not None:
            matched = records.loc[records["event_id"] == event_id].copy()
            expected_sensor_count = int(event.get("active_sensor_count", 0) or 0)
            matched_sensor_count = int(matched["sensor_name"].nunique()) if not matched.empty else 0
            if (
                _event_has_keys(event, ("deck", "start_time_utc", "end_time_utc"))
                and matched_sensor_count < expected_sensor_count
            ):
                matched = _resolve_event_records_by_group_window(records, event)
        else:
            matched = _resolve_event_records_by_group_window(records, event)
    return matched.reset_index(drop=True)


def _resolve_event_records_by_group_window(
    records: pd.DataFrame,
    event: Mapping[str, Any],
) -> pd.DataFrame:
    deck = event.get("deck")
    start_time = parse_utc_timestamp(event["start_time_utc"])
    end_time = parse_utc_timestamp(event["end_time_utc"])
    event_start_col = "event_start_time_utc" if "event_start_time_utc" in records else "start_time_utc"
    event_end_col = "event_end_time_utc" if "event_end_time_utc" in records else "end_time_utc"
    expected_sensor_count = int(event.get("active_sensor_count", 0) or 0)
    matched = records.loc[
        (records["deck"] == deck)
        & (records[event_start_col] == start_time)
        & (records[event_end_col] == end_time)
    ].copy()
    if not matched.empty and (
        expected_sensor_count == 0 or int(matched["sensor_name"].nunique()) >= expected_sensor_count
    ):
        return matched

    fallback = records.loc[
        (records["deck"] == deck)
        & (records["start_time_utc"] == start_time)
        & (records["end_time_utc"] <= end_time)
    ].copy()
    if not fallback.empty:
        fallback["event_start_time_utc"] = start_time
        fallback["event_end_time_utc"] = end_time
        fallback["event_id"] = str(
            event.get("event_id") or build_event_id(str(fallback["set_name"].iloc[0]), str(deck), start_time, end_time)
        )
    return fallback


def _event_has_keys(event: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in event for key in keys)


def _load_waveform_from_record(reader: AquinasReader, meta: pd.Series) -> pd.DataFrame:
    sensor_name = str(meta["sensor_name"])
    raw_df = reader.load_raw_file_prepped(sensor_name, str(meta["raw_file"]))
    start = int(meta["start_row_1based"]) - 1
    end = int(meta["end_row_1based"])

    timestamp_col = reader.match_column(raw_df, ["timestamp", "Timestamp"])
    if timestamp_col is None:
        raise KeyError(f"Raw waveform for sensor '{sensor_name}' is missing a timestamp column.")

    measure_columns = [column for column in raw_df.columns if column != timestamp_col]
    if not measure_columns:
        raise KeyError(f"Raw waveform for sensor '{sensor_name}' does not contain sensor values.")

    value_col = sensor_name if sensor_name in measure_columns else measure_columns[0]

    # Slice THEN convert — avoids copying the entire 100K+ row column
    ts_arr = raw_df[timestamp_col].iloc[start:end].to_numpy()
    val_arr = raw_df[value_col].iloc[start:end].to_numpy(dtype=float)

    # Drop rows where timestamp parsing failed (NaT)
    valid = ~pd.isna(ts_arr)
    n_dropped = int((~valid).sum())
    if n_dropped:
        warnings.warn(
            f"Sensor '{sensor_name}': dropped {n_dropped} row(s) with unparseable timestamps.",
            stacklevel=2,
        )
        ts_arr = ts_arr[valid]
        val_arr = val_arr[valid]

    return pd.DataFrame({"timestamp": ts_arr, sensor_name: val_arr})


def _empty_waveform_frame(sensor_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", sensor_name])
