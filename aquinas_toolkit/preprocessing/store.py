"""SQLite-backed preprocess stage storage and read APIs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from aquinas_toolkit.io import parse_sensor_name
from aquinas_toolkit.preprocessing.core import collapse_sensor_records


PREPROCESS_DB_NAME = "preprocess.sqlite"
PREPROCESS_SCHEMA_VERSION = 1

EVENT_COLUMNS = [
    "event_id",
    "set_name",
    "deck",
    "start_time_utc",
    "end_time_utc",
    "active_sensor_count",
    "active_sensors_json",
    "excluded_sensor_count",
    "excluded_sensors_json",
    "excluded_sensor_reasons_json",
    "reference_sensor",
    "rows_before_alignment",
    "rows_after_alignment",
    "discarded",
    "discard_reason",
    "zeroing_method",
]

EVENT_SENSOR_COLUMNS = [
    "event_id",
    "set_name",
    "deck",
    "sensor_name",
    "sensor_order",
    "sensor_status",
    "exclusion_reason",
    "exclusion_source",
    "is_reference",
    "record_uid",
    "raw_file",
    "start_row_1based",
    "end_row_1based",
    "start_time_utc",
    "end_time_utc",
    "duration",
    "temperature",
    "start_value",
    "end_value",
    "diff_value",
    "min_value",
    "max_value",
    "mean_value",
    "range_value",
]

ALIGNED_SAMPLE_COLUMNS = [
    "event_id",
    "set_name",
    "deck",
    "sensor_name",
    "sample_index",
    "timestamp_utc",
    "value",
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

SENSOR_QC_COLUMNS = [
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


def preprocess_store_path(path_or_stage_dir: str | Path) -> Path:
    """Resolve a preprocess store path from a stage directory or a DB path."""
    path = Path(path_or_stage_dir)
    if path.suffix == ".sqlite":
        return path
    return path / PREPROCESS_DB_NAME


class PreprocessStoreWriter:
    """Writer for the canonical preprocess SQLite artifact."""

    def __init__(
        self,
        path_or_stage_dir: str | Path,
        *,
        run_id: str,
        settings_payload: dict[str, Any],
        set_names: Sequence[str],
    ) -> None:
        self.path = preprocess_store_path(path_or_stage_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        _configure_connection(self.conn)
        _create_schema(self.conn)
        _write_stage_info(
            self.conn,
            run_id=run_id,
            settings_payload=settings_payload,
        )
        _write_sets(self.conn, set_names)

    def close(self) -> None:
        """Close the SQLite connection."""
        _close_sqlite_connection(self.conn)

    def write_set(
        self,
        *,
        sensor_records: pd.DataFrame,
        qc_report: pd.DataFrame,
        events: pd.DataFrame,
        event_sensors: pd.DataFrame,
        aligned_samples: pd.DataFrame,
    ) -> None:
        """Commit one set worth of preprocess outputs atomically."""
        with self.conn:
            _upsert_sensors(self.conn, sensor_records, event_sensors)
            _insert_dataframe(
                self.conn,
                "sensor_records",
                _prepare_sensor_records_frame(sensor_records),
                SENSOR_RECORD_COLUMNS,
            )
            _insert_dataframe(
                self.conn,
                "sensor_qc",
                _prepare_simple_frame(qc_report, SENSOR_QC_COLUMNS),
                SENSOR_QC_COLUMNS,
            )
            _insert_dataframe(
                self.conn,
                "events",
                _prepare_events_frame(events),
                EVENT_COLUMNS,
            )
            _insert_dataframe(
                self.conn,
                "event_sensors",
                _prepare_event_sensors_frame(event_sensors),
                EVENT_SENSOR_COLUMNS,
            )
            _insert_dataframe(
                self.conn,
                "aligned_samples",
                _prepare_aligned_samples_frame(aligned_samples),
                ALIGNED_SAMPLE_COLUMNS,
            )


class PreprocessStoreReader:
    """Read API for the canonical preprocess SQLite artifact."""

    def __init__(self, path_or_stage_dir: str | Path) -> None:
        self.path = preprocess_store_path(path_or_stage_dir)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the SQLite connection."""
        _close_sqlite_connection(self.conn)

    def __enter__(self) -> PreprocessStoreReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def load_stage_info(self) -> dict[str, Any]:
        """Return the stage metadata row as a plain dictionary."""
        row = self.conn.execute("SELECT * FROM stage_info").fetchone()
        if row is None:
            raise FileNotFoundError(f"Preprocess stage info not found in {self.path}.")
        payload = dict(row)
        payload["settings_json"] = json.loads(payload["settings_json"])
        return payload

    def list_events(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
        discarded: bool | None = None,
    ) -> pd.DataFrame:
        """Return event metadata rows from the preprocess store."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if set_name is not None:
            query += " AND set_name = ?"
            params.append(set_name)
        if deck is not None:
            query += " AND deck = ?"
            params.append(deck)
        if discarded is not None:
            query += " AND discarded = ?"
            params.append(int(discarded))
        query += " ORDER BY set_name, deck, start_time_utc, event_id"
        events = pd.read_sql_query(query, self.conn, params=params)
        if events.empty:
            return pd.DataFrame(
                columns=[
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
            )
        events["discarded"] = events["discarded"].astype(bool)
        events["active_sensors"] = events["active_sensors_json"].map(json.loads)
        events["excluded_sensors"] = events["excluded_sensors_json"].map(json.loads)
        events["excluded_sensor_reasons"] = events["excluded_sensor_reasons_json"].map(json.loads)
        return events.drop(
            columns=["active_sensors_json", "excluded_sensors_json", "excluded_sensor_reasons_json"]
        )

    def iter_retained_events(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
    ) -> pd.DataFrame:
        """Return retained event metadata for downstream stages."""
        return self.list_events(set_name=set_name, deck=deck, discarded=False)

    def load_event_sensors(self, event_id: str) -> pd.DataFrame:
        """Return one row per event/sensor pair in organizer order."""
        query = """
            SELECT *
            FROM event_sensors
            WHERE event_id = ?
            ORDER BY sensor_order, sensor_name
        """
        return pd.read_sql_query(query, self.conn, params=[event_id])

    def load_aligned_event(
        self,
        event_id: str,
        *,
        sensor_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load one retained event as a wide aligned matrix."""
        event_sensors = self.load_event_sensors(event_id)
        included_sensors = event_sensors.loc[event_sensors["sensor_status"] == "included"].copy()
        if sensor_names is None:
            ordered_sensor_names = included_sensors["sensor_name"].astype(str).tolist()
        else:
            requested = set(sensor_names)
            ordered_sensor_names = [
                sensor_name
                for sensor_name in event_sensors["sensor_name"].astype(str).tolist()
                if sensor_name in requested
            ]
            for sensor_name in sensor_names:
                if sensor_name not in ordered_sensor_names:
                    ordered_sensor_names.append(sensor_name)

        query = """
            SELECT sample_index, timestamp_utc, sensor_name, value
            FROM aligned_samples
            WHERE event_id = ?
            ORDER BY sample_index, sensor_name
        """
        params: list[Any] = [event_id]
        samples = pd.read_sql_query(query, self.conn, params=params)
        if sensor_names is not None and not samples.empty:
            samples = samples.loc[samples["sensor_name"].isin(set(sensor_names))].copy()

        if samples.empty:
            return pd.DataFrame(columns=["timestamp_utc", *ordered_sensor_names])

        wide = (
            samples.pivot(index=["sample_index", "timestamp_utc"], columns="sensor_name", values="value")
            .reset_index()
            .sort_values("sample_index", kind="mergesort")
            .drop(columns=["sample_index"])
        )
        wide["timestamp_utc"] = pd.to_datetime(wide["timestamp_utc"], utc=True, format="mixed")
        for sensor_name in ordered_sensor_names:
            if sensor_name not in wide.columns:
                wide[sensor_name] = float("nan")
        return wide[["timestamp_utc", *ordered_sensor_names]].reset_index(drop=True)

    def load_aligned_samples(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
        sensor_names: Sequence[str] | None = None,
        event_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Stream long-form aligned samples with optional event and sensor filters."""
        query = """
            SELECT sample.*
            FROM aligned_samples AS sample
            JOIN events AS event ON event.event_id = sample.event_id
            WHERE 1=1
        """
        params: list[Any] = []
        if set_name is not None:
            query += " AND event.set_name = ?"
            params.append(set_name)
        if deck is not None:
            query += " AND event.deck = ?"
            params.append(deck)
        if sensor_names:
            query += f" AND sample.sensor_name IN ({_placeholders(sensor_names)})"
            params.extend(sensor_names)
        if event_ids:
            query += f" AND sample.event_id IN ({_placeholders(event_ids)})"
            params.extend(event_ids)
        query += " ORDER BY event.set_name, event.deck, sample.event_id, sample.sample_index, sample.sensor_name"
        frame = pd.read_sql_query(query, self.conn, params=params)
        if not frame.empty:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, format="mixed")
        return frame


class LegacyPreprocessCsvReader:
    """Temporary reader for legacy CSV/GZ preprocess artifacts."""

    def __init__(self, stage_dir: str | Path) -> None:
        self.stage_dir = Path(stage_dir)
        self.manifest_path = self.stage_dir / "event_manifest.csv"
        self.sensor_records_path = self.stage_dir / "sensor_records.csv"
        self._sensor_records_cache: pd.DataFrame | None = None

    def close(self) -> None:
        """Compatibility no-op."""

    def __enter__(self) -> LegacyPreprocessCsvReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def load_stage_info(self) -> dict[str, Any]:
        """Return minimal compatibility metadata."""
        return {
            "stage_name": "preprocess",
            "schema_version": 0,
            "run_id": self.stage_dir.parent.parent.name,
            "created_at_utc": None,
            "settings_json": {"backend": "legacy_csv"},
        }

    def list_events(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
        discarded: bool | None = None,
    ) -> pd.DataFrame:
        """Return legacy event manifest rows in the canonical reader shape."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Legacy event manifest not found at {self.manifest_path}.")
        events = pd.read_csv(self.manifest_path)
        if events.empty:
            return pd.DataFrame(
                columns=[
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
            )
        events["discarded"] = events["discarded"].astype(bool)
        events["active_sensors"] = events["active_sensors"].fillna("").map(_split_semicolon_list)
        events["excluded_sensors"] = events["excluded_sensors"].fillna("").map(_split_semicolon_list)
        events["excluded_sensor_reasons"] = events["excluded_sensor_reasons"].fillna("").map(
            _split_semicolon_list
        )
        if set_name is not None:
            events = events.loc[events["set_name"] == set_name].copy()
        if deck is not None:
            events = events.loc[events["deck"] == deck].copy()
        if discarded is not None:
            events = events.loc[events["discarded"] == bool(discarded)].copy()
        return events.reset_index(drop=True)

    def iter_retained_events(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
    ) -> pd.DataFrame:
        return self.list_events(set_name=set_name, deck=deck, discarded=False)

    def load_event_sensors(self, event_id: str) -> pd.DataFrame:
        """Return one row per event/sensor from legacy sensor_records.csv."""
        sensor_records = self._load_sensor_records()
        matched = sensor_records.loc[sensor_records["event_id"] == event_id].copy()
        if matched.empty:
            return pd.DataFrame(columns=EVENT_SENSOR_COLUMNS)
        matched["start_time_utc"] = pd.to_datetime(matched["start_time_utc"], utc=True, format="mixed")
        matched["end_time_utc"] = pd.to_datetime(matched["end_time_utc"], utc=True, format="mixed")
        collapsed = collapse_sensor_records(matched)
        events = self.list_events()
        event_row = events.loc[events["event_id"] == event_id]
        reference_sensor = ""
        if not event_row.empty:
            reference_sensor = str(event_row.iloc[0]["reference_sensor"])

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
                    "record_uid": row.get("Record_UID"),
                    "raw_file": str(row["raw_file"]),
                    "start_row_1based": int(row["start_row_1based"]),
                    "end_row_1based": int(row["end_row_1based"]),
                    "start_time_utc": row["start_time_utc"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "end_time_utc": row["end_time_utc"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "duration": row.get("Duration"),
                    "temperature": row.get("Temperature"),
                    "start_value": row.get("Start_Value"),
                    "end_value": row.get("End_Value"),
                    "diff_value": row.get("Diff_Value"),
                    "min_value": row.get("Min_Value"),
                    "max_value": row.get("Max_Value"),
                    "mean_value": row.get("Mean_Value"),
                    "range_value": row.get("Range"),
                }
            )
        return pd.DataFrame(rows, columns=EVENT_SENSOR_COLUMNS)

    def _load_sensor_records(self) -> pd.DataFrame:
        """Load and cache legacy sensor records with stable dtypes."""
        if self._sensor_records_cache is None:
            if not self.sensor_records_path.is_file():
                raise FileNotFoundError(
                    f"Legacy sensor records not found at {self.sensor_records_path}."
                )
            self._sensor_records_cache = pd.read_csv(
                self.sensor_records_path,
                dtype={
                    "exclusion_reason": "string",
                    "exclusion_source": "string",
                },
                low_memory=False,
            )
        return self._sensor_records_cache

    def load_aligned_event(
        self,
        event_id: str,
        *,
        sensor_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load one legacy event as a wide aligned matrix."""
        events = self.list_events()
        event_row = events.loc[events["event_id"] == event_id]
        if event_row.empty:
            return pd.DataFrame(columns=["timestamp_utc"])
        event_row = event_row.iloc[0]
        aligned_path = _legacy_aligned_partition_path(
            self.stage_dir,
            set_name=str(event_row["set_name"]),
            deck=str(event_row["deck"]),
        )
        if aligned_path is None:
            return pd.DataFrame(columns=["timestamp_utc"])
        frame = pd.read_csv(aligned_path)
        frame = frame.loc[frame["event_id"] == event_id].copy()
        if frame.empty:
            ordered_sensors = sensor_names or event_row["active_sensors"]
            return pd.DataFrame(columns=["timestamp_utc", *ordered_sensors])
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, format="mixed")
        frame = frame.sort_values("sample_index", kind="mergesort").drop(columns=["event_id", "sample_index"])
        ordered_sensors = [
            column
            for column in frame.columns
            if column != "timestamp_utc"
        ]
        if sensor_names is not None:
            requested = list(sensor_names)
            for sensor_name in requested:
                if sensor_name not in frame.columns:
                    frame[sensor_name] = float("nan")
            ordered_sensors = requested
        return frame[["timestamp_utc", *ordered_sensors]].reset_index(drop=True)

    def load_aligned_samples(
        self,
        *,
        set_name: str | None = None,
        deck: str | None = None,
        sensor_names: Sequence[str] | None = None,
        event_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return long-form aligned samples from legacy partition files."""
        events = self.list_events(set_name=set_name, deck=deck, discarded=False)
        if event_ids is not None:
            events = events.loc[events["event_id"].isin(set(event_ids))].copy()
        if events.empty:
            return pd.DataFrame(columns=ALIGNED_SAMPLE_COLUMNS)

        frames: list[pd.DataFrame] = []
        for set_value, deck_value in (
            events[["set_name", "deck"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            aligned_path = _legacy_aligned_partition_path(self.stage_dir, set_name=set_value, deck=deck_value)
            if aligned_path is None:
                continue
            partition = pd.read_csv(aligned_path)
            partition = partition.loc[partition["event_id"].isin(events["event_id"])].copy()
            if partition.empty:
                continue
            melted = partition.melt(
                id_vars=["event_id", "sample_index", "timestamp_utc"],
                var_name="sensor_name",
                value_name="value",
            )
            melted["set_name"] = set_value
            melted["deck"] = deck_value
            melted = melted.dropna(subset=["value"]).reset_index(drop=True)
            frames.append(
                melted[
                    ["event_id", "set_name", "deck", "sensor_name", "sample_index", "timestamp_utc", "value"]
                ]
            )
        if not frames:
            return pd.DataFrame(columns=ALIGNED_SAMPLE_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        if sensor_names is not None:
            combined = combined.loc[combined["sensor_name"].isin(set(sensor_names))].copy()
        combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True, format="mixed")
        return combined.sort_values(
            ["set_name", "deck", "event_id", "sample_index", "sensor_name"],
            kind="mergesort",
        ).reset_index(drop=True)


def open_preprocess_store(path_or_stage_dir: str | Path) -> PreprocessStoreReader | LegacyPreprocessCsvReader:
    """Open the preprocess stage outputs using the canonical or legacy reader."""
    candidate_path = Path(path_or_stage_dir)
    sqlite_path = preprocess_store_path(candidate_path)
    if sqlite_path.is_file():
        return PreprocessStoreReader(sqlite_path)

    stage_dir = candidate_path if candidate_path.is_dir() else candidate_path.parent
    if (stage_dir / "event_manifest.csv").is_file() and (stage_dir / "sensor_records.csv").is_file():
        return LegacyPreprocessCsvReader(stage_dir)

    raise FileNotFoundError(
        f"No preprocess store found at {sqlite_path} and no legacy CSV artifacts found in {stage_dir}."
    )


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")


def _close_sqlite_connection(conn: sqlite3.Connection) -> None:
    """Checkpoint WAL state before closing so Windows callers can move/delete the DB cleanly."""
    try:
        if conn.in_transaction:
            conn.commit()
    except sqlite3.Error:
        pass
    for pragma in (
        "PRAGMA wal_checkpoint(TRUNCATE)",
        "PRAGMA journal_mode = DELETE",
    ):
        try:
            conn.execute(pragma).fetchall()
        except sqlite3.Error:
            pass
    conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE stage_info (
            stage_name TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            settings_json TEXT NOT NULL
        );

        CREATE TABLE sets (
            set_name TEXT PRIMARY KEY,
            set_order INTEGER NOT NULL
        );

        CREATE TABLE sensors (
            sensor_name TEXT PRIMARY KEY,
            deck TEXT,
            span TEXT,
            side TEXT,
            location TEXT,
            quantity TEXT,
            axis TEXT
        );

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            set_name TEXT NOT NULL REFERENCES sets(set_name),
            deck TEXT NOT NULL,
            start_time_utc TEXT NOT NULL,
            end_time_utc TEXT NOT NULL,
            active_sensor_count INTEGER NOT NULL,
            active_sensors_json TEXT NOT NULL,
            excluded_sensor_count INTEGER NOT NULL,
            excluded_sensors_json TEXT NOT NULL,
            excluded_sensor_reasons_json TEXT NOT NULL,
            reference_sensor TEXT NOT NULL,
            rows_before_alignment INTEGER NOT NULL,
            rows_after_alignment INTEGER NOT NULL,
            discarded INTEGER NOT NULL,
            discard_reason TEXT NOT NULL,
            zeroing_method TEXT NOT NULL
        );

        CREATE TABLE event_sensors (
            event_id TEXT NOT NULL REFERENCES events(event_id),
            set_name TEXT NOT NULL REFERENCES sets(set_name),
            deck TEXT NOT NULL,
            sensor_name TEXT NOT NULL REFERENCES sensors(sensor_name),
            sensor_order INTEGER NOT NULL,
            sensor_status TEXT NOT NULL,
            exclusion_reason TEXT NOT NULL,
            exclusion_source TEXT NOT NULL,
            is_reference INTEGER NOT NULL,
            record_uid TEXT,
            raw_file TEXT NOT NULL,
            start_row_1based INTEGER NOT NULL,
            end_row_1based INTEGER NOT NULL,
            start_time_utc TEXT NOT NULL,
            end_time_utc TEXT NOT NULL,
            duration REAL,
            temperature REAL,
            start_value REAL,
            end_value REAL,
            diff_value REAL,
            min_value REAL,
            max_value REAL,
            mean_value REAL,
            range_value REAL,
            PRIMARY KEY (event_id, sensor_name)
        );

        CREATE TABLE aligned_samples (
            event_id TEXT NOT NULL REFERENCES events(event_id),
            set_name TEXT NOT NULL REFERENCES sets(set_name),
            deck TEXT NOT NULL,
            sensor_name TEXT NOT NULL REFERENCES sensors(sensor_name),
            sample_index INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (event_id, sensor_name, sample_index)
        );

        CREATE TABLE sensor_records (
            table_row_index INTEGER NOT NULL,
            Record_UID TEXT,
            File TEXT,
            Start_Row INTEGER,
            End_Row INTEGER,
            Start_Time TEXT,
            End_Time TEXT,
            Duration REAL,
            Start_Value REAL,
            End_Value REAL,
            Diff_Value REAL,
            Min_Value REAL,
            Max_Value REAL,
            Mean_Value REAL,
            Range REAL,
            Temperature REAL,
            sensor_name TEXT NOT NULL REFERENCES sensors(sensor_name),
            dataset TEXT NOT NULL,
            set_name TEXT NOT NULL REFERENCES sets(set_name),
            deck TEXT NOT NULL,
            sensor_order INTEGER NOT NULL,
            start_time_utc TEXT NOT NULL,
            end_time_utc TEXT NOT NULL,
            raw_file TEXT NOT NULL,
            start_row_1based INTEGER NOT NULL,
            end_row_1based INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            sensor_status TEXT NOT NULL,
            exclusion_reason TEXT NOT NULL,
            exclusion_source TEXT NOT NULL
        );

        CREATE TABLE sensor_qc (
            set_name TEXT NOT NULL REFERENCES sets(set_name),
            sensor_name TEXT NOT NULL REFERENCES sensors(sensor_name),
            event_count INTEGER NOT NULL,
            sensor_status TEXT NOT NULL,
            exclusion_reason TEXT NOT NULL,
            exclusion_source TEXT NOT NULL,
            table_range_median REAL,
            table_range_nonzero_fraction REAL,
            table_mean_abs_median REAL,
            table_start_value_median REAL,
            table_end_value_median REAL,
            raw_range_spotcheck_median REAL,
            raw_to_table_range_ratio_spotcheck REAL,
            PRIMARY KEY (set_name, sensor_name)
        );

        CREATE INDEX idx_events_set_deck_discarded_start
            ON events(set_name, deck, discarded, start_time_utc);
        CREATE INDEX idx_event_sensors_sensor_event
            ON event_sensors(sensor_name, event_id);
        CREATE INDEX idx_aligned_samples_event_sample_sensor
            ON aligned_samples(event_id, sample_index, sensor_name);
        CREATE INDEX idx_aligned_samples_sensor_event_sample
            ON aligned_samples(sensor_name, event_id, sample_index);
        CREATE INDEX idx_sensor_records_set_sensor_start
            ON sensor_records(set_name, sensor_name, start_time_utc);
        """
    )


def _write_stage_info(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    settings_payload: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO stage_info (
            stage_name,
            schema_version,
            run_id,
            created_at_utc,
            settings_json
        )
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
        """,
        (
            "preprocess",
            PREPROCESS_SCHEMA_VERSION,
            run_id,
            json.dumps(settings_payload, sort_keys=True),
        ),
    )


def _write_sets(conn: sqlite3.Connection, set_names: Sequence[str]) -> None:
    conn.executemany(
        "INSERT INTO sets (set_name, set_order) VALUES (?, ?)",
        [(set_name, index) for index, set_name in enumerate(set_names, start=1)],
    )


def _upsert_sensors(
    conn: sqlite3.Connection,
    sensor_records: pd.DataFrame,
    event_sensors: pd.DataFrame,
) -> None:
    sensor_names = sorted(
        {
            str(sensor_name)
            for sensor_name in pd.concat(
                [
                    sensor_records.get("sensor_name", pd.Series(dtype=object)),
                    event_sensors.get("sensor_name", pd.Series(dtype=object)),
                ],
                ignore_index=True,
            ).dropna()
        }
    )
    if not sensor_names:
        return
    rows = []
    for sensor_name in sensor_names:
        parsed = parse_sensor_name(sensor_name)
        rows.append(
            (
                sensor_name,
                parsed["deck"],
                parsed["span"],
                parsed["side"],
                parsed["location"],
                parsed["quantity"],
                parsed["axis"],
            )
        )
    conn.executemany(
        """
        INSERT OR IGNORE INTO sensors (
            sensor_name, deck, span, side, location, quantity, axis
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_dataframe(
    conn: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    if frame.empty:
        return
    placeholders = ", ".join("?" for _ in columns)
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(
        query,
        [
            tuple(_normalize_sql_value(row[column]) for column in columns)
            for row in frame.to_dict("records")
        ],
    )


def _prepare_events_frame(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = events.copy()
    frame["active_sensors_json"] = frame["active_sensors"].map(json.dumps)
    frame["excluded_sensors_json"] = frame["excluded_sensors"].map(json.dumps)
    frame["excluded_sensor_reasons_json"] = frame["excluded_sensor_reasons"].map(json.dumps)
    frame = _prepare_simple_frame(frame, EVENT_COLUMNS).copy()
    if frame.empty:
        return frame
    frame["active_sensor_count"] = frame["active_sensor_count"].map(_to_int)
    frame["excluded_sensor_count"] = frame["excluded_sensor_count"].map(_to_int)
    frame["rows_before_alignment"] = frame["rows_before_alignment"].map(_to_int)
    frame["rows_after_alignment"] = frame["rows_after_alignment"].map(_to_int)
    frame["discarded"] = frame["discarded"].map(lambda value: int(bool(value)))
    return frame


def _prepare_event_sensors_frame(event_sensors: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_simple_frame(event_sensors, EVENT_SENSOR_COLUMNS).copy()
    if frame.empty:
        return frame
    integer_columns = [
        "sensor_order",
        "is_reference",
        "start_row_1based",
        "end_row_1based",
    ]
    for column in integer_columns:
        frame[column] = frame[column].map(_to_int)
    float_columns = [
        "duration",
        "temperature",
        "start_value",
        "end_value",
        "diff_value",
        "min_value",
        "max_value",
        "mean_value",
        "range_value",
    ]
    for column in float_columns:
        frame[column] = frame[column].map(_to_optional_float)
    return frame


def _prepare_aligned_samples_frame(samples: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_simple_frame(samples, ALIGNED_SAMPLE_COLUMNS).copy()
    if frame.empty:
        return frame
    frame["sample_index"] = frame["sample_index"].map(_to_int)
    frame["value"] = frame["value"].map(_to_optional_float)
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    return frame


def _prepare_sensor_records_frame(sensor_records: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_simple_frame(sensor_records, SENSOR_RECORD_COLUMNS).copy()
    if frame.empty:
        return frame
    frame["start_time_utc"] = frame["start_time_utc"].map(_to_timestamp_text)
    frame["end_time_utc"] = frame["end_time_utc"].map(_to_timestamp_text)
    integer_columns = [
        "table_row_index",
        "Start_Row",
        "End_Row",
        "sensor_order",
        "start_row_1based",
        "end_row_1based",
    ]
    for column in integer_columns:
        frame[column] = frame[column].map(_to_int)
    float_columns = [
        "Duration",
        "Start_Value",
        "End_Value",
        "Diff_Value",
        "Min_Value",
        "Max_Value",
        "Mean_Value",
        "Range",
        "Temperature",
    ]
    for column in float_columns:
        frame[column] = frame[column].map(_to_optional_float)
    return frame


def _prepare_simple_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    prepared = frame.copy()
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[list(columns)]


def _placeholders(values: Sequence[Any]) -> str:
    return ", ".join("?" for _ in values)


def _split_semicolon_list(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in str(value).split(";") if part]


def _legacy_aligned_partition_path(stage_dir: Path, *, set_name: str, deck: str) -> Path | None:
    normalized_deck = str(deck).strip().upper()
    deck_candidates = [
        normalized_deck,
        f"{normalized_deck}_DECK",
    ]
    root_candidates = [
        stage_dir / "aligned",
        stage_dir / "exports" / "aligned",
    ]
    suffixes = [".csv.gz", ".csv"]
    for root in root_candidates:
        for deck_label in deck_candidates:
            for suffix in suffixes:
                candidate = root / f"{set_name}__{deck_label}{suffix}"
                if candidate.is_file():
                    return candidate
    return None


def _to_int(value: Any) -> int:
    return int(value)


def _to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    return float(value)


def _to_timestamp_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalize_sql_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value
