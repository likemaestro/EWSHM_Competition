"""SQLite-backed storage and read APIs for the features stage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from aquinas_toolkit.preprocessing.store import _normalize_sql_value


FEATURES_DB_NAME = "features.sqlite"
FEATURES_SCHEMA_VERSION = 2

SENSOR_EVENT_FEATURE_COLUMNS = [
    "event_id",
    "set_name",
    "deck",
    "sensor_name",
    "sensor_order",
    "quantity",
    "axis",
    "sample_count",
    "aligned_duration_s",
    "table_duration",
    "table_start_value",
    "table_end_value",
    "table_diff_value",
    "table_min_value",
    "table_max_value",
    "table_mean_value",
    "table_range_value",
    "table_temperature",
    "waveform_mean",
    "waveform_std",
    "waveform_rms",
    "waveform_min",
    "waveform_max",
    "waveform_peak_to_peak",
    "waveform_energy",
    "waveform_crest_factor",
    "waveform_zero_crossing_rate",
    "waveform_skewness",
    "waveform_kurtosis",
]

DECK_MODAL_PEAK_COLUMNS = [
    "set_name",
    "deck",
    "feature_family",
    "quantity",
    "axis",
    "peak_rank",
    "frequency_hz",
    "singular_value",
    "frequency_index",
    "channel_count",
    "event_count",
]

DECK_MODE_SHAPE_COMPONENT_COLUMNS = [
    "set_name",
    "deck",
    "peak_rank",
    "sensor_name",
    "frequency_hz",
    "singular_value",
    "mode_shape_amplitude",
    "mode_shape_signed_component",
    "mode_shape_phase_deg",
    "span",
    "side",
    "location",
    "quantity",
    "axis",
    "position_label",
]

FEATURE_FAMILY_STATUS_COLUMNS = [
    "set_name",
    "deck",
    "feature_family",
    "status",
    "detail",
    "event_count",
    "channel_count",
]


def features_store_path(path_or_stage_dir: str | Path) -> Path:
    """Resolve a features store path from a stage directory or a DB path."""
    path = Path(path_or_stage_dir)
    if path.suffix == ".sqlite":
        return path
    return path / FEATURES_DB_NAME


class FeaturesStoreWriter:
    """Writer for the canonical features SQLite artifact."""

    def __init__(
        self,
        path_or_stage_dir: str | Path,
        *,
        run_id: str,
        preprocess_store_path: str,
        settings_payload: dict[str, Any],
    ) -> None:
        self.path = features_store_path(path_or_stage_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        _configure_features_connection(self.conn)
        self._create_schema()
        self.conn.execute(
            """
            INSERT INTO stage_info (
                stage_name,
                schema_version,
                run_id,
                created_at_utc,
                preprocess_store_path,
                settings_json
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
            (
                "features",
                FEATURES_SCHEMA_VERSION,
                run_id,
                preprocess_store_path,
                json.dumps(settings_payload, sort_keys=True),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

    def __enter__(self) -> FeaturesStoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def write_sensor_event_features(self, rows: list[dict[str, Any]]) -> None:
        """Append per-sensor per-event features."""
        self._insert_rows("sensor_event_features", rows, SENSOR_EVENT_FEATURE_COLUMNS)

    def write_deck_modal_peaks(self, rows: list[dict[str, Any]]) -> None:
        """Append deck-level FDD peak rows."""
        self._insert_rows("deck_modal_peaks", rows, DECK_MODAL_PEAK_COLUMNS)

    def write_deck_mode_shape_components(self, rows: list[dict[str, Any]]) -> None:
        """Append deck-level FDD mode-shape component rows."""
        self._insert_rows(
            "deck_mode_shape_components",
            rows,
            DECK_MODE_SHAPE_COMPONENT_COLUMNS,
        )

    def write_feature_family_status(self, rows: list[dict[str, Any]]) -> None:
        """Append per-family status rows."""
        self._insert_rows("feature_family_status", rows, FEATURE_FAMILY_STATUS_COLUMNS)

    def _insert_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        columns: Sequence[str],
    ) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        placeholders = ", ".join("?" for _ in columns)
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        with self.conn:
            self.conn.executemany(
                query,
                [
                    tuple(_normalize_sql_value(record[column]) for column in columns)
                    for record in frame.to_dict("records")
                ],
            )

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE stage_info (
                stage_name TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                preprocess_store_path TEXT NOT NULL,
                settings_json TEXT NOT NULL
            );

            CREATE TABLE sensor_event_features (
                event_id TEXT NOT NULL,
                set_name TEXT NOT NULL,
                deck TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                sensor_order INTEGER NOT NULL,
                quantity TEXT,
                axis TEXT,
                sample_count INTEGER NOT NULL,
                aligned_duration_s REAL,
                table_duration REAL,
                table_start_value REAL,
                table_end_value REAL,
                table_diff_value REAL,
                table_min_value REAL,
                table_max_value REAL,
                table_mean_value REAL,
                table_range_value REAL,
                table_temperature REAL,
                waveform_mean REAL,
                waveform_std REAL,
                waveform_rms REAL,
                waveform_min REAL,
                waveform_max REAL,
                waveform_peak_to_peak REAL,
                waveform_energy REAL,
                waveform_crest_factor REAL,
                waveform_zero_crossing_rate REAL,
                waveform_skewness REAL,
                waveform_kurtosis REAL,
                PRIMARY KEY (event_id, sensor_name)
            );

            CREATE TABLE deck_modal_peaks (
                set_name TEXT NOT NULL,
                deck TEXT NOT NULL,
                feature_family TEXT NOT NULL,
                quantity TEXT NOT NULL,
                axis TEXT NOT NULL,
                peak_rank INTEGER NOT NULL,
                frequency_hz REAL NOT NULL,
                singular_value REAL NOT NULL,
                frequency_index INTEGER NOT NULL,
                channel_count INTEGER NOT NULL,
                event_count INTEGER NOT NULL,
                PRIMARY KEY (set_name, deck, feature_family, peak_rank)
            );

            CREATE TABLE deck_mode_shape_components (
                set_name TEXT NOT NULL,
                deck TEXT NOT NULL,
                peak_rank INTEGER NOT NULL,
                sensor_name TEXT NOT NULL,
                frequency_hz REAL NOT NULL,
                singular_value REAL NOT NULL,
                mode_shape_amplitude REAL NOT NULL,
                mode_shape_signed_component REAL NOT NULL,
                mode_shape_phase_deg REAL NOT NULL,
                span TEXT,
                side TEXT,
                location TEXT,
                quantity TEXT,
                axis TEXT,
                position_label TEXT,
                PRIMARY KEY (set_name, deck, peak_rank, sensor_name)
            );

            CREATE TABLE feature_family_status (
                set_name TEXT NOT NULL,
                deck TEXT NOT NULL,
                feature_family TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                channel_count INTEGER NOT NULL,
                PRIMARY KEY (set_name, deck, feature_family)
            );

            CREATE INDEX idx_sensor_event_features_set_deck_sensor
                ON sensor_event_features(set_name, deck, sensor_name);
            CREATE INDEX idx_deck_modal_peaks_set_deck
                ON deck_modal_peaks(set_name, deck, feature_family);
            CREATE INDEX idx_feature_family_status_set_deck
                ON feature_family_status(set_name, deck);
            """
        )


class FeaturesStoreReader:
    """Read API for the canonical features SQLite artifact."""

    def __init__(self, path_or_stage_dir: str | Path) -> None:
        self.path = features_store_path(path_or_stage_dir)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

    def __enter__(self) -> FeaturesStoreReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def load_sensor_event_features(self) -> pd.DataFrame:
        """Return the canonical per-sensor feature table."""
        return pd.read_sql_query(
            """
            SELECT *
            FROM sensor_event_features
            ORDER BY set_name, deck, event_id, sensor_order
            """,
            self.conn,
        )

    def load_deck_modal_peaks(self) -> pd.DataFrame:
        """Return deck-level modal peak rows."""
        return pd.read_sql_query(
            """
            SELECT *
            FROM deck_modal_peaks
            ORDER BY set_name, deck, feature_family, peak_rank
            """,
            self.conn,
        )

    def load_deck_mode_shape_components(self) -> pd.DataFrame:
        """Return deck-level mode-shape component rows."""
        return pd.read_sql_query(
            """
            SELECT *
            FROM deck_mode_shape_components
            ORDER BY set_name, deck, peak_rank, sensor_name
            """,
            self.conn,
        )

    def load_feature_family_status(self) -> pd.DataFrame:
        """Return per-family execution status rows."""
        return pd.read_sql_query(
            """
            SELECT *
            FROM feature_family_status
            ORDER BY set_name, deck, feature_family
            """,
            self.conn,
        )


def open_features_store(path_or_stage_dir: str | Path) -> FeaturesStoreReader:
    """Open the features SQLite store for reading."""
    return FeaturesStoreReader(path_or_stage_dir)


def _configure_features_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
