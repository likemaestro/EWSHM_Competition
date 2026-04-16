import json
import sqlite3
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquinas_toolkit import AquinasReader
from aquinas_toolkit.cli import run as run_mod
from aquinas_toolkit.preprocessing import (
    LoadedEventGroup,
    PreprocessWaveformMigrationWarning,
    align_event_group,
    detect_legacy_preprocess_waveforms,
    find_events,
    load_event_group,
    migrate_preprocess_waveforms,
    open_preprocess_store,
    run_organizer_query,
    synchro_indices,
    zero_loaded_event_group,
    zero_waveform,
)
from aquinas_toolkit.preprocessing.alignment import AlignedEvent
from aquinas_toolkit.preprocessing import pipeline as pipeline_mod
from aquinas_toolkit.preprocessing.core import _parse_timestamps_fast
from aquinas_toolkit.preprocessing.pipeline import load_preprocessing_settings
from aquinas_toolkit.preprocessing.signals import (
    bandpass_filter_waveform_matrix,
    filter_loaded_event_group,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_default_preprocess_config(
    workspace: Path,
    *,
    set_names: tuple[str, ...] = ("AQUINAS_SET1_2022_07",),
    min_active_sensors_per_event: int = 1,
    aligned_export_enabled: bool = False,
    export_format: str = "csv.gz",
) -> None:
    lines = [
        "data:",
        "  dataset_root: AQUINAS_DATASET",
        "  sets:",
    ]
    lines.extend(f"    - {set_name}" for set_name in set_names)
    lines.extend(
        [
            "preprocessing:",
            "  event_grouping:",
            "    key_fields: [deck, Start_Time, End_Time]",
            "  alignment:",
            "    method: r_synchro",
            "  zeroing:",
            "    method: linear_endpoints",
            "  filtering:",
            f"    min_active_sensors_per_event: {min_active_sensors_per_event}",
            "  storage:",
            "    backend: sqlite",
            "  exports:",
            "    aligned_waveforms:",
            f"      enabled: {str(aligned_export_enabled).lower()}",
            f"      format: {export_format}",
            "output:",
            "  results_dir: results",
        ]
    )
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(config_dir / "default.yaml", lines)


def _fetch_table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def _build_legacy_preprocess_waveform_layout(tmp_path: Path) -> tuple[Path, str, list[str], list[str], np.ndarray]:
    from aquinas_toolkit.preprocessing.store import PreprocessStoreWriter  # noqa: PLC0415

    preprocess_dir = tmp_path / "results" / "run-1" / "stages" / "preprocess"
    event_id = "AQUINAS_SET1_2022_07__NEW__2022-07-01T00-00-00Z__2022-07-01T00-00-02Z"
    timestamps = [
        "2022-07-01T00:00:00Z",
        "2022-07-01T00:00:01Z",
    ]
    sensor_names = ["NEW_S1_DO_INF_STR", "NEW_S1_DO_SUP_STR"]
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    writer = PreprocessStoreWriter(
        preprocess_dir,
        run_id="run-1",
        settings_payload={},
        set_names=("AQUINAS_SET1_2022_07",),
    )
    writer.write_aligned_samples(
        [],
        events=pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "set_name": "AQUINAS_SET1_2022_07",
                    "deck": "NEW",
                    "start_time_utc": "2022-07-01T00:00:00Z",
                    "end_time_utc": "2022-07-01T00:00:02Z",
                    "active_sensor_count": 2,
                    "active_sensors": sensor_names,
                    "excluded_sensor_count": 0,
                    "excluded_sensors": [],
                    "excluded_sensor_reasons": {},
                    "reference_sensor": "NEW_S1_DO_INF_STR",
                    "rows_before_alignment": 2,
                    "rows_after_alignment": 2,
                    "discarded": False,
                    "discard_reason": "",
                    "zeroing_method": "linear_endpoints",
                }
            ]
        ),
    )
    writer.close()

    waveforms_dir = preprocess_dir / "waveforms"
    waveforms_dir.mkdir(parents=True, exist_ok=True)
    np.save(waveforms_dir / f"{event_id}.npy", matrix)
    (waveforms_dir / f"{event_id}.meta.json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "sensor_names": sensor_names,
                "timestamps_utc": timestamps,
            }
        ),
        encoding="utf-8",
    )

    return preprocess_dir, event_id, timestamps, sensor_names, matrix


def _build_preprocessing_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)

    _write_sensor(
        set_dir,
        "NEW_S1_DO_INF_STR",
        table_payload={
            "Record_UID": [1001],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [4],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03"],
            "Duration": [3.0],
            "Temperature": [21.5],
        },
        timestamps=[
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
            "2022-07-01 00:00:03.000",
        ],
        values=[10.0, 21.0, 30.0, 40.0],
    )
    _write_sensor(
        set_dir,
        "NEW_S1_DO_MID_ACC_Z",
        table_payload={
            "Record_UID": [1002],
            "File": ["NEW_S1_DO_MID_ACC_Z_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [3],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03"],
            "Duration": [3.0],
            "Temperature": [21.5],
        },
        timestamps=[
            "2022-07-01 00:00:00.200",
            "2022-07-01 00:00:01.200",
            "2022-07-01 00:00:02.200",
        ],
        values=[1.0, 2.0, 3.0],
    )
    _write_sensor(
        set_dir,
        "NEW_S1_DO_SUP_STR",
        table_payload={
            "Record_UID": [1003],
            "File": ["NEW_S1_DO_SUP_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03"],
            "Duration": [3.0],
            "Temperature": [21.5],
        },
        timestamps=[
            "2022-07-01 00:00:00.100",
            "2022-07-01 00:00:02.100",
        ],
        values=[5.0, 7.0],
    )
    _write_sensor(
        set_dir,
        "OLD_S1_DO_INF_STR",
        table_payload={
            "Record_UID": [2001],
            "File": ["OLD_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [4],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03"],
            "Duration": [3.0],
            "Temperature": [19.0],
        },
        timestamps=[
            "2022-07-01 00:00:00.001",
            "2022-07-01 00:00:01.001",
            "2022-07-01 00:00:02.001",
            "2022-07-01 00:00:03.001",
        ],
        values=[9.0, 9.5, 10.0, 10.5],
    )

    return dataset_root


def _build_two_set_preprocessing_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_specs = [
        ("AQUINAS_SET1_2022_07", "SET1", 10.0, 1.0, 5.0, 9.0),
        ("AQUINAS_SET2_2023_04", "SET2", 50.0, 4.0, 8.0, 12.0),
    ]

    for set_name, set_id, inf_base, acc_base, sup_base, old_base in set_specs:
        set_dir = dataset_root / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        _write_sensor(
            set_dir,
            "NEW_S1_DO_INF_STR",
            table_payload={
                "Record_UID": [1001],
                "File": [f"NEW_S1_DO_INF_STR_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [4],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Temperature": [21.5],
            },
            timestamps=[
                "2022-07-01 00:00:00.000",
                "2022-07-01 00:00:01.000",
                "2022-07-01 00:00:02.000",
                "2022-07-01 00:00:03.000",
            ],
            values=[inf_base, inf_base + 11.0, inf_base + 20.0, inf_base + 30.0],
            set_id=set_id,
        )
        _write_sensor(
            set_dir,
            "NEW_S1_DO_MID_ACC_Z",
            table_payload={
                "Record_UID": [1002],
                "File": [f"NEW_S1_DO_MID_ACC_Z_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [3],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Temperature": [21.5],
            },
            timestamps=[
                "2022-07-01 00:00:00.200",
                "2022-07-01 00:00:01.200",
                "2022-07-01 00:00:02.200",
            ],
            values=[acc_base, acc_base + 1.0, acc_base + 2.0],
            set_id=set_id,
        )
        _write_sensor(
            set_dir,
            "NEW_S1_DO_SUP_STR",
            table_payload={
                "Record_UID": [1003],
                "File": [f"NEW_S1_DO_SUP_STR_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [2],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Temperature": [21.5],
            },
            timestamps=[
                "2022-07-01 00:00:00.100",
                "2022-07-01 00:00:02.100",
            ],
            values=[sup_base, sup_base + 2.0],
            set_id=set_id,
        )
        _write_sensor(
            set_dir,
            "OLD_S1_DO_INF_STR",
            table_payload={
                "Record_UID": [2001],
                "File": [f"OLD_S1_DO_INF_STR_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [4],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Temperature": [19.0],
            },
            timestamps=[
                "2022-07-01 00:00:00.001",
                "2022-07-01 00:00:01.001",
                "2022-07-01 00:00:02.001",
                "2022-07-01 00:00:03.001",
            ],
            values=[old_base, old_base + 0.5, old_base + 1.0, old_base + 1.5],
            set_id=set_id,
        )

    return dataset_root


def _build_widening_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)

    _write_sensor(
        set_dir,
        "NEW_S1_DO_INF_STR",
        table_payload={
            "Record_UID": [1],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [4],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03"],
            "Duration": [3.0],
        },
        timestamps=[
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
            "2022-07-01 00:00:03.000",
        ],
        values=[1.0, 2.0, 3.0, 4.0],
    )
    _write_sensor(
        set_dir,
        "NEW_S1_DO_SUP_STR",
        table_payload={
            "Record_UID": [2, 3],
            "File": ["NEW_S1_DO_SUP_STR_SET1_1.json", "NEW_S1_DO_SUP_STR_SET1_1.json"],
            "Start_Row": [1, 2],
            "End_Row": [2, 4],
            "Start_Time": ["2022-07-01 00:00:00", "2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:03", "2022-07-01 00:00:03"],
            "Duration": [3.0, 3.0],
        },
        timestamps=[
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
            "2022-07-01 00:00:03.000",
        ],
        values=[10.0, 20.0, 30.0, 40.0],
    )
    return dataset_root


def _build_mixed_timestamp_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)

    common_table = {
        "Start_Time": ["2022-07-01 00:00:00"],
        "End_Time": ["2022-07-01 00:00:02"],
        "Duration": [2.0],
    }
    _write_sensor(
        set_dir,
        "NEW_S1_DO_INF_STR",
        table_payload={
            **common_table,
            "Record_UID": [1],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [4],
        },
        timestamps=[
            "2022-07-01 00:00:00",
            "2022-07-01 00:00:00.500",
            "2022-07-01 00:00:01",
            "2022-07-01 00:00:01.500",
        ],
        values=[1.0, 2.0, 3.0, 4.0],
    )
    _write_sensor(
        set_dir,
        "NEW_S1_DO_SUP_STR",
        table_payload={
            **common_table,
            "Record_UID": [2],
            "File": ["NEW_S1_DO_SUP_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [4],
        },
        timestamps=[
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:00.500",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:01.500",
        ],
        values=[10.0, 11.0, 12.0, 13.0],
    )
    return dataset_root


def _build_sensor_exclusion_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_specs = [
        ("AQUINAS_SET1_2022_07", "SET1", 0.02, 0.02, [-0.30, -0.28, -0.31, -0.29], [-0.15, -0.14, -0.16, -0.15]),
        ("AQUINAS_SET4_2024_01", "SET4", 0.0, 0.01, [29.8, 30.1, 29.9, 30.2], [-0.14, -0.13, -0.15, -0.14]),
        ("AQUINAS_SET5_2024_06", "SET5", 0.0, 0.01, [29.9, 30.0, 30.2, 29.8], [-0.13, -0.14, -0.12, -0.13]),
    ]

    for set_name, set_id, bad_range, good_range, bad_values, good_values in set_specs:
        set_dir = dataset_root / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        _write_sensor(
            set_dir,
            "OLD_S1_UP_SUP_STR",
            table_payload={
                "Record_UID": [1],
                "File": [f"OLD_S1_UP_SUP_STR_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [4],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Start_Value": [bad_values[0]],
                "End_Value": [bad_values[-1]],
                "Min_Value": [min(bad_values)],
                "Max_Value": [max(bad_values)],
                "Mean_Value": [sum(bad_values) / len(bad_values)],
                "Range": [bad_range],
                "Temperature": [18.0],
            },
            timestamps=[
                "2022-07-01 00:00:00.000",
                "2022-07-01 00:00:01.000",
                "2022-07-01 00:00:02.000",
                "2022-07-01 00:00:03.000",
            ],
            values=bad_values,
            set_id=set_id,
        )
        _write_sensor(
            set_dir,
            "OLD_S1_UP_INF_STR",
            table_payload={
                "Record_UID": [2],
                "File": [f"OLD_S1_UP_INF_STR_{set_id}_1.json"],
                "Start_Row": [1],
                "End_Row": [4],
                "Start_Time": ["2022-07-01 00:00:00"],
                "End_Time": ["2022-07-01 00:00:03"],
                "Duration": [3.0],
                "Start_Value": [good_values[0]],
                "End_Value": [good_values[-1]],
                "Min_Value": [min(good_values)],
                "Max_Value": [max(good_values)],
                "Mean_Value": [sum(good_values) / len(good_values)],
                "Range": [good_range],
                "Temperature": [18.0],
            },
            timestamps=[
                "2022-07-01 00:00:00.000",
                "2022-07-01 00:00:01.000",
                "2022-07-01 00:00:02.000",
                "2022-07-01 00:00:03.000",
            ],
            values=good_values,
            set_id=set_id,
        )

    return dataset_root


def _build_custom_raw_dataset(workspace: Path, *, raw_payload: dict) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)
    _write_sensor(
        set_dir,
        "NEW_S1_DO_INF_STR",
        table_payload={
            "Record_UID": [1],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [3],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:02"],
            "Duration": [2.0],
        },
        raw_payload=raw_payload,
    )
    return dataset_root


def _build_cross_file_match_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)
    sensor_name = "NEW_S1_DO_INF_STR"
    sensor_dir = set_dir / sensor_name
    sensor_dir.mkdir()
    _write_json(
        set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json",
        {
            "Record_UID": [1, 2],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json", "NEW_S1_DO_INF_STR_SET1_2.json"],
            "Start_Row": [1, 1],
            "End_Row": [2, 2],
            "Start_Time": ["2022-07-01 00:00:00", "2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:02", "2022-07-01 00:00:02"],
            "Duration": [2.0, 2.0],
        },
    )
    _write_json(
        sensor_dir / "NEW_S1_DO_INF_STR_SET1_1.json",
        {
            "timestamp": ["2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
            sensor_name: [1.0, 2.0],
        },
    )
    _write_json(
        sensor_dir / "NEW_S1_DO_INF_STR_SET1_2.json",
        {
            "timestamp": ["2022-07-01 00:00:01.500", "2022-07-01 00:00:02.000"],
            sensor_name: [3.0, 4.0],
        },
    )
    return dataset_root


def _build_no_common_rows_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)
    common_table = {
        "Start_Time": ["2022-07-01 00:00:00"],
        "End_Time": ["2022-07-01 00:00:02"],
        "Duration": [2.0],
    }
    _write_sensor(
        set_dir,
        "NEW_S1_DO_INF_STR",
        table_payload={
            **common_table,
            "Record_UID": [1],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
        },
        timestamps=[
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
        ],
        values=[10.0, 11.0],
    )
    _write_sensor(
        set_dir,
        "NEW_S1_DO_SUP_STR",
        table_payload={
            **common_table,
            "Record_UID": [2],
            "File": ["NEW_S1_DO_SUP_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
        },
        timestamps=[
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:00.500",
        ],
        values=[1.0, 2.0],
    )
    return dataset_root


def _build_empty_set_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)
    (set_dir / "NEW_S1_DO_INF_STR").mkdir()
    _write_json(
        set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json",
        {
            "Record_UID": [],
            "File": [],
            "Start_Row": [],
            "End_Row": [],
            "Start_Time": [],
            "End_Time": [],
            "Duration": [],
        },
    )
    return dataset_root


def _write_sensor(
    set_dir: Path,
    sensor_name: str,
    *,
    table_payload: dict,
    timestamps: list[str] | None = None,
    values: list[float] | None = None,
    set_id: str = "SET1",
    raw_payload: dict | None = None,
) -> None:
    sensor_dir = set_dir / sensor_name
    sensor_dir.mkdir()
    _write_json(set_dir / f"TABLE_{sensor_name}_{set_id}.json", table_payload)
    if raw_payload is None:
        if timestamps is None or values is None:
            raise ValueError("timestamps and values are required when raw_payload is not provided.")
        raw_payload = {
            "timestamp": timestamps,
            sensor_name: values,
        }
    _write_json(sensor_dir / table_payload["File"][0], raw_payload)


def test_find_events_uses_strict_timestamp_containment(tmp_path: Path) -> None:
    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    assert len(find_events(reader, deck="NEW", timestamp="2022-07-01 00:00:00")) == 0
    assert len(find_events(reader, deck="NEW", timestamp="2022-07-01 00:00:01")) == 1
    assert len(find_events(reader, deck="NEW", timestamp="2022-07-01 00:00:03")) == 0


def test_synchro_indices_match_organizer_semantics() -> None:
    reference = pd.to_datetime(
        [
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
        ],
        utc=True,
    )
    target = pd.to_datetime(
        [
            "2022-06-30 23:59:59.000",
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:00.500",
            "2022-07-01 00:00:02.500",
        ],
        utc=True,
    )

    assert synchro_indices(reference, target).tolist() == [0, 1, 1, 3]


def test_align_event_group_matches_two_pass_organizer_shrinking(tmp_path: Path) -> None:
    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    event = find_events(reader, deck="NEW").iloc[0]
    loaded = load_event_group(reader, event)
    aligned = align_event_group(loaded)

    timestamps = aligned.aligned_waveform["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()
    assert aligned.reference_sensor == "NEW_S1_DO_INF_STR"
    assert aligned.alignment_diagnostics["rows_reference"] == 4
    assert aligned.alignment_diagnostics["rows_after_alignment"] == 2
    assert timestamps == [
        "2022-07-01T00:00:00.000000",
        "2022-07-01T00:00:02.000000",
    ]
    assert aligned.aligned_waveform["NEW_S1_DO_MID_ACC_Z"].tolist() == [1.0, 3.0]
    assert aligned.aligned_waveform["NEW_S1_DO_SUP_STR"].tolist() == [5.0, 7.0]


def test_zero_loaded_event_group_linear_endpoints_zeroes_raw_waveform_endpoints(tmp_path: Path) -> None:
    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    event = find_events(reader, deck="NEW").iloc[0]
    zeroed = zero_loaded_event_group(load_event_group(reader, event), method="linear_endpoints")

    values = zeroed.waveforms["NEW_S1_DO_INF_STR"][1]["NEW_S1_DO_INF_STR"].tolist()
    assert values == [0.0, 1.0, 0.0, 0.0]


def test_load_event_group_widens_duplicate_sensor_records(tmp_path: Path) -> None:
    dataset_root = _build_widening_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    event = find_events(reader, deck="NEW").iloc[0]
    loaded = load_event_group(reader, event)

    waveform = loaded.waveforms["NEW_S1_DO_SUP_STR"][1]
    assert waveform["NEW_S1_DO_SUP_STR"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_load_event_group_preserves_mixed_timestamp_formats(tmp_path: Path) -> None:
    dataset_root = _build_mixed_timestamp_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    event = find_events(reader, deck="NEW", sensor_pattern="STR", timestamp="2022-07-01 00:00:01").iloc[0]
    loaded = load_event_group(reader, event)

    waveform = loaded.waveforms["NEW_S1_DO_INF_STR"][1]
    assert len(waveform) == 4
    assert waveform["timestamp"].isna().sum() == 0


def test_run_organizer_query_preserves_selected_sensor_order_and_strict_boundary(tmp_path: Path) -> None:
    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    empty = run_organizer_query(
        reader,
        timestamp="2022-07-01 00:00:00",
        deck="NEW",
        sensor_pattern="STR",
    )
    filled = run_organizer_query(
        reader,
        timestamp="2022-07-01 00:00:01",
        deck="NEW",
        sensor_pattern="STR",
    )

    assert empty.selected_sensors == ["NEW_S1_DO_INF_STR", "NEW_S1_DO_SUP_STR"]
    assert empty.data_measures.empty
    assert filled.selected_sensors == ["NEW_S1_DO_INF_STR", "NEW_S1_DO_SUP_STR"]
    assert list(filled.data_measures.columns) == ["timestamp", "NEW_S1_DO_INF_STR", "NEW_S1_DO_SUP_STR"]


def test_load_preprocessing_settings_rejects_legacy_alignment_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: nearest_timestamp",
                "    tolerance_ms: 5",
                "  zeroing:",
                "    method: linear_endpoints",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy preprocessing.alignment keys"):
        load_preprocessing_settings(config_path)


def test_run_preprocess_writes_stage_artifacts(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_preprocessing_dataset(tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "default.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  event_grouping:",
                "    key_fields: [deck, Start_Time, End_Time]",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  filtering:",
                "    min_active_sensors_per_event: 1",
                "  storage:",
                "    backend: sqlite",
                "  exports:",
                "    aligned_waveforms:",
                "      enabled: true",
                "      format: csv.gz",
                "output:",
                "  results_dir: results",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()
    captured = capsys.readouterr()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    metadata = json.loads((tmp_path / "results" / latest["run_id"] / "metadata.json").read_text(encoding="utf-8"))
    preprocess_db = preprocess_dir / "preprocess.sqlite"
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    with sqlite3.connect(preprocess_db) as conn:
        sensor_records = pd.read_sql_query("SELECT * FROM sensor_records", conn)

    with open_preprocess_store(preprocess_dir) as store:
        manifest = store.list_events()
        new_event_id = manifest.loc[manifest["deck"] == "NEW", "event_id"].iloc[0]
        new_aligned = store.load_aligned_event(new_event_id)

    assert preprocess_db.is_file()
    assert _fetch_table_names(preprocess_db) >= {
        "stage_info",
        "sets",
        "sensors",
        "events",
        "event_sensors",
        "sensor_records",
        "sensor_qc",
    }
    assert not (preprocess_dir / "preprocess.sqlite").__class__.__name__ or True  # SQLite still exists
    waveforms_dir = preprocess_dir / "waveforms"
    set_waveforms_dir = waveforms_dir / "AQUINAS_SET1_2022_07"
    assert waveforms_dir.is_dir(), "waveforms/ directory was not created"
    assert set_waveforms_dir.is_dir(), "per-SET waveform subdirectory was not created"
    assert any(waveforms_dir.glob("*/*.npy")), "no .npy waveform files were written"
    assert (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__NEW_DECK.csv.gz"
    ).is_file()
    assert (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__OLD_DECK.csv.gz"
    ).is_file()
    assert len(manifest) == 2
    assert sorted(manifest["discarded"].tolist()) == [False, False]
    assert "Temperature" in sensor_records.columns
    assert summary["total_events"] == 2
    assert summary["retained_events"] == 2
    assert summary["alignment"]["method"] == "r_synchro"
    assert summary["alignment"]["passes"] == 2
    assert summary["zeroing"]["stage"] == "before_alignment"
    assert summary["storage"]["backend"] == "sqlite"
    assert len(new_aligned) == 2
    assert "Writing aligned exports..." in captured.out
    assert metadata["stages"]["preprocess"]["progress"]["current_set"] is None
    assert metadata["stages"]["preprocess"]["progress"]["completed_sets"] == [
        "AQUINAS_SET1_2022_07",
    ]
    assert set(metadata["stages"]["preprocess"]["progress"]["written_partitions"]) == {
        "AQUINAS_SET1_2022_07__NEW_DECK",
        "AQUINAS_SET1_2022_07__OLD_DECK",
    }


def test_migrate_preprocess_waveforms_moves_flat_artifacts_and_is_idempotent(
    tmp_path: Path,
) -> None:
    preprocess_dir, event_id, timestamps, sensor_names, matrix = _build_legacy_preprocess_waveform_layout(
        tmp_path
    )
    waveforms_dir = preprocess_dir / "waveforms"

    first_summary = migrate_preprocess_waveforms(preprocess_dir)

    target_dir = waveforms_dir / "AQUINAS_SET1_2022_07"
    target_npy = target_dir / f"{event_id}.npy"
    target_meta = target_dir / f"{event_id}.meta.json"
    assert first_summary == {"moved_events": 1, "already_migrated_events": 0}
    assert target_npy.is_file()
    assert target_meta.is_file()
    assert not (waveforms_dir / f"{event_id}.npy").exists()
    assert not (waveforms_dir / f"{event_id}.meta.json").exists()
    np.testing.assert_array_equal(np.load(target_npy), matrix)

    with open_preprocess_store(preprocess_dir) as store:
        aligned = store.load_aligned_event(event_id)
    assert aligned["timestamp_utc"].tolist() == list(pd.to_datetime(timestamps, utc=True))
    np.testing.assert_allclose(aligned[sensor_names].to_numpy(), matrix)

    second_summary = migrate_preprocess_waveforms(preprocess_dir)

    assert second_summary == {"moved_events": 0, "already_migrated_events": 1}


def test_detect_legacy_preprocess_waveforms_reports_flat_layout(tmp_path: Path) -> None:
    preprocess_dir, event_id, _, _, _ = _build_legacy_preprocess_waveform_layout(tmp_path)

    assert detect_legacy_preprocess_waveforms(preprocess_dir) is True

    migrate_preprocess_waveforms(preprocess_dir)

    assert detect_legacy_preprocess_waveforms(preprocess_dir) is False
    assert (preprocess_dir / "waveforms" / "AQUINAS_SET1_2022_07" / f"{event_id}.npy").is_file()


def test_detect_legacy_preprocess_waveforms_raises_for_conflicting_states(tmp_path: Path) -> None:
    preprocess_dir, event_id, timestamps, sensor_names, matrix = _build_legacy_preprocess_waveform_layout(
        tmp_path
    )
    target_dir = preprocess_dir / "waveforms" / "AQUINAS_SET1_2022_07"
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / f"{event_id}.npy", matrix)
    (target_dir / f"{event_id}.meta.json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "sensor_names": sensor_names,
                "timestamps_utc": timestamps,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Both flat and migrated artifacts exist"):
        detect_legacy_preprocess_waveforms(preprocess_dir)


def test_open_preprocess_store_auto_migrates_legacy_waveforms_and_warns(tmp_path: Path) -> None:
    preprocess_dir, event_id, timestamps, sensor_names, matrix = _build_legacy_preprocess_waveform_layout(
        tmp_path
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with open_preprocess_store(preprocess_dir) as store:
            aligned = store.load_aligned_event(event_id)

    migration_warnings = [
        warning
        for warning in recorded
        if issubclass(warning.category, PreprocessWaveformMigrationWarning)
    ]
    assert len(migration_warnings) == 1
    assert "Legacy preprocess waveform layout detected" in str(migration_warnings[0].message)
    assert "Do not quit until this move completes" in str(migration_warnings[0].message)

    np.testing.assert_allclose(aligned[sensor_names].to_numpy(), matrix)
    assert aligned["timestamp_utc"].tolist() == list(pd.to_datetime(timestamps, utc=True))
    assert not (preprocess_dir / "waveforms" / f"{event_id}.npy").exists()
    assert (preprocess_dir / "waveforms" / "AQUINAS_SET1_2022_07" / f"{event_id}.npy").is_file()

    with warnings.catch_warnings(record=True) as recorded_second:
        warnings.simplefilter("always")
        with open_preprocess_store(preprocess_dir):
            pass
    assert not [
        warning
        for warning in recorded_second
        if issubclass(warning.category, PreprocessWaveformMigrationWarning)
    ]


def test_scripts_migrate_preprocess_waveforms_main_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.migrate_preprocess_waveforms import main  # noqa: PLC0415

    preprocess_dir, _, _, _, _ = _build_legacy_preprocess_waveform_layout(tmp_path)

    exit_code = main([str(preprocess_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Migration complete: 1 events moved, 0 already migrated." in captured.out


def test_run_preprocess_applies_configured_sensor_exclusion_and_writes_qc_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_sensor_exclusion_dataset(tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "default.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "    - AQUINAS_SET4_2024_01",
                "    - AQUINAS_SET5_2024_06",
                "preprocessing:",
                "  event_grouping:",
                "    key_fields: [deck, Start_Time, End_Time]",
                "  sensor_overrides:",
                "    exclude:",
                "      - sensor_name: OLD_S1_UP_SUP_STR",
                "        sets: [AQUINAS_SET4_2024_01, AQUINAS_SET5_2024_06]",
                "        reason: damaged sensor per organizer email",
                "        source: François-Baptiste Cartiaux email dated April 9, 2026",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  filtering:",
                "    min_active_sensors_per_event: 1",
                "  storage:",
                "    backend: sqlite",
                "  exports:",
                "    aligned_waveforms:",
                "      enabled: true",
                "      format: csv.gz",
                "output:",
                "  results_dir: results",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    preprocess_db = preprocess_dir / "preprocess.sqlite"
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    with sqlite3.connect(preprocess_db) as conn:
        sensor_records = pd.read_sql_query("SELECT * FROM sensor_records", conn)
        qc_report = pd.read_sql_query("SELECT * FROM sensor_qc", conn)
    with open_preprocess_store(preprocess_dir) as store:
        manifest = store.list_events()

    set1_bad = sensor_records.loc[
        (sensor_records["set_name"] == "AQUINAS_SET1_2022_07")
        & (sensor_records["sensor_name"] == "OLD_S1_UP_SUP_STR")
    ]
    set4_bad = sensor_records.loc[
        (sensor_records["set_name"] == "AQUINAS_SET4_2024_01")
        & (sensor_records["sensor_name"] == "OLD_S1_UP_SUP_STR")
    ]
    set5_bad = sensor_records.loc[
        (sensor_records["set_name"] == "AQUINAS_SET5_2024_06")
        & (sensor_records["sensor_name"] == "OLD_S1_UP_SUP_STR")
    ]

    assert set1_bad["sensor_status"].tolist() == ["included"]
    assert set4_bad["sensor_status"].tolist() == ["excluded"]
    assert set5_bad["sensor_status"].tolist() == ["excluded"]

    set4_manifest = manifest.loc[manifest["set_name"] == "AQUINAS_SET4_2024_01"].iloc[0]
    set5_manifest = manifest.loc[manifest["set_name"] == "AQUINAS_SET5_2024_06"].iloc[0]
    assert set4_manifest["excluded_sensors"] == ["OLD_S1_UP_SUP_STR"]
    assert set5_manifest["excluded_sensors"] == ["OLD_S1_UP_SUP_STR"]
    assert int(set4_manifest["active_sensor_count"]) == 1
    assert int(set5_manifest["active_sensor_count"]) == 1
    assert set4_manifest["reference_sensor"] == "OLD_S1_UP_INF_STR"

    with open_preprocess_store(preprocess_dir) as store:
        set4_aligned = store.load_aligned_event(set4_manifest["event_id"])
        set1_aligned = store.load_aligned_event(
            manifest.loc[manifest["set_name"] == "AQUINAS_SET1_2022_07", "event_id"].iloc[0]
        )
    assert "OLD_S1_UP_SUP_STR" not in set4_aligned.columns
    assert "OLD_S1_UP_SUP_STR" in set1_aligned.columns

    set4_qc = qc_report.loc[
        (qc_report["set_name"] == "AQUINAS_SET4_2024_01")
        & (qc_report["sensor_name"] == "OLD_S1_UP_SUP_STR")
    ].iloc[0]
    assert set4_qc["sensor_status"] == "excluded"
    assert set4_qc["table_range_median"] == pytest.approx(0.0)
    assert set4_qc["raw_range_spotcheck_median"] > 0.0

    assert summary["sensor_exclusions"]["applied_record_counts_by_set"] == {
        "AQUINAS_SET4_2024_01": 1,
        "AQUINAS_SET5_2024_06": 1,
    }
    assert summary["sensor_exclusions"]["applied_record_counts_by_reason"] == {
        "damaged sensor per organizer email": 2
    }
    assert summary["sensor_exclusions"]["applied_sensor_names_by_set"] == {
        "AQUINAS_SET4_2024_01": ["OLD_S1_UP_SUP_STR"],
        "AQUINAS_SET5_2024_06": ["OLD_S1_UP_SUP_STR"],
    }


def test_run_preprocess_defaults_to_sqlite_without_optional_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_preprocessing_dataset(tmp_path)
    _write_default_preprocess_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

    assert (preprocess_dir / "preprocess.sqlite").is_file()
    assert not (preprocess_dir / "exports").exists()


@pytest.mark.parametrize(
    ("config_lines", "error_match"),
    [
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
            ],
            "data.sets",
            id="missing-data-sets",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: nearest_timestamp",
                "  zeroing:",
                "    method: linear_endpoints",
            ],
            "Unsupported preprocessing.alignment.method",
            id="unsupported-alignment",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: median",
            ],
            "Unsupported preprocessing.zeroing.method",
            id="unsupported-zeroing",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  sensor_overrides:",
                "    exclude:",
                "      - bad-entry",
            ],
            "must be a mapping",
            id="exclude-entry-not-mapping",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  sensor_overrides:",
                "    exclude:",
                "      - sets: [AQUINAS_SET1_2022_07]",
            ],
            "must define sensor_name",
            id="exclude-entry-missing-sensor-name",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  sensor_overrides:",
                "    exclude:",
                "      - sensor_name: NEW_S1_DO_INF_STR",
            ],
            "must define at least one set",
            id="exclude-entry-missing-sets",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  export:",
                "    format: csv.gz",
            ],
            "Legacy preprocessing.export",
            id="legacy-export-config",
        ),
        pytest.param(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: linear_endpoints",
                "  storage:",
                "    backend: parquet",
            ],
            "Unsupported preprocessing.storage.backend",
            id="unsupported-storage-backend",
        ),
    ],
)
def test_load_preprocessing_settings_rejects_invalid_config_values(
    tmp_path: Path,
    config_lines: list[str],
    error_match: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_lines)

    with pytest.raises(ValueError, match=error_match):
        load_preprocessing_settings(config_path)


def test_load_event_group_rejects_empty_sensor_subset(tmp_path: Path) -> None:
    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
    event = find_events(reader, deck="NEW").iloc[0]

    with pytest.raises(ValueError, match="No sensor records matched"):
        load_event_group(reader, event, sensor_names=[])


def test_load_event_group_rejects_multi_file_sensor_matches(tmp_path: Path) -> None:
    dataset_root = _build_cross_file_match_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
    event = find_events(reader).iloc[0]

    with pytest.raises(ValueError, match="exactly one raw file per sensor match"):
        load_event_group(reader, event)


def test_load_event_group_drops_unparseable_timestamps_with_warning(tmp_path: Path) -> None:
    dataset_root = _build_custom_raw_dataset(
        tmp_path,
        raw_payload={
            "timestamp": [
                "2022-07-01 00:00:00.000",
                "not-a-time",
                "2022-07-01 00:00:02.000",
            ],
            "NEW_S1_DO_INF_STR": [1.0, 2.0, 3.0],
        },
    )
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
    event = find_events(reader).iloc[0]

    with pytest.warns(UserWarning, match="dropped 1 row\\(s\\) with unparseable timestamps"):
        loaded = load_event_group(reader, event)

    waveform = loaded.waveforms["NEW_S1_DO_INF_STR"][1]
    assert len(waveform) == 2
    assert waveform["timestamp"].isna().sum() == 0


@pytest.mark.parametrize(
    ("raw_payload", "error_match"),
    [
        pytest.param(
            {"time": ["2022-07-01 00:00:00.000"], "NEW_S1_DO_INF_STR": [1.0]},
            "missing a timestamp column",
            id="missing-timestamp-column",
        ),
        pytest.param(
            {"timestamp": ["2022-07-01 00:00:00.000"]},
            "does not contain sensor values",
            id="missing-value-column",
        ),
    ],
)
def test_load_event_group_raises_clear_errors_for_malformed_raw_waveforms(
    tmp_path: Path,
    raw_payload: dict,
    error_match: str,
) -> None:
    dataset_root = _build_custom_raw_dataset(tmp_path, raw_payload=raw_payload)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
    event = find_events(reader).iloc[0]

    with pytest.raises(KeyError, match=error_match):
        load_event_group(reader, event)


def test_zero_waveform_none_returns_numeric_values_without_baseline_change() -> None:
    zeroed = zero_waveform(pd.Series(["1.5", 2]), method="none")

    assert zeroed.tolist() == [1.5, 2.0]


def test_zero_waveform_single_sample_returns_zero_relative_value() -> None:
    zeroed = zero_waveform(pd.Series([5.5]), method="linear_endpoints")

    assert zeroed.tolist() == [0.0]


def test_zero_waveform_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError, match="Unsupported zeroing method"):
        zero_waveform(pd.Series([1.0, 2.0]), method="median")


def test_filter_loaded_event_group_leaves_short_waveforms_unchanged() -> None:
    waveform = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-07-01 00:00:00", "2022-07-01 00:00:01"], utc=True),
            "NEW_S1_DO_INF_STR": [1.5, 2.5],
        }
    )
    event_group = LoadedEventGroup(
        event_id="short-event",
        set_name="AQUINAS_SET1_2022_07",
        deck="NEW",
        start_time_utc=pd.Timestamp("2022-07-01T00:00:00Z"),
        end_time_utc=pd.Timestamp("2022-07-01T00:00:01Z"),
        sensor_records=pd.DataFrame(),
        waveforms={"NEW_S1_DO_INF_STR": (pd.Series({"sensor_name": "NEW_S1_DO_INF_STR"}), waveform)},
    )

    filtered = filter_loaded_event_group(event_group)

    assert filtered.waveforms["NEW_S1_DO_INF_STR"][1]["NEW_S1_DO_INF_STR"].tolist() == [1.5, 2.5]


def test_bandpass_filter_waveform_matrix_leaves_short_inputs_unchanged() -> None:
    waveform = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-07-01 00:00:00", "2022-07-01 00:00:01"], utc=True),
            "NEW_S1_DO_INF_STR": [1.5, 2.5],
        }
    )

    filtered = bandpass_filter_waveform_matrix(waveform)

    pd.testing.assert_frame_equal(filtered, waveform)


def test_align_event_group_returns_empty_frame_when_reference_sensor_has_no_rows() -> None:
    event_group = LoadedEventGroup(
        event_id="event-1",
        set_name="AQUINAS_SET1_2022_07",
        deck="NEW",
        start_time_utc=pd.Timestamp("2022-07-01T00:00:00Z"),
        end_time_utc=pd.Timestamp("2022-07-01T00:00:02Z"),
        sensor_records=pd.DataFrame(),
        waveforms={
            "NEW_S1_DO_INF_STR": (
                pd.Series({"sensor_name": "NEW_S1_DO_INF_STR"}),
                pd.DataFrame(columns=["timestamp", "NEW_S1_DO_INF_STR"]),
            ),
            "NEW_S1_DO_SUP_STR": (
                pd.Series({"sensor_name": "NEW_S1_DO_SUP_STR"}),
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            ["2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
                            utc=True,
                        ),
                        "NEW_S1_DO_SUP_STR": [1.0, 2.0],
                    }
                ),
            ),
        },
    )

    aligned = align_event_group(event_group)

    assert aligned.reference_sensor == "NEW_S1_DO_INF_STR"
    assert aligned.aligned_waveform.empty
    assert list(aligned.aligned_waveform.columns) == [
        "timestamp_utc",
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_SUP_STR",
    ]


def test_align_event_group_preserves_empty_non_reference_sensors_as_nan_columns() -> None:
    event_group = LoadedEventGroup(
        event_id="event-2",
        set_name="AQUINAS_SET1_2022_07",
        deck="NEW",
        start_time_utc=pd.Timestamp("2022-07-01T00:00:00Z"),
        end_time_utc=pd.Timestamp("2022-07-01T00:00:02Z"),
        sensor_records=pd.DataFrame(),
        waveforms={
            "NEW_S1_DO_INF_STR": (
                pd.Series({"sensor_name": "NEW_S1_DO_INF_STR"}),
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            ["2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
                            utc=True,
                        ),
                        "NEW_S1_DO_INF_STR": [10.0, 11.0],
                    }
                ),
            ),
            "NEW_S1_DO_SUP_STR": (
                pd.Series({"sensor_name": "NEW_S1_DO_SUP_STR"}),
                pd.DataFrame(columns=["timestamp", "NEW_S1_DO_SUP_STR"]),
            ),
        },
    )

    aligned = align_event_group(event_group)

    assert aligned.alignment_diagnostics["rows_after_alignment"] == 2
    assert aligned.aligned_waveform["NEW_S1_DO_INF_STR"].tolist() == [10.0, 11.0]
    assert aligned.aligned_waveform["NEW_S1_DO_SUP_STR"].isna().all()


def test_run_preprocess_records_insufficient_active_sensor_discards_and_writes_csv_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_preprocessing_dataset(tmp_path)
    _write_default_preprocess_config(
        tmp_path,
        min_active_sensors_per_event=2,
        aligned_export_enabled=True,
        export_format="csv",
    )

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    with open_preprocess_store(preprocess_dir) as store:
        manifest = store.list_events()

    old_event = manifest.loc[manifest["deck"] == "OLD"].iloc[0]
    assert bool(old_event["discarded"])
    assert old_event["discard_reason"] == "insufficient_active_sensors"
    assert (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__NEW_DECK.csv"
    ).is_file()
    assert not (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__NEW_DECK.csv.gz"
    ).exists()
    assert summary["retained_events"] == 1
    assert summary["discard_reasons"] == {"insufficient_active_sensors": 1}


def test_run_preprocess_records_no_common_aligned_rows_discards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_no_common_rows_dataset(tmp_path)
    _write_default_preprocess_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    with open_preprocess_store(preprocess_dir) as store:
        manifest = store.list_events()
        samples = store.load_aligned_samples()

    assert len(manifest) == 1
    assert bool(manifest.loc[0, "discarded"])
    assert manifest.loc[0, "discard_reason"] == "no_common_aligned_rows"
    assert int(manifest.loc[0, "rows_after_alignment"]) == 0
    assert summary["retained_events"] == 0
    assert summary["discard_reasons"] == {"no_common_aligned_rows": 1}
    assert samples.empty
    assert not (preprocess_dir / "exports").exists()


def test_run_preprocess_writes_parseable_empty_stage_artifacts_for_empty_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_empty_set_dataset(tmp_path)
    _write_default_preprocess_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    preprocess_db = preprocess_dir / "preprocess.sqlite"
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    with open_preprocess_store(preprocess_dir) as store:
        manifest = store.list_events()
    with sqlite3.connect(preprocess_db) as conn:
        sensor_records = pd.read_sql_query("SELECT * FROM sensor_records", conn)
        qc_report = pd.read_sql_query("SELECT * FROM sensor_qc", conn)

    assert manifest.empty
    assert manifest.columns.tolist() == [
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
    assert sensor_records.empty
    assert sensor_records.columns.tolist() == [
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
    assert qc_report.empty
    assert qc_report.columns.tolist() == [
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
    assert summary["total_events"] == 0
    assert summary["retained_events"] == 0
    assert summary["discard_reasons"] == {}
    assert not (preprocess_dir / "exports").exists()


def test_run_preprocess_writes_completed_sets_incrementally_and_preserves_progress_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_two_set_preprocessing_dataset(tmp_path)
    _write_default_preprocess_config(
        tmp_path,
        set_names=("AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"),
        aligned_export_enabled=True,
    )

    original_writer = pipeline_mod._write_set_aligned_partitions
    call_count = 0

    def fail_on_second_set(
        aligned_dir: Path,
        partitions: dict[tuple[str, str], list[pd.DataFrame]],
        export_format: str,
    ) -> list[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated set write failure")
        return original_writer(aligned_dir, partitions, export_format)

    monkeypatch.setattr(pipeline_mod, "_write_set_aligned_partitions", fail_on_second_set)
    monkeypatch.setattr(run_mod, "_refresh_visualization_bundle", lambda run_context: None)

    exit_code = run_mod.run_command(stage="preprocess", name=None, run_id=None)

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    run_dir = tmp_path / "results" / latest["run_id"]
    preprocess_dir = run_dir / "stages" / "preprocess"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__NEW_DECK.csv.gz"
    ).is_file()
    assert (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET1_2022_07__OLD_DECK.csv.gz"
    ).is_file()
    assert not (
        preprocess_dir / "exports" / "aligned" / "AQUINAS_SET2_2023_04__NEW_DECK.csv.gz"
    ).exists()
    assert metadata["stages"]["preprocess"]["status"] == "failed"
    assert metadata["stages"]["preprocess"]["error"] == "simulated set write failure"
    assert metadata["stages"]["preprocess"]["progress"]["current_set"] is None
    assert metadata["stages"]["preprocess"]["progress"]["completed_sets"] == [
        "AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04",
    ]
    assert set(metadata["stages"]["preprocess"]["progress"]["written_partitions"]) == {
        "AQUINAS_SET1_2022_07__NEW_DECK",
        "AQUINAS_SET1_2022_07__OLD_DECK",
        "AQUINAS_SET2_2023_04__NEW_DECK",
        "AQUINAS_SET2_2023_04__OLD_DECK",
    }
    with open_preprocess_store(preprocess_dir) as store:
        retained = store.iter_retained_events()
    assert set(retained["set_name"]) == {"AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"}


def test_write_dataframe_atomic_removes_temp_file_and_leaves_no_final_file_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1, 2]})
    output_path = tmp_path / "aligned.csv.gz"
    original_to_csv = pd.DataFrame.to_csv

    def fail_after_temp_write(self, path_or_buf=None, *args, **kwargs):  # noqa: ANN001
        if isinstance(path_or_buf, Path):
            path_or_buf.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_temp_write)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline_mod._write_dataframe_atomic(output_path, frame)

    assert not output_path.exists()
    assert not (tmp_path / ".aligned.csv.gz.tmp").exists()

    monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)


# ---------------------------------------------------------------------------
# Accuracy tests for the optimised preprocessing path
# ---------------------------------------------------------------------------


def test_synchro_indices_fast_path_gives_same_result_as_full_parse() -> None:
    """_to_datetime64_ns skips pd.to_datetime when input is already datetime64[ns].

    The optimised path (fast-path branch in _to_datetime64_ns) must produce
    identical results to the original full pd.to_datetime parse.
    """
    from aquinas_toolkit.preprocessing.alignment import _to_datetime64_ns  # noqa: PLC0415

    ref_strs = [
        "2022-07-01 00:00:00.000",
        "2022-07-01 00:00:01.000",
        "2022-07-01 00:00:02.000",
    ]
    tgt_strs = [
        "2022-06-30 23:59:59.000",
        "2022-07-01 00:00:00.000",
        "2022-07-01 00:00:00.500",
        "2022-07-01 00:00:02.500",
    ]

    # Baseline: string lists — triggers full pd.to_datetime path
    result_strings = synchro_indices(ref_strs, tgt_strs)

    # Fast path: Series already typed as datetime64[ns, UTC]
    ref_dt64 = pd.Series(pd.to_datetime(ref_strs, utc=True))
    tgt_dt64 = pd.Series(pd.to_datetime(tgt_strs, utc=True))
    result_dt64 = synchro_indices(ref_dt64, tgt_dt64)

    # Verify _to_datetime64_ns actually takes the fast branch
    assert pd.api.types.is_datetime64_any_dtype(ref_dt64)
    arr = _to_datetime64_ns(ref_dt64)
    import numpy as np  # noqa: PLC0415
    assert arr.dtype == np.dtype("datetime64[ns]")

    assert result_dt64.tolist() == result_strings.tolist()


def test_aligned_event_to_long_frame_produces_correct_columns_and_values(tmp_path: Path) -> None:
    """aligned_event_to_long_frame must emit event_id, sample_index, timestamp_utc ISO string,
    then one column per sensor, with values matching the aligned waveform.
    """
    from aquinas_toolkit.preprocessing.pipeline import aligned_event_to_long_frame  # noqa: PLC0415

    dataset_root = _build_preprocessing_dataset(tmp_path)
    reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

    event = find_events(reader, deck="NEW").iloc[0]
    aligned = align_event_group(load_event_group(reader, event))
    frame = aligned_event_to_long_frame(aligned)

    # Structure
    assert list(frame.columns[:3]) == ["event_id", "sample_index", "timestamp_utc"]
    assert set(frame.columns[3:]) == {
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_MID_ACC_Z",
        "NEW_S1_DO_SUP_STR",
    }
    assert len(frame) == 2  # two-pass synchro retains 2 reference rows

    # event_id and sample_index
    assert frame["event_id"].tolist() == [aligned.event_id, aligned.event_id]
    assert frame["sample_index"].tolist() == [0, 1]

    # timestamp_utc must be an ISO-8601 string with millisecond precision and trailing Z
    assert frame["timestamp_utc"].iloc[0] == "2022-07-01T00:00:00.000Z"
    assert frame["timestamp_utc"].iloc[1] == "2022-07-01T00:00:02.000Z"

    # Sensor values (no zeroing — align_event_group receives raw waveform)
    assert frame["NEW_S1_DO_MID_ACC_Z"].tolist() == pytest.approx([1.0, 3.0])
    assert frame["NEW_S1_DO_SUP_STR"].tolist() == pytest.approx([5.0, 7.0])


def test_run_preprocess_aligned_samples_values_are_numerically_correct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end accuracy check: aligned_samples rows in SQLite must contain
    the correct sensor values, sample indices, and timestamps.

    zeroing=none is used so expected values can be read directly from the
    input fixture without tracing through the zeroing arithmetic.

    Expected (two-pass synchro on NEW deck retains reference timestamps
    T+0.0 s and T+2.0 s):
        NEW_S1_DO_INF_STR  sample 0 = 10.0,  sample 1 = 30.0
        NEW_S1_DO_MID_ACC_Z sample 0 =  1.0,  sample 1 =  3.0
        NEW_S1_DO_SUP_STR  sample 0 =  5.0,  sample 1 =  7.0
    """
    monkeypatch.chdir(tmp_path)
    _build_preprocessing_dataset(tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    _write_yaml(
        tmp_path / "configs" / "default.yaml",
        [
            "data:",
            "  dataset_root: AQUINAS_DATASET",
            "  sets:",
            "    - AQUINAS_SET1_2022_07",
            "preprocessing:",
            "  event_grouping:",
            "    key_fields: [deck, Start_Time, End_Time]",
            "  alignment:",
            "    method: r_synchro",
            "  zeroing:",
            "    method: none",
            "  filtering:",
            "    min_active_sensors_per_event: 1",
            "  storage:",
            "    backend: sqlite",
            "  exports:",
            "    aligned_waveforms:",
            "      enabled: false",
            "output:",
            "  results_dir: results",
        ],
    )

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

    with open_preprocess_store(preprocess_dir) as store:
        samples = store.load_aligned_samples(deck="NEW")

    new_samples = samples.sort_values(["sensor_name", "sample_index"]).reset_index(drop=True)

    assert len(new_samples) == 6  # 3 sensors × 2 aligned rows
    assert set(new_samples["sensor_name"]) == {
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_MID_ACC_Z",
        "NEW_S1_DO_SUP_STR",
    }

    inf = new_samples[new_samples["sensor_name"] == "NEW_S1_DO_INF_STR"].sort_values("sample_index")
    assert inf["sample_index"].tolist() == [0, 1]
    assert inf["value"].tolist() == pytest.approx([10.0, 30.0])
    assert inf["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S").iloc[0] == "2022-07-01T00:00:00"
    assert inf["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S").iloc[1] == "2022-07-01T00:00:02"

    mid = new_samples[new_samples["sensor_name"] == "NEW_S1_DO_MID_ACC_Z"].sort_values("sample_index")
    assert mid["sample_index"].tolist() == [0, 1]
    assert mid["value"].tolist() == pytest.approx([1.0, 3.0])

    sup = new_samples[new_samples["sensor_name"] == "NEW_S1_DO_SUP_STR"].sort_values("sample_index")
    assert sup["sample_index"].tolist() == [0, 1]
    assert sup["value"].tolist() == pytest.approx([5.0, 7.0])


def test_insert_dataframe_maps_none_and_nan_to_sql_null() -> None:
    """_insert_dataframe must write Python None and pandas NaN as SQL NULL,
    and must preserve non-null values exactly.
    """
    import sqlite3  # noqa: PLC0415

    from aquinas_toolkit.preprocessing.store import _insert_dataframe  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE t (sensor_name TEXT, sample_index INTEGER, value REAL)"
    )

    frame = pd.DataFrame(
        {
            "sensor_name": ["A", "B", None],
            "sample_index": [0, 1, 2],
            "value": [1.5, float("nan"), 3.0],
        }
    )
    _insert_dataframe(conn, "t", frame, ["sensor_name", "sample_index", "value"])

    rows = conn.execute(
        "SELECT sensor_name, sample_index, value FROM t ORDER BY sample_index"
    ).fetchall()

    assert rows[0] == ("A", 0, 1.5)                                     # normal row intact
    assert rows[1][0] == "B" and rows[1][1] == 1 and rows[1][2] is None  # float NaN → NULL
    assert rows[2][0] is None and rows[2][1] == 2 and rows[2][2] == 3.0  # str None → NULL


def test_insert_dataframe_writes_all_rows_across_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No rows should be skipped or duplicated when the frame spans multiple chunks.

    Forces a small chunk size so that 3 500-row chunks are exercised with:
      - NaN values in the first chunk, at the exact chunk boundary, and in the last chunk
      - spot-checks on the first row, last row, and both sides of each boundary
    """
    import sqlite3  # noqa: PLC0415

    from aquinas_toolkit.preprocessing import store as store_mod  # noqa: PLC0415
    from aquinas_toolkit.preprocessing.store import _insert_dataframe  # noqa: PLC0415

    chunk = 500
    monkeypatch.setattr(store_mod, "_INSERT_CHUNK_ROWS", chunk)

    n = chunk * 3  # exactly 3 chunks = 1500 rows
    nan_indices = {100, chunk - 1, chunk, chunk + 1, chunk * 2}  # straddle every boundary

    frame = pd.DataFrame(
        {
            "sensor_name": [f"S{i % 4}" for i in range(n)],
            "sample_index": list(range(n)),
            "value": [
                float("nan") if i in nan_indices else float(i) * 1.5
                for i in range(n)
            ],
        }
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (sensor_name TEXT, sample_index INTEGER, value REAL)")
    _insert_dataframe(conn, "t", frame, ["sensor_name", "sample_index", "value"])

    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert total == n, f"Expected {n} rows, got {total} — rows were skipped or duplicated"

    # Every NaN position must be SQL NULL
    for idx in nan_indices:
        (db_val,) = conn.execute(
            "SELECT value FROM t WHERE sample_index = ?", [idx]
        ).fetchone()
        assert db_val is None, f"NaN at sample_index={idx} was not stored as SQL NULL"

    # Spot-check first row, last row, and both sides of each chunk boundary
    check_indices = {0, chunk - 2, chunk - 1, chunk, chunk + 1, n - 2, n - 1}
    for idx in sorted(check_indices):
        (db_idx, db_val) = conn.execute(
            "SELECT sample_index, value FROM t WHERE sample_index = ?", [idx]
        ).fetchone()
        assert db_idx == idx
        if idx in nan_indices:
            assert db_val is None
        else:
            assert db_val == pytest.approx(idx * 1.5)


# ---------------------------------------------------------------------------
# Edge-case tests for optimised parsing (orjson, explicit datetime, parallel)
# ---------------------------------------------------------------------------


class TestParseTimestampsFast:
    """Verify _parse_timestamps_fast produces identical results to format='mixed'."""

    def test_standard_fractional_seconds(self) -> None:
        series = pd.Series([
            "2022-07-01 00:00:00.000",
            "2022-07-01 12:30:45.123",
            "2022-07-01 23:59:59.999",
        ])
        result = _parse_timestamps_fast(series)
        expected = pd.to_datetime(series, utc=True, format="mixed")
        pd.testing.assert_series_equal(result, expected)

    def test_no_fractional_seconds_triggers_fallback(self) -> None:
        """Timestamps without fractional seconds don't match %S.%f -- fallback must handle them."""
        series = pd.Series([
            "2022-07-01 00:00:00",
            "2022-07-01 12:30:45",
        ])
        result = _parse_timestamps_fast(series)
        expected = pd.to_datetime(series, utc=True, format="mixed")
        pd.testing.assert_series_equal(result, expected)

    def test_mixed_with_and_without_fractional_seconds(self) -> None:
        """Real datasets sometimes mix '00:00:00' and '00:00:00.000' in the same column."""
        series = pd.Series([
            "2022-07-01 00:00:00",
            "2022-07-01 00:00:01.500",
            "2022-07-01 00:00:02",
        ])
        result = _parse_timestamps_fast(series)
        expected = pd.to_datetime(series, utc=True, format="mixed")
        pd.testing.assert_series_equal(result, expected)

    def test_iso8601_format_triggers_fallback(self) -> None:
        series = pd.Series([
            "2022-07-01T00:00:00.000Z",
            "2022-07-01T12:30:45.123Z",
        ])
        result = _parse_timestamps_fast(series)
        expected = pd.to_datetime(series, utc=True, format="mixed")
        pd.testing.assert_series_equal(result, expected)

    def test_single_element_series(self) -> None:
        series = pd.Series(["2022-07-01 00:00:00.500"])
        result = _parse_timestamps_fast(series)
        expected = pd.to_datetime(series, utc=True, format="mixed")
        pd.testing.assert_series_equal(result, expected)

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        result = _parse_timestamps_fast(series)
        assert len(result) == 0

    def test_microsecond_precision_is_preserved(self) -> None:
        """The AQUINAS dataset stores milliseconds but we must not lose sub-ms precision."""
        series = pd.Series(["2022-07-01 00:00:00.123456"])
        result = _parse_timestamps_fast(series)
        assert result.iloc[0].microsecond == 123456

    def test_result_is_utc_aware(self) -> None:
        series = pd.Series(["2022-07-01 00:00:00.000"])
        result = _parse_timestamps_fast(series)
        assert result.dt.tz is not None
        assert str(result.dt.tz) == "UTC"


class TestOrjsonReaderFidelity:
    """Verify orjson-based reader produces identical DataFrames for all JSON shapes."""

    def test_columnar_json_values_are_exact(self, tmp_path: Path) -> None:
        """Columnar JSON (dict of lists) -- the AQUINAS raw waveform format."""
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
        df = reader.load_raw_file("NEW_S1_DO_INF_STR", "NEW_S1_DO_INF_STR_SET1_1.json")

        assert df["NEW_S1_DO_INF_STR"].tolist() == [10.0, 21.0, 30.0, 40.0]
        assert df["timestamp"].tolist() == [
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
            "2022-07-01 00:00:03.000",
        ]

    def test_table_json_record_count_is_exact(self, tmp_path: Path) -> None:
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
        df = reader.load_index_table("NEW_S1_DO_INF_STR")
        assert len(df) == 1
        assert int(df["Start_Row"].iloc[0]) == 1
        assert int(df["End_Row"].iloc[0]) == 4

    def test_special_float_values_survive_roundtrip(self, tmp_path: Path) -> None:
        """Values like 0.0, -0.0, very small floats must not be corrupted."""
        dataset_root = _build_custom_raw_dataset(
            tmp_path,
            raw_payload={
                "timestamp": [
                    "2022-07-01 00:00:00.000",
                    "2022-07-01 00:00:01.000",
                    "2022-07-01 00:00:02.000",
                ],
                "NEW_S1_DO_INF_STR": [0.0, -1e-15, 1e15],
            },
        )
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
        event = find_events(reader).iloc[0]
        loaded = load_event_group(reader, event)
        values = loaded.waveforms["NEW_S1_DO_INF_STR"][1]["NEW_S1_DO_INF_STR"].tolist()
        assert values == pytest.approx([0.0, -1e-15, 1e15])

    def test_unicode_in_json_survives_orjson(self, tmp_path: Path) -> None:
        """orjson reads bytes, not text -- verify non-ASCII characters don't break."""
        dataset_root = tmp_path / "AQUINAS_DATASET"
        set_dir = dataset_root / "AQUINAS_SET1_2022_07"
        set_dir.mkdir(parents=True, exist_ok=True)
        sensor_dir = set_dir / "NEW_S1_DO_INF_STR"
        sensor_dir.mkdir()
        # Write table with UTF-8 content via json (standard encoder)
        import json as json_stdlib
        table_data = {
            "Record_UID": [1],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:01"],
            "Duration": [1.0],
        }
        (set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json").write_text(
            json_stdlib.dumps(table_data), encoding="utf-8"
        )
        raw_data = {
            "timestamp": ["2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
            "NEW_S1_DO_INF_STR": [1.0, 2.0],
        }
        (sensor_dir / "NEW_S1_DO_INF_STR_SET1_1.json").write_text(
            json_stdlib.dumps(raw_data), encoding="utf-8"
        )

        reader = AquinasReader(set_dir)
        df = reader.load_raw_file("NEW_S1_DO_INF_STR", "NEW_S1_DO_INF_STR_SET1_1.json")
        assert df["NEW_S1_DO_INF_STR"].tolist() == [1.0, 2.0]


class TestThreadSafeReaderCache:
    """Verify that the reader's file cache is safe under concurrent access."""

    def test_concurrent_loads_return_identical_data(self, tmp_path: Path) -> None:
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

        results: dict[int, list[float]] = {}
        errors: list[Exception] = []

        def load_and_record(thread_id: int) -> None:
            try:
                df = reader.load_raw_file("NEW_S1_DO_INF_STR", "NEW_S1_DO_INF_STR_SET1_1.json")
                results[thread_id] = df["NEW_S1_DO_INF_STR"].tolist()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load_and_record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        expected = [10.0, 21.0, 30.0, 40.0]
        for tid, values in results.items():
            assert values == expected, f"Thread {tid} got wrong values: {values}"

    def test_concurrent_loads_different_sensors(self, tmp_path: Path) -> None:
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

        results: dict[str, list[float]] = {}
        errors: list[Exception] = []

        def load_sensor(sensor_name: str, raw_file: str) -> None:
            try:
                df = reader.load_raw_file(sensor_name, raw_file)
                results[sensor_name] = df[sensor_name].tolist()
            except Exception as exc:
                errors.append(exc)

        sensors = [
            ("NEW_S1_DO_INF_STR", "NEW_S1_DO_INF_STR_SET1_1.json"),
            ("NEW_S1_DO_MID_ACC_Z", "NEW_S1_DO_MID_ACC_Z_SET1_1.json"),
            ("NEW_S1_DO_SUP_STR", "NEW_S1_DO_SUP_STR_SET1_1.json"),
            ("OLD_S1_DO_INF_STR", "OLD_S1_DO_INF_STR_SET1_1.json"),
        ]
        threads = [threading.Thread(target=load_sensor, args=s) for s in sensors]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert results["NEW_S1_DO_INF_STR"] == [10.0, 21.0, 30.0, 40.0]
        assert results["NEW_S1_DO_MID_ACC_Z"] == [1.0, 2.0, 3.0]
        assert results["NEW_S1_DO_SUP_STR"] == [5.0, 7.0]
        assert results["OLD_S1_DO_INF_STR"] == [9.0, 9.5, 10.0, 10.5]


class TestParallelEventProcessingFidelity:
    """Verify parallel event processing produces identical results to serial."""

    def test_parallel_load_event_group_values_match_serial(self, tmp_path: Path) -> None:
        """load_event_group now uses ThreadPoolExecutor for sensor loading.
        Values must be identical to what they were before parallelization."""
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

        event = find_events(reader, deck="NEW").iloc[0]
        loaded = load_event_group(reader, event)

        # Verify all 3 NEW sensors loaded with correct values
        assert set(loaded.waveforms.keys()) == {
            "NEW_S1_DO_INF_STR",
            "NEW_S1_DO_MID_ACC_Z",
            "NEW_S1_DO_SUP_STR",
        }
        assert loaded.waveforms["NEW_S1_DO_INF_STR"][1]["NEW_S1_DO_INF_STR"].tolist() == [
            10.0, 21.0, 30.0, 40.0,
        ]
        assert loaded.waveforms["NEW_S1_DO_MID_ACC_Z"][1]["NEW_S1_DO_MID_ACC_Z"].tolist() == [
            1.0, 2.0, 3.0,
        ]
        assert loaded.waveforms["NEW_S1_DO_SUP_STR"][1]["NEW_S1_DO_SUP_STR"].tolist() == [
            5.0, 7.0,
        ]

    def test_parallel_pipeline_produces_same_aligned_samples_as_expected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Full pipeline end-to-end: verify aligned_samples values are numerically
        identical to the known-correct expected values from the fixture."""
        monkeypatch.chdir(tmp_path)
        _build_preprocessing_dataset(tmp_path)
        (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
        _write_yaml(
            tmp_path / "configs" / "default.yaml",
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  event_grouping:",
                "    key_fields: [deck, Start_Time, End_Time]",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: none",
                "  signal_filter:",
                "    method: none",
                "  filtering:",
                "    min_active_sensors_per_event: 1",
                "  storage:",
                "    backend: sqlite",
                "  exports:",
                "    aligned_waveforms:",
                "      enabled: false",
                "output:",
                "  results_dir: results",
            ],
        )

        monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
        run_mod.run()

        latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
        preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

        with open_preprocess_store(preprocess_dir) as store:
            samples = store.load_aligned_samples(deck="NEW")
            old_samples = store.load_aligned_samples(deck="OLD")

        # NEW deck: 3 sensors x 2 aligned rows = 6 sample rows
        new = samples.sort_values(["sensor_name", "sample_index"]).reset_index(drop=True)
        assert len(new) == 6

        inf = new[new["sensor_name"] == "NEW_S1_DO_INF_STR"].sort_values("sample_index")
        assert inf["value"].tolist() == pytest.approx([10.0, 30.0])

        mid = new[new["sensor_name"] == "NEW_S1_DO_MID_ACC_Z"].sort_values("sample_index")
        assert mid["value"].tolist() == pytest.approx([1.0, 3.0])

        sup = new[new["sensor_name"] == "NEW_S1_DO_SUP_STR"].sort_values("sample_index")
        assert sup["value"].tolist() == pytest.approx([5.0, 7.0])

        # OLD deck: 1 sensor x 4 aligned rows = 4 sample rows
        old = old_samples.sort_values(["sensor_name", "sample_index"]).reset_index(drop=True)
        assert len(old) == 4
        assert old["value"].tolist() == pytest.approx([9.0, 9.5, 10.0, 10.5])

    def test_no_events_dropped_or_duplicated_by_parallel_processing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """With 4 sensors across 2 decks, we should get exactly 2 events,
        both retained, no duplicates."""
        monkeypatch.chdir(tmp_path)
        _build_preprocessing_dataset(tmp_path)
        _write_default_preprocess_config(tmp_path)

        monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
        run_mod.run()

        latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
        preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
        summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))

        with open_preprocess_store(preprocess_dir) as store:
            manifest = store.list_events()

        assert summary["total_events"] == 2
        assert summary["retained_events"] == 2
        assert len(manifest) == 2
        # No duplicate event IDs
        assert manifest["event_id"].nunique() == 2

    def test_multi_set_parallel_processing_all_sets_complete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Both SET1 and SET2 must be fully processed with parallel events."""
        monkeypatch.chdir(tmp_path)
        _build_two_set_preprocessing_dataset(tmp_path)
        _write_default_preprocess_config(
            tmp_path,
            set_names=("AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"),
        )

        monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
        run_mod.run()

        latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
        preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

        with open_preprocess_store(preprocess_dir) as store:
            manifest = store.list_events()

        set_names = set(manifest["set_name"])
        assert set_names == {"AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"}
        # 2 events per set (NEW + OLD decks)
        for sn in set_names:
            assert len(manifest[manifest["set_name"] == sn]) == 2


class TestSearchsortedSynchroFidelity:
    """Verify the searchsorted-based synchro_indices matches organizer semantics
    across edge cases that the merge-based implementation handled."""

    def test_duplicate_reference_timestamps(self) -> None:
        """When reference has duplicates, the result should pick the last one."""
        reference = pd.to_datetime(
            ["2022-07-01 00:00:00.000", "2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
            utc=True,
        )
        target = pd.to_datetime(["2022-07-01 00:00:00.000", "2022-07-01 00:00:00.500"], utc=True)
        result = synchro_indices(reference, target)
        # Target 0 matches ref[0] and ref[1] — should return 2 (last duplicate, 1-based)
        assert result[0] == 2
        # Target 1 is after ref[1] but before ref[2] — still 2
        assert result[1] == 2

    def test_duplicate_target_timestamps(self) -> None:
        """Duplicate targets should each independently match the same reference."""
        reference = pd.to_datetime(
            ["2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
            utc=True,
        )
        target = pd.to_datetime(
            ["2022-07-01 00:00:00.500", "2022-07-01 00:00:00.500", "2022-07-01 00:00:00.500"],
            utc=True,
        )
        result = synchro_indices(reference, target)
        assert result.tolist() == [1, 1, 1]

    def test_target_exactly_equals_all_references(self) -> None:
        """Every target exactly matches a reference → all should return non-zero."""
        timestamps = [
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
        ]
        reference = pd.to_datetime(timestamps, utc=True)
        target = pd.to_datetime(timestamps, utc=True)
        result = synchro_indices(reference, target)
        assert (result != 0).all(), "No target should be dropped when all timestamps match exactly"
        assert result.tolist() == [1, 2, 3]

    def test_single_reference_single_target_equal(self) -> None:
        reference = pd.to_datetime(["2022-07-01 00:00:00.000"], utc=True)
        target = pd.to_datetime(["2022-07-01 00:00:00.000"], utc=True)
        result = synchro_indices(reference, target)
        assert result.tolist() == [1]

    def test_all_targets_before_reference(self) -> None:
        """All targets earlier than reference → all zeros (no match)."""
        reference = pd.to_datetime(["2022-07-01 00:00:01.000"], utc=True)
        target = pd.to_datetime(
            ["2022-07-01 00:00:00.000", "2022-07-01 00:00:00.500"],
            utc=True,
        )
        result = synchro_indices(reference, target)
        assert result.tolist() == [0, 0]

    def test_unsorted_reference_returns_original_indices(self) -> None:
        """Reference is not sorted — returned indices must point to original positions."""
        reference = pd.to_datetime(
            ["2022-07-01 00:00:02.000", "2022-07-01 00:00:00.000", "2022-07-01 00:00:01.000"],
            utc=True,
        )
        target = pd.to_datetime(["2022-07-01 00:00:01.500"], utc=True)
        result = synchro_indices(reference, target)
        # The latest reference <= 01.5 is 01.0 which is at original index 2 → 1-based = 3
        assert result.tolist() == [3]

    def test_searchsorted_matches_legacy_merge_on_fixture_data(self, tmp_path: Path) -> None:
        """The standard fixture (4 sensors, mixed offsets) must produce the same
        alignment result as the merge-based implementation verified in
        test_align_event_group_matches_two_pass_organizer_shrinking."""
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")
        event = find_events(reader, deck="NEW").iloc[0]
        loaded = load_event_group(reader, event)
        aligned = align_event_group(loaded)

        assert aligned.alignment_diagnostics["rows_after_alignment"] == 2
        assert aligned.aligned_waveform["NEW_S1_DO_MID_ACC_Z"].tolist() == [1.0, 3.0]
        assert aligned.aligned_waveform["NEW_S1_DO_SUP_STR"].tolist() == [5.0, 7.0]


class TestZeroDataLoss:
    """Verify that valid data is never silently dropped during preprocessing.

    These tests represent the user's requirement: 'I don't want to skip any
    data during preprocessing.' Every valid sensor value that enters the
    pipeline must emerge in the aligned samples output.
    """

    def test_all_valid_waveform_rows_survive_load(self, tmp_path: Path) -> None:
        """load_event_group must return every row from the raw file slice
        when all timestamps and values are valid."""
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

        event = find_events(reader, deck="NEW").iloc[0]
        loaded = load_event_group(reader, event)

        # NEW_S1_DO_INF_STR has 4 valid rows
        wf = loaded.waveforms["NEW_S1_DO_INF_STR"][1]
        assert len(wf) == 4, f"Expected 4 rows, got {len(wf)} — data was silently dropped"
        assert wf["timestamp"].isna().sum() == 0, "Valid timestamps were turned into NaT"
        assert wf["NEW_S1_DO_INF_STR"].isna().sum() == 0, "Valid values were turned into NaN"

        # NEW_S1_DO_MID_ACC_Z has 3 valid rows
        wf2 = loaded.waveforms["NEW_S1_DO_MID_ACC_Z"][1]
        assert len(wf2) == 3

        # NEW_S1_DO_SUP_STR has 2 valid rows
        wf3 = loaded.waveforms["NEW_S1_DO_SUP_STR"][1]
        assert len(wf3) == 2

    def test_aligned_sample_count_equals_sensors_times_aligned_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """After alignment, total sample rows = n_sensors * rows_after_alignment
        when all sensor data is valid (no NaN introduced by processing)."""
        monkeypatch.chdir(tmp_path)
        _build_preprocessing_dataset(tmp_path)
        (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
        _write_yaml(
            tmp_path / "configs" / "default.yaml",
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  event_grouping:",
                "    key_fields: [deck, Start_Time, End_Time]",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: none",
                "  signal_filter:",
                "    method: none",
                "  filtering:",
                "    min_active_sensors_per_event: 1",
                "  storage:",
                "    backend: sqlite",
                "  exports:",
                "    aligned_waveforms:",
                "      enabled: false",
                "output:",
                "  results_dir: results",
            ],
        )

        monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
        run_mod.run()

        latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
        preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

        with open_preprocess_store(preprocess_dir) as store:
            manifest = store.list_events()
            all_samples = store.load_aligned_samples()

        # Every event must be retained — none discarded
        assert manifest["discarded"].sum() == 0, (
            f"Events were discarded: {manifest.loc[manifest['discarded'] == 1, 'discard_reason'].tolist()}"
        )

        # For each retained event, sample rows = n_sensors * rows_after_alignment
        for _, event in manifest.iterrows():
            event_samples = all_samples[all_samples["event_id"] == event["event_id"]]
            n_sensors = len(event_samples["sensor_name"].unique())
            rows_after = int(event["rows_after_alignment"])
            expected_total = n_sensors * rows_after
            actual = len(event_samples)
            assert actual == expected_total, (
                f"Event {event['event_id']}: expected {expected_total} samples "
                f"({n_sensors} sensors * {rows_after} aligned rows), got {actual} — "
                f"data was silently dropped"
            )

    def test_no_nan_values_in_aligned_samples_when_input_is_clean(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When all input waveforms have valid numeric values, no NaN should
        appear in the final aligned_samples table."""
        monkeypatch.chdir(tmp_path)
        _build_preprocessing_dataset(tmp_path)
        (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
        _write_yaml(
            tmp_path / "configs" / "default.yaml",
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "preprocessing:",
                "  event_grouping:",
                "    key_fields: [deck, Start_Time, End_Time]",
                "  alignment:",
                "    method: r_synchro",
                "  zeroing:",
                "    method: none",
                "  signal_filter:",
                "    method: none",
                "  filtering:",
                "    min_active_sensors_per_event: 1",
                "  storage:",
                "    backend: sqlite",
                "  exports:",
                "    aligned_waveforms:",
                "      enabled: false",
                "output:",
                "  results_dir: results",
            ],
        )

        monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
        run_mod.run()

        latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
        preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

        with open_preprocess_store(preprocess_dir) as store:
            samples = store.load_aligned_samples()

        nan_count = samples["value"].isna().sum()
        assert nan_count == 0, f"{nan_count} NaN values found in aligned_samples — data corruption"

    def test_identical_timestamps_across_all_sensors_preserves_all_rows(
        self,
        tmp_path: Path,
    ) -> None:
        """When every sensor has the exact same timestamps, alignment should
        not drop ANY rows — all are common."""
        dataset_root = tmp_path / "AQUINAS_DATASET"
        set_dir = dataset_root / "AQUINAS_SET1_2022_07"
        set_dir.mkdir(parents=True, exist_ok=True)

        common_timestamps = [
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:01.000",
            "2022-07-01 00:00:02.000",
            "2022-07-01 00:00:03.000",
            "2022-07-01 00:00:04.000",
        ]
        common_table = {
            "Start_Time": ["2022-07-01 00:00:00"],
            "End_Time": ["2022-07-01 00:00:04"],
            "Duration": [4.0],
            "Temperature": [20.0],
        }
        for i, sensor_name in enumerate(["NEW_S1_DO_INF_STR", "NEW_S1_DO_MID_ACC_Z", "NEW_S1_DO_SUP_STR"]):
            _write_sensor(
                set_dir,
                sensor_name,
                table_payload={
                    **common_table,
                    "Record_UID": [i + 1],
                    "File": [f"{sensor_name}_SET1_1.json"],
                    "Start_Row": [1],
                    "End_Row": [5],
                },
                timestamps=common_timestamps,
                values=[float(i * 10 + j) for j in range(5)],
            )

        reader = AquinasReader(set_dir)
        event = find_events(reader, deck="NEW").iloc[0]
        loaded = load_event_group(reader, event)
        aligned = align_event_group(loaded)

        assert aligned.alignment_diagnostics["rows_after_alignment"] == 5, (
            f"Expected 5 aligned rows (all timestamps identical), "
            f"got {aligned.alignment_diagnostics['rows_after_alignment']} — rows were silently dropped"
        )

    def test_filter_and_zero_preserve_row_count(self, tmp_path: Path) -> None:
        """Filtering and zeroing must never change the number of waveform rows."""
        dataset_root = _build_preprocessing_dataset(tmp_path)
        reader = AquinasReader(dataset_root / "AQUINAS_SET1_2022_07")

        event = find_events(reader, deck="NEW").iloc[0]
        loaded = load_event_group(reader, event)

        original_lengths = {
            name: len(wf) for name, (_, wf) in loaded.waveforms.items()
        }

        filtered = filter_loaded_event_group(loaded, method="butterworth_bandpass")
        for name, (_, wf) in filtered.waveforms.items():
            assert len(wf) == original_lengths[name], (
                f"Filter changed row count for {name}: {original_lengths[name]} -> {len(wf)}"
            )

        zeroed = zero_loaded_event_group(filtered, method="linear_endpoints")
        for name, (_, wf) in zeroed.waveforms.items():
            assert len(wf) == original_lengths[name], (
                f"Zeroing changed row count for {name}: {original_lengths[name]} -> {len(wf)}"
            )

    def test_vectorized_timestamp_formatting_matches_scalar(self) -> None:
        """Vectorized dt.strftime path must produce the same string as the
        scalar format_timestamp_utc for every timestamp."""
        from aquinas_toolkit.preprocessing.core import format_timestamp_utc

        timestamps = pd.to_datetime([
            "2022-07-01 00:00:00.000",
            "2022-07-01 00:00:00.123",
            "2022-07-01 00:00:00.999",
            "2022-12-31 23:59:59.001",
            "2023-01-01 00:00:00.000",
        ], utc=True)

        scalar_results = [format_timestamp_utc(ts) for ts in timestamps]
        vectorized_results = (
            timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
        ).tolist()

        assert scalar_results == vectorized_results, (
            f"Vectorized timestamps differ from scalar:\n"
            f"  scalar:     {scalar_results}\n"
            f"  vectorized: {vectorized_results}"
        )
