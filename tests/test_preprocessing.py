import json
from pathlib import Path

import numpy as np

from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.preprocessing import (
    bandpass_filter_waveform_matrix,
    filter_records_by_min_duration,
    find_common_sensor_events,
    load_common_event_waveform_matrix,
    summarize_min_duration_filter,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "AQUINAS_SET1_2022_07"
    dataset_dir.mkdir()

    acc_z_mid = dataset_dir / "NEW_S1_DO_MID_ACC_Z"
    acc_z_mid.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_MID_ACC_Z_SET1.json",
        {
            "Record_UID": [1, 2, 3],
            "File": [
                "NEW_S1_DO_MID_ACC_Z_SET1_1.json",
                "NEW_S1_DO_MID_ACC_Z_SET1_1.json",
                "NEW_S1_DO_MID_ACC_Z_SET1_1.json",
            ],
            "Start_Row": [1, 1, 1],
            "End_Row": [2, 2, 2],
            "Duration": [29.9, 30.0, 45.0],
            "Start_Time": [
                "2022-07-01 00:00:00",
                "2022-07-01 00:01:00",
                "2022-07-01 00:02:00",
            ],
            "End_Time": [
                "2022-07-01 00:00:29",
                "2022-07-01 00:01:30",
                "2022-07-01 00:02:45",
            ],
        },
    )
    _write_json(
        acc_z_mid / "NEW_S1_DO_MID_ACC_Z_SET1_1.json",
        {
            "timestamp": [
                "2022-07-01 00:00:00.000",
                "2022-07-01 00:00:00.010",
                "2022-07-01 00:00:00.020",
                "2022-07-01 00:00:00.030",
                "2022-07-01 00:00:00.040",
                "2022-07-01 00:00:00.050",
            ],
            "value": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        },
    )

    acc_z_int = dataset_dir / "NEW_S1_DO_INT_ACC_Z"
    acc_z_int.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_INT_ACC_Z_SET1.json",
        {
            "Record_UID": [10, 11],
            "File": [
                "NEW_S1_DO_INT_ACC_Z_SET1_1.json",
                "NEW_S1_DO_INT_ACC_Z_SET1_1.json",
            ],
            "Start_Row": [1, 1],
            "End_Row": [2, 2],
            "Duration": [10.0, 35.0],
            "Start_Time": ["2022-07-01 00:00:10", "2022-07-01 00:01:00"],
            "End_Time": ["2022-07-01 00:00:20", "2022-07-01 00:01:30"],
        },
    )
    _write_json(
        acc_z_int / "NEW_S1_DO_INT_ACC_Z_SET1_1.json",
        {
            "timestamp": [
                "2022-07-01 00:00:10.000",
                "2022-07-01 00:00:10.010",
                "2022-07-01 00:00:10.020",
                "2022-07-01 00:00:10.030",
            ],
            "value": [10.0, 10.5, 11.0, 11.5],
        },
    )

    old_acc_z_mid = dataset_dir / "OLD_S1_DO_MID_ACC_Z"
    old_acc_z_mid.mkdir()
    _write_json(
        dataset_dir / "TABLE_OLD_S1_DO_MID_ACC_Z_SET1.json",
        {
            "Record_UID": [101],
            "File": ["OLD_S1_DO_MID_ACC_Z_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [3],
            "Duration": [20.0],
            "Start_Time": ["2022-07-01 00:03:00"],
            "End_Time": ["2022-07-01 00:03:20"],
        },
    )
    _write_json(
        old_acc_z_mid / "OLD_S1_DO_MID_ACC_Z_SET1_1.json",
        {
            "timestamp": [
                "2022-07-01 00:03:00.000",
                "2022-07-01 00:03:00.010",
                "2022-07-01 00:03:00.020",
            ],
            "value": [4.0, 4.5, 5.0],
        },
    )

    old_acc_z_int = dataset_dir / "OLD_S1_DO_INT_ACC_Z"
    old_acc_z_int.mkdir()
    _write_json(
        dataset_dir / "TABLE_OLD_S1_DO_INT_ACC_Z_SET1.json",
        {
            "Record_UID": [111],
            "File": ["OLD_S1_DO_INT_ACC_Z_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [3],
            "Duration": [20.0],
            "Start_Time": ["2022-07-01 00:03:00"],
            "End_Time": ["2022-07-01 00:03:20"],
        },
    )
    _write_json(
        old_acc_z_int / "OLD_S1_DO_INT_ACC_Z_SET1_1.json",
        {
            "timestamp": [
                "2022-07-01 00:03:00.000",
                "2022-07-01 00:03:00.010",
                "2022-07-01 00:03:00.020",
            ],
            "value": [6.0, 6.5, 7.0],
        },
    )

    acc_y_mid = dataset_dir / "NEW_S1_DO_MID_ACC_Y"
    acc_y_mid.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_MID_ACC_Y_SET1.json",
        {
            "Record_UID": [20],
            "File": ["NEW_S1_DO_MID_ACC_Y_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
            "Duration": [60.0],
            "Start_Time": ["2022-07-01 00:01:00"],
            "End_Time": ["2022-07-01 00:02:00"],
        },
    )

    strain = dataset_dir / "NEW_S1_DO_INF_STR"
    strain.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json",
        {
            "Record_UID": [30],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
            "Duration": [90.0],
            "Start_Time": ["2022-07-01 00:01:00"],
            "End_Time": ["2022-07-01 00:02:30"],
        },
    )

    return dataset_dir


def test_filter_records_by_min_duration_keeps_only_acc_z_at_or_above_threshold(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    reader = AquinasReader(dataset_dir)

    filtered = filter_records_by_min_duration(reader, min_duration_seconds=30.0)

    assert filtered["sensor_name"].tolist() == [
        "NEW_S1_DO_INT_ACC_Z",
        "NEW_S1_DO_MID_ACC_Z",
        "NEW_S1_DO_MID_ACC_Z",
    ]
    assert filtered["Record_UID"].tolist() == [11, 2, 3]
    assert filtered["Duration"].tolist() == [35.0, 30.0, 45.0]


def test_summarize_min_duration_filter_reports_kept_and_removed_counts(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    reader = AquinasReader(dataset_dir)

    summary = summarize_min_duration_filter(reader, min_duration_seconds=30.0, deck="NEW")

    assert summary[["sensor_name", "kept_count", "removed_count"]].to_dict("records") == [
        {
            "sensor_name": "NEW_S1_DO_INT_ACC_Z",
            "kept_count": 1,
            "removed_count": 1,
        },
        {
            "sensor_name": "NEW_S1_DO_MID_ACC_Z",
            "kept_count": 2,
            "removed_count": 1,
        },
    ]
    assert summary["min_duration_seconds"].tolist() == [30.0, 30.0]


def test_find_common_sensor_events_returns_long_shared_acc_z_events(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    reader = AquinasReader(dataset_dir)

    common_events = find_common_sensor_events(reader, min_duration_seconds=30.0, deck="NEW")

    assert common_events[["Start_Time", "End_Time", "deck", "channel_count"]].to_dict("records") == [
        {
            "Start_Time": "2022-07-01 00:01:00",
            "End_Time": "2022-07-01 00:01:30",
            "deck": "NEW",
            "channel_count": 2,
        }
    ]


def test_find_common_sensor_events_can_separate_old_and_new_decks(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    reader = AquinasReader(dataset_dir)

    new_events = find_common_sensor_events(reader, min_duration_seconds=10.0, deck="NEW")
    old_events = find_common_sensor_events(reader, min_duration_seconds=10.0, deck="OLD")
    all_events = find_common_sensor_events(reader, min_duration_seconds=10.0)

    assert new_events[["Start_Time", "End_Time"]].to_dict("records") == [
        {"Start_Time": "2022-07-01 00:01:00", "End_Time": "2022-07-01 00:01:30"},
    ]
    assert old_events[["Start_Time", "End_Time"]].to_dict("records") == [
        {"Start_Time": "2022-07-01 00:03:00", "End_Time": "2022-07-01 00:03:20"}
    ]
    assert all_events.empty


def test_load_common_event_waveform_matrix_builds_multichannel_matrix(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    reader = AquinasReader(dataset_dir)

    matrix = load_common_event_waveform_matrix(
        reader,
        start_time="2022-07-01 00:01:00",
        end_time="2022-07-01 00:01:30",
        sensor_names=["NEW_S1_DO_INT_ACC_Z", "NEW_S1_DO_MID_ACC_Z"],
    )

    assert matrix.columns.tolist() == [
        "timestamp",
        "NEW_S1_DO_INT_ACC_Z",
        "NEW_S1_DO_MID_ACC_Z",
    ]
    assert matrix.shape == (2, 3)


def test_bandpass_filter_waveform_matrix_suppresses_out_of_band_content() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)
    signal = np.sin(2 * np.pi * 3.0 * time) + 0.8 * np.sin(2 * np.pi * 30.0 * time)
    filtered = bandpass_filter_waveform_matrix(signal[:, None], sampling_rate_hz=sampling_rate_hz)

    spectrum = np.abs(np.fft.rfft(filtered[:, 0]))
    freqs = np.fft.rfftfreq(filtered.shape[0], d=1.0 / sampling_rate_hz)
    amp_3hz = spectrum[np.argmin(np.abs(freqs - 3.0))]
    amp_30hz = spectrum[np.argmin(np.abs(freqs - 30.0))]

    assert amp_3hz > 5 * amp_30hz