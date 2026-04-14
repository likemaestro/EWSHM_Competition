import json
import math
import shutil
import sqlite3
import sys
import warnings
from warnings import WarningMessage
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.errors import DtypeWarning

from aquinas_toolkit.cli import run as run_mod
from aquinas_toolkit.feature_extraction import (
    open_features_store,
    annotate_mode_shape_locations,
    run_acc_z_fdd_from_preprocess_store,
    frequency_domain_decomposition,
    summarize_fdd_mode_shapes,
    summarize_fdd_peaks,
)
from aquinas_toolkit.preprocessing import LegacyPreprocessCsvReader, open_preprocess_store


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_pipeline_config(workspace: Path) -> None:
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        config_dir / "default.yaml",
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
            "      enabled: false",
            "      format: csv.gz",
            "features:",
            "  sampling_rate_hz: 100.0",
            "  modal_analysis:",
            "    enabled: true",
            "    quantity: ACC",
            "    axis: Z",
            "    min_common_events: 2",
            "    max_events: 5",
            "    low_hz: 0.5",
            "    high_hz: 20.0",
            "    nperseg: 128",
            "    noverlap: 64",
            "    n_peaks: 3",
            "output:",
            "  results_dir: results",
        ],
    )


def _build_feature_stage_dataset(workspace: Path) -> Path:
    dataset_root = workspace / "AQUINAS_DATASET"
    set_dir = dataset_root / "AQUINAS_SET1_2022_07"
    set_dir.mkdir(parents=True, exist_ok=True)

    first_start = pd.Timestamp("2022-07-01T00:00:00Z")
    second_start = pd.Timestamp("2022-07-01T00:01:00Z")
    sample_count = 200
    dt_seconds = 0.01
    time_axis = np.arange(sample_count, dtype=float) * dt_seconds

    all_timestamps = _event_timestamps(first_start, sample_count, dt_seconds) + _event_timestamps(
        second_start, sample_count, dt_seconds
    )

    new_acc_1 = np.concatenate(
        [
            np.sin(2 * np.pi * 3.0 * time_axis),
            1.05 * np.sin(2 * np.pi * 3.0 * time_axis + 0.1),
        ]
    )
    new_acc_2 = np.concatenate(
        [
            0.8 * np.sin(2 * np.pi * 3.0 * time_axis + 0.3),
            0.9 * np.sin(2 * np.pi * 3.0 * time_axis + 0.5),
        ]
    )
    new_str = np.concatenate(
        [
            0.2 + 0.01 * np.arange(sample_count),
            0.5 + 0.01 * np.arange(sample_count),
        ]
    )
    old_acc = np.concatenate(
        [
            0.7 * np.sin(2 * np.pi * 3.0 * time_axis - 0.2),
            0.65 * np.sin(2 * np.pi * 3.0 * time_axis - 0.1),
        ]
    )

    _write_sensor_with_two_events(
        set_dir,
        sensor_name="NEW_S1_DO_MID_ACC_Z",
        values=new_acc_1.tolist(),
        timestamps=all_timestamps,
    )
    _write_sensor_with_two_events(
        set_dir,
        sensor_name="NEW_S1_UP_MID_ACC_Z",
        values=new_acc_2.tolist(),
        timestamps=all_timestamps,
    )
    _write_sensor_with_two_events(
        set_dir,
        sensor_name="NEW_S1_DO_INF_STR",
        values=new_str.tolist(),
        timestamps=all_timestamps,
    )
    _write_sensor_with_two_events(
        set_dir,
        sensor_name="OLD_S1_DO_MID_ACC_Z",
        values=old_acc.tolist(),
        timestamps=all_timestamps,
    )

    return dataset_root


def _event_timestamps(start: pd.Timestamp, sample_count: int, dt_seconds: float) -> list[str]:
    return [
        (start + pd.to_timedelta(index * dt_seconds, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for index in range(sample_count)
    ]


def _write_sensor_with_two_events(
    set_dir: Path,
    *,
    sensor_name: str,
    values: list[float],
    timestamps: list[str],
) -> None:
    sensor_dir = set_dir / sensor_name
    sensor_dir.mkdir()
    file_name = f"{sensor_name}_SET1_1.json"
    table_path = set_dir / f"TABLE_{sensor_name}_SET1.json"

    sample_count = len(values) // 2
    first_segment = values[:sample_count]
    second_segment = values[sample_count:]
    start_times = [timestamps[0], timestamps[sample_count]]
    end_times = [timestamps[sample_count - 1], timestamps[-1]]
    duration = round((sample_count - 1) * 0.01, 2)

    _write_json(
        table_path,
        {
            "Record_UID": [1, 2],
            "File": [file_name, file_name],
            "Start_Row": [1, sample_count + 1],
            "End_Row": [sample_count, sample_count * 2],
            "Start_Time": start_times,
            "End_Time": end_times,
            "Duration": [duration, duration],
            "Start_Value": [first_segment[0], second_segment[0]],
            "End_Value": [first_segment[-1], second_segment[-1]],
            "Diff_Value": [first_segment[-1] - first_segment[0], second_segment[-1] - second_segment[0]],
            "Min_Value": [float(np.min(first_segment)), float(np.min(second_segment))],
            "Max_Value": [float(np.max(first_segment)), float(np.max(second_segment))],
            "Mean_Value": [float(np.mean(first_segment)), float(np.mean(second_segment))],
            "Range": [
                float(np.max(first_segment) - np.min(first_segment)),
                float(np.max(second_segment) - np.min(second_segment)),
            ],
            "Temperature": [21.0, 22.0],
        },
    )
    _write_json(
        sensor_dir / file_name,
        {
            "timestamp": timestamps,
            sensor_name: values,
        },
    )


def test_frequency_domain_decomposition_finds_dominant_mode_near_target_frequency() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)

    matrix = np.column_stack(
        [
            1.0 * np.sin(2 * np.pi * 3.2 * time),
            0.7 * np.sin(2 * np.pi * 3.2 * time + 0.3),
            1.2 * np.sin(2 * np.pi * 3.2 * time - 0.4),
            0.6 * np.sin(2 * np.pi * 3.2 * time + 0.8),
        ]
    )
    matrix += 0.15 * np.column_stack(
        [
            np.sin(2 * np.pi * 11.0 * time),
            np.sin(2 * np.pi * 11.0 * time + 0.2),
            np.sin(2 * np.pi * 11.0 * time - 0.1),
            np.sin(2 * np.pi * 11.0 * time + 0.4),
        ]
    )

    result = frequency_domain_decomposition(matrix, sampling_rate_hz=sampling_rate_hz)
    peaks = summarize_fdd_peaks(result["frequencies_hz"], result["singular_values"], n_peaks=3)

    assert any(abs(freq - 3.2) < 0.3 for freq in peaks["frequency_hz"])


def test_frequency_domain_decomposition_supports_multiple_event_matrices() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 20.0, 1.0 / sampling_rate_hz)

    event_a = np.column_stack(
        [np.sin(2 * np.pi * 2.5 * time), 0.8 * np.sin(2 * np.pi * 2.5 * time + 0.2)]
    )
    event_b = np.column_stack(
        [1.1 * np.sin(2 * np.pi * 2.5 * time - 0.1), 0.9 * np.sin(2 * np.pi * 2.5 * time + 0.4)]
    )

    result = frequency_domain_decomposition(
        [event_a, event_b],
        sampling_rate_hz=sampling_rate_hz,
        nperseg=512,
        noverlap=256,
    )

    assert result["singular_values"].shape[1] == 2
    assert result["mode_shapes"].shape[1:] == (2, 2)


def test_summarize_fdd_mode_shapes_reports_normalized_components() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)
    matrix = np.column_stack(
        [
            1.0 * np.sin(2 * np.pi * 4.0 * time),
            0.5 * np.sin(2 * np.pi * 4.0 * time + 0.4),
            0.8 * np.sin(2 * np.pi * 4.0 * time - 0.2),
        ]
    )

    result = frequency_domain_decomposition(matrix, sampling_rate_hz=sampling_rate_hz)
    peak_table, mode_shape_table = summarize_fdd_mode_shapes(
        result["frequencies_hz"],
        result["singular_values"],
        result["mode_shapes"],
        channel_names=["ch1", "ch2", "ch3"],
        n_peaks=2,
    )

    assert not peak_table.empty
    assert set(mode_shape_table["channel"]) == {"ch1", "ch2", "ch3"}
    peak_groups = mode_shape_table.groupby("peak_rank")["mode_shape_amplitude"].max().tolist()
    assert all(np.isclose(value, 1.0) for value in peak_groups)
    signed_groups = mode_shape_table.groupby("peak_rank")["mode_shape_signed_component"].apply(
        lambda values: np.max(np.abs(values.to_numpy()))
    )
    assert all(np.isclose(value, 1.0) for value in signed_groups)


def test_annotate_mode_shape_locations_extracts_structural_position_fields() -> None:
    mode_shape_table = pd.DataFrame(
        {
            "peak_rank": [1, 1],
            "frequency_hz": [3.2, 3.2],
            "singular_value": [1.0, 1.0],
            "channel": ["OLD_S1_DO_INT_ACC_Z", "NEW_S2_UP_MID_ACC_Z"],
            "mode_shape_amplitude": [1.0, 0.8],
            "mode_shape_signed_component": [1.0, -0.8],
            "mode_shape_phase_deg": [0.0, 180.0],
        }
    )

    annotated = annotate_mode_shape_locations(mode_shape_table)

    assert list(annotated["deck"]) == ["OLD", "NEW"]
    assert list(annotated["span"]) == ["S1", "S2"]
    assert list(annotated["side"]) == ["DO", "UP"]
    assert list(annotated["location"]) == ["INT", "MID"]
    assert list(annotated["quantity"]) == ["ACC", "ACC"]
    assert list(annotated["axis"]) == ["Z", "Z"]
    assert list(annotated["position_label"]) == ["S1_DO_INT", "S2_UP_MID"]


def test_run_features_creates_sqlite_feature_store_and_modal_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    features_dir = tmp_path / "results" / latest["run_id"] / "stages" / "features"
    features_db = features_dir / "features.sqlite"

    assert features_db.is_file()
    with open_features_store(features_dir) as store:
        sensor_features = store.load_sensor_event_features()
        modal_peaks = store.load_deck_modal_peaks()
        mode_shapes = store.load_deck_mode_shape_components()
        family_status = store.load_feature_family_status()

    assert {"ACC", "STR"} <= set(sensor_features["quantity"].dropna())
    assert {"waveform_rms", "table_temperature"} <= set(sensor_features.columns)
    assert sensor_features["sample_count"].min() > 0
    assert set(sensor_features["deck"]) == {"NEW", "OLD"}
    assert not modal_peaks.empty
    assert set(modal_peaks["deck"]) == {"NEW"}
    assert not mode_shapes.empty

    new_status = family_status.loc[family_status["deck"] == "NEW"].iloc[0]
    old_status = family_status.loc[family_status["deck"] == "OLD"].iloc[0]
    assert new_status["status"] == "completed"
    assert old_status["status"] == "skipped"
    assert "insufficient common ACC_Z" in old_status["detail"]


def test_run_features_prints_stage_progress_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])
    run_mod.run()

    captured = capsys.readouterr()
    assert "Loading preprocess artifacts..." in captured.out
    assert "Extracting sensor-event features..." in captured.out
    assert "MODAL 1/" in captured.out
    assert "Running ACC_Z FDD..." in captured.out
    assert "Writing feature store..." in captured.out
    assert "skipped" in captured.out
    assert "Timing breakdown (features)" not in captured.out


def test_run_features_verbose_prints_timing_breakdown_and_writes_debug_timings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features", "--verbose"])
    run_mod.run()

    captured = capsys.readouterr()
    assert "Timing breakdown (features)" in captured.out
    assert "load_aligned_event_s" in captured.out
    assert "write_sensor_event_features_s" in captured.out

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    debug_log = tmp_path / "results" / latest["run_id"] / "debug.log"
    log_text = debug_log.read_text(encoding="utf-8")
    assert "event=TIMING" in log_text
    assert "stage=features" in log_text
    assert "phase=load_aligned_event_s" in log_text
    assert "phase=load_event_sensors_s" in log_text
    assert "phase=write_sensor_event_features_s" in log_text
    assert "phase=total_stage_s" in log_text


def test_run_acc_z_fdd_from_preprocess_store_reads_canonical_preprocess_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"

    with open_preprocess_store(preprocess_dir) as store:
        summary = run_acc_z_fdd_from_preprocess_store(
            store,
            set_name="AQUINAS_SET1_2022_07",
            deck="NEW",
            min_common_events=2,
            max_events=5,
            sampling_rate_hz=100.0,
            low_hz=0.5,
            high_hz=20.0,
            nperseg=128,
            noverlap=64,
            n_peaks=3,
        )

    assert summary["deck"] == "NEW"
    assert not summary["peak_table"].empty
    assert not summary["available_events"].empty
    assert not summary["selected_events"].empty
    assert len(summary["aligned_events"]) == len(summary["selected_events"])
    assert len(summary["channel_names"]) >= 2


def test_run_features_can_read_legacy_preprocess_csv_artifacts_temporarily(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    original_run_dir = tmp_path / "results" / latest["run_id"]
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    _materialize_legacy_preprocess_artifacts(preprocess_dir)

    legacy_run_id = f"{latest['run_id']}_legacy_csv"
    legacy_run_dir = tmp_path / "results" / legacy_run_id
    shutil.copytree(original_run_dir, legacy_run_dir)

    legacy_preprocess_dir = legacy_run_dir / "stages" / "preprocess"
    (legacy_preprocess_dir / "preprocess.sqlite").unlink()
    for sidecar in ("preprocess.sqlite-wal", "preprocess.sqlite-shm"):
        sidecar_path = legacy_preprocess_dir / sidecar
        if sidecar_path.exists():
            sidecar_path.unlink()

    latest["run_id"] = legacy_run_id
    (tmp_path / "results" / "latest.json").write_text(json.dumps(latest), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])
    run_mod.run()

    features_dir = legacy_run_dir / "stages" / "features"
    with open_features_store(features_dir) as store:
        sensor_features = store.load_sensor_event_features()
        family_status = store.load_feature_family_status()

    assert not sensor_features.empty
    assert "table_duration" in sensor_features.columns
    assert set(family_status["status"]) >= {"completed", "skipped"}


def test_legacy_preprocess_reader_caches_sensor_records_and_avoids_dtype_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _build_feature_stage_dataset(tmp_path)
    _write_feature_pipeline_config(tmp_path)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    _materialize_legacy_preprocess_artifacts(preprocess_dir)

    legacy_run_id = f"{latest['run_id']}_legacy_cache"
    legacy_run_dir = tmp_path / "results" / legacy_run_id
    shutil.copytree(tmp_path / "results" / latest["run_id"], legacy_run_dir)
    legacy_preprocess_dir = legacy_run_dir / "stages" / "preprocess"
    (legacy_preprocess_dir / "preprocess.sqlite").unlink()
    for sidecar in ("preprocess.sqlite-wal", "preprocess.sqlite-shm"):
        sidecar_path = legacy_preprocess_dir / sidecar
        if sidecar_path.exists():
            sidecar_path.unlink()

    read_calls: list[Path] = []
    original_read_csv = pd.read_csv

    def tracking_read_csv(*args, **kwargs):  # noqa: ANN001
        path_arg = args[0] if args else kwargs.get("filepath_or_buffer")
        if Path(path_arg) == legacy_preprocess_dir / "sensor_records.csv":
            read_calls.append(Path(path_arg))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracking_read_csv)

    recorded_warnings: list[WarningMessage] = []
    with LegacyPreprocessCsvReader(legacy_preprocess_dir) as reader:
        event_ids = reader.iter_retained_events()["event_id"].tolist()
        assert event_ids
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            first = reader.load_event_sensors(event_ids[0])
            second = reader.load_event_sensors(event_ids[0])
        recorded_warnings = list(recorded)

    assert not first.empty
    pd.testing.assert_frame_equal(first, second)
    assert len(read_calls) == 1
    assert not [warning for warning in recorded_warnings if isinstance(warning.message, DtypeWarning)]


def _materialize_legacy_preprocess_artifacts(preprocess_dir: Path) -> None:
    preprocess_db = preprocess_dir / "preprocess.sqlite"
    with open_preprocess_store(preprocess_dir) as store:
        events = store.list_events()
        retained = store.iter_retained_events()
    with sqlite3.connect(preprocess_db) as conn:
        sensor_records = pd.read_sql_query("SELECT * FROM sensor_records", conn)

    legacy_manifest = events.copy()
    legacy_manifest["active_sensors"] = legacy_manifest["active_sensors"].map(";".join)
    legacy_manifest["excluded_sensors"] = legacy_manifest["excluded_sensors"].map(";".join)
    legacy_manifest["excluded_sensor_reasons"] = legacy_manifest["excluded_sensor_reasons"].map(";".join)
    legacy_manifest.to_csv(preprocess_dir / "event_manifest.csv", index=False)
    sensor_records.to_csv(preprocess_dir / "sensor_records.csv", index=False)

    aligned_dir = preprocess_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    with open_preprocess_store(preprocess_dir) as store:
        for set_name, deck in (
            retained[["set_name", "deck"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            partition_events = retained.loc[
                (retained["set_name"] == set_name) & (retained["deck"] == deck)
            ]
            frames = []
            for event_id in partition_events["event_id"]:
                frame = store.load_aligned_event(event_id)
                frame = frame.copy()
                frame.insert(0, "sample_index", range(len(frame)))
                frame.insert(0, "event_id", event_id)
                frames.append(frame)
            partition = pd.concat(frames, ignore_index=True)
            partition["timestamp_utc"] = partition["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
            partition.to_csv(aligned_dir / f"{set_name}__{deck}_DECK.csv.gz", index=False, compression="gzip")
