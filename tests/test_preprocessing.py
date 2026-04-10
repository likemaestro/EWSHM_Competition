import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from aquinas_toolkit import AquinasReader
from aquinas_toolkit.cli import run as run_mod
from aquinas_toolkit.preprocessing import (
    LoadedEventGroup,
    align_event_group,
    find_events,
    load_event_group,
    run_organizer_query,
    synchro_indices,
    zero_loaded_event_group,
    zero_waveform,
)
from aquinas_toolkit.preprocessing.pipeline import load_preprocessing_settings


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_default_preprocess_config(
    workspace: Path,
    *,
    set_names: tuple[str, ...] = ("AQUINAS_SET1_2022_07",),
    min_active_sensors_per_event: int = 1,
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
            "  export:",
            f"    format: {export_format}",
            "    partition_by: [set_name, deck]",
            "output:",
            "  results_dir: results",
        ]
    )
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(config_dir / "default.yaml", lines)


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


def test_run_preprocess_writes_stage_artifacts(monkeypatch, tmp_path: Path) -> None:
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
                "  export:",
                "    format: csv.gz",
                "    partition_by: [set_name, deck]",
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

    manifest = pd.read_csv(preprocess_dir / "event_manifest.csv")
    sensor_records = pd.read_csv(preprocess_dir / "sensor_records.csv")
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))
    new_aligned = pd.read_csv(preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__NEW.csv.gz")

    assert (preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__NEW.csv.gz").is_file()
    assert (preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__OLD.csv.gz").is_file()
    assert len(manifest) == 2
    assert sorted(manifest["discarded"].tolist()) == [False, False]
    assert "Temperature" in sensor_records.columns
    assert summary["total_events"] == 2
    assert summary["retained_events"] == 2
    assert summary["alignment"]["method"] == "r_synchro"
    assert summary["alignment"]["passes"] == 2
    assert summary["zeroing"]["stage"] == "before_alignment"
    assert len(new_aligned) == 2


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
                "  export:",
                "    format: csv.gz",
                "    partition_by: [set_name, deck]",
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

    manifest = pd.read_csv(preprocess_dir / "event_manifest.csv")
    sensor_records = pd.read_csv(preprocess_dir / "sensor_records.csv")
    qc_report = pd.read_csv(preprocess_dir / "sensor_qc_report.csv")
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))

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
    assert set4_manifest["excluded_sensors"] == "OLD_S1_UP_SUP_STR"
    assert set5_manifest["excluded_sensors"] == "OLD_S1_UP_SUP_STR"
    assert int(set4_manifest["active_sensor_count"]) == 1
    assert int(set5_manifest["active_sensor_count"]) == 1
    assert set4_manifest["reference_sensor"] == "OLD_S1_UP_INF_STR"

    set4_aligned = pd.read_csv(preprocess_dir / "aligned" / "AQUINAS_SET4_2024_01__OLD.csv.gz")
    set1_aligned = pd.read_csv(preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__OLD.csv.gz")
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
        export_format="csv",
    )

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    preprocess_dir = tmp_path / "results" / latest["run_id"] / "stages" / "preprocess"
    manifest = pd.read_csv(preprocess_dir / "event_manifest.csv")
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))

    old_event = manifest.loc[manifest["deck"] == "OLD"].iloc[0]
    assert bool(old_event["discarded"])
    assert old_event["discard_reason"] == "insufficient_active_sensors"
    assert (preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__NEW.csv").is_file()
    assert not (preprocess_dir / "aligned" / "AQUINAS_SET1_2022_07__NEW.csv.gz").exists()
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
    manifest = pd.read_csv(preprocess_dir / "event_manifest.csv")
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(manifest) == 1
    assert bool(manifest.loc[0, "discarded"])
    assert manifest.loc[0, "discard_reason"] == "no_common_aligned_rows"
    assert int(manifest.loc[0, "rows_after_alignment"]) == 0
    assert summary["retained_events"] == 0
    assert summary["discard_reasons"] == {"no_common_aligned_rows": 1}
    assert list((preprocess_dir / "aligned").iterdir()) == []


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

    manifest = pd.read_csv(preprocess_dir / "event_manifest.csv")
    sensor_records = pd.read_csv(preprocess_dir / "sensor_records.csv")
    qc_report = pd.read_csv(preprocess_dir / "sensor_qc_report.csv")
    summary = json.loads((preprocess_dir / "summary.json").read_text(encoding="utf-8"))

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
    assert list((preprocess_dir / "aligned").iterdir()) == []
