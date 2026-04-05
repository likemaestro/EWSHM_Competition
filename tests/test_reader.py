import json
from pathlib import Path

from aquinas_toolkit import AquinasReader


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "AQUINAS_SET1_2022_07"
    dataset_dir.mkdir()

    inf_sensor = dataset_dir / "NEW_S1_DO_INF_STR"
    inf_sensor.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json",
        {
            "Record_UID": [101, 102],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json", "NEW_S1_DO_INF_STR_SET1_1.json"],
            "Start_Row": [2, 3],
            "End_Row": [3, 4],
            "Tag": [["baseline"], ["follow_up"]],
        },
    )
    _write_json(
        inf_sensor / "NEW_S1_DO_INF_STR_SET1_1.json",
        {
            "timestamp": ["2022-07-01 00:00:00.000", "2022-07-01 00:00:00.010", "2022-07-01 00:00:00.020", "2022-07-01 00:00:00.030"],
            "value": [10.0, 20.0, 30.0, 40.0],
        },
    )

    acc_sensor = dataset_dir / "NEW_S1_DO_MID_ACC_Z"
    acc_sensor.mkdir()
    _write_json(
        dataset_dir / "TABLE_NEW_S1_DO_MID_ACC_Z_SET1.json",
        {
            "Record_UID": [201],
            "File": ["NEW_S1_DO_MID_ACC_Z_SET1_1.json"],
            "Start_Row": [1],
            "End_Row": [2],
            "Tag": [["acc"]],
        },
    )
    _write_json(
        acc_sensor / "NEW_S1_DO_MID_ACC_Z_SET1_1.json",
        {
            "timestamp": ["2022-07-01 00:00:00.000", "2022-07-01 00:00:00.010", "2022-07-01 00:00:00.020"],
            "value": [1.0, 2.0, 3.0],
        },
    )

    return dataset_dir


def test_list_sensor_names_and_summary(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)

    assert reader.list_sensor_names() == [
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_MID_ACC_Z",
    ]

    summary = reader.summary()
    assert summary["sensor_name"].tolist() == [
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_MID_ACC_Z",
    ]
    assert summary["sensor_dir_exists"].tolist() == [True, True]


def test_load_index_table_unwraps_singleton_lists(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)
    index_df = reader.load_index_table("NEW_S1_DO_INF_STR")

    assert index_df["Record_UID"].tolist() == [101, 102]
    assert index_df["Tag"].tolist() == ["baseline", "follow_up"]


def test_read_record_uses_one_based_rows_and_record_uid_lookup(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)
    meta, waveform = reader.read_record("NEW_S1_DO_INF_STR", record_uid=102)

    assert meta["Record_UID"] == 102
    assert waveform["value"].tolist() == [30.0, 40.0]


def test_load_all_index_tables_adds_sensor_and_dataset_columns(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)
    combined = reader.load_all_index_tables()

    assert set(combined["sensor_name"]) == {
        "NEW_S1_DO_INF_STR",
        "NEW_S1_DO_MID_ACC_Z",
    }
    assert set(combined["dataset"]) == {"AQUINAS_SET1_2022_07"}


def test_summarize_sensor_records_parses_sensor_metadata_and_counts(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)
    summary = reader.summarize_sensor_records()

    assert summary.to_dict("records") == [
        {
            "dataset": "AQUINAS_SET1_2022_07",
            "sensor_name": "NEW_S1_DO_INF_STR",
            "deck": "NEW",
            "span": "S1",
            "side": "DO",
            "location": "INF",
            "quantity": "STR",
            "axis": None,
            "record_count": 2,
        },
        {
            "dataset": "AQUINAS_SET1_2022_07",
            "sensor_name": "NEW_S1_DO_MID_ACC_Z",
            "deck": "NEW",
            "span": "S1",
            "side": "DO",
            "location": "MID",
            "quantity": "ACC",
            "axis": "Z",
            "record_count": 1,
        },
    ]


def test_summarize_sensor_records_supports_quantity_and_axis_filters(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    reader = AquinasReader(dataset_dir)
    summary = reader.summarize_sensor_records(quantity="acc", axis="z")

    assert summary["sensor_name"].tolist() == ["NEW_S1_DO_MID_ACC_Z"]
    assert summary["record_count"].tolist() == [1]
