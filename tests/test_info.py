import json
import sys
from pathlib import Path

import pytest

from aquinas_toolkit.cli import info as info_mod


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_info_set(dataset_root: Path, set_name: str) -> Path:
    set_dir = dataset_root / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    acc_sensor = set_dir / "NEW_S1_DO_MID_ACC_Z"
    acc_sensor.mkdir()
    _write_json(
        set_dir / "TABLE_NEW_S1_DO_MID_ACC_Z_SET1.json",
        {
            "Record_UID": [201, 202],
            "File": ["NEW_S1_DO_MID_ACC_Z_SET1_1.json", "NEW_S1_DO_MID_ACC_Z_SET1_2.json"],
            "Start_Row": [1, 1],
            "End_Row": [2, 2],
            "Start_Time": ["2022-07-01 00:00:00.000", "2022-07-01 00:01:00.000"],
        },
    )

    strain_sensor = set_dir / "NEW_S1_DO_INF_STR"
    strain_sensor.mkdir()
    _write_json(
        set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json",
        {
            "Record_UID": [101, 102],
            "File": ["NEW_S1_DO_INF_STR_SET1_1.json", "NEW_S1_DO_INF_STR_SET1_2.json"],
            "Start_Row": [1, 1],
            "End_Row": [2, 2],
            "Start_Time": ["2022-07-01 00:00:05.000", "2022-07-01 00:01:05.000"],
        },
    )

    return set_dir


def test_info_fails_when_dataset_root_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    with pytest.raises(SystemExit) as exc_info:
        info_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Dataset folder not found" in captured.err


def test_info_renders_dataset_summary_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _build_info_set(dataset_root, "AQUINAS_SET1_2022_07")
    _build_info_set(dataset_root, "AQUINAS_SET2_2023_04")
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    info_mod.run()

    captured = capsys.readouterr()
    assert "AQUINAS Dataset" in captured.out
    assert "Dataset root" in captured.out
    assert "Monthly sets" in captured.out
    assert "AQUINAS_SET1_2022_07" in captured.out
    assert "AQUINAS_SET2_2023_04" in captured.out
    assert "2 (1 ACC, 1 STR)" in captured.out
    assert "~2 per sensor" in captured.out
    assert "ok" in captured.out


def test_info_reports_broken_set_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _build_info_set(dataset_root, "AQUINAS_SET1_2022_07")
    (dataset_root / "AQUINAS_SET2_2023_04").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    info_mod.run()

    captured = capsys.readouterr()
    assert "AQUINAS_SET1_2022_07" in captured.out
    assert "AQUINAS_SET2_2023_04" in captured.out
    assert "missing tables" in captured.out
