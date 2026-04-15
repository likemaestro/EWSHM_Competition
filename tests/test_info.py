import json
import sys
from pathlib import Path

import pytest

from aquinas_toolkit.cli import info as info_mod


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_default_config(workspace: Path, set_names: list[str]) -> None:
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rendered_sets = "\n".join(f"    - {name}" for name in set_names)
    (config_dir / "default.yaml").write_text(
        "data:\n"
        "  dataset_root: AQUINAS_DATASET\n"
        "  sets:\n"
        f"{rendered_sets}\n",
        encoding="utf-8",
    )


def _build_info_set(dataset_root: Path, set_name: str, *, include_start_time: bool = True) -> Path:
    set_dir = dataset_root / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    acc_table = {
        "Record_UID": [201, 202],
        "File": ["NEW_S1_DO_MID_ACC_Z_SET1_1.json", "NEW_S1_DO_MID_ACC_Z_SET1_2.json"],
        "Start_Row": [1, 1],
        "End_Row": [2, 2],
    }
    strain_table = {
        "Record_UID": [101, 102],
        "File": ["NEW_S1_DO_INF_STR_SET1_1.json", "NEW_S1_DO_INF_STR_SET1_2.json"],
        "Start_Row": [1, 1],
        "End_Row": [2, 2],
    }
    if include_start_time:
        acc_table["Start_Time"] = ["2022-07-01 00:00:00.000", "2022-07-01 00:01:00.000"]
        strain_table["Start_Time"] = ["2022-07-01 00:00:05.000", "2022-07-01 00:01:05.000"]

    acc_sensor = set_dir / "NEW_S1_DO_MID_ACC_Z"
    acc_sensor.mkdir()
    _write_json(set_dir / "TABLE_NEW_S1_DO_MID_ACC_Z_SET1.json", acc_table)

    strain_sensor = set_dir / "NEW_S1_DO_INF_STR"
    strain_sensor.mkdir()
    _write_json(set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json", strain_table)

    return set_dir


def _build_malformed_info_set(dataset_root: Path, set_name: str) -> Path:
    set_dir = dataset_root / set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    (set_dir / "NEW_S1_DO_INF_STR").mkdir()
    (set_dir / "TABLE_NEW_S1_DO_INF_STR_SET1.json").write_text("123", encoding="utf-8")
    return set_dir


def test_info_fails_when_dataset_root_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path, ["AQUINAS_SET1_2022_07"])
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    with pytest.raises(SystemExit) as exc_info:
        info_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Dataset is missing or incomplete" in captured.err


def test_info_fails_when_dataset_root_has_no_set_folders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path, ["AQUINAS_SET1_2022_07"])
    (tmp_path / "AQUINAS_DATASET").mkdir()
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    with pytest.raises(SystemExit) as exc_info:
        info_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Dataset is missing or incomplete" in captured.err


def test_info_renders_dataset_summary_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _write_default_config(tmp_path, ["AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"])
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


def test_info_renders_unknown_period_when_start_time_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _write_default_config(tmp_path, ["AQUINAS_SET1_2022_07"])
    _build_info_set(dataset_root, "AQUINAS_SET1_2022_07", include_start_time=False)
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    info_mod.run()

    captured = capsys.readouterr()
    assert "AQUINAS_SET1_2022_07" in captured.out
    assert "unknown" in captured.out
    assert "ok" in captured.out


def test_info_reports_broken_set_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _write_default_config(tmp_path, ["AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04"])
    _build_info_set(dataset_root, "AQUINAS_SET1_2022_07")
    (dataset_root / "AQUINAS_SET2_2023_04").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["aquinas", "info"])

    info_mod.run()

    captured = capsys.readouterr()
    assert "AQUINAS_SET1_2022_07" in captured.out
    assert "AQUINAS_SET2_2023_04" in captured.out
    assert "missing tables" in captured.out


def test_info_keeps_rendering_good_sets_when_one_set_has_malformed_tables(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _build_info_set(dataset_root, "AQUINAS_SET1_2022_07")
    _build_malformed_info_set(dataset_root, "AQUINAS_SET2_2023_04")
    rows: list[dict[str, str]] = []
    for set_dir in sorted(dataset_root.glob("AQUINAS_SET*")):
        try:
            rows.append(info_mod._summarize_set(set_dir))
        except Exception as exc:
            rows.append(
                {
                    "set_name": set_dir.name,
                    "sensors": "-",
                    "events": "-",
                    "period": "-",
                    "status": info_mod._short_error_message(exc),
                    "level": "error",
                }
            )

    assert rows == [
        {
            "set_name": "AQUINAS_SET1_2022_07",
            "sensors": "2 (1 ACC, 1 STR)",
            "events": "~2 per sensor",
            "period": "2022-07-01 00:00:05.000 .. 2022-07-01 00:01:05.000",
            "status": "ok",
            "level": "success",
        },
        {
            "set_name": "AQUINAS_SET2_2023_04",
            "sensors": "-",
            "events": "-",
            "period": "-",
            "status": "Unsupported JSON structure in "
            + str(dataset_root / "AQUINAS_SET2_2023_04" / "TABLE_NEW_S1_DO_INF_STR_SET1.json"),
            "level": "error",
        },
    ]
