import json
import re
import sys
from pathlib import Path

import pytest

from aquinas_toolkit.cli import run as run_mod
from aquinas_toolkit.cli import viz as viz_mod
from aquinas_toolkit.utils.run_management import create_run
from aquinas_toolkit.visualization import (
    build_bridge_geometry,
    build_sensor_layout,
    build_visualization_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_default_config(workspace: Path) -> None:
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "configs" / "default.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  dataset_root: AQUINAS_DATASET",
                "  sets:",
                "    - AQUINAS_SET1_2022_07",
                "    - AQUINAS_SET2_2023_04",
                "    - AQUINAS_SET3_2023_08",
                "output:",
                "  results_dir: results",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_visualization_dataset(root: Path, dataset_name: str, offset: float) -> None:
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    set_match = re.search(r"(SET\d+)", dataset_name)
    if set_match is None:
        raise ValueError(f"Could not derive set label from {dataset_name}")
    set_label = set_match.group(1)
    date_prefix = {
        "SET1": "2022-07-01",
        "SET2": "2023-04-01",
        "SET3": "2023-08-01",
    }[set_label]

    sensors = {
        "OLD_S1_DO_MID_ACC_Z": {
            "records": [
                ("00:00:00", "00:00:10", 0.1 + offset, 1.0 + offset, 10.0 + offset, 0.5 + offset),
                ("00:00:20", "00:00:30", 0.2 + offset, 1.1 + offset, 10.5 + offset, 0.6 + offset),
            ]
        },
        "NEW_S1_DO_MID_ACC_Z": {
            "records": [
                ("00:00:00", "00:00:10", 0.15 + offset, 1.2 + offset, 10.2 + offset, 0.45 + offset),
                ("00:00:20", "00:00:30", 0.25 + offset, 1.15 + offset, 10.8 + offset, 0.65 + offset),
            ]
        },
        "OLD_S1_DO_INF_STR": {
            "records": [
                ("00:00:00", "00:00:10", 0.03 + offset, -0.8 - offset, 11.2 + offset, 0.08 + offset),
                ("00:00:20", "00:00:30", 0.05 + offset, -0.7 - offset, 11.7 + offset, 0.1 + offset),
            ]
        },
        "NEW_S1_DO_INF_STR": {
            "records": [
                ("00:00:00", "00:00:10", 0.04 + offset, -0.75 - offset, 11.1 + offset, 0.09 + offset),
                ("00:00:20", "00:00:30", 0.06 + offset, -0.65 - offset, 11.6 + offset, 0.12 + offset),
            ]
        },
        "OLD_S1_DO_INT_ACC_Y": {
            "records": [
                ("00:00:00", "00:00:10", 0.12 + offset, 0.4 + offset, 9.9 + offset, 0.55 + offset),
            ]
        },
        "NEW_S1_DO_INT_ACC_Y": {
            "records": [
                ("00:00:00", "00:00:10", 0.18 + offset, 0.5 + offset, 9.7 + offset, 0.52 + offset),
            ]
        },
    }

    for sensor_name, definition in sensors.items():
        sensor_dir = dataset_dir / sensor_name
        sensor_dir.mkdir(parents=True, exist_ok=True)
        table_name = f"TABLE_{sensor_name}_{set_label}.json"
        raw_name = f"{sensor_name}_{set_label}_1.json"
        table_payload = {
            "Record_UID": [],
            "File": [],
            "Start_Row": [],
            "End_Row": [],
            "Start_Time": [],
            "End_Time": [],
            "Duration": [],
            "Mean_Value": [],
            "Temperature": [],
            "Range": [],
        }
        raw_payload = {"timestamp": [], sensor_name: []}

        for row_index, record in enumerate(definition["records"], start=1):
            start_time, end_time, range_value, mean_value, temperature, amplitude = record
            table_payload["Record_UID"].append(1000 + row_index)
            table_payload["File"].append(raw_name)
            table_payload["Start_Row"].append((row_index - 1) * 4 + 1)
            table_payload["End_Row"].append((row_index - 1) * 4 + 4)
            table_payload["Start_Time"].append(f"{date_prefix} {start_time}")
            table_payload["End_Time"].append(f"{date_prefix} {end_time}")
            table_payload["Duration"].append(10.0)
            table_payload["Mean_Value"].append(mean_value)
            table_payload["Temperature"].append(temperature)
            table_payload["Range"].append(range_value)

            for step in range(4):
                raw_payload["timestamp"].append(
                    f"{date_prefix}T{start_time[:5]}:{step * 10:02d}.000Z"
                )
                raw_payload[sensor_name].append(round(mean_value + amplitude * step, 6))

        _write_json(dataset_dir / table_name, table_payload)
        _write_json(sensor_dir / raw_name, raw_payload)


def _prepare_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    dataset_root = tmp_path / "AQUINAS_DATASET"
    _build_visualization_dataset(dataset_root, "AQUINAS_SET1_2022_07", 0.00)
    _build_visualization_dataset(dataset_root, "AQUINAS_SET2_2023_04", 0.10)
    _build_visualization_dataset(dataset_root, "AQUINAS_SET3_2023_08", 0.25)


def test_build_sensor_layout_rejects_upstream_acc_y() -> None:
    with pytest.raises(ValueError):
        build_sensor_layout(["OLD_S1_UP_INT_ACC_Y"])


def test_build_sensor_layout_exports_mount_metadata() -> None:
    layout = build_sensor_layout(
        [
            "OLD_S1_UP_SUP_STR",
            "OLD_S1_DO_INF_STR",
            "OLD_S1_DO_INT_ACC_Y",
            "NEW_S2_UP_SHE_STR",
        ]
    )

    old_sup = next(row for row in layout if row["sensor_id"] == "OLD_S1_UP_SUP_STR")
    old_inf = next(row for row in layout if row["sensor_id"] == "OLD_S1_DO_INF_STR")
    old_acc_y = next(row for row in layout if row["sensor_id"] == "OLD_S1_DO_INT_ACC_Y")
    new_she = next(row for row in layout if row["sensor_id"] == "NEW_S2_UP_SHE_STR")

    assert old_sup["mount_surface"] == "top_slab_exterior"
    assert old_sup["surface_normal"] == {"x": 0.0, "y": 1.0, "z": 0.0}
    assert old_sup["glyph_orientation"] == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert old_sup["local_position"]["y"] > old_sup["anchor_local"]["y"]
    assert old_sup["compact_z"] == pytest.approx(0.1733, abs=1e-4)

    assert old_inf["mount_surface"] == "bottom_slab_exterior"
    assert old_inf["surface_normal"] == {"x": 0.0, "y": -1.0, "z": 0.0}
    assert old_inf["local_position"]["y"] < old_inf["anchor_local"]["y"]
    assert old_inf["compact_z"] == pytest.approx(0.0982, abs=1e-4)

    assert old_acc_y["mount_surface"] == "web_outer_face"
    assert old_acc_y["glyph_orientation"] == {"x": 0.0, "y": 0.0, "z": 1.0}
    assert old_acc_y["surface_normal"] == {"x": 0.0, "y": 0.0, "z": -1.0}
    assert old_acc_y["compact_z"] == pytest.approx(0.0851, abs=1e-3)

    assert new_she["mount_surface"] == "web_outer_face"
    assert new_she["glyph_orientation"]["x"] < 0
    assert new_she["glyph_orientation"]["y"] > 0
    assert new_she["surface_normal"]["z"] > 0


def test_build_bridge_geometry_exports_shared_cross_section() -> None:
    geometry = build_bridge_geometry()
    cross_section = geometry["cross_section"]
    first_segment = geometry["deck_meshes"][0]["segments"][0]

    assert geometry["world"]["meters_per_normalized_unit"] == 45.0
    assert geometry["world"]["bridge_length_m"] == 90.0
    assert cross_section["depth"] == pytest.approx(2.0 / 45.0)
    assert cross_section["top_slab_width"] == pytest.approx(7.5 / 45.0)
    assert cross_section["bottom_slab_width"] == pytest.approx(4.7 / 45.0)
    assert cross_section["slab_thickness"] == pytest.approx(0.30 / 45.0)
    assert cross_section["web_thickness"] == pytest.approx(0.35 / 45.0)
    assert cross_section["overhang_width"] == pytest.approx(1.75 / 45.0)
    assert geometry["view_modes"]["compact"]["deck_centers"] == {"OLD": 0.14, "NEW": -0.14}
    assert geometry["view_modes"]["exploded"]["deck_centers"] == {"OLD": 0.22, "NEW": -0.22}
    assert first_segment["x_start"] == 0.0
    assert first_segment["x_end"] == 1.0


def test_build_visualization_artifacts_exports_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path, monkeypatch)
    run_context = create_run(name="viewer")

    result = build_visualization_artifacts(run_context, include_waveforms=True)

    assert result.output_dir.is_dir()
    assert result.index_path.is_file()
    assert result.manifest_path.is_file()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2026-04-13"
    assert manifest["default_dataset"] == "AQUINAS_SET3_2023_08"
    assert manifest["supported_measurement_families"] == ["ALL", "ACC", "STR"]

    index_html = (result.output_dir / "index.html").read_text(encoding="utf-8")
    viewer_css = (result.output_dir / "viewer.css").read_text(encoding="utf-8")
    assert (result.output_dir / "logo.png").is_file()
    assert "./logo.png" in index_html
    assert "3D View" in index_html
    assert "Sensor Analysis" in index_html
    assert "Datasets" in index_html
    assert "Manrope" in viewer_css

    sensor_layout = json.loads((result.output_dir / "sensor_layout.json").read_text(encoding="utf-8"))
    old_inf = next(row for row in sensor_layout if row["sensor_id"] == "OLD_S1_DO_INF_STR")
    old_acc_y = next(row for row in sensor_layout if row["sensor_id"] == "OLD_S1_DO_INT_ACC_Y")
    assert old_inf["homologous_sensor_id"] == "NEW_S1_DO_INF_STR"
    assert old_inf["section"] == "MID"
    assert old_inf["mount_surface"] == "bottom_slab_exterior"
    assert old_inf["local_position"]["y"] < 0
    assert old_acc_y["axis_or_fibre"] == "Y"
    assert old_acc_y["section"] == "INT"
    assert old_acc_y["glyph_orientation"] == {"x": 0.0, "y": 0.0, "z": 1.0}

    bridge_geometry = json.loads((result.output_dir / "bridge_geometry.json").read_text(encoding="utf-8"))
    assert bridge_geometry["cross_section"]["web_top_outer_width"] == pytest.approx(4.0 / 45.0)
    assert bridge_geometry["cross_section"]["inner_bottom_width"] == pytest.approx(4.0 / 45.0)
    assert bridge_geometry["world"]["bridge_length_m"] == 90.0

    metrics = json.loads((result.output_dir / "sensor_metrics.json").read_text(encoding="utf-8"))
    assert any(
        row["sensor_id"] == "OLD_S1_DO_MID_ACC_Z"
        and row["dataset"] == "AQUINAS_SET1_2022_07"
        and row["metric_id"] == "event_count"
        and row["value"] == 2
        for row in metrics
    )

    event_groups = json.loads((result.output_dir / "event_groups.json").read_text(encoding="utf-8"))
    shared_event = next(
        row
        for row in event_groups
        if row["dataset"] == "AQUINAS_SET1_2022_07"
        and row["deck"] == "OLD"
        and row["start_time_utc"] == "2022-07-01T00:00:00Z"
    )
    assert shared_event["sensor_count"] == 3
    assert "Record_UID" not in json.dumps(event_groups)
    assert shared_event["waveform_preview_path"] is not None
    assert (result.output_dir / shared_event["waveform_preview_path"]).is_file()


def test_viz_cli_build_command_writes_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_workspace(tmp_path, monkeypatch)
    run_context = create_run(name="viewer")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "viz", "build", "--run-id", run_context.run_id, "--set", "AQUINAS_SET2_2023_04"],
    )

    viz_mod.run()

    captured = capsys.readouterr()
    assert "Visualization Bundle" in captured.out
    assert "Viewer index" in captured.out
    assert (tmp_path / "results" / run_context.run_id / "visualization" / "manifest.json").is_file()


def test_viz_cli_open_uses_local_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path, monkeypatch)
    run_context = create_run(name="viewer")
    build_visualization_artifacts(run_context)

    captured: dict[str, object] = {}

    def fake_serve_bundle(
        *,
        bundle_dir: Path,
        host: str,
        port: int,
        open_browser: bool,
    ) -> None:
        captured["bundle_dir"] = bundle_dir
        captured["host"] = host
        captured["port"] = port
        captured["open_browser"] = open_browser

    monkeypatch.setattr(viz_mod, "_serve_bundle", fake_serve_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "viz", "open", "--run-id", run_context.run_id, "--port", "8765"],
    )

    viz_mod.run()

    assert captured["bundle_dir"] == tmp_path / "results" / run_context.run_id / "visualization"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["open_browser"] is True


def test_run_preprocess_also_builds_visualization_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess", "--name", "viewer"])

    run_mod.run()

    results_root = tmp_path / "results"
    run_id = json.loads((results_root / "latest.json").read_text(encoding="utf-8"))["run_id"]
    bundle_dir = results_root / run_id / "visualization"
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "index.html").is_file()
