from datetime import UTC, datetime
from pathlib import Path

import pytest

from aquinas_toolkit.utils import run_management


def _write_default_config(workspace: Path, *, results_dir: str = "results") -> None:
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        f"output:\n  results_dir: {results_dir}\n",
        encoding="utf-8",
    )


def test_generate_run_id_uses_readable_windows_safe_timestamp(tmp_path: Path) -> None:
    run_id = run_management.generate_run_id(
        tmp_path,
        now=datetime(2026, 3, 31, 21, 45, 0, tzinfo=UTC),
    )

    assert run_id == "2026-03-31T21-45-00Z"


def test_generate_run_id_adds_suffix_when_timestamp_collides(tmp_path: Path) -> None:
    (tmp_path / "2026-03-31T21-45-00Z").mkdir()

    run_id = run_management.generate_run_id(
        tmp_path,
        now=datetime(2026, 3, 31, 21, 45, 0, tzinfo=UTC),
    )

    assert run_id == "2026-03-31T21-45-00Z-02"


def test_get_results_dir_uses_default_config_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path, results_dir="artifacts")

    results_dir = run_management.get_results_dir()

    assert results_dir == tmp_path / "artifacts"


def test_get_results_dir_falls_back_to_results_when_output_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("model:\n  type: pca\n", encoding="utf-8")

    results_dir = run_management.get_results_dir()

    assert results_dir == tmp_path / "results"


def test_create_run_uses_configured_results_dir_and_initializes_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path, results_dir="artifacts")

    run_context = run_management.create_run(name="custom-root")
    metadata = run_management.read_metadata(run_context.run_dir)

    assert run_context.results_dir == tmp_path / "artifacts"
    assert run_context.config_path == run_context.run_dir / "config.yaml"
    assert metadata["name"] == "custom-root"
    assert metadata["stages"]["preprocess"]["status"] == "not_started"
    assert (run_context.results_dir / "latest.json").is_file()


def test_resolve_run_fails_when_config_snapshot_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_context.config_path.unlink()

    with pytest.raises(run_management.RunManagementError, match="missing config.yaml"):
        run_management.resolve_run(run_context.run_id)


def test_validate_stage_can_run_rejects_completed_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_management.mark_stage_started(run_context.run_dir, "preprocess")
    run_management.mark_stage_completed(run_context.run_dir, "preprocess")

    with pytest.raises(run_management.RunManagementError, match="already completed"):
        run_management.validate_stage_can_run(run_context.run_dir, "preprocess")


def test_validate_stage_can_run_rejects_running_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_management.mark_stage_started(run_context.run_dir, "preprocess")

    with pytest.raises(run_management.RunManagementError, match="already marked as running"):
        run_management.validate_stage_can_run(run_context.run_dir, "preprocess")


def test_mark_stage_failed_preserves_started_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_management.mark_stage_started(run_context.run_dir, "preprocess")
    started_at = run_management.read_metadata(run_context.run_dir)["stages"]["preprocess"][
        "started_at_utc"
    ]

    run_management.mark_stage_failed(run_context.run_dir, "preprocess", "bad data")
    stage_state = run_management.read_metadata(run_context.run_dir)["stages"]["preprocess"]

    assert stage_state["status"] == "failed"
    assert stage_state["error"] == "bad data"
    assert stage_state["started_at_utc"] == started_at
    assert stage_state["completed_at_utc"] is not None
