import json
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


def _write_config_body(workspace: Path, body: str) -> None:
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(body, encoding="utf-8")


def _complete_stage(run_dir: Path, stage: str) -> None:
    run_management.mark_stage_started(run_dir, stage)
    run_management.mark_stage_completed(run_dir, stage)


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


@pytest.mark.parametrize(
    ("body", "expected_path", "error_match"),
    [
        pytest.param("{invalid\n", None, "Could not parse config file", id="invalid-yaml"),
        pytest.param("output: results\n", "results", None, id="non-mapping-output"),
        pytest.param("output:\n  results_dir: '   '\n", "results", None, id="blank-results-dir"),
    ],
)
def test_get_results_dir_handles_invalid_or_missing_output_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: str,
    expected_path: str | None,
    error_match: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_config_body(tmp_path, body)

    if error_match is not None:
        with pytest.raises(run_management.RunManagementError, match=error_match):
            run_management.get_results_dir()
        return

    results_dir = run_management.get_results_dir()
    assert results_dir == tmp_path / expected_path


def test_get_results_dir_falls_back_to_results_when_output_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_config_body(tmp_path, "model:\n  type: pca\n")

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


def test_create_run_snapshots_explicit_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    custom_config = tmp_path / "configs" / "old_deck_all_sets.yaml"
    custom_config.write_text(
        "output:\n  results_dir: custom-results\npreprocessing:\n  sensor_selection:\n"
        "    decks: [OLD]\n",
        encoding="utf-8",
    )

    run_context = run_management.create_run(name="old-deck", config_path=custom_config)

    assert run_context.results_dir == tmp_path / "custom-results"
    assert run_context.config_path.read_text(encoding="utf-8") == custom_config.read_text(
        encoding="utf-8"
    )
    metadata = run_management.read_metadata(run_context.run_dir)
    assert metadata["name"] == "old-deck"


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


@pytest.mark.parametrize(
    ("latest_payload", "error_match"),
    [
        pytest.param([], "must contain a JSON object", id="non-object"),
        pytest.param({"updated_at_utc": "2026-04-10T10:00:00Z"}, "missing run_id", id="missing-run-id"),
    ],
)
def test_resolve_run_rejects_invalid_latest_pointer_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    latest_payload: object,
    error_match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "latest.json").write_text(json.dumps(latest_payload), encoding="utf-8")

    with pytest.raises(run_management.RunManagementError, match=error_match):
        run_management.resolve_run()


def test_resolve_run_fails_when_latest_pointer_targets_run_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_context.metadata_path.unlink()

    with pytest.raises(run_management.RunManagementError, match="missing metadata.json"):
        run_management.resolve_run()


def test_resolve_run_falls_back_to_newest_run_when_pointer_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run(name="new-deck")

    # Point latest.json at a run folder that no longer exists.
    run_management.write_latest_pointer(run_context.results_dir, "does-not-exist")

    with pytest.warns(UserWarning, match="falling back to newest run"):
        resolved = run_management.resolve_run()

    assert resolved.run_id == run_context.run_id


def test_resolve_run_still_raises_when_no_valid_run_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    # Remove the only run's config so no valid fallback candidate remains.
    run_context.config_path.unlink()
    run_management.write_latest_pointer(run_context.results_dir, "does-not-exist")

    with pytest.raises(run_management.RunManagementError, match="missing run"):
        run_management.resolve_run()


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


@pytest.mark.parametrize(
    ("stage", "completed_stages", "error_match"),
    [
        pytest.param("features", (), "requires completed 'preprocess' outputs", id="features"),
        pytest.param("train", ("preprocess",), "requires completed 'features' outputs", id="train"),
        pytest.param(
            "score",
            ("preprocess", "features"),
            "requires completed 'train' outputs",
            id="score",
        ),
    ],
)
def test_validate_stage_can_run_enforces_stage_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    completed_stages: tuple[str, ...],
    error_match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    for completed_stage in completed_stages:
        _complete_stage(run_context.run_dir, completed_stage)

    with pytest.raises(run_management.RunManagementError, match=error_match):
        run_management.validate_stage_can_run(run_context.run_dir, stage)


def test_mark_stage_completed_rejects_stage_that_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()

    with pytest.raises(run_management.RunManagementError, match="not currently running"):
        run_management.mark_stage_completed(run_context.run_dir, "preprocess")


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


def test_mark_stage_terminal_transitions_preserve_progress_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_management.mark_stage_started(run_context.run_dir, "preprocess")
    run_management.write_stage_progress(
        run_context.run_dir,
        "preprocess",
        {
            "current_set": "AQUINAS_SET1_2022_07",
            "completed_sets": [],
            "written_partitions": [],
        },
    )

    run_management.mark_stage_completed(run_context.run_dir, "preprocess")
    completed_state = run_management.read_metadata(run_context.run_dir)["stages"]["preprocess"]

    assert completed_state["status"] == "completed"
    assert completed_state["progress"] == {
        "current_set": "AQUINAS_SET1_2022_07",
        "completed_sets": [],
        "written_partitions": [],
    }

    run_management.mark_stage_started(run_context.run_dir, "features")
    run_management.write_stage_progress(
        run_context.run_dir,
        "features",
        {
            "current_set": "chunk-1",
            "completed_sets": ["chunk-0"],
            "written_partitions": ["chunk-0"],
        },
    )
    run_management.mark_stage_failed(run_context.run_dir, "features", "boom")
    failed_state = run_management.read_metadata(run_context.run_dir)["stages"]["features"]

    assert failed_state["status"] == "failed"
    assert failed_state["progress"] == {
        "current_set": "chunk-1",
        "completed_sets": ["chunk-0"],
        "written_partitions": ["chunk-0"],
    }


def test_mark_stage_started_allows_retry_after_failure_and_clears_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run()
    run_management.mark_stage_started(run_context.run_dir, "features")
    run_management.mark_stage_failed(run_context.run_dir, "features", "boom")

    failed_state = run_management.read_metadata(run_context.run_dir)["stages"]["features"]
    assert failed_state["status"] == "failed"
    assert failed_state["error"] == "boom"
    assert failed_state["completed_at_utc"] is not None

    run_management.mark_stage_started(run_context.run_dir, "features")
    stage_state = run_management.read_metadata(run_context.run_dir)["stages"]["features"]

    assert stage_state["status"] == "running"
    assert stage_state["error"] is None
    assert stage_state["started_at_utc"] is not None
    assert stage_state["completed_at_utc"] is None
