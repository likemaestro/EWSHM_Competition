import json
import re
import sys
from pathlib import Path

import pytest

from aquinas_toolkit.cli import run as run_mod
from aquinas_toolkit.cli.main import main
from aquinas_toolkit.utils import run_management


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_default_config(
    workspace: Path,
    *,
    body: str = "output:\n  results_dir: results\n",
) -> None:
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(body, encoding="utf-8")


def test_main_shows_usage_when_no_subcommand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas"])

    main()

    captured = capsys.readouterr()
    assert "AQUINAS CLI" in captured.out
    assert "Usage: aquinas <command>" in captured.out
    assert "run" in captured.out
    assert "info" in captured.out
    assert "viz" in captured.out
    assert captured.err == ""


def test_main_shows_usage_for_help_alias(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas", "help"])

    main()

    captured = capsys.readouterr()
    assert "AQUINAS CLI" in captured.out
    assert "Usage: aquinas <command>" in captured.out
    assert "help" in captured.out
    assert captured.err == ""


def test_main_shows_usage_for_help_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas", "--help"])

    main()

    captured = capsys.readouterr()
    assert "AQUINAS CLI" in captured.out
    assert "Usage: aquinas <command>" in captured.out
    assert captured.err == ""


def test_main_fails_for_unknown_subcommand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas", "unknown"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown command: unknown" in captured.err
    assert "AQUINAS CLI" in captured.out


def test_run_help_mentions_name_and_run_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "AQUINAS RUN" in captured.out
    assert "Usage: aquinas run" in captured.out
    assert "--name" in captured.out
    assert "--run-id" in captured.out
    assert "--verbose" in captured.out


def test_viz_help_mentions_build_and_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aquinas_toolkit.cli import viz as viz_mod

    monkeypatch.setattr(sys, "argv", ["aquinas", "viz", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        viz_mod.run()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "AQUINAS VIZ" in captured.out
    assert "aquinas viz build" in captured.out
    assert "aquinas viz open" in captured.out


def test_run_full_pipeline_creates_run_and_marks_preprocess_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(
        run_mod,
        "_execute_stage",
        lambda stage, run_context: None if stage == "preprocess" else (_ for _ in ()).throw(
            run_mod.StageNotImplementedError("Not yet implemented")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["aquinas", "run"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Running full pipeline for run" in captured.out
    assert "STEP  pipeline 1/4 completed (preprocess)" in captured.out
    assert "START preprocess Run" in captured.out
    assert "START features Run" in captured.out
    assert "Pipeline progress" not in captured.out
    assert "aquinas viz open" in captured.out
    assert "Not yet implemented" in captured.err

    results_dir = tmp_path / "results"
    latest = _read_json(results_dir / "latest.json")
    run_id = latest["run_id"]
    metadata = _read_json(results_dir / run_id / "metadata.json")
    assert metadata["stages"]["preprocess"]["status"] == "completed"
    assert metadata["stages"]["features"]["status"] == "failed"


def test_run_preprocess_creates_snapshot_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(
        tmp_path,
        body="output:\n  results_dir: results\nmodel:\n  type: pca\n",
    )

    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "preprocess", "--name", "baseline-run"],
    )

    run_mod.run()

    captured = capsys.readouterr()
    assert "Created Run" in captured.out
    assert "Run ID" in captured.out
    assert "Run directory" in captured.out
    assert "Config snapshot" in captured.out
    assert "DONE" in captured.out
    assert "preprocess" in captured.out
    assert "aquinas viz open" in captured.out

    results_dir = tmp_path / "results"
    latest = _read_json(results_dir / "latest.json")
    run_id = latest["run_id"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(?:-\d{2})?", run_id)

    run_dir = results_dir / run_id
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == (
        "output:\n  results_dir: results\nmodel:\n  type: pca\n"
    )

    metadata = _read_json(run_dir / "metadata.json")
    assert metadata["run_id"] == run_id
    assert metadata["name"] == "baseline-run"
    assert metadata["stages"]["preprocess"]["status"] == "completed"
    assert metadata["stages"]["preprocess"]["error"] is None
    assert (run_dir / "stages" / "preprocess").is_dir()
    assert not (run_dir / "stages" / "features").exists()
    assert not (run_dir / "stages" / "train").exists()
    assert not (run_dir / "stages" / "score").exists()


def test_run_preprocess_rejects_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "preprocess", "--run-id", "existing-run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "`--run-id` cannot be used" in captured.err


def test_run_invalid_stage_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "bogus"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
    assert "AQUINAS RUN" in captured.out


def test_run_features_uses_latest_pointer_and_run_snapshot_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(
        tmp_path,
        body="output:\n  results_dir: results\nmodel:\n  type: pca\n",
    )

    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    (tmp_path / "configs" / "default.yaml").write_text(
        "output:\n  results_dir: results\nmodel:\n  type: isolation_forest\n",
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def fake_execute(stage: str, run_context: run_management.RunContext) -> None:
        captured["stage"] = stage
        captured["config_text"] = run_context.config_path.read_text(encoding="utf-8")

    monkeypatch.setattr(run_mod, "_execute_stage", fake_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])
    run_mod.run()

    assert captured["stage"] == "features"
    assert "type: pca" in captured["config_text"]
    assert "type: isolation_forest" not in captured["config_text"]

    latest = _read_json(tmp_path / "results" / "latest.json")
    metadata = _read_json(tmp_path / "results" / latest["run_id"] / "metadata.json")
    assert metadata["stages"]["features"]["status"] == "completed"


def test_run_verbose_flag_is_propagated_to_stage_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    captured: dict[str, bool] = {}

    def fake_execute(stage: str, run_context: run_management.RunContext) -> None:
        captured["verbose"] = run_context.verbose

    monkeypatch.setattr(run_mod, "_execute_stage", fake_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess", "--verbose"])
    run_mod.run()

    assert captured["verbose"] is True


def test_run_features_explicit_run_id_overrides_latest_and_updates_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess", "--name", "first"])
    run_mod.run()
    first_run_id = _read_json(tmp_path / "results" / "latest.json")["run_id"]

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess", "--name", "second"])
    run_mod.run()
    second_run_id = _read_json(tmp_path / "results" / "latest.json")["run_id"]
    assert second_run_id != first_run_id

    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--run-id", first_run_id],
    )
    run_mod.run()

    latest = _read_json(tmp_path / "results" / "latest.json")
    assert latest["run_id"] == first_run_id


def test_run_features_rejects_name_for_existing_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--name", "should-fail"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "`--name` can only be used when creating a new run" in captured.err


def test_run_full_pipeline_executes_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    executed_stages: list[str] = []

    def fake_execute(stage: str, run_context: run_management.RunContext) -> None:
        executed_stages.append(stage)

    monkeypatch.setattr(run_mod, "_execute_stage", fake_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "--name", "full-run"])

    run_mod.run()

    assert executed_stages == ["preprocess", "features", "train", "score"]
    latest = _read_json(tmp_path / "results" / "latest.json")
    metadata = _read_json(tmp_path / "results" / latest["run_id"] / "metadata.json")
    assert all(metadata["stages"][stage]["status"] == "completed" for stage in executed_stages)


def test_run_full_pipeline_stops_after_train_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    executed_stages: list[str] = []

    def fake_execute(stage: str, run_context: run_management.RunContext) -> None:
        executed_stages.append(stage)
        if stage == "train":
            raise RuntimeError("train boom")

    monkeypatch.setattr(run_mod, "_execute_stage", fake_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "train boom" in captured.err
    assert "STEP  pipeline 2/4 completed (features)" in captured.out
    assert "STEP  pipeline 3/4 completed (train)" not in captured.out
    assert "STEP  pipeline 4/4 completed (score)" not in captured.out
    assert "START train Run" in captured.out
    assert executed_stages == ["preprocess", "features", "train"]

    latest = _read_json(tmp_path / "results" / "latest.json")
    metadata = _read_json(tmp_path / "results" / latest["run_id"] / "metadata.json")
    assert metadata["stages"]["preprocess"]["status"] == "completed"
    assert metadata["stages"]["features"]["status"] == "completed"
    assert metadata["stages"]["train"]["status"] == "failed"
    assert metadata["stages"]["train"]["error"] == "train boom"
    assert metadata["stages"]["score"]["status"] == "not_started"


def test_run_stage_failure_records_error_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    def fail_execute(stage: str, run_context: run_management.RunContext) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_mod, "_execute_stage", fail_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "boom" in captured.err

    latest = _read_json(tmp_path / "results" / "latest.json")
    metadata = _read_json(tmp_path / "results" / latest["run_id"] / "metadata.json")
    assert metadata["stages"]["features"]["status"] == "failed"
    assert metadata["stages"]["features"]["error"] == "boom"


def test_run_failed_stage_can_be_retried_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    run_id = _read_json(tmp_path / "results" / "latest.json")["run_id"]

    def fail_execute(stage: str, run_context: run_management.RunContext) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_mod, "_execute_stage", fail_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features", "--run-id", run_id])

    with pytest.raises(SystemExit):
        run_mod.run()

    failed_state = _read_json(tmp_path / "results" / run_id / "metadata.json")["stages"]["features"]
    assert failed_state["status"] == "failed"
    assert failed_state["error"] == "boom"
    assert failed_state["completed_at_utc"] is not None

    observed_running_state: dict[str, str | None] = {}

    def succeed_execute(stage: str, run_context: run_management.RunContext) -> None:
        observed_running_state.update(_read_json(run_context.metadata_path)["stages"]["features"])

    monkeypatch.setattr(run_mod, "_execute_stage", succeed_execute)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features", "--run-id", run_id])
    run_mod.run()

    assert observed_running_state["status"] == "running"
    assert observed_running_state["error"] is None
    assert observed_running_state["completed_at_utc"] is None

    stage_state = _read_json(tmp_path / "results" / run_id / "metadata.json")["stages"]["features"]
    assert stage_state["status"] == "completed"
    assert stage_state["error"] is None
    assert stage_state["started_at_utc"] is not None
    assert stage_state["completed_at_utc"] is not None


def test_run_features_fails_clearly_when_latest_pointer_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "No active run pointer found" in captured.err


def test_run_features_fails_clearly_when_latest_pointer_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "latest.json").write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Could not parse active run pointer" in captured.err


def test_run_features_fails_clearly_when_latest_pointer_is_missing_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "latest.json").write_text(
        json.dumps({"updated_at_utc": "2026-04-10T10:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "features"])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "missing run_id" in captured.err


def test_run_invalid_explicit_run_id_does_not_change_latest_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest_before = _read_json(tmp_path / "results" / "latest.json")

    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--run-id", "missing-run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Run 'missing-run' was not found" in captured.err
    latest_after = _read_json(tmp_path / "results" / "latest.json")
    assert latest_after == latest_before


def test_run_features_enforces_prerequisite_stage_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    run_context = run_management.create_run(name="needs-preprocess")

    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--run-id", run_context.run_id],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "requires completed 'preprocess' outputs" in captured.err


def test_run_completed_stage_cannot_be_re_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = _read_json(tmp_path / "results" / "latest.json")
    run_id = latest["run_id"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--run-id", run_id],
    )
    run_mod.run()

    monkeypatch.setattr(
        sys,
        "argv",
        ["aquinas", "run", "features", "--run-id", run_id],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_mod.run()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "already completed" in captured.err


def test_main_dispatches_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "--help"])
    monkeypatch.setattr(run_mod, "run", fake_run)

    main()

    assert called is True


def test_run_single_stage_does_not_show_pipeline_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])

    run_mod.run()

    captured = capsys.readouterr()
    assert "Pipeline progress" not in captured.out
    assert "STEP  pipeline" not in captured.out


def test_run_writes_debug_log_even_without_verbose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_default_config(tmp_path)
    monkeypatch.setattr(run_mod, "_execute_stage", lambda stage, run_context: None)
    monkeypatch.setattr(sys, "argv", ["aquinas", "run", "preprocess"])
    run_mod.run()

    latest = _read_json(tmp_path / "results" / "latest.json")
    debug_log = tmp_path / "results" / latest["run_id"] / "debug.log"
    assert debug_log.is_file()
    log_text = debug_log.read_text(encoding="utf-8")
    assert "event=RUN_START" in log_text
    assert "event=STAGE_START" in log_text
    assert "stage=preprocess" in log_text
    assert "event=TIMING" in log_text
    assert "phase=stage_total_s" in log_text
