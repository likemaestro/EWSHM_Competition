from io import StringIO

from aquinas_toolkit.cli import terminal


def test_build_console_honors_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    buffer = StringIO()
    console = terminal.build_console(file=buffer, force_terminal=True)

    console.print("[header]AQUINAS[/header]")

    assert "\x1b[" not in buffer.getvalue()


def test_render_run_summary_plain_text_contains_core_fields(tmp_path) -> None:
    view = terminal.render_run_summary(
        title="Created Run",
        run_id="2026-03-31T21-45-00Z",
        run_dir=tmp_path / "results" / "2026-03-31T21-45-00Z",
        config_path=tmp_path / "results" / "2026-03-31T21-45-00Z" / "config.yaml",
    )

    plain_text = str(view)
    assert "Created Run" in plain_text
    assert "Run ID" in plain_text
    assert "Run directory" in plain_text
    assert "Config snapshot" in plain_text


def test_render_viz_summary_plain_text_contains_core_fields(tmp_path) -> None:
    view = terminal.render_viz_summary(
        run_id="2026-03-31T21-45-00Z",
        output_dir=tmp_path / "results" / "2026-03-31T21-45-00Z" / "visualization",
        manifest_path=tmp_path / "results" / "2026-03-31T21-45-00Z" / "visualization" / "manifest.json",
        index_path=tmp_path / "results" / "2026-03-31T21-45-00Z" / "visualization" / "index.html",
    )

    plain_text = str(view)
    assert "Run ID" in plain_text
    assert "Bundle directory" in plain_text
    assert "Manifest" in plain_text
    assert "Viewer index" in plain_text
