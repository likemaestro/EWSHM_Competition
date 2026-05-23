import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import pytest

from aquinas_toolkit.dataset_fetch import (
    DatasetArchiveSource,
    DatasetFetchError,
    _download_archive,
    fetch_dataset,
)
from aquinas_toolkit.utils.dataset_config import DatasetLayout, inspect_dataset_layout, load_dataset_layout
from aquinas_toolkit.utils.dataset_paths import find_workspace_root


def _build_dataset_zip(zip_path: Path, *, set_names: tuple[str, ...], root_dir_name: str = "AQUINAS_DATASET") -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{root_dir_name}/README.md", "dataset bundle")
        for set_name in set_names:
            archive.writestr(f"{root_dir_name}/{set_name}/TABLE_SENSOR_SET1.json", "{}")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_load_dataset_layout_defaults_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    layout = load_dataset_layout(config_path=tmp_path / "missing.yaml")
    assert layout.dataset_root == find_workspace_root() / "AQUINAS_DATASET"
    assert len(layout.set_names) == 5
    assert layout.set_names[0] == "AQUINAS_SET1_2022_07"


def test_load_dataset_layout_prefers_current_workspace_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text("data:\n  dataset_root: custom_data\n", encoding="utf-8")
    monkeypatch.chdir(workspace)

    layout = load_dataset_layout()

    assert layout.dataset_root == workspace / "custom_data"


def test_inspect_dataset_layout_reports_stub_root(tmp_path: Path) -> None:
    layout = DatasetLayout(
        dataset_root=tmp_path / "AQUINAS_DATASET",
        set_names=("AQUINAS_SET1_2022_07",),
    )
    layout.dataset_root.mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "README.md").write_text("stub", encoding="utf-8")

    status = inspect_dataset_layout(layout)

    assert status.dataset_root_exists is True
    assert status.dataset_root_is_stub is True
    assert status.dataset_is_complete is False
    assert status.missing_set_names == ("AQUINAS_SET1_2022_07",)


def test_fetch_dataset_extracts_valid_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_names = ("AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04")
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(
        dataset_root=tmp_path / "AQUINAS_DATASET",
        set_names=set_names,
    )

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    result = fetch_dataset(layout, source=source)
    assert result == layout.dataset_root
    assert (result / set_names[0]).is_dir()
    assert (result / set_names[1]).is_dir()


def test_fetch_dataset_requires_force_when_destination_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    layout.dataset_root.mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / set_names[0]).mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "junk.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    with pytest.raises(DatasetFetchError) as exc_info:
        fetch_dataset(layout, source=source)

    assert "Use --force to replace it" in str(exc_info.value)


def test_fetch_dataset_force_aborts_on_wrong_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    layout.dataset_root.mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / set_names[0]).mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "junk.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _: "NOPE")
    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    with pytest.raises(DatasetFetchError) as exc_info:
        fetch_dataset(layout, source=source, force=True, assume_yes=False)

    assert "Overwrite aborted by user" in str(exc_info.value)


def test_fetch_dataset_force_yes_replaces_existing_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    layout.dataset_root.mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "junk.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    fetch_dataset(layout, source=source, force=True, assume_yes=True)

    assert not (layout.dataset_root / "junk.txt").exists()
    assert (layout.dataset_root / set_names[0]).is_dir()


def test_fetch_dataset_repairs_incomplete_existing_root_without_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04")
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    (layout.dataset_root / set_names[0]).mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "partial.txt").write_text("old", encoding="utf-8")

    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "OVERWRITE")
    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    fetch_dataset(layout, source=source)

    assert prompts
    assert "Dataset root is incomplete" in prompts[0]
    assert "Missing set folders: AQUINAS_SET2_2023_04" in prompts[0]
    assert not (layout.dataset_root / "partial.txt").exists()
    assert (layout.dataset_root / set_names[0]).is_dir()
    assert (layout.dataset_root / set_names[1]).is_dir()


def test_fetch_dataset_incomplete_root_abort_keeps_existing_contents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07", "AQUINAS_SET2_2023_04")
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    (layout.dataset_root / set_names[0]).mkdir(parents=True, exist_ok=True)
    sentinel = layout.dataset_root / "partial.txt"
    sentinel.write_text("keep", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _: "NOPE")
    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    with pytest.raises(DatasetFetchError) as exc_info:
        fetch_dataset(layout, source=source)

    assert "Dataset repair aborted by user" in str(exc_info.value)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (layout.dataset_root / set_names[0]).is_dir()
    assert not (layout.dataset_root / set_names[1]).exists()


def test_fetch_dataset_creates_configured_destination_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names, root_dir_name="downloaded-root")
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(
        dataset_root=tmp_path / "custom" / "AQUINAS_DATASET",
        set_names=set_names,
    )

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    result = fetch_dataset(layout, source=source)

    assert result == layout.dataset_root
    assert result.is_dir()
    assert (result / set_names[0]).is_dir()


def test_fetch_dataset_treats_readme_only_root_as_fresh_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    layout.dataset_root.mkdir(parents=True, exist_ok=True)
    (layout.dataset_root / "README.md").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    result = fetch_dataset(layout, source=source)

    assert result == layout.dataset_root
    assert (layout.dataset_root / "README.md").read_text(encoding="utf-8") == "dataset bundle"
    assert (layout.dataset_root / set_names[0]).is_dir()


def test_fetch_dataset_treats_empty_root_as_fresh_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_names = ("AQUINAS_SET1_2022_07",)
    local_zip = tmp_path / "source.zip"
    _build_dataset_zip(local_zip, set_names=set_names)
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256=_sha256_of(local_zip),
    )
    layout = DatasetLayout(dataset_root=tmp_path / "AQUINAS_DATASET", set_names=set_names)
    layout.dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch._download_archive",
        lambda src, destination: shutil.copy2(local_zip, destination),
    )

    result = fetch_dataset(layout, source=source)

    assert result == layout.dataset_root
    assert (layout.dataset_root / set_names[0]).is_dir()


class _FakeResponse:
    def __init__(self, payload: bytes, *, content_length: str | None) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _FakeProgress:
    def __init__(self) -> None:
        self.added_total: int | None = None
        self.updates: list[int] = []

    def __enter__(self) -> "_FakeProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def add_task(self, description: str, *, total: int | None, start: bool = True) -> int:
        assert "Downloading dataset archive" in description
        assert start is True
        self.added_total = total
        return 1

    def update(self, task_id: int, *, advance: int) -> None:
        assert task_id == 1
        self.updates.append(advance)


def test_download_archive_tracks_known_content_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"a" * (1024 * 1024 + 123)
    progress = _FakeProgress()
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256="0" * 64,
    )

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch.urlopen",
        lambda request, timeout=300: _FakeResponse(payload, content_length=str(len(payload))),
    )
    monkeypatch.setattr("aquinas_toolkit.dataset_fetch.terminal.build_download_progress", lambda transient=True: progress)

    destination = tmp_path / "download.zip"
    _download_archive(source, destination)

    assert destination.read_bytes() == payload
    assert progress.added_total == len(payload)
    assert sum(progress.updates) == len(payload)


def test_download_archive_handles_unknown_content_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"abc123"
    progress = _FakeProgress()
    source = DatasetArchiveSource(
        share_url="https://example.com/source.zip",
        sha256="0" * 64,
    )

    monkeypatch.setattr(
        "aquinas_toolkit.dataset_fetch.urlopen",
        lambda request, timeout=300: _FakeResponse(payload, content_length=None),
    )
    monkeypatch.setattr("aquinas_toolkit.dataset_fetch.terminal.build_download_progress", lambda transient=True: progress)

    destination = tmp_path / "download.zip"
    _download_archive(source, destination)

    assert destination.read_bytes() == payload
    assert progress.added_total is None
    assert sum(progress.updates) == len(payload)
