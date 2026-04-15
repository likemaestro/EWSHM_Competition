"""Tests for dataset path helpers used by notebooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from aquinas_toolkit.utils import dataset_paths


def test_find_repo_root_returns_project_root() -> None:
    repo_root = dataset_paths.find_repo_root()

    assert (repo_root / "aquinas_toolkit").is_dir()
    assert (repo_root / "README.md").is_file()


def test_find_workspace_root_prefers_current_workspace_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "configs" / "default.yaml").write_text("data:\n  dataset_root: AQUINAS_DATASET\n")
    (workspace / "subdir").mkdir()
    monkeypatch.chdir(workspace / "subdir")

    assert dataset_paths.find_workspace_root() == workspace


def test_find_workspace_root_falls_back_to_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert dataset_paths.find_workspace_root() == dataset_paths.find_repo_root()


def test_find_dataset_root_returns_repo_dataset_dir() -> None:
    dataset_root = dataset_paths.find_dataset_root()

    assert dataset_root.name == "AQUINAS_DATASET"
    assert dataset_root.is_dir()


def test_find_dataset_root_raises_when_dataset_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "repo"
    fake_module = fake_root / "aquinas_toolkit" / "utils" / "dataset_paths.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr(dataset_paths, "__file__", str(fake_module))

    with pytest.raises(FileNotFoundError, match="Missing dataset folder"):
        dataset_paths.find_dataset_root()


def test_list_dataset_dirs_returns_sorted_set_directories(tmp_path: Path) -> None:
    dataset_root = tmp_path / "AQUINAS_DATASET"
    dataset_root.mkdir()
    (dataset_root / "AQUINAS_SET2_2023_04").mkdir()
    (dataset_root / "AQUINAS_SET1_2022_07").mkdir()
    (dataset_root / "notes").mkdir()

    dataset_dirs = dataset_paths.list_dataset_dirs(dataset_root)

    assert [path.name for path in dataset_dirs] == [
        "AQUINAS_SET1_2022_07",
        "AQUINAS_SET2_2023_04",
    ]


def test_list_dataset_dirs_raises_when_no_sets_exist(tmp_path: Path) -> None:
    dataset_root = tmp_path / "AQUINAS_DATASET"
    dataset_root.mkdir()

    with pytest.raises(FileNotFoundError, match="No AQUINAS_SET\\* folders found"):
        dataset_paths.list_dataset_dirs(dataset_root)
