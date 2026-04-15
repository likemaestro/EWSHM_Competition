"""Dataset path helpers for notebooks and exploratory workflows."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def find_repo_root() -> Path:
    """Return the repository root anchored from this installed source file."""
    module_path = Path(__file__).resolve()

    for parent in module_path.parents:
        if (parent / "aquinas_toolkit").is_dir():
            return parent

    raise FileNotFoundError(f"Could not determine repo root from module path: {module_path}")


def find_workspace_root() -> Path:
    """Return the active workspace root, falling back to the installed repo root."""
    current = Path.cwd().resolve()

    for parent in (current, *current.parents):
        if (parent / DEFAULT_CONFIG_PATH).is_file():
            return parent

    return find_repo_root()


def find_dataset_root() -> Path:
    """Return the repo-level AQUINAS dataset directory."""
    dataset_root = find_repo_root() / "AQUINAS_DATASET"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {dataset_root}")

    return dataset_root


def list_dataset_dirs(dataset_root: Path) -> list[Path]:
    """Return all monthly AQUINAS set directories in sorted order."""
    dataset_dirs = sorted(path for path in dataset_root.glob("AQUINAS_SET*") if path.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError(f"No AQUINAS_SET* folders found under {dataset_root}")

    return dataset_dirs
