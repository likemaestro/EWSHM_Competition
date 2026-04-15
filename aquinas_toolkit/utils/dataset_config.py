"""Dataset path and set-list resolution from run/workspace configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aquinas_toolkit.utils.dataset_paths import find_workspace_root

DEFAULT_DATASET_ROOT = Path("AQUINAS_DATASET")
DEFAULT_SET_NAMES = (
    "AQUINAS_SET1_2022_07",
    "AQUINAS_SET2_2023_04",
    "AQUINAS_SET3_2023_08",
    "AQUINAS_SET4_2024_01",
    "AQUINAS_SET5_2024_06",
)
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved local dataset root and required AQUINAS set directories."""

    dataset_root: Path
    set_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetLayoutStatus:
    """Resolved dataset state for CLI checks and fetch logic."""

    layout: DatasetLayout
    missing_set_names: tuple[str, ...]
    dataset_root_exists: bool
    dataset_root_is_stub: bool

    @property
    def dataset_is_complete(self) -> bool:
        """Return whether the dataset root and all configured set folders exist."""
        return self.dataset_root_exists and not self.missing_set_names


def load_dataset_layout(config_path: Path | None = None) -> DatasetLayout:
    """Resolve dataset layout from YAML config, with sensible defaults."""
    workspace_root = find_workspace_root()
    resolved_config_path = config_path or (workspace_root / DEFAULT_CONFIG_PATH)
    config = _load_yaml_config(resolved_config_path)
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        data_config = {}

    dataset_root_value = data_config.get("dataset_root", str(DEFAULT_DATASET_ROOT))
    if not isinstance(dataset_root_value, str) or not dataset_root_value.strip():
        dataset_root_value = str(DEFAULT_DATASET_ROOT)

    set_names_value = data_config.get("sets")
    set_names = _coerce_set_names(set_names_value)
    dataset_root = Path(dataset_root_value)
    if not dataset_root.is_absolute():
        dataset_root = workspace_root / dataset_root

    return DatasetLayout(dataset_root=dataset_root, set_names=set_names)


def find_missing_set_names(layout: DatasetLayout) -> list[str]:
    """Return configured set names that are not currently available on disk."""
    if not layout.dataset_root.is_dir():
        return list(layout.set_names)

    return [set_name for set_name in layout.set_names if not (layout.dataset_root / set_name).is_dir()]


def dataset_is_complete(layout: DatasetLayout) -> bool:
    """Return whether dataset root and all configured set folders exist."""
    return not find_missing_set_names(layout)


def inspect_dataset_layout(layout: DatasetLayout) -> DatasetLayoutStatus:
    """Return detailed dataset-root status for CLI display and validation."""
    dataset_root_exists = layout.dataset_root.exists()
    missing_set_names = tuple(find_missing_set_names(layout))
    dataset_root_is_stub = dataset_root_exists and is_stub_dataset_root(layout.dataset_root)
    return DatasetLayoutStatus(
        layout=layout,
        missing_set_names=missing_set_names,
        dataset_root_exists=dataset_root_exists,
        dataset_root_is_stub=dataset_root_is_stub,
    )


def is_stub_dataset_root(dataset_root: Path) -> bool:
    """Return whether the dataset root only contains placeholder bootstrap files."""
    if not dataset_root.is_dir():
        return False

    allowed_names = {"README.md", ".gitkeep"}
    for child in dataset_root.iterdir():
        if child.name not in allowed_names:
            return False
    return True


def _coerce_set_names(raw_value: Any) -> tuple[str, ...]:
    if not isinstance(raw_value, list):
        return DEFAULT_SET_NAMES

    cleaned = tuple(
        str(item).strip()
        for item in raw_value
        if isinstance(item, str) and item.strip()
    )
    if not cleaned:
        return DEFAULT_SET_NAMES

    return cleaned


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}

    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed

