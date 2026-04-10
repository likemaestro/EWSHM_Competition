"""Smoke tests: every subpackage is importable and the public API is stable."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "aquinas_toolkit",
        "aquinas_toolkit.io",
        "aquinas_toolkit.cli",
        "aquinas_toolkit.preprocessing",
        "aquinas_toolkit.feature_extraction",
        "aquinas_toolkit.training",
        "aquinas_toolkit.scoring",
    ],
)
def test_subpackage_is_importable(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    assert mod is not None


def test_public_api_exports_aquinas_reader() -> None:
    from aquinas_toolkit import AquinasReader

    assert callable(AquinasReader)


def test_io_reexports_aquinas_reader() -> None:
    from aquinas_toolkit.io import AquinasReader

    assert callable(AquinasReader)


def test_cli_reexports_main() -> None:
    from aquinas_toolkit.cli import main

    assert callable(main)


def test_preprocessing_reexports_public_api() -> None:
    from aquinas_toolkit.preprocessing import align_event_group, find_events, run_preprocessing

    assert callable(find_events)
    assert callable(align_event_group)
    assert callable(run_preprocessing)


def test_package_import_is_lazy_for_plotting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "aquinas_toolkit", raising=False)
    monkeypatch.delitem(sys.modules, "aquinas_toolkit.utils", raising=False)
    monkeypatch.delitem(sys.modules, "aquinas_toolkit.utils.plotting", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    mod = importlib.import_module("aquinas_toolkit")

    assert mod is not None
    assert "matplotlib" not in sys.modules
    assert "aquinas_toolkit.utils.plotting" not in sys.modules
