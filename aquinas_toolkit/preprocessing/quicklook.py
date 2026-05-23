"""Quick inspection helpers for preprocess neural-input arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NN_INPUTS_DIR_NAME = "nn_inputs"
NN_METADATA_DIR_NAME = "metadata"


@dataclass(frozen=True)
class NNInputArrays:
    """Loaded event-aligned neural-input arrays."""

    nn_inputs_dir: Path
    metadata_dir: Path
    strain: np.ndarray
    acc: np.ndarray
    temperature: np.ndarray
    event_ids: np.ndarray
    frequency_bins: np.ndarray
    sensor_ids: dict[str, list[str]]


def nn_inputs_dir(preprocess_dir: str | Path) -> Path:
    """Return the neural-input artifact directory for a preprocess stage."""
    return Path(preprocess_dir) / NN_INPUTS_DIR_NAME


def nn_input_metadata_dir(preprocess_dir: str | Path) -> Path:
    """Return the neural-input metadata directory for a preprocess stage."""
    return nn_inputs_dir(preprocess_dir) / NN_METADATA_DIR_NAME


def load_nn_input_arrays(preprocess_dir: str | Path) -> NNInputArrays:
    """Load split neural-input arrays and validate their shared event axis."""
    directory = nn_inputs_dir(preprocess_dir)
    metadata_directory = nn_input_metadata_dir(preprocess_dir)
    required = {
        "strain": directory / "strain_inputs.npy",
        "acc": directory / "acc_inputs.npy",
        "temperature": directory / "temperature_inputs.npy",
        "event_ids": directory / "event_ids.npy",
        "frequency_bins": metadata_directory / "frequency_bins.npy",
        "sensor_ids": metadata_directory / "sensor_ids.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing preprocess NN input artifact(s): " + ", ".join(missing))

    strain = np.load(required["strain"])
    acc = np.load(required["acc"])
    temperature = np.load(required["temperature"])
    event_ids = np.load(required["event_ids"], allow_pickle=False)
    frequency_bins = np.load(required["frequency_bins"])
    sensor_ids = json.loads(required["sensor_ids"].read_text(encoding="utf-8"))

    _validate_event_axis(
        strain=strain,
        acc=acc,
        temperature=temperature,
        event_ids=event_ids,
    )
    return NNInputArrays(
        nn_inputs_dir=directory,
        metadata_dir=metadata_directory,
        strain=strain,
        acc=acc,
        temperature=temperature,
        event_ids=event_ids,
        frequency_bins=frequency_bins,
        sensor_ids=sensor_ids,
    )


def summarize_nn_inputs(preprocess_dir: str | Path) -> dict[str, Any]:
    """Return shape and finite-value diagnostics for split NN inputs."""
    arrays = load_nn_input_arrays(preprocess_dir)
    return {
        "nn_inputs_dir": str(arrays.nn_inputs_dir),
        "metadata_dir": str(arrays.metadata_dir),
        "event_count": int(arrays.event_ids.shape[0]),
        "shapes": {
            "strain_inputs": list(arrays.strain.shape),
            "acc_inputs": list(arrays.acc.shape),
            "temperature_inputs": list(arrays.temperature.shape),
            "event_ids": list(arrays.event_ids.shape),
            "frequency_bins": list(arrays.frequency_bins.shape),
        },
        "finite_counts": {
            "strain_inputs": _finite_count(arrays.strain),
            "acc_inputs": _finite_count(arrays.acc),
            "temperature_inputs": _finite_count(arrays.temperature),
        },
        "nan_counts": {
            "strain_inputs": int(np.isnan(arrays.strain).sum()),
            "acc_inputs": int(np.isnan(arrays.acc).sum()),
            "temperature_inputs": int(np.isnan(arrays.temperature).sum()),
        },
        "channels": arrays.sensor_ids,
    }


def load_sensor_map(preprocess_dir: str | Path) -> pd.DataFrame:
    """Load the NN sensor map from neural-input metadata or the preprocess report."""
    preprocess_path = Path(preprocess_dir)
    candidates = [
        nn_input_metadata_dir(preprocess_path) / "sensor_map.csv",
        preprocess_path / "report" / "sensor_map.csv",
    ]
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No sensor_map.csv found under {preprocess_path}")


def random_event_indices(preprocess_dir: str | Path, *, count: int, seed: int = 2026) -> list[int]:
    """Return deterministic random event indices for quicklook plotting."""
    arrays = load_nn_input_arrays(preprocess_dir)
    event_count = int(arrays.event_ids.shape[0])
    if count <= 0 or event_count == 0:
        return []
    rng = np.random.default_rng(seed)
    size = min(count, event_count)
    return sorted(int(value) for value in rng.choice(event_count, size=size, replace=False))


def plot_nn_input_event(
    preprocess_dir: str | Path,
    *,
    event_index: int,
    output: str | Path | None = None,
) -> Path:
    """Plot one event's strain clips, ACC spectra, and temperature context."""
    arrays = load_nn_input_arrays(preprocess_dir)
    event_count = int(arrays.event_ids.shape[0])
    if event_index < 0 or event_index >= event_count:
        raise IndexError(f"event_index {event_index} is outside [0, {event_count})")

    output_path = _resolve_output_path(
        arrays.nn_inputs_dir,
        event_index=event_index,
        output=output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strain_channels = arrays.sensor_ids.get("strain", [])
    acc_channels = arrays.sensor_ids.get("acc_z", [])
    event_id = str(arrays.event_ids[event_index])
    temperature_value = float(arrays.temperature[event_index, 0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for channel_index in range(arrays.strain.shape[2]):
        label = _channel_label(strain_channels, channel_index)
        axes[0].plot(arrays.strain[event_index, :, channel_index], label=label, linewidth=1.0)
    axes[0].set_title("Strain clips")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Zeroed strain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    x_axis = arrays.frequency_bins if len(arrays.frequency_bins) else np.arange(arrays.acc.shape[1])
    for channel_index in range(arrays.acc.shape[2]):
        label = _channel_label(acc_channels, channel_index)
        axes[1].plot(x_axis, arrays.acc[event_index, :, channel_index], label=label, linewidth=1.0)
    axes[1].set_title("ACC-Z FFT magnitude")
    axes[1].set_xlabel("Frequency (Hz)" if len(arrays.frequency_bins) else "Frequency bin")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"NN input quicklook | row {event_index} | {event_id} | temp={temperature_value:.3g}"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _validate_event_axis(
    *,
    strain: np.ndarray,
    acc: np.ndarray,
    temperature: np.ndarray,
    event_ids: np.ndarray,
) -> None:
    event_counts = {
        "strain_inputs": int(strain.shape[0]),
        "acc_inputs": int(acc.shape[0]),
        "temperature_inputs": int(temperature.shape[0]),
        "event_ids": int(event_ids.shape[0]),
    }
    if len(set(event_counts.values())) != 1:
        raise ValueError(f"NN input arrays do not share the same event axis: {event_counts}")


def _finite_count(values: np.ndarray) -> int:
    return int(np.isfinite(values).sum())


def _channel_label(channel_names: list[str], channel_index: int) -> str:
    if channel_index < len(channel_names):
        return channel_names[channel_index]
    return f"channel_{channel_index}"


def _resolve_output_path(
    nn_dir: Path,
    *,
    event_index: int,
    output: str | Path | None,
) -> Path:
    if output is None:
        return nn_dir / "quicklook" / f"event_{event_index:04d}.png"

    output_path = Path(output)
    if output_path.suffix.lower() == ".png":
        return output_path
    return output_path / f"event_{event_index:04d}.png"
