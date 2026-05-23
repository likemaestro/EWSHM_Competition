"""Training data preparation for split preprocess NN inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aquinas_toolkit.preprocessing import load_nn_input_arrays
from aquinas_toolkit.utils.run_management import RunContext, stage_output_dir


@dataclass(frozen=True)
class SplitSettings:
    """Deterministic train/validation/test split settings."""

    train: float = 0.70
    validation: float = 0.20
    test: float = 0.10


@dataclass(frozen=True)
class StandardizationSettings:
    """Training data standardization settings."""

    enabled: bool = True


@dataclass(frozen=True)
class TrainingDataSettings:
    """Settings for training-stage data preparation."""

    random_seed: int = 2026
    split: SplitSettings = SplitSettings()
    standardization: StandardizationSettings = StandardizationSettings()


@dataclass(frozen=True)
class TrainingDataPrepResult:
    """Summary of training data preparation outputs."""

    train_indices_path: Path
    val_indices_path: Path
    test_indices_path: Path
    split_manifest_path: Path
    normalization_stats_path: Path | None
    event_count: int


def run_training(run_context: RunContext) -> None:
    """Prepare deterministic NN training inputs for an existing run."""
    prepare_training_data(run_context)


def prepare_training_data(run_context: RunContext) -> TrainingDataPrepResult:
    """Create split indices and train-only normalization stats for NN arrays."""
    settings = load_training_data_settings(run_context.config_path)
    preprocess_dir = stage_output_dir(run_context.run_dir, "preprocess")
    train_dir = stage_output_dir(run_context.run_dir, "train")
    splits_dir = train_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_nn_input_arrays(preprocess_dir)
    event_count = int(arrays.event_ids.shape[0])
    if event_count == 0:
        raise ValueError("Cannot prepare training data because no NN input events were retained.")

    train_idx, val_idx, test_idx = make_split_indices(
        event_count,
        settings=settings,
    )

    train_indices_path = splits_dir / "train_indices.npy"
    val_indices_path = splits_dir / "val_indices.npy"
    test_indices_path = splits_dir / "test_indices.npy"
    np.save(train_indices_path, train_idx)
    np.save(val_indices_path, val_idx)
    np.save(test_indices_path, test_idx)

    normalization_stats_path: Path | None = None
    if settings.standardization.enabled:
        normalization_stats_path = train_dir / "normalization_stats.npz"
        save_normalization_stats(
            normalization_stats_path,
            strain=arrays.strain,
            acc=arrays.acc,
            temperature=arrays.temperature,
            train_indices=train_idx,
        )

    split_manifest_path = splits_dir / "split_manifest.json"
    manifest = {
        "event_count": event_count,
        "settings": asdict(settings),
        "counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "index_files": {
            "train": str(train_indices_path),
            "validation": str(val_indices_path),
            "test": str(test_indices_path),
        },
        "normalization_stats_path": str(normalization_stats_path)
        if normalization_stats_path is not None
        else None,
        "source_nn_inputs_dir": str(arrays.nn_inputs_dir),
        "standardization_axes": {
            "strain": "mean/std over training events and samples, per strain channel",
            "acc": "mean/std over training events and frequency bins, per ACC channel",
            "temperature": "mean/std over training events, per scalar column",
        },
    }
    split_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return TrainingDataPrepResult(
        train_indices_path=train_indices_path,
        val_indices_path=val_indices_path,
        test_indices_path=test_indices_path,
        split_manifest_path=split_manifest_path,
        normalization_stats_path=normalization_stats_path,
        event_count=event_count,
    )


def load_training_data_settings(config_path: Path) -> TrainingDataSettings:
    """Parse training data-preparation settings from a run config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training_config = config.get("training") or {}
    split_config = training_config.get("split") or {}
    standardization_config = training_config.get("standardization") or {}

    settings = TrainingDataSettings(
        random_seed=int(training_config.get("random_seed", 2026)),
        split=SplitSettings(
            train=float(split_config.get("train", 0.70)),
            validation=float(split_config.get("validation", 0.20)),
            test=float(split_config.get("test", 0.10)),
        ),
        standardization=StandardizationSettings(
            enabled=bool(standardization_config.get("enabled", True)),
        ),
    )
    _validate_training_data_settings(settings)
    return settings


def make_split_indices(
    event_count: int,
    *,
    settings: TrainingDataSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/validation/test event indices."""
    if event_count <= 0:
        raise ValueError("event_count must be positive.")

    counts = _split_counts(event_count, settings.split)
    rng = np.random.default_rng(settings.random_seed)
    indices = rng.permutation(event_count).astype(np.int32, copy=False)

    train_end = counts["train"]
    val_end = train_end + counts["validation"]
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def save_normalization_stats(
    path: Path,
    *,
    strain: np.ndarray,
    acc: np.ndarray,
    temperature: np.ndarray,
    train_indices: np.ndarray,
) -> None:
    """Save train-only normalization statistics for split NN input arrays."""
    if len(train_indices) == 0:
        raise ValueError("Cannot fit normalization statistics without training rows.")
    path.parent.mkdir(parents=True, exist_ok=True)

    strain_train = strain[train_indices]
    acc_train = acc[train_indices]
    temperature_train = temperature[train_indices]
    strain_mean, strain_std = _mean_std(strain_train, axis=(0, 1), keepdims=True)
    acc_mean, acc_std = _mean_std(acc_train, axis=(0, 1), keepdims=True)
    temperature_mean, temperature_std = _mean_std(temperature_train, axis=0, keepdims=True)

    np.savez(
        path,
        strain_mean=strain_mean.astype(np.float32, copy=False),
        strain_std=strain_std.astype(np.float32, copy=False),
        acc_mean=acc_mean.astype(np.float32, copy=False),
        acc_std=acc_std.astype(np.float32, copy=False),
        temperature_mean=temperature_mean.astype(np.float32, copy=False),
        temperature_std=temperature_std.astype(np.float32, copy=False),
    )


def _validate_training_data_settings(settings: TrainingDataSettings) -> None:
    fractions = [settings.split.train, settings.split.validation, settings.split.test]
    if any(value < 0 for value in fractions):
        raise ValueError("training split fractions must be non-negative.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("training split fractions must sum to 1.0.")


def _split_counts(event_count: int, split: SplitSettings) -> dict[str, int]:
    labels = ("train", "validation", "test")
    fractions = np.array([split.train, split.validation, split.test], dtype=float)
    raw_counts = fractions * event_count
    counts = np.floor(raw_counts).astype(int)
    remainder = event_count - int(counts.sum())
    if remainder:
        order = np.argsort(raw_counts - counts)[::-1]
        for index in order[:remainder]:
            counts[index] += 1

    if event_count > 0 and counts[0] == 0:
        donor_indices = [index for index in range(1, len(counts)) if counts[index] > 0]
        if donor_indices:
            donor = donor_indices[0]
            counts[donor] -= 1
        counts[0] = 1

    return {label: int(count) for label, count in zip(labels, counts, strict=True)}


def _mean_std(
    values: np.ndarray,
    *,
    axis: int | tuple[int, ...],
    keepdims: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis, keepdims=keepdims)
    std = np.nanstd(values, axis=axis, keepdims=keepdims)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std
