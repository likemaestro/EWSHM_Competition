"""Neural-network-ready preprocessing artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquinas_toolkit.io import parse_sensor_name
from aquinas_toolkit.preprocessing.store import open_preprocess_store


def strain_peak_window_bounds(
    *,
    peak_idx: int,
    signal_length: int,
    peak_window_half_samples: int,
) -> tuple[int, int] | None:
    """Return fixed-length strain window bounds shifted inside signal limits."""
    target_length = peak_window_half_samples * 2
    if target_length <= 0 or signal_length < target_length:
        return None

    start = peak_idx - peak_window_half_samples
    stop = peak_idx + peak_window_half_samples
    if start < 0:
        start = 0
        stop = target_length
    elif stop > signal_length:
        stop = signal_length
        start = signal_length - target_length
    return start, stop


@dataclass(frozen=True)
class StrainInputSettings:
    """Settings for strain event clipping."""

    peak_window_half_samples: int = 100
    locations: tuple[str, ...] = ("INF", "SHE", "SUP")


@dataclass(frozen=True)
class AccInputSettings:
    """Settings for acceleration frequency-domain inputs."""

    min_aligned_samples: int = 500
    low_hz: float = 0.5
    high_hz: float = 20.0


@dataclass(frozen=True)
class NeuralInputSettings:
    """Settings for the canonical preprocessing neural tensor."""

    decks: tuple[str, ...] = ()
    sampling_rate_hz: float = 100.0
    strain: StrainInputSettings = StrainInputSettings()
    acc: AccInputSettings = AccInputSettings()


@dataclass(frozen=True)
class NeuralInputResult:
    """Summary of neural-input artifact generation."""

    output_path: Path
    report_dir: Path
    sample_count: int
    feature_count: int
    strain_channel_count: int
    acc_channel_count: int
    frequency_bin_count: int


def build_neural_inputs(
    preprocess_dir: Path,
    *,
    settings: NeuralInputSettings,
) -> NeuralInputResult:
    """Build the canonical flattened neural input tensor from preprocess outputs."""
    output_path = preprocess_dir / "neural_inputs.npy"
    report_dir = preprocess_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open_preprocess_store(preprocess_dir) as store:
        sensors = store.list_sensors()
        retained_events = store.iter_retained_events()
        selected_retained_events = _filter_events_by_deck(retained_events, decks=settings.decks)

        sensor_map, strain_sensors, acc_sensors = _build_sensor_map(
            sensors,
            decks=settings.decks,
            strain_locations=settings.strain.locations,
        )
        sensor_map.to_csv(report_dir / "sensor_map.csv", index=False)

        required_sensor_names = strain_sensors + acc_sensors

        candidates: list[dict[str, Any]] = []
        acc_lengths: list[int] = []
        complete_coverage_events = 0
        incomplete_coverage_events = 0
        packaging_rejected_events = 0

        for event in selected_retained_events.itertuples(index=False):
            event_id = str(event.event_id)
            event_sensors = store.load_event_sensors(event_id)
            aligned = store.load_aligned_event(event_id)
            if not _has_required_sensor_coverage(aligned, required_sensor_names):
                incomplete_coverage_events += 1
                continue

            complete_coverage_events += 1
            result = _prepare_candidate(
                event=event,
                aligned=aligned,
                event_sensors=event_sensors,
                strain_sensors=strain_sensors,
                acc_sensors=acc_sensors,
                settings=settings,
            )
            if result["status"] != "keep":
                packaging_rejected_events += 1
                continue

            candidates.append(result)
            acc_lengths.append(int(result["acc_time"].shape[0]))

    frequency_bins = _frequency_bins(acc_lengths, settings=settings)
    tensor_rows: list[np.ndarray] = []
    event_ids: list[str] = []
    temperature_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        strain_flat = candidate["strain_window"].astype(np.float32, copy=False).reshape(-1)
        acc_frequency = _acc_frequency_block(
            candidate["acc_time"],
            n_fft=max(acc_lengths),
            frequency_bins=frequency_bins,
            settings=settings,
        )
        temperature = np.array([candidate["temperature_mean"]], dtype=np.float32)
        tensor_rows.append(np.concatenate([strain_flat, acc_frequency.reshape(-1), temperature]))
        event_ids.append(candidate["event_id"])
        temperature_rows.append(
            {
                "event_id": candidate["event_id"],
                "temperature_mean": candidate["temperature_mean"],
                "temperature_sensor_count": candidate["temperature_sensor_count"],
            }
        )

    if tensor_rows:
        neural_inputs = np.vstack(tensor_rows).astype(np.float32, copy=False)
    else:
        neural_inputs = np.empty((0, 0), dtype=np.float32)

    np.save(output_path, neural_inputs)
    np.save(report_dir / "event_ids.npy", np.array(event_ids, dtype=str))
    np.save(report_dir / "frequency_bins.npy", frequency_bins.astype(np.float32, copy=False))
    np.save(report_dir / "valid_lengths.npy", np.array(acc_lengths, dtype=np.int32))
    pd.DataFrame(temperature_rows).to_csv(report_dir / "temperature_metadata.csv", index=False)
    _write_metadata_files(
        report_dir,
        settings=settings,
        neural_inputs=neural_inputs,
        total_retained_preprocess_events=len(selected_retained_events),
        complete_coverage_events=complete_coverage_events,
        incomplete_coverage_events=incomplete_coverage_events,
        packaging_rejected_events=packaging_rejected_events,
        strain_sensors=strain_sensors,
        acc_sensors=acc_sensors,
        frequency_bins=frequency_bins,
    )

    return NeuralInputResult(
        output_path=output_path,
        report_dir=report_dir,
        sample_count=int(neural_inputs.shape[0]),
        feature_count=int(neural_inputs.shape[1]) if neural_inputs.ndim == 2 else 0,
        strain_channel_count=len(strain_sensors),
        acc_channel_count=len(acc_sensors),
        frequency_bin_count=len(frequency_bins),
    )


def _filter_events_by_deck(retained_events: pd.DataFrame, *, decks: tuple[str, ...]) -> pd.DataFrame:
    if not decks or retained_events.empty:
        return retained_events.copy()
    return retained_events.loc[retained_events["deck"].astype(str).isin(decks)].copy()


def _has_required_sensor_coverage(aligned: pd.DataFrame, required_sensor_names: list[str]) -> bool:
    return set(required_sensor_names).issubset(set(aligned.columns))


def _build_sensor_map(
    sensors: pd.DataFrame,
    *,
    decks: tuple[str, ...],
    strain_locations: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    selected = sensors.copy()
    if decks:
        selected = selected.loc[selected["deck"].isin(decks)].copy()
    selected = selected.sort_values(["sensor_order", "sensor_name"], kind="mergesort")

    selected_rows = list(selected.itertuples(index=False))
    strain_sensors = [
        str(row.sensor_name)
        for row in selected_rows
        if parse_sensor_name(str(row.sensor_name))["quantity"] == "STR"
        and str(parse_sensor_name(str(row.sensor_name))["location"]) in strain_locations
    ]
    acc_sensors = [
        str(row.sensor_name)
        for row in selected_rows
        if parse_sensor_name(str(row.sensor_name))["quantity"] == "ACC"
        and parse_sensor_name(str(row.sensor_name))["axis"] == "Z"
    ]
    strain_index = {sensor_name: index for index, sensor_name in enumerate(strain_sensors)}
    acc_index = {sensor_name: index for index, sensor_name in enumerate(acc_sensors)}

    for row in selected_rows:
        sensor_name = str(row.sensor_name)
        parsed = parse_sensor_name(sensor_name)
        quantity = parsed["quantity"]
        axis = parsed["axis"]
        location = parsed["location"]
        is_strain = quantity == "STR" and str(location) in strain_locations
        is_acc_z = quantity == "ACC" and axis == "Z"
        include = is_strain or is_acc_z
        if is_strain:
            model_channel_index = strain_index[sensor_name]
            global_index = model_channel_index
            model_channel_id = f"STR{model_channel_index:02d}"
        elif is_acc_z:
            model_channel_index = acc_index[sensor_name]
            global_index = len(strain_sensors) + model_channel_index
            model_channel_id = f"ACCZ{model_channel_index:02d}"
        else:
            model_channel_index = None
            global_index = None
            model_channel_id = ""

        rows.append(
            {
                "sensor_name": sensor_name,
                "sensor_type": quantity,
                "deck": parsed["deck"],
                "span": parsed["span"],
                "side": parsed["side"],
                "location": location,
                "axis_or_type": axis or location,
                "source_sensor_order": int(row.sensor_order),
                "model_channel_index": model_channel_index,
                "global_model_channel_index": global_index,
                "model_channel_id": model_channel_id,
                "include_flag": bool(include),
                "notes": "" if include else "excluded_from_neural_input",
            }
        )

    return pd.DataFrame(rows), strain_sensors, acc_sensors


def _prepare_candidate(
    *,
    event: Any,
    aligned: pd.DataFrame,
    event_sensors: pd.DataFrame,
    strain_sensors: list[str],
    acc_sensors: list[str],
    settings: NeuralInputSettings,
) -> dict[str, Any]:
    if aligned.empty:
        return {"status": "discard", "discard_reason": "empty_aligned_event", "details": []}

    missing_columns = sorted((set(strain_sensors) | set(acc_sensors)) - set(aligned.columns))
    if missing_columns:
        return {
            "status": "discard",
            "discard_reason": "missing_aligned_columns",
            "details": missing_columns,
        }

    strain_values = aligned[strain_sensors].to_numpy(dtype=float)
    acc_values = aligned[acc_sensors].to_numpy(dtype=float)
    if not np.isfinite(strain_values).all() or not np.isfinite(acc_values).all():
        return {"status": "discard", "discard_reason": "nonfinite_values", "details": []}

    if len(acc_values) < settings.acc.min_aligned_samples:
        return {
            "status": "discard",
            "discard_reason": "acc_too_short",
            "details": [int(len(acc_values)), int(settings.acc.min_aligned_samples)],
        }

    if strain_values.size == 0 or acc_values.size == 0:
        return {"status": "discard", "discard_reason": "empty_required_matrix", "details": []}

    peak_flat_index = int(np.argmax(np.abs(strain_values)))
    peak_row = int(np.unravel_index(peak_flat_index, strain_values.shape)[0])
    window_bounds = strain_peak_window_bounds(
        peak_idx=peak_row,
        signal_length=len(strain_values),
        peak_window_half_samples=settings.strain.peak_window_half_samples,
    )
    if window_bounds is None:
        return {
            "status": "discard",
            "discard_reason": "strain_window_out_of_bounds",
            "details": [peak_row, len(strain_values)],
        }
    start, end = window_bounds

    temperatures = pd.to_numeric(
        event_sensors.loc[event_sensors["sensor_name"].isin(strain_sensors + acc_sensors), "temperature"],
        errors="coerce",
    ).dropna()
    temperature_mean = float(temperatures.mean()) if not temperatures.empty else float("nan")

    return {
        "status": "keep",
        "event_id": str(event.event_id),
        "strain_window": strain_values[start:end, :],
        "acc_time": acc_values,
        "temperature_mean": temperature_mean,
        "temperature_sensor_count": int(len(temperatures)),
    }


def _frequency_bins(acc_lengths: list[int], *, settings: NeuralInputSettings) -> np.ndarray:
    if not acc_lengths:
        return np.empty((0,), dtype=np.float32)
    n_fft = max(acc_lengths)
    bins = np.fft.rfftfreq(n_fft, d=1.0 / settings.sampling_rate_hz)
    mask = (bins >= settings.acc.low_hz) & (bins <= settings.acc.high_hz)
    return bins[mask]


def _acc_frequency_block(
    acc_time: np.ndarray,
    *,
    n_fft: int,
    frequency_bins: np.ndarray,
    settings: NeuralInputSettings,
) -> np.ndarray:
    padded = np.zeros((n_fft, acc_time.shape[1]), dtype=np.float32)
    padded[: acc_time.shape[0], :] = acc_time.astype(np.float32, copy=False)
    spectrum = np.abs(np.fft.rfft(padded, n=n_fft, axis=0))
    bins = np.fft.rfftfreq(n_fft, d=1.0 / settings.sampling_rate_hz)
    mask = np.isin(bins, frequency_bins)
    return spectrum[mask, :].astype(np.float32, copy=False)


def _write_metadata_files(
    report_dir: Path,
    *,
    settings: NeuralInputSettings,
    neural_inputs: np.ndarray,
    total_retained_preprocess_events: int,
    complete_coverage_events: int,
    incomplete_coverage_events: int,
    packaging_rejected_events: int,
    strain_sensors: list[str],
    acc_sensors: list[str],
    frequency_bins: np.ndarray,
) -> None:
    strain_window_samples = settings.strain.peak_window_half_samples * 2
    strain_width = strain_window_samples * len(strain_sensors)
    acc_width = len(frequency_bins) * len(acc_sensors)
    slices = {
        "strain": {
            "start": 0,
            "stop": strain_width,
            "shape": [
                strain_window_samples,
                len(strain_sensors),
            ],
        },
        "acc_z_frequency": {
            "start": strain_width,
            "stop": strain_width + acc_width,
            "shape": [len(frequency_bins), len(acc_sensors)],
        },
        "temperature": {
            "start": strain_width + acc_width,
            "stop": strain_width + acc_width + 1,
            "shape": [1],
        },
    }
    (report_dir / "input_slices.json").write_text(
        json.dumps(slices, indent=2, sort_keys=True), encoding="utf-8"
    )
    (report_dir / "sensor_ids.json").write_text(
        json.dumps({"strain": strain_sensors, "acc_z": acc_sensors}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "neural_inputs_shape": list(neural_inputs.shape),
        "total_retained_preprocess_events": total_retained_preprocess_events,
        "events_with_complete_selected_sensor_coverage": complete_coverage_events,
        "events_excluded_incomplete_selected_sensor_coverage": incomplete_coverage_events,
        "events_excluded_packaging_constraints": packaging_rejected_events,
        "retained_events": int(neural_inputs.shape[0]),
        "settings": asdict(settings),
    }
    (report_dir / "neural_input_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (report_dir / "preprocessing_summary.md").write_text(
        "\n".join(
            [
                "# Preprocessing Summary",
                "",
                f"- Neural input shape: `{tuple(neural_inputs.shape)}`",
                f"- Retained preprocess events checked: `{total_retained_preprocess_events}`",
                f"- Events with complete selected-sensor coverage: `{complete_coverage_events}`",
                f"- Events excluded for incomplete selected-sensor coverage: `{incomplete_coverage_events}`",
                f"- Events excluded by packaging constraints: `{packaging_rejected_events}`",
                f"- Retained events: `{int(neural_inputs.shape[0])}`",
                f"- Strain channels: `{len(strain_sensors)}`",
                f"- ACC-Z channels: `{len(acc_sensors)}`",
                f"- Frequency bins: `{len(frequency_bins)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
