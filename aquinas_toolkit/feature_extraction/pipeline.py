"""Features stage orchestration over the canonical preprocess SQLite store."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aquinas_toolkit.feature_extraction.store import FeaturesStoreWriter, features_store_path
from aquinas_toolkit.feature_extraction.workflow import (
    collect_preprocessed_event_matrices,
    run_acc_z_fdd_from_event_matrices,
)
from aquinas_toolkit.preprocessing.store import open_preprocess_store, preprocess_store_path
from aquinas_toolkit.utils.run_management import RunContext, stage_output_dir


@dataclass(frozen=True)
class ModalAnalysisSettings:
    """Settings for the optional ACC_Z FDD feature family."""

    enabled: bool = True
    quantity: str = "ACC"
    axis: str = "Z"
    min_common_events: int = 2
    max_events: int | None = 5
    low_hz: float = 0.5
    high_hz: float = 20.0
    nperseg: int = 1024
    noverlap: int = 512
    n_peaks: int = 3


@dataclass(frozen=True)
class FeatureSettings:
    """Runtime settings for the features stage."""

    sampling_rate_hz: float = 100.0
    modal_analysis: ModalAnalysisSettings = ModalAnalysisSettings()


def run_features(run_context: RunContext) -> None:
    """Execute the features stage from the preprocess SQLite store."""
    settings = load_feature_settings(run_context.config_path)
    preprocess_dir = stage_output_dir(run_context.run_dir, "preprocess")
    preprocess_db_path = preprocess_store_path(preprocess_dir)
    legacy_manifest_path = preprocess_dir / "event_manifest.csv"
    if not preprocess_db_path.is_file() and not legacy_manifest_path.is_file():
        raise FileNotFoundError(
            f"Canonical preprocess store not found at {preprocess_db_path}, and no legacy "
            f"preprocess CSV artifacts were found in {preprocess_dir}. "
            "Run preprocess successfully before feature extraction."
        )

    features_dir = stage_output_dir(run_context.run_dir, "features")
    writer = FeaturesStoreWriter(
        features_store_path(features_dir),
        run_id=run_context.run_id,
        preprocess_store_path=str(preprocess_db_path if preprocess_db_path.is_file() else preprocess_dir),
        settings_payload=asdict(settings),
    )

    try:
        with open_preprocess_store(preprocess_dir) as preprocess_store:
            sensor_feature_rows = _build_sensor_event_feature_rows(
                preprocess_store,
                sampling_rate_hz=settings.sampling_rate_hz,
            )
            writer.write_sensor_event_features(sensor_feature_rows)

            family_status_rows, peak_rows, component_rows = _build_modal_feature_rows(
                preprocess_store,
                settings=settings,
            )
            writer.write_feature_family_status(family_status_rows)
            writer.write_deck_modal_peaks(peak_rows)
            writer.write_deck_mode_shape_components(component_rows)
    finally:
        writer.close()


def load_feature_settings(config_path: Path) -> FeatureSettings:
    """Parse the snapped run config into feature-stage settings."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    features = config.get("features") or {}
    modal_analysis = features.get("modal_analysis") or {}

    return FeatureSettings(
        sampling_rate_hz=float(features.get("sampling_rate_hz", 100.0)),
        modal_analysis=ModalAnalysisSettings(
            enabled=bool(modal_analysis.get("enabled", True)),
            quantity=str(modal_analysis.get("quantity", "ACC")),
            axis=str(modal_analysis.get("axis", "Z")),
            min_common_events=int(modal_analysis.get("min_common_events", 2)),
            max_events=(
                None
                if modal_analysis.get("max_events") in {None, "null"}
                else int(modal_analysis.get("max_events", 5))
            ),
            low_hz=float(modal_analysis.get("low_hz", 0.5)),
            high_hz=float(modal_analysis.get("high_hz", 20.0)),
            nperseg=int(modal_analysis.get("nperseg", 1024)),
            noverlap=int(modal_analysis.get("noverlap", 512)),
            n_peaks=int(modal_analysis.get("n_peaks", 3)),
        ),
    )


def _build_sensor_event_feature_rows(
    preprocess_store: Any,
    *,
    sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retained_events = preprocess_store.iter_retained_events()
    for event in retained_events.itertuples(index=False):
        aligned_event = preprocess_store.load_aligned_event(event.event_id)
        event_sensors = preprocess_store.load_event_sensors(event.event_id)
        event_sensors = event_sensors.loc[event_sensors["sensor_status"] == "included"].copy()
        if not aligned_event.empty:
            aligned_event = aligned_event.copy()
            aligned_event["timestamp_utc"] = pd.to_datetime(
                aligned_event["timestamp_utc"],
                utc=True,
                format="mixed",
            )

        for sensor in event_sensors.itertuples(index=False):
            sensor_name = str(sensor.sensor_name)
            metadata = _parse_sensor_name(sensor_name)
            waveform_stats = _compute_waveform_statistics(
                timestamps=aligned_event.get("timestamp_utc", pd.Series(dtype="datetime64[ns, UTC]")),
                values=aligned_event.get(sensor_name, pd.Series(dtype=float)),
                sampling_rate_hz=sampling_rate_hz,
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "set_name": event.set_name,
                    "deck": event.deck,
                    "sensor_name": sensor_name,
                    "sensor_order": int(sensor.sensor_order),
                    "quantity": metadata["quantity"],
                    "axis": metadata["axis"],
                    "sample_count": waveform_stats["sample_count"],
                    "aligned_duration_s": waveform_stats["aligned_duration_s"],
                    "table_duration": _to_optional_float(sensor.duration),
                    "table_start_value": _to_optional_float(sensor.start_value),
                    "table_end_value": _to_optional_float(sensor.end_value),
                    "table_diff_value": _to_optional_float(sensor.diff_value),
                    "table_min_value": _to_optional_float(sensor.min_value),
                    "table_max_value": _to_optional_float(sensor.max_value),
                    "table_mean_value": _to_optional_float(sensor.mean_value),
                    "table_range_value": _to_optional_float(sensor.range_value),
                    "table_temperature": _to_optional_float(sensor.temperature),
                    "waveform_mean": waveform_stats["mean"],
                    "waveform_std": waveform_stats["std"],
                    "waveform_rms": waveform_stats["rms"],
                    "waveform_min": waveform_stats["min"],
                    "waveform_max": waveform_stats["max"],
                    "waveform_peak_to_peak": waveform_stats["peak_to_peak"],
                    "waveform_energy": waveform_stats["energy"],
                    "waveform_crest_factor": waveform_stats["crest_factor"],
                    "waveform_zero_crossing_rate": waveform_stats["zero_crossing_rate"],
                    "waveform_skewness": waveform_stats["skewness"],
                    "waveform_kurtosis": waveform_stats["kurtosis"],
                }
            )
    return rows


def _build_modal_feature_rows(
    preprocess_store: Any,
    *,
    settings: FeatureSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    retained_events = preprocess_store.iter_retained_events()
    if retained_events.empty:
        return [], [], []

    family_status_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for set_name, deck in (
        retained_events[["set_name", "deck"]]
        .drop_duplicates()
        .sort_values(["set_name", "deck"], kind="mergesort")
        .itertuples(index=False, name=None)
    ):
        if not settings.modal_analysis.enabled:
            family_status_rows.append(
                {
                    "set_name": set_name,
                    "deck": deck,
                    "feature_family": "acc_z_fdd",
                    "status": "skipped",
                    "detail": "disabled in config",
                    "event_count": 0,
                    "channel_count": 0,
                }
            )
            continue

        event_rows = retained_events.loc[
            (retained_events["set_name"] == set_name)
            & (retained_events["deck"] == deck)
        ].copy()
        collection = collect_preprocessed_event_matrices(
            preprocess_store,
            set_name=set_name,
            deck=deck,
            quantity=settings.modal_analysis.quantity,
            axis=settings.modal_analysis.axis,
            min_common_events=settings.modal_analysis.min_common_events,
            max_events=settings.modal_analysis.max_events,
        )
        if not collection.aligned_events:
            family_status_rows.append(
                {
                    "set_name": set_name,
                    "deck": deck,
                    "feature_family": "acc_z_fdd",
                    "status": "skipped",
                    "detail": collection.detail,
                    "event_count": len(event_rows),
                    "channel_count": len(collection.channel_names),
                }
            )
            continue

        summary = run_acc_z_fdd_from_event_matrices(
            collection.aligned_events,
            channel_names=collection.channel_names,
            sampling_rate_hz=settings.sampling_rate_hz,
            low_hz=settings.modal_analysis.low_hz,
            high_hz=settings.modal_analysis.high_hz,
            nperseg=settings.modal_analysis.nperseg,
            noverlap=settings.modal_analysis.noverlap,
            n_peaks=settings.modal_analysis.n_peaks,
        )
        peak_table = summary["peak_table"].copy()
        mode_shape_locations = summary["mode_shape_locations"].copy()
        family_status_rows.append(
            {
                "set_name": set_name,
                "deck": deck,
                "feature_family": "acc_z_fdd",
                "status": "completed",
                "detail": f"computed from {len(collection.aligned_events)} retained events",
                "event_count": len(collection.aligned_events),
                "channel_count": len(collection.channel_names),
            }
        )
        for peak_rank, peak in enumerate(peak_table.itertuples(index=False), start=1):
            peak_rows.append(
                {
                    "set_name": set_name,
                    "deck": deck,
                    "peak_rank": peak_rank,
                    "frequency_hz": float(peak.frequency_hz),
                    "singular_value": float(peak.singular_value),
                    "frequency_index": int(peak.frequency_index),
                    "channel_count": len(collection.channel_names),
                    "event_count": len(collection.aligned_events),
                }
            )
        for component in mode_shape_locations.itertuples(index=False):
            component_rows.append(
                {
                    "set_name": set_name,
                    "deck": deck,
                    "peak_rank": int(component.peak_rank),
                    "sensor_name": component.channel,
                    "frequency_hz": float(component.frequency_hz),
                    "singular_value": float(component.singular_value),
                    "mode_shape_amplitude": float(component.mode_shape_amplitude),
                    "mode_shape_signed_component": float(component.mode_shape_signed_component),
                    "mode_shape_phase_deg": float(component.mode_shape_phase_deg),
                    "span": component.span,
                    "side": component.side,
                    "location": component.location,
                    "quantity": component.quantity,
                    "axis": component.axis,
                    "position_label": component.position_label,
                }
            )

    return family_status_rows, peak_rows, component_rows


def _compute_waveform_statistics(
    *,
    timestamps: pd.Series,
    values: pd.Series,
    sampling_rate_hz: float,
) -> dict[str, Any]:
    if values.empty:
        return _empty_waveform_statistics()

    numeric = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric.notna()
    numeric = numeric.loc[valid_mask].astype(float).reset_index(drop=True)
    if numeric.empty:
        return _empty_waveform_statistics()

    if timestamps.empty:
        valid_timestamps = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        valid_timestamps = pd.to_datetime(
            timestamps.loc[valid_mask].reset_index(drop=True),
            utc=True,
            format="mixed",
        )

    sample_count = int(len(numeric))
    if len(valid_timestamps) >= 2:
        aligned_duration_s = float(
            (valid_timestamps.iloc[-1] - valid_timestamps.iloc[0]).total_seconds()
        )
    elif sample_count >= 2:
        aligned_duration_s = float((sample_count - 1) / sampling_rate_hz)
    else:
        aligned_duration_s = 0.0

    array = numeric.to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(np.square(array))))
    max_abs = float(np.max(np.abs(array)))
    std_value = float(numeric.std()) if sample_count > 1 else None
    skewness = float(numeric.skew()) if sample_count > 2 else None
    kurtosis = float(numeric.kurt()) if sample_count > 3 else None

    return {
        "sample_count": sample_count,
        "aligned_duration_s": aligned_duration_s,
        "mean": float(numeric.mean()),
        "std": std_value,
        "rms": rms,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "peak_to_peak": float(numeric.max() - numeric.min()),
        "energy": float(np.sum(np.square(array))),
        "crest_factor": (max_abs / rms) if rms > 0 else None,
        "zero_crossing_rate": _zero_crossing_rate(array),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def _empty_waveform_statistics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "aligned_duration_s": None,
        "mean": None,
        "std": None,
        "rms": None,
        "min": None,
        "max": None,
        "peak_to_peak": None,
        "energy": None,
        "crest_factor": None,
        "zero_crossing_rate": None,
        "skewness": None,
        "kurtosis": None,
    }


def _zero_crossing_rate(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    signs = np.sign(values)
    crossings = np.sum(signs[1:] * signs[:-1] < 0)
    return float(crossings / (len(values) - 1))


def _parse_sensor_name(sensor_name: str) -> dict[str, str | None]:
    parts = sensor_name.split("_")
    return {
        "deck": parts[0] if len(parts) > 0 else None,
        "span": parts[1] if len(parts) > 1 else None,
        "side": parts[2] if len(parts) > 2 else None,
        "location": parts[3] if len(parts) > 3 else None,
        "quantity": parts[4] if len(parts) > 4 else None,
        "axis": parts[5] if len(parts) > 5 else None,
    }


def _to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    return float(value)
