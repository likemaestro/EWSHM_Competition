"""Record-level QC for neural preprocessing inputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquinas_toolkit.io import parse_sensor_name


EVENT_QC_COLUMNS = [
    "set_id",
    "sensor_name",
    "sensor_type",
    "event_id",
    "Start_Time",
    "End_Time",
    "Duration",
    "N",
    "T_N",
    "Range",
    "qc_status",
    "discard_reason",
    "warning_reason",
    "mad_score_max",
]

SENSOR_QC_COLUMNS = [
    "set_id",
    "sensor_name",
    "sensor_type",
    "n_total_records",
    "n_keep",
    "n_warning",
    "n_discard",
    "n_available_records",
    "n_coverage_missing",
    "coverage_missing_rate",
    "n_true_qc_failures",
    "true_failure_rate",
    "discard_rate",
    "warning_rate",
    "main_discard_reasons",
    "sensor_status",
]

COVERAGE_MISSING_REASON = "not_available_for_global_event"

TRUE_QC_FAILURE_REASONS = {
    "missing_row_range",
    "invalid_row_range",
    "waveform_load_failed",
    "nan_values",
    "timestamp_error",
    "flat_signal",
    "strain_window_out_of_bounds",
    "acc_short_duration",
}


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
class QCSettings:
    """Settings for deterministic QC and MAD warning rules."""

    flat_range_tolerance: float = 1e-12
    mad_warning_threshold: float = 3.5
    mad_severe_threshold: float = 5.0
    severe_plot_limit: int = 5
    sanity_plot_limit: int = 5


@dataclass(frozen=True)
class QCResult:
    """QC outputs used by neural input packaging."""

    event_qc: pd.DataFrame
    retained_event_ids: set[str]
    retained_sensor_names_by_event: dict[str, set[str]]


def run_neural_record_qc(
    preprocess_store: Any,
    *,
    retained_events: pd.DataFrame,
    required_sensor_names: list[str],
    sampling_rate_hz: float,
    peak_window_half_samples: int,
    acc_min_aligned_samples: int,
    settings: QCSettings,
    output_dir: Path,
) -> QCResult:
    """Run deterministic QC and MAD warning rules for selected preprocessing records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_plots_dir = output_dir / "flagged_plots"
    flagged_plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    waveform_examples: dict[str, list[dict[str, Any]]] = {}

    for event in retained_events.itertuples(index=False):
        event_id = str(event.event_id)
        event_sensors = preprocess_store.load_event_sensors(event_id)
        aligned = preprocess_store.load_aligned_event(event_id)
        for sensor_name in required_sensor_names:
            sensor_rows = event_sensors.loc[event_sensors["sensor_name"] == sensor_name]
            if sensor_rows.empty:
                rows.append(
                    _base_row(
                        event=event,
                        sensor_name=sensor_name,
                        qc_status="discard",
                        discard_reason=COVERAGE_MISSING_REASON,
                    )
                )
                continue

            sensor = sensor_rows.iloc[0]
            row, values = _qc_one_record(
                event=event,
                sensor=sensor,
                aligned=aligned,
                sampling_rate_hz=sampling_rate_hz,
                peak_window_half_samples=peak_window_half_samples,
                acc_min_aligned_samples=acc_min_aligned_samples,
                flat_range_tolerance=settings.flat_range_tolerance,
            )
            rows.append(row)
            if row["qc_status"] == "discard" or row["qc_status"] == "keep":
                _add_plot_example(waveform_examples, row, values, settings=settings)

    event_qc = pd.DataFrame(rows)
    if event_qc.empty:
        event_qc = pd.DataFrame(columns=EVENT_QC_COLUMNS)
    event_qc = _apply_mad_warnings(event_qc, settings=settings)
    _write_qc_outputs(output_dir, event_qc)
    _write_qc_plots(flagged_plots_dir, event_qc, waveform_examples, settings=settings)

    accepted = event_qc.loc[event_qc["qc_status"].isin(["keep", "warning"])].copy()
    retained_sensor_names_by_event = {
        event_id: set(group["sensor_name"].astype(str))
        for event_id, group in accepted.groupby("event_id", sort=False)
    }
    required = set(required_sensor_names)
    retained_event_ids = {
        event_id
        for event_id, sensor_names in retained_sensor_names_by_event.items()
        if required.issubset(sensor_names)
    }
    return QCResult(
        event_qc=event_qc,
        retained_event_ids=retained_event_ids,
        retained_sensor_names_by_event=retained_sensor_names_by_event,
    )


def _qc_one_record(
    *,
    event: Any,
    sensor: pd.Series,
    aligned: pd.DataFrame,
    sampling_rate_hz: float,
    peak_window_half_samples: int,
    acc_min_aligned_samples: int,
    flat_range_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    sensor_name = str(sensor["sensor_name"])
    parsed = parse_sensor_name(sensor_name)
    sensor_type = _sensor_type(sensor_name)
    duration = _to_float(sensor.get("duration"))
    range_value = _to_float(sensor.get("range_value"))
    mean_value = _to_float(sensor.get("mean_value"))
    diff_value = _to_float(sensor.get("diff_value"))
    start_row = _to_int_or_none(sensor.get("start_row_1based"))
    end_row = _to_int_or_none(sensor.get("end_row_1based"))

    if start_row is None or end_row is None:
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="missing_row_range",
            ),
            np.array([], dtype=float),
        )
    if end_row <= start_row:
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                n=end_row - start_row + 1,
                t_n=(end_row - start_row + 1) / sampling_rate_hz,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="invalid_row_range",
            ),
            np.array([], dtype=float),
        )

    n_samples = end_row - start_row + 1
    t_n = n_samples / sampling_rate_hz
    if sensor_name not in aligned.columns or "timestamp_utc" not in aligned.columns:
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                n=n_samples,
                t_n=t_n,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="waveform_load_failed",
            ),
            np.array([], dtype=float),
        )

    timestamps = pd.to_datetime(aligned["timestamp_utc"], utc=True, errors="coerce")
    values = aligned[sensor_name].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                n=n_samples,
                t_n=t_n,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="nan_values",
            ),
            values,
        )
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                n=n_samples,
                t_n=t_n,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="timestamp_error",
            ),
            values,
        )

    signal_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
    table_flat = range_value is not None and abs(range_value) <= flat_range_tolerance
    signal_flat = signal_range <= flat_range_tolerance
    if table_flat or signal_flat:
        return (
            _base_row(
                event=event,
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                duration=duration,
                n=n_samples,
                t_n=t_n,
                range_value=range_value,
                mean_value=mean_value,
                diff_value=diff_value,
                qc_status="discard",
                discard_reason="flat_signal",
                rms=_rms(values),
                peak_abs=_peak_abs(values),
            ),
            values,
        )

    if parsed["quantity"] == "STR":
        peak_idx = int(np.argmax(np.abs(values)))
        window_bounds = strain_peak_window_bounds(
            peak_idx=peak_idx,
            signal_length=len(values),
            peak_window_half_samples=peak_window_half_samples,
        )
        if window_bounds is None:
            return (
                _base_row(
                    event=event,
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    duration=duration,
                    n=n_samples,
                    t_n=t_n,
                    range_value=range_value,
                    mean_value=mean_value,
                    diff_value=diff_value,
                    qc_status="discard",
                    discard_reason="strain_window_out_of_bounds",
                    rms=_rms(values),
                    peak_abs=_peak_abs(values),
                ),
                values,
            )
    elif parsed["quantity"] == "ACC" and parsed["axis"] == "Z":
        min_duration_s = acc_min_aligned_samples / sampling_rate_hz
        if n_samples < acc_min_aligned_samples or t_n < min_duration_s:
            return (
                _base_row(
                    event=event,
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    duration=duration,
                    n=n_samples,
                    t_n=t_n,
                    range_value=range_value,
                    mean_value=mean_value,
                    diff_value=diff_value,
                    qc_status="discard",
                    discard_reason="acc_short_duration",
                    rms=_rms(values),
                    peak_abs=_peak_abs(values),
                ),
                values,
            )

    return (
        _base_row(
            event=event,
            sensor_name=sensor_name,
            sensor_type=sensor_type,
            duration=duration,
            n=n_samples,
            t_n=t_n,
            range_value=range_value,
            mean_value=mean_value,
            diff_value=diff_value,
            qc_status="keep",
            rms=_rms(values),
            peak_abs=_peak_abs(values),
        ),
        values,
    )


def _apply_mad_warnings(event_qc: pd.DataFrame, *, settings: QCSettings) -> pd.DataFrame:
    frame = event_qc.copy()
    frame["_rms"] = pd.to_numeric(frame.get("_rms"), errors="coerce")
    frame["_peak_abs"] = pd.to_numeric(frame.get("_peak_abs"), errors="coerce")
    feature_columns = {
        "Range": "Range",
        "Mean_Value": "_mean_value",
        "Diff_Value": "_diff_value",
        "RMS": "_rms",
        "peak_abs": "_peak_abs",
    }
    frame["mad_score_max"] = 0.0
    for _, group_index in frame.groupby(["set_id", "sensor_name"], sort=False).groups.items():
        idx = list(group_index)
        for column in feature_columns.values():
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame.loc[idx, column], errors="coerce")
            valid = values.notna() & (frame.loc[idx, "qc_status"] != "discard")
            if valid.sum() < 2:
                continue
            scores = _robust_scores(values.loc[valid])
            score_index = values.loc[valid].index
            frame.loc[score_index, "mad_score_max"] = np.maximum(
                frame.loc[score_index, "mad_score_max"].to_numpy(dtype=float),
                np.abs(scores.to_numpy(dtype=float)),
            )

    warning_mask = (frame["qc_status"] != "discard") & (
        frame["mad_score_max"] > settings.mad_warning_threshold
    )
    frame.loc[warning_mask, "qc_status"] = "warning"
    severe_mask = (frame["qc_status"] != "discard") & (
        frame["mad_score_max"] > settings.mad_severe_threshold
    )
    frame.loc[severe_mask, "warning_reason"] = "severe_mad_outlier"
    frame["mad_score_max"] = frame["mad_score_max"].fillna(0.0)
    return frame


def _robust_scores(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if mad and not pd.isna(mad):
        return 0.6745 * (values - median) / mad
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr and not pd.isna(iqr):
        return 0.7413 * (values - median) / iqr
    return pd.Series(0.0, index=values.index)


def _write_qc_outputs(output_dir: Path, event_qc: pd.DataFrame) -> None:
    event_qc[EVENT_QC_COLUMNS].to_csv(output_dir / "event_qc_report.csv", index=False)
    sensor_report = _sensor_report(event_qc)
    sensor_report.to_csv(output_dir / "sensor_qc_report.csv", index=False)
    discarded = event_qc.loc[event_qc["qc_status"] == "discard", [
        "set_id",
        "sensor_name",
        "sensor_type",
        "event_id",
        "discard_reason",
    ]]
    discarded.to_csv(output_dir / "discarded_events.csv", index=False)
    summary = _summary_payload(event_qc)
    (output_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sensor_report(event_qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (set_id, sensor_name), group in event_qc.groupby(["set_id", "sensor_name"], sort=False):
        n_total = int(len(group))
        n_keep = int((group["qc_status"] == "keep").sum())
        n_warning = int((group["qc_status"] == "warning").sum())
        n_discard = int((group["qc_status"] == "discard").sum())
        coverage_missing_mask = group["discard_reason"] == COVERAGE_MISSING_REASON
        true_failure_mask = group["discard_reason"].isin(TRUE_QC_FAILURE_REASONS)
        n_coverage_missing = int(coverage_missing_mask.sum())
        n_available = n_total - n_coverage_missing
        n_true_qc_failures = int(true_failure_mask.sum())
        discard_rate = n_discard / n_total if n_total else 0.0
        warning_rate = n_warning / n_total if n_total else 0.0
        coverage_missing_rate = n_coverage_missing / n_total if n_total else 0.0
        true_failure_rate = n_true_qc_failures / n_available if n_available else 0.0
        reasons = [
            reason for reason in group["discard_reason"].astype(str).tolist() if reason
        ]
        if true_failure_rate > 0.20:
            status = "exclude"
        elif warning_rate > 0.20:
            status = "warning"
        else:
            status = "good"
        rows.append(
            {
                "set_id": set_id,
                "sensor_name": sensor_name,
                "sensor_type": str(group["sensor_type"].iloc[0]),
                "n_total_records": n_total,
                "n_keep": n_keep,
                "n_warning": n_warning,
                "n_discard": n_discard,
                "n_available_records": n_available,
                "n_coverage_missing": n_coverage_missing,
                "coverage_missing_rate": coverage_missing_rate,
                "n_true_qc_failures": n_true_qc_failures,
                "true_failure_rate": true_failure_rate,
                "discard_rate": discard_rate,
                "warning_rate": warning_rate,
                "main_discard_reasons": ";".join(
                    reason for reason, _count in Counter(reasons).most_common(3)
                ),
                "sensor_status": status,
            }
        )
    return pd.DataFrame(rows, columns=SENSOR_QC_COLUMNS)


def _summary_payload(event_qc: pd.DataFrame) -> dict[str, Any]:
    total = int(len(event_qc))
    kept = int((event_qc["qc_status"] == "keep").sum())
    warning = int((event_qc["qc_status"] == "warning").sum())
    discarded = int((event_qc["qc_status"] == "discard").sum())
    coverage_missing = int((event_qc["discard_reason"] == COVERAGE_MISSING_REASON).sum())
    true_qc_failures = int(event_qc["discard_reason"].isin(TRUE_QC_FAILURE_REASONS).sum())
    return {
        "total_records": total,
        "kept_records": kept,
        "warning_records": warning,
        "discarded_records": discarded,
        "discard_rate": discarded / total if total else 0.0,
        "warning_rate": warning / total if total else 0.0,
        "coverage_missing_records": coverage_missing,
        "true_qc_failure_records": true_qc_failures,
        "coverage_missing_rate": coverage_missing / total if total else 0.0,
        "true_failure_rate": true_qc_failures / (total - coverage_missing)
        if total > coverage_missing
        else 0.0,
        "counts_by_sensor_type": event_qc["sensor_type"].value_counts().to_dict(),
        "counts_by_discard_reason": (
            event_qc.loc[event_qc["discard_reason"].astype(str) != "", "discard_reason"]
            .value_counts()
            .to_dict()
        ),
        "counts_by_set": event_qc["set_id"].value_counts().to_dict(),
    }


def _write_qc_plots(
    flagged_plots_dir: Path,
    event_qc: pd.DataFrame,
    waveform_examples: dict[str, list[dict[str, Any]]],
    *,
    settings: QCSettings,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    folders = [
        "strain_window_out_of_bounds",
        "acc_short_duration",
        "flat_signal",
        "timestamp_error",
        "severe_mad_outlier",
        "sanity_check_retained",
    ]
    for folder in folders:
        (flagged_plots_dir / folder).mkdir(parents=True, exist_ok=True)

    for reason in folders[:-2]:
        for index, example in enumerate(waveform_examples.get(reason, [])[: settings.severe_plot_limit]):
            _save_waveform_plot(flagged_plots_dir / reason / f"{index:03d}.png", example, plt)

    severe = event_qc.loc[event_qc["warning_reason"] == "severe_mad_outlier"].head(
        settings.severe_plot_limit
    )
    for index, row in enumerate(severe.itertuples(index=False)):
        example = {
            "row": row._asdict(),
            "values": np.array([], dtype=float),
        }
        _save_waveform_plot(flagged_plots_dir / "severe_mad_outlier" / f"{index:03d}.png", example, plt)

    sanity = waveform_examples.get("keep", [])[: settings.sanity_plot_limit]
    for index, example in enumerate(sanity):
        _save_waveform_plot(flagged_plots_dir / "sanity_check_retained" / f"{index:03d}.png", example, plt)


def _save_waveform_plot(path: Path, example: dict[str, Any], plt: Any) -> None:
    row = example["row"]
    values = example["values"]
    fig, ax = plt.subplots(figsize=(8, 3))
    if len(values):
        ax.plot(values)
    else:
        ax.text(0.5, 0.5, "No waveform preview", ha="center", va="center")
    reason = row.get("discard_reason") or row.get("warning_reason") or ""
    ax.set_title(
        f"{row.get('set_id')} {row.get('sensor_name')} {row.get('event_id')} "
        f"{row.get('qc_status')} {reason}"
    )
    ax.set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _add_plot_example(
    examples: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    values: np.ndarray,
    *,
    settings: QCSettings,
) -> None:
    key = row["discard_reason"] if row["qc_status"] == "discard" else "keep"
    limit = settings.severe_plot_limit if key != "keep" else settings.sanity_plot_limit
    bucket = examples.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append({"row": row, "values": values})


def _base_row(
    *,
    event: Any,
    sensor_name: str,
    sensor_type: str | None = None,
    duration: float | None = None,
    n: int | None = None,
    t_n: float | None = None,
    range_value: float | None = None,
    mean_value: float | None = None,
    diff_value: float | None = None,
    qc_status: str,
    discard_reason: str = "",
    warning_reason: str = "",
    rms: float | None = None,
    peak_abs: float | None = None,
) -> dict[str, Any]:
    parsed_type = sensor_type or _sensor_type(sensor_name)
    return {
        "set_id": str(event.set_name),
        "sensor_name": sensor_name,
        "sensor_type": parsed_type,
        "event_id": str(event.event_id),
        "Start_Time": str(event.start_time_utc),
        "End_Time": str(event.end_time_utc),
        "Duration": duration,
        "N": n,
        "T_N": t_n,
        "Range": range_value,
        "qc_status": qc_status,
        "discard_reason": discard_reason,
        "warning_reason": warning_reason,
        "mad_score_max": 0.0,
        "_rms": rms,
        "_peak_abs": peak_abs,
        "_mean_value": mean_value,
        "_diff_value": diff_value,
    }


def _sensor_type(sensor_name: str) -> str:
    parsed = parse_sensor_name(sensor_name)
    if parsed["quantity"] == "ACC" and parsed["axis"] == "Z":
        return "ACC_Z"
    if parsed["quantity"] == "STR":
        location = parsed["location"]
        return f"{location}_STR" if location else "STR"
    return str(parsed["quantity"])


def _rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def _peak_abs(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _to_int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
