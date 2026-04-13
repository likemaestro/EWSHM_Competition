"""Artifact export pipeline for the offline AQUINAS bridge viewer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aquinas_toolkit.io import AquinasReader
from aquinas_toolkit.utils.run_management import RunContext, read_metadata
from aquinas_toolkit.visualization.layout import build_bridge_geometry, build_sensor_layout

SCHEMA_VERSION = "2026-04-13"
VIEWER_ASSET_NAMES = ("index.html", "viewer.css", "viewer.js")
MAX_WAVEFORM_PREVIEWS_PER_DECK = 2
MAX_WAVEFORM_SAMPLES = 240

METRIC_DEFINITIONS = {
    "event_count": {
        "label": "Event Count",
        "description": "Number of trigger events recorded for the sensor in this dataset.",
    },
    "mean_range": {
        "label": "Mean Range",
        "description": "Average per-event range from the AQUINAS index tables.",
    },
    "mean_abs_mean_value": {
        "label": "Mean Absolute Mean Value",
        "description": "Average absolute mean value across recorded events.",
    },
    "mean_duration": {
        "label": "Mean Duration",
        "description": "Average trigger duration across recorded events.",
    },
    "mean_temperature": {
        "label": "Mean Temperature",
        "description": "Average temperature recorded for the sensor events.",
    },
}


@dataclass(frozen=True)
class VisualizationBuildResult:
    """Resolved locations for a built visualization artifact bundle."""

    run_id: str
    output_dir: Path
    manifest_path: Path
    index_path: Path
    dataset_names: tuple[str, ...]


def build_visualization_artifacts(
    run_context: RunContext,
    *,
    set_names: list[str] | None = None,
    output_dir: Path | None = None,
    include_waveforms: bool = False,
) -> VisualizationBuildResult:
    """Build a portable visualization bundle for a run."""
    run_config = _load_run_config(run_context.config_path)
    dataset_root, selected_sets = _resolve_dataset_selection(run_config, set_names)
    readers = {set_name: AquinasReader(dataset_root / set_name) for set_name in selected_sets}

    # The exported schema should reflect the sensors that actually exist in
    # the selected AQUINAS sets rather than a hardcoded 48-channel list.
    sensor_names = sorted({sensor for reader in readers.values() for sensor in reader.list_sensor_names()})
    sensor_layout = build_sensor_layout(sensor_names)
    layout_lookup = {row["sensor_id"]: row for row in sensor_layout}
    bridge_geometry = build_bridge_geometry()

    output_path = output_dir or (run_context.run_dir / "visualization")
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_viewer_assets(output_path)

    # Everything below feeds the stable viewer contract. Later scoring
    # stages can replace proxy metrics without changing the frontend shape.
    summary_rows, event_rows = _summarize_readers(readers, layout_lookup)
    metric_rows = _build_metric_rows(summary_rows)
    trend_rows = _build_trend_rows(metric_rows)
    correlation_rows = _build_correlations(trend_rows, layout_lookup)
    event_group_rows = _build_event_groups(event_rows)

    if include_waveforms:
        event_group_rows = _attach_waveform_previews(
            output_dir=output_path,
            readers=readers,
            event_rows=event_rows,
            event_groups=event_group_rows,
        )

    metadata = read_metadata(run_context.run_dir)
    dataset_catalog = [
        {
            "dataset": set_name,
            "label": _dataset_label(set_name),
            "position": index,
        }
        for index, set_name in enumerate(selected_sets, start=1)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_built_at_utc": _utc_now(),
        "run": {
            "run_id": run_context.run_id,
            "name": metadata.get("name"),
            "created_at_utc": metadata.get("created_at_utc"),
            "git_commit": metadata.get("git_commit"),
            "git_dirty": metadata.get("git_dirty"),
        },
        "available_datasets": dataset_catalog,
        "default_dataset": dataset_catalog[-1]["dataset"] if dataset_catalog else None,
        "supported_measurement_families": ["ALL", "ACC", "STR"],
        "metric_catalog": _build_metric_catalog(),
        "notes": {
            "viewer_mode": "offline analytical schematic",
            "metric_source": "AQUINAS index-table proxy metrics",
            "waveform_previews_included": include_waveforms,
        },
        "files": {
            "bridge_geometry": "bridge_geometry.json",
            "sensor_layout": "sensor_layout.json",
            "sensor_metrics": "sensor_metrics.json",
            "sensor_trends": "sensor_trends.json",
            "event_groups": "event_groups.json",
            "correlations": "correlations.json",
        },
    }

    _write_json(output_path / "bridge_geometry.json", bridge_geometry)
    _write_json(output_path / "sensor_layout.json", sensor_layout)
    _write_json(output_path / "sensor_metrics.json", metric_rows)
    _write_json(output_path / "sensor_trends.json", trend_rows)
    _write_json(output_path / "event_groups.json", event_group_rows)
    _write_json(output_path / "correlations.json", correlation_rows)
    _write_json(output_path / "manifest.json", manifest)

    return VisualizationBuildResult(
        run_id=run_context.run_id,
        output_dir=output_path,
        manifest_path=output_path / "manifest.json",
        index_path=output_path / "index.html",
        dataset_names=tuple(selected_sets),
    )


def _load_run_config(config_path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"Could not parse run config at {config_path}: {exc}") from exc


def _resolve_dataset_selection(
    run_config: dict[str, Any],
    set_names: list[str] | None,
) -> tuple[Path, list[str]]:
    # The run snapshot config remains the source of truth so a viewer built
    # later still matches the exact datasets the run was configured for.
    data_config = run_config.get("data") or {}
    raw_root = data_config.get("dataset_root", "AQUINAS_DATASET")
    dataset_root = Path(raw_root)
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    configured_sets = data_config.get("sets") or sorted(path.name for path in dataset_root.glob("AQUINAS_SET*"))
    if not configured_sets:
        raise RuntimeError(f"No AQUINAS sets configured or found under {dataset_root}.")

    if set_names is None:
        selected_sets = list(configured_sets)
    else:
        missing = [set_name for set_name in set_names if set_name not in configured_sets]
        if missing:
            raise RuntimeError(
                f"Requested dataset(s) not found in run config: {', '.join(missing)}."
            )
        selected_sets = list(dict.fromkeys(set_names))

    for set_name in selected_sets:
        set_dir = dataset_root / set_name
        if not set_dir.is_dir():
            raise RuntimeError(f"Configured dataset directory does not exist: {set_dir}")

    return dataset_root, selected_sets


def _summarize_readers(
    readers: dict[str, AquinasReader],
    layout_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []

    for set_name, reader in readers.items():
        dataset_label = _dataset_label(set_name)
        for sensor_id in reader.list_sensor_names():
            sensor_layout = layout_lookup[sensor_id]
            index_df = reader.load_index_table(sensor_id).copy()

            # Preserve the row index from the table so waveform previews can
            # later re-open the exact record through `AquinasReader`.
            index_df = index_df.reset_index().rename(columns={"index": "row_index"})

            range_series = _numeric_series(index_df, "Range")
            mean_value_series = _numeric_series(index_df, "Mean_Value")
            duration_series = _numeric_series(index_df, "Duration")
            temperature_series = _numeric_series(index_df, "Temperature")

            summary_rows.append(
                {
                    "dataset": set_name,
                    "dataset_label": dataset_label,
                    "dataset_position": _dataset_position(set_name),
                    "sensor_id": sensor_id,
                    "measurement_family": sensor_layout["measurement_family"],
                    "event_count": int(len(index_df)),
                    "mean_range": _safe_mean(range_series),
                    "mean_abs_mean_value": _safe_mean(mean_value_series.abs()),
                    "mean_duration": _safe_mean(duration_series),
                    "mean_temperature": _safe_mean(temperature_series),
                }
            )

            start_times = _to_iso_series(index_df, "Start_Time")
            end_times = _to_iso_series(index_df, "End_Time")
            event_frames.append(
                pd.DataFrame(
                    {
                        "dataset": set_name,
                        "dataset_label": dataset_label,
                        "dataset_position": _dataset_position(set_name),
                        "sensor_id": sensor_id,
                        "row_index": index_df["row_index"].astype(int),
                        "deck": sensor_layout["deck"],
                        "span": sensor_layout["span"],
                        "section": sensor_layout["section"],
                        "measurement_family": sensor_layout["measurement_family"],
                        "start_time_utc": start_times,
                        "end_time_utc": end_times,
                        "duration_seconds": duration_series,
                        "temperature_c": temperature_series,
                    }
                )
            )

    event_rows = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    event_rows = event_rows.dropna(subset=["start_time_utc", "end_time_utc"]).reset_index(drop=True)
    return summary_rows, event_rows


def _build_metric_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for row in summary_rows:
        family = row["measurement_family"]
        for metric_id in METRIC_DEFINITIONS:
            value = row[metric_id]
            if value is None:
                continue
            metrics.append(
                {
                    "sensor_id": row["sensor_id"],
                    "dataset": row["dataset"],
                    "dataset_label": row["dataset_label"],
                    "dataset_position": row["dataset_position"],
                    "measurement_family": family,
                    "metric_id": metric_id,
                    "value": _round_number(value),
                    "unit": _metric_unit(metric_id, family),
                    "source_stage": "proxy_metadata",
                    "status_band": "mid",
                }
            )

    metric_frame = pd.DataFrame(metrics)
    if metric_frame.empty:
        return []

    # Status bands are relative within dataset + metric + family. They are
    # visual cues for exploration, not absolute health classifications.
    for _, group in metric_frame.groupby(["dataset", "metric_id", "measurement_family"]):
        if len(group) == 1:
            metric_frame.loc[group.index, "status_band"] = "mid"
            continue
        low = group["value"].quantile(0.2)
        high = group["value"].quantile(0.8)
        for index, value in group["value"].items():
            if value <= low:
                metric_frame.at[index, "status_band"] = "low"
            elif value >= high:
                metric_frame.at[index, "status_band"] = "high"
            else:
                metric_frame.at[index, "status_band"] = "mid"

    return [dict(row) for row in metric_frame.to_dict(orient="records")]


def _build_trend_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_frame = pd.DataFrame(metric_rows)
    if metric_frame.empty:
        return []

    trend_rows: list[dict[str, Any]] = []
    grouped = metric_frame.sort_values("dataset_position").groupby(["sensor_id", "metric_id"])
    for (sensor_id, metric_id), group in grouped:
        first = group.iloc[0]
        trend_rows.append(
            {
                "sensor_id": sensor_id,
                "measurement_family": first["measurement_family"],
                "metric_id": metric_id,
                "unit": first["unit"],
                "points": [
                    {
                        "dataset": record["dataset"],
                        "dataset_label": record["dataset_label"],
                        "dataset_position": int(record["dataset_position"]),
                        "value": _round_number(record["value"]),
                        "status_band": record["status_band"],
                    }
                    for record in group.to_dict(orient="records")
                ],
            }
        )

    return trend_rows


def _build_correlations(
    trend_rows: list[dict[str, Any]],
    layout_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not trend_rows:
        return []

    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_DEFINITIONS:
        metric_trends = [row for row in trend_rows if row["metric_id"] == metric_id]
        if len(metric_trends) < 2:
            continue

        series_map = {
            row["sensor_id"]: {point["dataset"]: point["value"] for point in row["points"]}
            for row in metric_trends
        }
        dataset_order = sorted({dataset for points in series_map.values() for dataset in points})

        # Correlations are computed over the exported trend vectors across
        # monthly datasets. This keeps the bundle lightweight and aligned
        # with the viewer's comparison use case.
        correlation_frame = pd.DataFrame(
            {
                sensor_id: [values.get(dataset) for dataset in dataset_order]
                for sensor_id, values in series_map.items()
            },
            index=dataset_order,
        )
        correlation_matrix = correlation_frame.corr(method="pearson", min_periods=3)

        for source_sensor in correlation_matrix.columns:
            ranked = (
                correlation_matrix[source_sensor]
                .drop(labels=[source_sensor], errors="ignore")
                .dropna()
                .sort_values(key=lambda series: series.abs(), ascending=False)
                .head(3)
            )
            source_layout = layout_lookup[source_sensor]
            for rank, (target_sensor, coefficient) in enumerate(ranked.items(), start=1):
                target_layout = layout_lookup[target_sensor]
                rows.append(
                    {
                        "sensor_id": source_sensor,
                        "target_sensor_id": target_sensor,
                        "metric_id": metric_id,
                        "correlation": _round_number(float(coefficient)),
                        "rank": rank,
                        "same_section": source_layout["section_key"] == target_layout["section_key"],
                        "same_deck": source_layout["deck"] == target_layout["deck"],
                        "cross_deck_homologous": (
                            source_layout["homologous_sensor_id"] == target_sensor
                        ),
                        "same_measurement_family": (
                            source_layout["measurement_family"]
                            == target_layout["measurement_family"]
                        ),
                    }
                )

    return rows


def _build_event_groups(event_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if event_rows.empty:
        return []

    # Record_UID is sensor-local in AQUINAS, so event grouping is keyed by
    # dataset + deck + time window instead.
    grouped = event_rows.groupby(["dataset", "deck", "start_time_utc", "end_time_utc"], sort=True)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        dataset, deck, start_time_utc, end_time_utc = keys
        rows.append(
            {
                "event_group_id": _event_group_id(dataset, deck, start_time_utc, end_time_utc),
                "dataset": dataset,
                "dataset_label": group["dataset_label"].iloc[0],
                "dataset_position": int(group["dataset_position"].iloc[0]),
                "deck": deck,
                "start_time_utc": start_time_utc,
                "end_time_utc": end_time_utc,
                "sensor_count": int(group["sensor_id"].nunique()),
                "sensor_ids": sorted(group["sensor_id"].unique().tolist()),
                "spans": sorted(group["span"].dropna().unique().tolist()),
                "measurement_families": sorted(group["measurement_family"].dropna().unique().tolist()),
                "waveform_preview_path": None,
            }
        )
    return rows


def _attach_waveform_previews(
    *,
    output_dir: Path,
    readers: dict[str, AquinasReader],
    event_rows: pd.DataFrame,
    event_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    waveform_dir = output_dir / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    mutable_groups = [dict(row) for row in event_groups]
    grouped_rows = event_rows.groupby(["dataset", "deck", "start_time_utc", "end_time_utc"], sort=True)
    candidate_groups = {
        key: frame.sort_values(["sensor_id", "row_index"]).reset_index(drop=True)
        for key, frame in grouped_rows
    }

    selections: set[tuple[str, str, str, str]] = set()
    for dataset in sorted(event_rows["dataset"].unique()):
        dataset_rows = event_rows[event_rows["dataset"] == dataset]
        for deck in sorted(dataset_rows["deck"].unique()):
            subset = [
                (key, frame)
                for key, frame in candidate_groups.items()
                if key[0] == dataset and key[1] == deck
            ]
            subset.sort(key=lambda item: (-item[1]["sensor_id"].nunique(), item[0][2]))

            # Only export a few deck-scoped previews per dataset to keep the
            # offline bundle small and fast to open.
            for key, _ in subset[:MAX_WAVEFORM_PREVIEWS_PER_DECK]:
                selections.add(key)

    for group in mutable_groups:
        key = (
            group["dataset"],
            group["deck"],
            group["start_time_utc"],
            group["end_time_utc"],
        )
        if key not in selections:
            continue

        frame = candidate_groups[key]
        traces = []
        reader = readers[group["dataset"]]
        for record in frame.itertuples(index=False):
            _, waveform = reader.read_record(record.sensor_id, row_index=int(record.row_index))
            timestamp_column = waveform.columns[0]
            value_column = waveform.columns[1]
            traces.append(
                {
                    "sensor_id": record.sensor_id,
                    "measurement_family": record.measurement_family,
                    "timestamps": _downsample(waveform[timestamp_column].astype(str).tolist()),
                    "values": _downsample(
                        pd.to_numeric(waveform[value_column], errors="coerce")
                        .fillna(0.0)
                        .round(6)
                        .tolist()
                    ),
                }
            )

        slug = (
            f"{group['dataset']}_{group['deck']}_{group['start_time_utc']}"
            .replace(":", "-")
            .replace(".", "-")
        )
        file_name = f"{slug}.json"
        relative_path = Path("waveforms") / file_name
        _write_json(
            output_dir / relative_path,
            {
                "event_group_id": group["event_group_id"],
                "dataset": group["dataset"],
                "deck": group["deck"],
                "start_time_utc": group["start_time_utc"],
                "end_time_utc": group["end_time_utc"],
                "traces": traces,
            },
        )
        group["waveform_preview_path"] = str(relative_path).replace("\\", "/")

    return mutable_groups


def _copy_viewer_assets(output_dir: Path) -> None:
    source_dir = Path(files("aquinas_toolkit.visualization").joinpath("viewer_assets"))
    for asset_name in VIEWER_ASSET_NAMES:
        shutil.copy2(source_dir / asset_name, output_dir / asset_name)


def _build_metric_catalog() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": metric_id,
            "label": metric["label"],
            "description": metric["description"],
            "units_by_family": {
                "ACC": _metric_unit(metric_id, "ACC"),
                "STR": _metric_unit(metric_id, "STR"),
            },
        }
        for metric_id, metric in METRIC_DEFINITIONS.items()
    ]


def _metric_unit(metric_id: str, measurement_family: str) -> str:
    if metric_id in {"event_count"}:
        return "records"
    if metric_id in {"mean_duration"}:
        return "s"
    if metric_id in {"mean_temperature"}:
        return "deg C"
    return "g" if measurement_family == "ACC" else "mm/m"


def _numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column_name], errors="coerce")


def _to_iso_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series([None] * len(frame))
    series = pd.to_datetime(frame[column_name], errors="coerce", utc=True)
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_mean(series: pd.Series) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return float(cleaned.mean())


def _round_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        return round(float(value), 6)
    return value


def _dataset_label(dataset_name: str) -> str:
    match = re.search(r"AQUINAS_(SET\d+)_", dataset_name)
    return match.group(1) if match else dataset_name


def _dataset_position(dataset_name: str) -> int:
    match = re.search(r"AQUINAS_SET(\d+)_", dataset_name)
    return int(match.group(1)) if match else 0


def _event_group_id(dataset: str, deck: str, start_time_utc: str, end_time_utc: str) -> str:
    return f"{dataset}:{deck}:{start_time_utc}:{end_time_utc}"


def _downsample(values: list[Any], max_samples: int = MAX_WAVEFORM_SAMPLES) -> list[Any]:
    if len(values) <= max_samples:
        return values
    step = max(1, math.ceil(len(values) / max_samples))
    sampled = values[::step]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
