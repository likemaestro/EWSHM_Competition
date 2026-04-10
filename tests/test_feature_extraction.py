import numpy as np
import pandas as pd

from aquinas_toolkit.feature_extraction import (
    annotate_mode_shape_locations,
    frequency_domain_decomposition,
    summarize_fdd_mode_shapes,
    summarize_fdd_peaks,
)


def test_frequency_domain_decomposition_finds_dominant_mode_near_target_frequency() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)

    matrix = np.column_stack(
        [
            1.0 * np.sin(2 * np.pi * 3.2 * time),
            0.7 * np.sin(2 * np.pi * 3.2 * time + 0.3),
            1.2 * np.sin(2 * np.pi * 3.2 * time - 0.4),
            0.6 * np.sin(2 * np.pi * 3.2 * time + 0.8),
        ]
    )
    matrix += 0.15 * np.column_stack(
        [
            np.sin(2 * np.pi * 11.0 * time),
            np.sin(2 * np.pi * 11.0 * time + 0.2),
            np.sin(2 * np.pi * 11.0 * time - 0.1),
            np.sin(2 * np.pi * 11.0 * time + 0.4),
        ]
    )

    result = frequency_domain_decomposition(matrix, sampling_rate_hz=sampling_rate_hz)
    peaks = summarize_fdd_peaks(result["frequencies_hz"], result["singular_values"], n_peaks=3)

    assert any(abs(freq - 3.2) < 0.3 for freq in peaks["frequency_hz"])


def test_frequency_domain_decomposition_supports_multiple_event_matrices() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 20.0, 1.0 / sampling_rate_hz)

    event_a = np.column_stack(
        [np.sin(2 * np.pi * 2.5 * time), 0.8 * np.sin(2 * np.pi * 2.5 * time + 0.2)]
    )
    event_b = np.column_stack(
        [1.1 * np.sin(2 * np.pi * 2.5 * time - 0.1), 0.9 * np.sin(2 * np.pi * 2.5 * time + 0.4)]
    )

    result = frequency_domain_decomposition(
        [event_a, event_b],
        sampling_rate_hz=sampling_rate_hz,
        nperseg=512,
        noverlap=256,
    )

    assert result["singular_values"].shape[1] == 2
    assert result["mode_shapes"].shape[1:] == (2, 2)


def test_summarize_fdd_mode_shapes_reports_normalized_components() -> None:
    sampling_rate_hz = 100.0
    time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)
    matrix = np.column_stack(
        [
            1.0 * np.sin(2 * np.pi * 4.0 * time),
            0.5 * np.sin(2 * np.pi * 4.0 * time + 0.4),
            0.8 * np.sin(2 * np.pi * 4.0 * time - 0.2),
        ]
    )

    result = frequency_domain_decomposition(matrix, sampling_rate_hz=sampling_rate_hz)
    peak_table, mode_shape_table = summarize_fdd_mode_shapes(
        result["frequencies_hz"],
        result["singular_values"],
        result["mode_shapes"],
        channel_names=["ch1", "ch2", "ch3"],
        n_peaks=2,
    )

    assert not peak_table.empty
    assert set(mode_shape_table["channel"]) == {"ch1", "ch2", "ch3"}
    peak_groups = mode_shape_table.groupby("peak_rank")["mode_shape_amplitude"].max().tolist()
    assert all(np.isclose(value, 1.0) for value in peak_groups)
    signed_groups = mode_shape_table.groupby("peak_rank")["mode_shape_signed_component"].apply(
        lambda values: np.max(np.abs(values.to_numpy()))
    )
    assert all(np.isclose(value, 1.0) for value in signed_groups)


def test_annotate_mode_shape_locations_extracts_structural_position_fields() -> None:
    mode_shape_table = pd.DataFrame(
        {
            "peak_rank": [1, 1],
            "frequency_hz": [3.2, 3.2],
            "singular_value": [1.0, 1.0],
            "channel": ["OLD_S1_DO_INT_ACC_Z", "NEW_S2_UP_MID_ACC_Z"],
            "mode_shape_amplitude": [1.0, 0.8],
            "mode_shape_signed_component": [1.0, -0.8],
            "mode_shape_phase_deg": [0.0, 180.0],
        }
    )

    annotated = annotate_mode_shape_locations(mode_shape_table)

    assert list(annotated["deck"]) == ["OLD", "NEW"]
    assert list(annotated["span"]) == ["S1", "S2"]
    assert list(annotated["side"]) == ["DO", "UP"]
    assert list(annotated["location"]) == ["INT", "MID"]
    assert list(annotated["quantity"]) == ["ACC", "ACC"]
    assert list(annotated["axis"]) == ["Z", "Z"]
    assert list(annotated["position_label"]) == ["S1_DO_INT", "S2_UP_MID"]