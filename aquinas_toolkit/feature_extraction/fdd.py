"""Frequency Domain Decomposition helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, stft


def frequency_domain_decomposition(
    waveform_matrices: pd.DataFrame | np.ndarray | Sequence[pd.DataFrame | np.ndarray],
    sampling_rate_hz: float = 100.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    window: str = "hann",
) -> dict[str, np.ndarray]:
    """Compute FDD singular-value spectra from one or more multichannel events."""
    if isinstance(waveform_matrices, (pd.DataFrame, np.ndarray)):
        matrices = [waveform_matrices]
    else:
        matrices = list(waveform_matrices)

    if not matrices:
        raise ValueError("At least one waveform matrix is required for FDD.")

    average_psd = None
    frequencies = None
    channel_count = None

    for matrix in matrices:
        array = _as_matrix(matrix)
        if channel_count is None:
            channel_count = array.shape[1]
        elif array.shape[1] != channel_count:
            raise ValueError("All waveform matrices must have the same channel count.")

        freqs, event_psd = _estimate_cross_spectral_density(
            array,
            sampling_rate_hz=sampling_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        if average_psd is None:
            average_psd = event_psd
            frequencies = freqs
        else:
            average_psd += event_psd

    average_psd = average_psd / len(matrices)
    singular_values = np.zeros((average_psd.shape[0], average_psd.shape[1]))
    mode_shapes = np.zeros_like(average_psd, dtype=np.complex128)

    for freq_index in range(average_psd.shape[0]):
        eigenvalues, eigenvectors = np.linalg.eigh(average_psd[freq_index])
        order = np.argsort(eigenvalues)[::-1]
        singular_values[freq_index] = eigenvalues[order].real
        mode_shapes[freq_index] = eigenvectors[:, order]

    return {
        "frequencies_hz": frequencies,
        "singular_values": singular_values,
        "mode_shapes": mode_shapes,
        "spectral_density_matrices": average_psd,
    }


def summarize_fdd_peaks(
    frequencies_hz: np.ndarray,
    singular_values: np.ndarray,
    frequency_band_hz: tuple[float, float] = (0.5, 20.0),
    n_peaks: int = 5,
) -> pd.DataFrame:
    """Return dominant peaks from the first FDD singular-value curve."""
    if singular_values.ndim != 2 or singular_values.shape[1] == 0:
        raise ValueError("singular_values must be a 2D array with at least one column.")

    low_hz, high_hz = frequency_band_hz
    mask = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    band_freqs = frequencies_hz[mask]
    first_curve = singular_values[mask, 0]

    peak_indices, _ = find_peaks(first_curve)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(first_curve))])

    ordered = peak_indices[np.argsort(first_curve[peak_indices])[::-1][:n_peaks]]
    peak_table = pd.DataFrame(
        {
            "frequency_hz": band_freqs[ordered],
            "singular_value": first_curve[ordered],
        }
    )
    return peak_table.sort_values("frequency_hz").reset_index(drop=True)


def summarize_fdd_mode_shapes(
    frequencies_hz: np.ndarray,
    singular_values: np.ndarray,
    mode_shapes: np.ndarray,
    channel_names: Sequence[str] | None = None,
    frequency_band_hz: tuple[float, float] = (0.5, 20.0),
    n_peaks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return picked FDD peaks together with normalized mode-shape components.

    The returned peak table includes the frequency-bin index so the associated
    first singular vector can be retrieved reproducibly from the FDD result.
    """
    peak_table = summarize_fdd_peaks(
        frequencies_hz,
        singular_values,
        frequency_band_hz=frequency_band_hz,
        n_peaks=n_peaks,
    )
    peak_table = peak_table.copy()
    peak_table["frequency_index"] = peak_table["frequency_hz"].apply(
        lambda freq: int(np.argmin(np.abs(frequencies_hz - freq)))
    )

    channel_count = mode_shapes.shape[1]
    if channel_names is None:
        channel_names = [f"channel_{index + 1}" for index in range(channel_count)]
    if len(channel_names) != channel_count:
        raise ValueError("channel_names length must match the number of mode-shape channels.")

    rows: list[dict[str, object]] = []
    for peak_rank, peak in enumerate(peak_table.itertuples(index=False), start=1):
        vector = mode_shapes[peak.frequency_index, :, 0]
        amplitudes = np.abs(vector)
        max_amplitude = np.max(amplitudes)
        if max_amplitude == 0:
            normalized = amplitudes
        else:
            normalized = amplitudes / max_amplitude
        phases_deg = np.rad2deg(np.angle(vector))

        for channel_name, amplitude, phase_deg in zip(
            channel_names, normalized, phases_deg, strict=True
        ):
            rows.append(
                {
                    "peak_rank": peak_rank,
                    "frequency_hz": peak.frequency_hz,
                    "singular_value": peak.singular_value,
                    "channel": channel_name,
                    "mode_shape_amplitude": float(amplitude),
                    "mode_shape_phase_deg": float(phase_deg),
                }
            )

    mode_shape_table = pd.DataFrame(rows)
    return peak_table, mode_shape_table


def _estimate_cross_spectral_density(
    waveform_matrix: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
) -> tuple[np.ndarray, np.ndarray]:
    if waveform_matrix.shape[0] < 2:
        raise ValueError("Waveform matrix must contain at least two samples.")

    segment_length = min(nperseg, waveform_matrix.shape[0])
    overlap = min(noverlap, max(segment_length - 1, 0))
    frequencies_hz, _, zxx = stft(
        waveform_matrix.T,
        fs=sampling_rate_hz,
        window=window,
        nperseg=segment_length,
        noverlap=overlap,
        boundary=None,
        padded=False,
        axis=-1,
    )
    spectra = np.transpose(zxx, (1, 2, 0))
    spectral_density = np.einsum("fsc,fsd->fcd", spectra, np.conjugate(spectra))
    spectral_density /= spectra.shape[1]
    return frequencies_hz, spectral_density


def _as_matrix(waveform_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(waveform_matrix, pd.DataFrame):
        numeric = waveform_matrix.drop(columns=["timestamp"], errors="ignore")
        array = numeric.to_numpy(dtype=float)
    else:
        array = np.asarray(waveform_matrix, dtype=float)

    if array.ndim != 2:
        raise ValueError("waveform_matrix must be a 2D array or DataFrame.")
    if array.shape[1] < 2:
        raise ValueError("waveform_matrix must contain at least two channels for FDD.")
    return array