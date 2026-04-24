# Temperature Correlation Discussion Summary

This document summarizes our working discussion regarding the temperature correlations observed in the `AQUINAS` dataset, reconciling initial data-driven hypotheses with the ground truth provided in the organizer's documentation, and extending the analysis to preprocessed waveform features. See `notebooks/misc/A_temperature_correlations.ipynb` (raw TABLE-based analysis) and `notebooks/misc/B_preprocessed_temperature_correlations.ipynb` (raw-vs-processed comparison, all 48 sensors) for the supporting code.

---

## 1. The Strain "Lock-Up" Illusion
**Initial Hypothesis:** The strong negative correlation between strain `Mean_Value` and temperature (which became extreme in Set 4) looked like a structural boundary condition locking up. It appeared as if the bridge could no longer expand freely in the winter, causing massive locked-in compressive thermal stresses.

**Ground Truth (Correction):** The organizer notes state that the strain sensors use **intensity-modulated fiber-optic technology** ("optical strands") and are **not temperature-compensated**.
* **Conclusion:** The massive drifts and correlations are largely a *hardware measurement artifact*. The fiber optic cables themselves expand and contract with temperature, heavily shifting the baseline. It is not necessarily indicative of dangerous structural stress.

**Quantitative confirmation (Notebook B):** Raw `Mean_Value` temperature correlations average ≈ −0.58 across strain sensors. After the full preprocessing pipeline (bandpass filter → linear-endpoints zeroing → Synchro alignment), aligned waveform mean correlations collapse to ≤ |0.11|, confirming that baseline zeroing effectively removes the hardware-induced drift.

---

## 2. The `OLD_S1_UP_SUP_STR` Anomaly
**Initial Hypothesis:** In Sets 1-3, `OLD_S1_UP_SUP_STR` showed strong positive correlations, unlike most other sensors. We theorized this was a real structural response to solar radiation creating a massive vertical thermal gradient on the upper superstructure of the older deck.

**Ground Truth (Correction):** The organizer confirmed this specific sensor was **damaged between SET3 and SET4**. Its baseline jumps wildly to ~30 and its range collapses to 0.0 later on.
* **Conclusion:** While localized heating exists, the severe deviations and ultimate failure of this channel are due to a degrading hardware failure, not a unique structural phenomenon. The pipeline configuration explicitly excludes this sensor for SET4 and SET5.

**Confirmed in Notebook B:** The preprocess store validation check (`OLD_S1_UP_SUP_STR` absent from SET4/SET5 event-sensor rows) passes cleanly, verifying end-to-end enforcement of the exclusion policy across 35,203 retained events and 622,177 event-sensor records.

---

## 3. Acceleration Dynamics
**Observation:** Acceleration sensors show strictly positive correlations with temperature across both decks (higher temperatures = higher raw mean accelerations, ≈ +0.80 average).
* **Conclusion:** This is structurally valid. Pavement, structural materials, and elastomeric bearings soften at higher temperatures, reducing stiffness and increasing dynamic vibration amplitudes under traffic loads. The OLD deck exhibits greater sensitivity to this softening than the NEW deck.

**After preprocessing:** Amplitude metrics (`waveform_std`, `waveform_rms`, `waveform_energy`, `waveform_peak_to_peak`) retain small positive residual correlations in the 0.1–0.3 range, primarily in ACC sensors at mid-span and interior locations. This residual thermal sensitivity in acceleration amplitudes is a genuine structural signal (temperature-dependent stiffness) rather than a measurement artifact, and will inform the normalization strategy for health scoring.

---

## 4. Why Baseline Zeroing is Critical
Because the strain sensor baselines drift so severely with uncompensated temperature changes, the absolute `Mean_Value` across sets is a fundamentally biased metric for long-term health scoring.

* **Pipeline strategy:** As directed by the organizer's `AQUINAS_Explorer.R` methodology, the toolkit applies per-sensor endpoint-line subtraction (`zeroing_method = linear_endpoints`) before Synchro alignment.
* **Focus for health scoring:** Amplitude-based features of the zeroed, aligned waveform (`std`, `rms`, `energy`, `peak_to_peak`) are the defensible temperature-independent alternatives to raw `Mean_Value`. These are computed and stored in the feature extraction stage.

---

## 5. Preprocessing Pipeline Summary
The full pipeline applied to each acquired event is:

1. **Bandpass filter** — 0.5–20 Hz Butterworth (`filter_loaded_event_group()`)
2. **Zeroing** — linear endpoint-line subtraction (`zero_loaded_event_group()`)
3. **Synchronization/alignment** — organizer Synchro two-pass, first sensor as reference, no interpolation (`align_event_group()`)

Applied to the full dataset (5 acquisition sets, July 2022 – June 2024), this produced 35,203 retained and aligned events stored in `preprocess.sqlite`. The largest raw-to-processed correlation shifts (Δ > 0.9) are concentrated in SET4 (January 2024 — winter minimum temperature), consistent with the fiber-optic strain artifact being strongest in cold conditions.

---

## 6. Open Items
* **Temperature normalization** — the residual 0.1–0.3 positive correlation in ACC amplitude metrics (temperature-dependent structural stiffness) should be modeled or normalized before building the health score. This is deferred to the training/scoring stages.
* **OMA / modal tracking** — Operational Modal Analysis across sets can now be performed on the aligned, zeroed waveforms. Expected bridge frequencies are 2–10 Hz, consistent with the 0.5–20 Hz bandpass. Reference: DOI `10.1016/j.prostr.2024.09.248`.
