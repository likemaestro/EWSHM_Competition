# notebooks/

Jupyter notebooks for exploration, visualisation, and presentation.

## Folder layout

- Top-level numbered notebooks (`01_`, `02_`, ...) are the main project
  storyline from raw data inspection through scoring.
- `misc/` holds supporting or one-off analyses that do not belong in the
  main numbered sequence.

## Naming convention

Notebooks are numbered sequentially so they tell a story from
raw data exploration through to the final health score:

| Notebook | Purpose | Status |
|---|---|---|
| `01_sensor_overview` | Dataset structure, raw waveform plots | Done |
| `02_preprocessing` | Filtering, zeroing, alignment | Done |
| `03_feature_extraction` | FDD, peak picking, and mode shapes | Done |
| `04_anomaly_detection` | Unsupervised outlier and trend detection | TODO |
| `05_health_scoring` | Final structural health score computation | TODO |

Supporting notebooks in `misc/` use alphabetical prefixes so they are clearly
distinguished from the numbered main storyline. Continue the series as
`A_`, `B_`, `C_`, and so on:

| Notebook | Purpose |
|---|---|
| `misc/A_temperature_correlations` | Exploratory temperature-correlation analysis |

## How to run

```bash
# from the repo root
pip install -e .
jupyter lab notebooks/
```

## Important rules

- **Notebooks are for exploration and presentation**, not for
  reusable logic. If you write a function that will be used in
  more than one notebook, move it to `aquinas_toolkit/`.
- Always import from the toolkit: `from aquinas_toolkit import AquinasReader`.
- The dataset path is `../AQUINAS_DATASET/AQUINAS_SET*` (relative
  to this folder).
- For config-setting meanings, see [configs/README.md](../configs/README.md).
- For preprocessing API semantics and the Python vs
  `AQUINAS_Explorer.R` adaptation notes, see
  [aquinas_toolkit/preprocessing/README.md](../aquinas_toolkit/preprocessing/README.md).

## 02_preprocessing Notes

`02_preprocessing.ipynb` is a consumer of the preprocessing API, not a second
implementation of preprocessing logic. It walks through the single pipeline in
order: **signal filtering → zeroing → alignment**.

- **Event discovery:** `find_events()` groups records by deck and exact
  `Start_Time`/`End_Time` window. `timestamp=` uses strict containment
  (`Start_Time < timestamp < End_Time`); boundary values return nothing.
- **Duration filtering:** `summarize_min_duration_filter()` runs across all
  five SETs to report keep/drop counts. `find_common_sensor_events()` then
  restricts to events present in every selected sensor.
- **Signal filtering:** `filter_loaded_event_group()` applies a 0.5–20 Hz
  zero-phase Butterworth band-pass filter before any baseline or timing
  correction. A raw-vs-filtered overlay is shown for up to three ACC_Z channels.
- **Zeroing:** `zero_loaded_event_group()` subtracts the linear baseline
  (endpoint-to-endpoint straight line) from each filtered sensor slice.
  Before/after plots are shown in separate cells so the team can compare them
  without stacked axes. Both a SET2 strain example and a SET1 ACC_Z example
  are included.
- **Alignment:** `align_event_group()` implements the organizer `Synchro()`
  workflow — first sensor as reference, two shrinking passes, no interpolation.
  Alignment diagnostics are displayed.
- **Export smoke-test:** `export_aligned_event()` writes one event to CSV and
  the result is read back to confirm the artifact.

Organizer-specific timestamp notes:

- The SET2 strain example uses a real event starting at
  `2023-04-20 07:04:05`; the notebook queries with timestamp `07:04:10`
  (strictly inside the window).
- The organizer's acceleration screenshot shows `2022-09-01 17:51:55`,
  which is not present in the released dataset. The notebook uses the real
  SET1 upstream `ACC_Z` event at `2022-07-30 18:36:53` as a documented
  substitute. `sensor_pattern=` uses wildcard matching when the pattern
  contains `*`, `?`, or `[]`; otherwise it is a case-insensitive substring
  filter.

## 03_feature_extraction Notes

`03_feature_extraction.ipynb` preserves Mohsen's FDD-focused feature
workflow. Signal conditioning (band-pass filtering, zeroing, alignment) is
handled entirely by the preprocessing stage described in notebook 02; this
notebook takes the conditioned waveforms and derives modal features.

- Runs `run_acc_z_fdd_workflow()` from `aquinas_toolkit.feature_extraction`
  for each SET/deck combination.
- Extracts dominant FDD peaks from the first singular-value curve inside the
  0.5–20 Hz band.
- Displays signed and absolute mode-shape summaries annotated by structural
  location (deck side, span, position).
- Plots first-singular-value spectra with picked-peak overlays.
- Reusable logic (FDD computation, peak picking, mode-shape annotation) lives
  in `aquinas_toolkit/feature_extraction/`; the notebook is thin wrappers
  and display only.
