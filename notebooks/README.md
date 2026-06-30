# notebooks/

Jupyter notebooks for exploration, visualisation, and presentation.

## Folder layout

- Top-level numbered notebooks (`01_`, `02_`, ...) are the main project
  storyline from raw data inspection through scoring.
- `misc/` holds supporting analyses, diagnostics, and experiment notebooks
  that feed the main numbered sequence.
- `azrmirz_fncs/` holds the notebook-backed neural-network preparation,
  training, evaluation, cross-SET inference, and outlier-inspection scripts
  used by notebooks 04 and 05.

## Naming convention

Notebooks are numbered sequentially so they tell a story from
raw data exploration through to the final health score:

| Notebook | Purpose | Status |
| --- | --- | --- |
| `01_sensor_overview` | Dataset structure, raw waveform plots | Done |
| `02_preprocessing` | Filtering, zeroing, alignment | Done |
| `03_feature_extraction` | FDD, peak picking, and mode shapes | Done |
| `04_anomaly_detection` | Unsupervised reconstruction-error anomaly detection | Done |
| `05_health_scoring` | Notebook-backed structural health score narrative | Done |

Supporting notebooks in `misc/` use alphabetical prefixes so they are clearly
distinguished from the numbered main storyline. Continue the series as
`A_`, `B_`, `C_`, and so on:

| Notebook | Purpose |
| --- | --- |
| `misc/A_temperature_correlations` | Exploratory temperature-correlation analysis |
| `misc/B_preprocessed_temperature_correlations` | Temperature checks after preprocessing |
| `misc/C_strain_processing` | Strain-channel processing experiments |
| `misc/D_statistical_trend_analysis` | Statistical trend analysis across sets |
| `misc/E_preprocessing_neural_inputs` | NN input tensor inspection |
| `misc/F_checking_preprocessed_26052026` | Preprocessed output diagnostics |
| `misc/G_testing_attention_model` | Attention-model experimentation |
| `misc/G_testing_attention_model_OLD` | Archived attention-model experiment kept for traceability |
| `misc/H_testing_models` | Toolkit model-helper experiments |
| `misc/I_delete_afterwards` | Scratch notebook retained outside the main storyline |

Notebook-backed neural-network scripts live in `azrmirz_fncs/`:

- `architecture_1_v3.py`
- `prepare_per_channel_samples_new_deck.py`
- `train_architecture_1_v3.py`
- `evaluate_architecture_1_v3.py`
- `inference_cross_set_v3.py`
- `inspect_outlier_event.py`

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

- **Event discovery:** exploratory `find_events()` calls can use exact
  windows, while the batch preprocess run now uses the configured
  `shared_start` policy by default: deck + `Start_Time`, with event end set
  to the maximum grouped `End_Time`. `timestamp=` uses strict containment
  (`Start_Time < timestamp < End_Time`); boundary values return nothing.
- **Duration filtering:** `summarize_min_duration_filter()` runs across all
  five SETs to report keep/drop counts. `find_common_sensor_events()` then
  restricts to events present in every selected sensor.
- **Signal filtering:** `filter_loaded_event_group()` applies signal-specific
  filtering before any baseline or timing correction. In the current v1
  workflow, strain uses `none` while ACC_Z uses a 0.5–20 Hz zero-phase
  Butterworth band-pass filter. A raw-vs-filtered overlay is shown for up to
  three ACC_Z channels.
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
- Also includes a run-backed section that reads snapped preprocess outputs via
  `open_preprocess_store()` and re-runs ACC_Z FDD from `preprocess.sqlite`
  or the temporary legacy CSV / CSV.GZ compatibility path.
- Extracts dominant FDD peaks from the first singular-value curve inside the
  0.5–20 Hz band.
- Displays signed and absolute mode-shape summaries annotated by structural
  location (deck side, span, position).
- Plots first-singular-value spectra with picked-peak overlays.
- Reusable logic (FDD computation, peak picking, mode-shape annotation) lives
  in `aquinas_toolkit/feature_extraction/`; the notebook is thin wrappers
  and display only.

## 04_anomaly_detection Notes

`04_anomaly_detection.ipynb` is the presentation index for the unsupervised
neural reconstruction-error workflow. It points to the attention-autoencoder
experiments in `misc/G_testing_attention_model.ipynb`,
`misc/H_testing_models.ipynb`, `aquinas_toolkit/models/`, and the v3 training
script in `azrmirz_fncs/`.

The notebook does not duplicate the model-training code. It explains the
current method: train an unsupervised reconstruction model on baseline
behavior, compute reconstruction errors, and use those errors as anomaly
evidence.

## 05_health_scoring Notes

`05_health_scoring.ipynb` is the presentation index for the current synthetic
health-score interpretation. It points to the SET1-trained baseline,
fixed-normalization cross-SET inference, percentile thresholding, and
per-event reconstruction-error summaries produced by the v3 evaluation and
inference scripts in `azrmirz_fncs/`.

The notebook is the final story layer: it ties reconstruction-error
distributions, threshold exceedance, temperature context, and monthly
cross-SET comparisons into the structural health-score narrative.
