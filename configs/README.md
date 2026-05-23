# configs/

Pipeline configuration files in YAML format.

## How it works

- The v1 CLI reads `configs/default.yaml` as the active workspace config.
- `aquinas run` and `aquinas run preprocess` snapshot that file into
  `results/<run_id>/config.yaml`.
- Later stages always read the selected run's `config.yaml`, not the
  current workspace copy. This keeps resumes deterministic even if
  `configs/default.yaml` changes afterward.

## Files

| File | Purpose |
| --- | --- |
| `default.yaml` | Active working configuration used when creating new runs |

## Working with variants

If you want to keep multiple config variants, store them next to
`default.yaml` and copy the desired variant over `configs/default.yaml`
before starting a new run.

## Active vs placeholder config

- `data`, `preprocessing`, `features`, `training`, and `output` are active
  in v1 and are read by the current code.
- Full NN model architecture settings and `scoring` remain future-stage
  concerns; the current `training` section controls data splits and
  standardization only.
- Some active-looking preprocessing keys are still mostly declarative
  records of the fixed v1 contract rather than fully generic hooks.
  These are called out below.

## Field-by-field glossary

### `data`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
| --- | --- | --- | --- | --- | --- |
| `data.dataset_root` | `AQUINAS_DATASET` | Path to the dataset root. Absolute paths are used as-is. Relative paths are resolved against the active workspace root when a local `configs/default.yaml` is present in the current directory tree; otherwise the installed repository root is used so the `aquinas` command still behaves sensibly when launched from elsewhere. A stub root that only contains placeholder files such as `README.md` or `.gitkeep` is treated as an empty bootstrap destination by `aquinas data fetch`; `aquinas data status` and `aquinas data verify` report that state explicitly. | Yes | Which dataset folder `run_preprocessing()` opens for each SET, and where `aquinas data fetch` installs the archive | `results/<run_id>/config.yaml`, CLI run summary, and `aquinas info` |
| `data.sets` | All five AQUINAS SET folders | Ordered list of monthly datasets to process. v1 expects valid AQUINAS set folder names such as `AQUINAS_SET1_2022_07`. Dataset availability is defined by the presence of every configured set directory under `data.dataset_root`. | Yes | Which SETs are looped over, and in what order, during preprocessing, and whether the CLI considers the dataset complete enough to run | `results/<run_id>/config.yaml`, `sensor_records.csv`, `event_manifest.csv`, `summary.json`, and CLI bootstrap messages |

The public archive URL and SHA256 used by `aquinas data fetch` are
static code constants in `aquinas_toolkit/dataset_source.py`. They are
intentionally not exposed as user-tunable config keys.

When the archive host exposes a total byte size, `aquinas data fetch`
uses it to show Rich download progress with bytes transferred, speed,
and ETA. When the host omits that header, the CLI still shows live
bytes transferred and speed without a reliable ETA.

### `preprocessing`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
| --- | --- | --- | --- | --- | --- |
| `preprocessing.sampling_rate_hz` | `100.0` | Sampling rate used for filter design, timestamp-to-sample reasoning, and FFT bin calculation. | Yes | ACC_Z filtering and neural-input frequency bins | `summary.json` and `report/neural_input_summary.json` |
| `preprocessing.sensor_selection.decks` | `[OLD]` | Selected deck subset for preprocessing and neural-input packaging. | Yes | Sensor inclusion before event processing and neural-input event filtering | `sensor_records.csv`, `summary.json`, and `report/neural_input_summary.json` |
| `preprocessing.strain.locations` | `[INF, SUP]` | Strain locations included in the neural-input sensor set. `SHE_STR` is excluded as the inclined strain family for the current NN contract. | Yes | Sensor inclusion and strain channel ordering | `sensor_records.csv`, `report/sensor_map.csv`, `nn_inputs/metadata/sensor_ids.json`, and `report/sensor_ids.json` |
| `preprocessing.strain.filter.method` | `none` | Strain filtering policy. Current runtime support requires `none`. | Yes | Strain preprocessing before alignment | `summary.json` and notebook plots |
| `preprocessing.strain.zeroing.method` | `linear_endpoints` | Strain baseline-removal method before alignment. Supported runtime values are `none` and `linear_endpoints`. | Yes | Strain preprocessing before alignment | `event_manifest.csv` and `summary.json` |
| `preprocessing.strain.peak_window_half_samples` | `100` | Half-width of the fixed-length strain window used in neural-input packaging. Total window length is `2 * peak_window_half_samples`. | Yes | Neural-input strain tensor width | `nn_inputs/metadata/input_shapes.json` and `report/neural_input_summary.json` |
| `preprocessing.acc.axis` | `Z` | Acceleration axis included in preprocessing and neural-input packaging. The current NN contract uses `Z`, so `ACC_Y` remains excluded by default. | Yes | Acceleration sensor inclusion and ACC channel ordering | `sensor_records.csv`, `summary.json`, and `nn_inputs/metadata/sensor_ids.json` |
| `preprocessing.acc.min_aligned_samples` | `500` | Minimum aligned ACC_Z length required for neural-input packaging. | Yes | Packaging rejection of too-short ACC_Z events | `report/neural_input_summary.json` |
| `preprocessing.acc.filter.method` | `butterworth_bandpass` | ACC_Z filtering policy. Supported runtime values are `none` and `butterworth_bandpass`. | Yes | ACC_Z preprocessing before alignment | `summary.json` and notebook plots |
| `preprocessing.acc.filter.low_hz` / `high_hz` / `order` | `0.5 / 20.0 / 4` | ACC_Z Butterworth band-pass parameters. | Yes | ACC_Z time-domain filtering | `summary.json` |
| `preprocessing.acc.zeroing.method` | `linear_endpoints` | ACC_Z baseline-removal method after filtering and before alignment. Supported runtime values are `none` and `linear_endpoints`. | Yes | ACC_Z preprocessing before alignment | `event_manifest.csv` and `summary.json` |
| `preprocessing.acc.frequency_transform.low_hz` / `high_hz` | `0.5 / 20.0` | Frequency range kept from the ACC_Z FFT magnitude block used in neural inputs. | Yes | Neural-input ACC_Z frequency-bin width | `nn_inputs/metadata/input_shapes.json`, `nn_inputs/metadata/frequency_bins.npy`, and `report/frequency_bins.npy` |
| `preprocessing.event_grouping.method` | `shared_start` | Event grouping policy. `shared_start` groups by deck and `Start_Time`, then records the event end as the maximum grouped `End_Time`; `exact_window` preserves the legacy `Start_Time + End_Time` grouping for comparison. | Yes | Which sensor records reach one alignment step and therefore NN event coverage | `summary.json`, `preprocess.sqlite`, and [preprocessing/README](../aquinas_toolkit/preprocessing/README.md) |
| `preprocessing.event_grouping.key_fields` | `[deck, Start_Time]` | Human-readable trace of the active grouping key. The runtime behavior is controlled by `preprocessing.event_grouping.method`. | Metadata | Summary metadata and team-facing traceability | `summary.json` |
| `preprocessing.event_grouping.group_end_policy` | `max_end` | Human-readable trace of the event-level end-time policy for `shared_start`. | Metadata | Audit trail for grouped event windows | `summary.json` |
| `preprocessing.sensor_overrides.exclude[].sensor_name` | `OLD_S1_UP_SUP_STR` | Exact sensor name to exclude for specified SETs. | Yes | Inclusion/exclusion status before grouping, reference selection, and export | `sensor_records.csv`, `event_manifest.csv`, `summary.json` |
| `preprocessing.sensor_overrides.exclude[].sets` | `AQUINAS_SET4_2024_01`, `AQUINAS_SET5_2024_06` | SET names where the exclusion applies. | Yes | Set-specific damaged-sensor policy | `sensor_records.csv`, `summary.json` |
| `preprocessing.sensor_overrides.exclude[].reason` | `damaged sensor per organizer email` | Human-readable reason stored with excluded records. | Yes | Audit trail for why a sensor was excluded | `sensor_records.csv`, `event_manifest.csv`, `summary.json` |
| `preprocessing.sensor_overrides.exclude[].source` | `François-Baptiste Cartiaux email dated April 9, 2026` | Source note stored with the exclusion. | Yes | Provenance and audit trail | `sensor_records.csv`, `summary.json` |
| `preprocessing.alignment.method` | `r_synchro` | Alignment algorithm. `r_synchro` is the only supported runtime value and mirrors the organizer helper's first-selected-reference `Synchro()` workflow. The key is kept intentionally so additional organizer-compatible methods can be added later without changing the config shape. | Yes | Cross-sensor sample matching and retained aligned rows | `summary.json`, `event_manifest.csv`, and alignment diagnostics in aligned events |
| `preprocessing.zeroing.method` | `linear_endpoints` | Legacy fallback zeroing key. It is still read as a default for signal-specific zeroing when `preprocessing.strain.zeroing.method` or `preprocessing.acc.zeroing.method` are omitted. | Compatibility only | Fallback config resolution | `results/<run_id>/config.yaml` and `summary.json` |
| `preprocessing.filtering.min_active_sensors_per_event` | `1` | Minimum number of included sensor records required before alignment begins. This is a pre-alignment filter only. | Yes | Early discard of events with too few included sensors | `event_manifest.csv` and `summary.json` |
| `preprocessing.storage.backend` | `sqlite` | Canonical preprocess storage backend. v1 currently supports only `sqlite`. | Yes | Canonical preprocess store layout | `summary.json` and `preprocess.sqlite` |
| `preprocessing.exports.aligned_waveforms.enabled` | `false` | Whether optional aligned CSV or CSV.GZ exports are written in addition to the canonical SQLite store and waveform artifacts. | Yes | Presence of optional aligned exports | `summary.json` and `results/<run_id>/stages/preprocess/exports/aligned/` |
| `preprocessing.exports.aligned_waveforms.format` | `csv.gz` | Output format for optional aligned waveform exports. Supported values are `csv` and `csv.gz`. | Yes | Filename suffix and compression of optional aligned exports | `summary.json` and `results/<run_id>/stages/preprocess/exports/aligned/` |

Legacy preprocessing alignment keys from the previous nearest-timestamp
contract are no longer accepted. Config loading now fails if
`reference_sensor`, `tolerance_ms`, or `drop_unmatched_rows` appear
under `preprocessing.alignment`.

Legacy top-level preprocessing filter and export keys are also no longer the
authoritative interface. Signal-specific settings under `preprocessing.strain`
and `preprocessing.acc`, plus storage and export settings under
`preprocessing.storage` and `preprocessing.exports.aligned_waveforms`, define
the current v1 contract.

### `training`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
| --- | --- | --- | --- | --- | --- |
| `training.random_seed` | `2026` | Seed for deterministic train/validation/test index generation. | Yes | Reproducible event split order | `stages/train/splits/split_manifest.json` |
| `training.split.train` / `validation` / `test` | `0.70 / 0.20 / 0.10` | Fractions for event-level split indices. Values must be non-negative and sum to `1.0`. | Yes | Sizes of `train_indices.npy`, `val_indices.npy`, and `test_indices.npy` | `stages/train/splits/split_manifest.json` |
| `training.standardization.enabled` | `true` | Whether to fit train-only normalization statistics for strain, ACC, and temperature inputs. | Yes | Presence of `stages/train/normalization_stats.npz` | `stages/train/splits/split_manifest.json` |

The training stage currently prepares data only. It does not yet train the
attention or latent-space NN architectures.

### `output`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
| --- | --- | --- | --- | --- | --- |
| `output.results_dir` | `results` | Root directory where run folders and `latest.json` are written. Relative paths follow the same workspace-root resolution as `data.dataset_root`, preferring a local workspace config and otherwise falling back to the installed repository root. | Yes | Run creation, run lookup, latest pointer, and stage output placement | CLI run summary, `results/latest.json`, and the on-disk run layout |

## How to verify a run used a setting

- `results/<run_id>/config.yaml` is the authoritative snapshot of the
  settings that run used.
- `results/<run_id>/metadata.json` confirms stage status and run
  identity, but not every setting's effect.
- `results/<run_id>/stages/preprocess/summary.json` records the
  alignment, zeroing, grouping, exclusion, and export settings that the
  preprocess stage ran with.
- `results/<run_id>/stages/preprocess/event_manifest.csv` shows
  per-event effects such as reference sensor choice, discard reasons,
  excluded sensors, and zeroing method.
- `results/<run_id>/stages/preprocess/sensor_records.csv`,
  `event_manifest.csv`, and `summary.json` show how configured sensor
  exclusions were applied in that run.

## Cross-links

- See [preprocessing/README](../aquinas_toolkit/preprocessing/README.md)
  for the exact preprocessing semantics, API behavior, and the Python
  vs `AQUINAS_Explorer.R` adaptation notes.
- See [docs/README](../docs/README.md) for organizer Q&A and email
  provenance behind the current preprocessing defaults.
