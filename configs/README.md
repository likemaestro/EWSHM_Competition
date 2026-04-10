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
|---|---|
| `default.yaml` | Active working configuration used when creating new runs |

## Working with variants

If you want to keep multiple config variants, store them next to
`default.yaml` and copy the desired variant over `configs/default.yaml`
before starting a new run.

## Active vs placeholder config

- `data`, `preprocessing`, and `output` are active in v1 and are read by
  the current code.
- The commented `features`, `model`, and `scoring` sections in
  `default.yaml` are placeholders for future stages only. They are not
  consumed by the current pipeline.
- Some active-looking preprocessing keys are still mostly declarative
  records of the fixed v1 contract rather than fully generic hooks.
  These are called out below.

## Field-by-field glossary

### `data`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
|---|---|---|---|---|---|
| `data.dataset_root` | `AQUINAS_DATASET` | Path to the dataset root. Absolute paths are used as-is. Relative paths are resolved against the current workspace root; in normal repo-root usage this makes the path repo-relative. | Yes | Which dataset folder `run_preprocessing()` opens for each SET | `results/<run_id>/config.yaml`, CLI run summary, and `aquinas info` |
| `data.sets` | All five AQUINAS SET folders | Ordered list of monthly datasets to process. v1 expects valid AQUINAS set folder names such as `AQUINAS_SET1_2022_07`. | Yes | Which SETs are looped over, and in what order, during preprocessing | `results/<run_id>/config.yaml`, `sensor_records.csv`, `event_manifest.csv`, `summary.json` |

### `preprocessing`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
|---|---|---|---|---|---|
| `preprocessing.event_grouping.key_fields` | `[deck, Start_Time, End_Time]` | Records the fixed v1 event-grouping policy. v1 grouping is deck-aware and uses exact event windows. This key is not yet a generic regrouping hook. | Recorded, not used to change grouping behavior | Summary metadata and team-facing traceability | `summary.json` and [preprocessing/README](../aquinas_toolkit/preprocessing/README.md) |
| `preprocessing.sensor_overrides.exclude[].sensor_name` | `OLD_S1_UP_SUP_STR` | Exact sensor name to exclude for specified SETs. | Yes | Inclusion/exclusion status before grouping, reference selection, and export | `sensor_records.csv`, `event_manifest.csv`, `summary.json`, `sensor_qc_report.csv` |
| `preprocessing.sensor_overrides.exclude[].sets` | `AQUINAS_SET4_2024_01`, `AQUINAS_SET5_2024_06` | SET names where the exclusion applies. | Yes | Set-specific damaged-sensor policy | `sensor_records.csv`, `summary.json`, `sensor_qc_report.csv` |
| `preprocessing.sensor_overrides.exclude[].reason` | `damaged sensor per organizer email` | Human-readable reason stored with excluded records. | Yes | Audit trail for why a sensor was excluded | `sensor_records.csv`, `event_manifest.csv`, `summary.json` |
| `preprocessing.sensor_overrides.exclude[].source` | `François-Baptiste Cartiaux email dated April 9, 2026` | Source note stored with the exclusion. | Yes | Provenance and audit trail | `sensor_records.csv`, `sensor_qc_report.csv`, `summary.json` |
| `preprocessing.alignment.method` | `r_synchro` | Alignment algorithm. `r_synchro` is the only supported runtime value and mirrors the organizer helper's first-selected-reference `Synchro()` workflow. The key is kept intentionally so additional organizer-compatible methods can be added later without changing the config shape. | Yes | Cross-sensor sample matching and retained aligned rows | `summary.json`, `event_manifest.csv`, and alignment diagnostics in aligned events |
| `preprocessing.zeroing.method` | `linear_endpoints` | Baseline-removal method applied before alignment. Supported runtime values are `none` and `linear_endpoints`. | Yes | Per-sensor raw-slice baseline removal before organizer synchronization | `event_manifest.csv`, `summary.json`, notebook plots |
| `preprocessing.filtering.min_active_sensors_per_event` | `1` | Minimum number of included sensor records required before alignment begins. This is a pre-alignment filter only. | Yes | Early discard of events with too few included sensors | `event_manifest.csv` and `summary.json` |
| `preprocessing.export.format` | `csv.gz` | Output file format for aligned stage exports. v1 supports `csv.gz` and `csv`. | Yes | Filename suffix and compression of aligned partition outputs | `results/<run_id>/stages/preprocess/aligned/` and `summary.json` |
| `preprocessing.export.partition_by` | `[set_name, deck]` | Records the fixed v1 aligned-export partitioning shape. v1 writes one aligned file per `(set_name, deck)` partition and does not yet support arbitrary partition schemes. | Recorded, not used to change partition logic | Summary metadata and team-facing traceability | `summary.json` and aligned filenames |

Legacy preprocessing alignment keys from the previous nearest-timestamp
contract are no longer accepted. Config loading now fails if
`reference_sensor`, `tolerance_ms`, or `drop_unmatched_rows` appear
under `preprocessing.alignment`.

### `output`

| Key | Default | v1 meaning and supported values | Consumed now | Affects | Verify |
|---|---|---|---|---|---|
| `output.results_dir` | `results` | Root directory where run folders and `latest.json` are written. Relative paths are resolved against the current workspace root. | Yes | Run creation, run lookup, latest pointer, and stage output placement | CLI run summary, `results/latest.json`, and the on-disk run layout |

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
- `results/<run_id>/stages/preprocess/sensor_qc_report.csv` shows the
  QC evidence supporting configured sensor exclusions.

## Cross-links

- See [preprocessing/README](../aquinas_toolkit/preprocessing/README.md)
  for the exact preprocessing semantics, API behavior, and the Python
  vs `AQUINAS_Explorer.R` adaptation notes.
- See [docs/README](../docs/README.md) for organizer Q&A and email
  provenance behind the current preprocessing defaults.
