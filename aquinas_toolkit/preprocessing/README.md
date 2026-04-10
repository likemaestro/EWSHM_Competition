# preprocessing/

## Purpose

Prepare raw 100 Hz waveforms for feature extraction. This stage sits
between the reader (raw data in) and the feature extractor (compact
vectors out).

## Status

Implemented for v1.

Pipeline order: **signal filtering → zeroing → alignment**

1. Zero-phase Butterworth band-pass filter applied to each raw sensor
   waveform (default 0.5–20 Hz, order 4).
2. Per-sensor linear-endpoint baseline removal (zeroing).
3. Organizer `Synchro()` timestamp alignment across sensors.

## Interface

- **Input:** raw waveform DataFrames from `AquinasReader`, grouped by
  exact event windows within each deck
- **Output:** filtered, zeroed, aligned per-event waveform tables plus
  manifest and diagnostics artifacts under
  `results/<run_id>/stages/preprocess/`

## Public API

```python
from aquinas_toolkit.preprocessing import (
    # Pipeline helpers — use these in the correct order
    filter_loaded_event_group,    # step 1: signal filtering
    zero_loaded_event_group,      # step 2: baseline removal
    align_event_group,            # step 3: timestamp alignment
    # Event discovery and loading
    find_events,
    load_event_group,
    load_timestamp_query_frames,
    # Batch pipeline and export
    run_preprocessing,
    export_aligned_event,
    # Organizer parity
    run_organizer_query,
    synchro_indices,
    # Signal helpers (also used by feature extraction)
    bandpass_filter_waveform_matrix,
    filter_records_by_min_duration,
    find_common_sensor_events,
    load_common_event_waveform_matrix,
    summarize_min_duration_filter,
    # Low-level
    zero_waveform,
    SIGNAL_FILTER_METHODS,
    # Dataclasses
    AlignedEvent,
    LoadedEventGroup,
    OrganizerQueryResult,
)
```

Key symbols:

| Symbol | Kind | Purpose |
|---|---|---|
| `filter_loaded_event_group()` | function | Apply zero-phase band-pass filter to each raw sensor slice (first conditioning step) |
| `zero_loaded_event_group()` | function | Apply baseline removal to each filtered sensor slice before alignment |
| `zero_waveform()` | function | Apply a zeroing method to a single waveform array |
| `align_event_group()` | function | Two-pass `Synchro()` alignment, first-selected reference, no interpolation |
| `find_events()` | function | Group records by `set + deck + Start_Time + End_Time` with optional timestamp/sensor filters |
| `load_event_group()` | function | Load all raw waveforms that belong to one grouped event |
| `load_timestamp_query_frames()` | function | Reproduce organizer-style timestamp selection for one deck/sensor subset |
| `run_organizer_query()` | function | Return organizer-style aligned `DataMesures` output for one timestamp query |
| `synchro_indices()` | function | Low-level helper: compute the shared row indices from one synchronization pass |
| `export_aligned_event()` | function | Export one aligned event as a CSV or CSV.GZ artifact |
| `run_preprocessing()` | function | Execute the full preprocess stage for a snapped pipeline run |
| `bandpass_filter_waveform_matrix()` | function | Apply a zero-phase Butterworth band-pass filter to a multichannel wide-format matrix |
| `filter_records_by_min_duration()` | function | Filter index-table rows by a minimum `Duration` threshold |
| `summarize_min_duration_filter()` | function | Report keep/remove counts after minimum-duration filtering |
| `find_common_sensor_events()` | function | Find events present in every selected sensor after duration filtering |
| `load_common_event_waveform_matrix()` | function | Load one event as a wide multichannel waveform matrix |
| `AlignedEvent` | dataclass | Output of `align_event_group()` |
| `LoadedEventGroup` | dataclass | Output of `load_event_group()` |
| `OrganizerQueryResult` | dataclass | Output of `run_organizer_query()` |

## Event Selection Semantics

- `find_events()` groups records by exact
  `set + deck + Start_Time + End_Time` before applying filters.
- `deck="OLD"` or `deck="NEW"` is an exact filter on the derived deck
  token. It is not partial matching on the full sensor name.
- `timestamp=` in `find_events()` is a strict containment query:
  an event is returned only when `Start_Time < timestamp < End_Time`.
  It is not a nearest-event lookup or an inclusive boundary check.
- `sensor_pattern=` filters on sensor names after deck filtering.
  If the pattern contains `*`, `?`, or `[]`, it uses shell-style
  wildcard matching. Otherwise it behaves like a case-insensitive
  substring filter.
- `sensor_pattern="STR"` means all strain sensors.
- `sensor_pattern="ACC_Z"` means all Z-acceleration sensors.
- `sensor_pattern="*UP*ACC_Z*"` means upstream Z-acceleration sensors
  such as `OLD_S1_UP_INT_ACC_Z` and `OLD_S2_UP_MID_ACC_Z`.
- `load_event_group()` then loads every raw waveform slice that belongs
  to the grouped event after those filters have been applied.

## Signal Filtering, Zeroing, and Alignment Semantics

### Pipeline order

The preprocessing pipeline conditions each raw event in three sequential
steps, in this order:

1. **Signal filtering** — `filter_loaded_event_group()` applies a
   zero-phase Butterworth band-pass filter (`butterworth_bandpass`,
   default 0.5–20 Hz, order 4) independently to each raw sensor
   waveform. Filtering on the raw signal ensures no baseline or
   alignment artefacts contaminate the passband.
2. **Zeroing** — `zero_loaded_event_group()` applies linear-endpoint
   baseline removal to each filtered sensor slice. Zeroing after
   filtering keeps the baseline estimate within the filtered band.
3. **Alignment** — `align_event_group()` / `organizer_align_sensor_frames()`
   implement organizer `Synchro()` behavior: the first selected sensor
   becomes the reference seed, alignment runs for exactly two shrinking
   passes, and no interpolation is used.

### Additional semantics

- `load_timestamp_query_frames()` mirrors the organizer helper's
  deck-plus-sensor selection order and widens duplicate sensor matches
  with `min(Start_Row):max(End_Row)` before waveform loading.
- Config-driven exclusions are applied before waveform loading in the
  batch preprocess stage, so an excluded sensor never enters the
  signal-conditioning or synchronization loop for that stage run.
- `linear_endpoints` subtracts, for each sensor slice independently, the
  straight baseline line connecting that slice's earliest and latest
  retained sample.
- Supported zeroing methods: `none` and `linear_endpoints`.
- Supported signal filter methods: `none` and `butterworth_bandpass`.
- `min_active_sensors_per_event=1` is the minimum inclusion filter
  before waveform loading. An event can still be discarded later if
  organizer-style synchronization leaves zero common rows.

## Python vs `AQUINAS_Explorer.R`

- The current Python implementation is a clean native conversion of the
  organizer helper's core behavior, not a nearest-timestamp adaptation.
- Matching points that were explicitly corrected during the cutover:
  strict timestamp containment, organizer sensor ordering, raw-slice
  widening, zero-before-alignment, exact `Synchro()` semantics, and the
  two-pass shrinking loop.
- Parity was validated against the real local R runtime across five
  representative probes: `SET1 OLD STR 2022-07-02 03:34:40`,
  `SET1 OLD ACC_Z 2022-07-30 18:36:52`,
  `SET1 OLD ACC_Z 2022-07-30 18:36:53`,
  `SET1 NEW STR 2022-07-01 02:39:08.500`,
  and `SET4 OLD INF_STR 2024-01-01 13:34:04`.

## Design Decisions From Organizer Q&A

| Source / date | Organizer point | Adopted implementation | Affected default or artifact | Status |
|---|---|---|---|---|
| Email, April 2, 2026 | Recording is triggered per deck with a 5-second pre-trigger buffer and a quiet-tail stop rule | Keep preprocessing deck-specific and preserve full raw record duration | Event grouping and waveform loading | Implemented now |
| Meeting Q&A + `AQUINAS_Explorer.R`, April 9, 2026 | Logger polling causes slight sensor time shifts | Use the organizer `Synchro()` workflow with first-selected reference and two shrinking passes | `alignment.method = r_synchro`, alignment diagnostics | Implemented now |
| Meeting Q&A, April 9, 2026 | Synchronize without interpolation | Keep organizer alignment discrete and non-interpolating | `align_event_group()` and aligned exports | Implemented now |
| Meeting Q&A + `AQUINAS_Explorer.R`, April 9, 2026 | Zeroing is flexible; endpoint-line subtraction is the shared helper behavior | Make organizer endpoint subtraction the runtime default, applied after band-pass filtering | `zeroing.method = linear_endpoints` | Implemented now |
| Meeting Q&A, April 9, 2026 | Missing or incomplete records can be discarded if justified | Keep discard reasons explicit in stage artifacts | `event_manifest.csv`, `summary.json` | Implemented now |
| Organizer email, April 9, 2026 | One sensor was damaged between SET3 and SET4 and should be discarded for SET4 and SET5 only | Add a config-driven exclusion policy rather than hardcoding it in the algorithm | `preprocessing.sensor_overrides.exclude` | Implemented now |
| Local dataset validation, April 9, 2026 guidance | `OLD_S1_UP_SUP_STR` matches the warning: TABLE `Range` becomes `0.0` throughout SET4 and SET5 while raw slices still vary and the baseline shifts sharply | Emit a report-only QC artifact that validates the exclusion and keeps the decision auditable | `sensor_qc_report.csv` | Implemented now |
| Meeting Q&A, April 9, 2026 | Temperature is not hardware-compensated | Preserve temperature metadata but defer active compensation | `sensor_records.csv` | Deferred but acknowledged |
| Follow-up Q&A, source date pending | Expected frequencies are around 2–10 Hz | Use 0.5–20 Hz band-pass as the default passband in both the preprocessing pipeline and the FDD analysis notebook | `signal_filter.low_hz`, `signal_filter.high_hz` | Implemented now |

<!-- TODO: consider writing aligned waveforms to a SQLite database instead of
     CSV/CSV.GZ. Subsequent stages (feature extraction, training, scoring)
     query by event, sensor, set, and deck. Indexed SQL lookups would be
     faster than scanning compressed CSV files on each stage run. -->

## Performance

For each event, `load_event_group()` reads one waveform slice per active
sensor. Each sensor's raw data is stored in numbered sequential batch files
(roughly 8.8 MB each, covering between 2 and 226 events). Without caching,
the same file would be read and re-parsed once per event that references it.

`AquinasReader.load_raw_file()` caches each parsed file for the lifetime of
the reader instance (one per SET). This reduces JSON reads from O(records) to
O(raw files per sensor), which is 29 to 226 times fewer reads depending on
the set.

See `aquinas_toolkit/io/README.md` for the measured numbers and caching rules.

## Implemented Now

- Band-pass filtering before zeroing and alignment (zero-phase Butterworth,
  configurable passband, default 0.5–20 Hz)
- Exact event grouping with `set + deck + Start_Time + End_Time`
- Organizer-style strict timestamp containment for timestamp queries
- Organizer `Synchro()` alignment without interpolation
- Zeroing after filtering with organizer `linear_endpoints` as the default
- Exclusion-aware manifests, sensor-record statuses, and summary counts
- QC reporting for the damaged-sensor override
- Manifest, sensor-record, aligned-export, and summary artifacts
- Local Python-vs-R parity validation against the original helper
- Notebook examples that exercise the package API instead of embedding logic

## Which config fields are true knobs in v1

- `signal_filter.method` is active; `butterworth_bandpass` and `none`
  are the supported values. Set to `none` to skip filtering.
- `signal_filter.low_hz`, `signal_filter.high_hz`, and
  `signal_filter.order` are active and control the Butterworth passband.
- `alignment.method` is active; `r_synchro` is the only supported value.
  Retained as an extension point for future organizer-compatible methods.
- Legacy alignment keys (`reference_sensor`, `tolerance_ms`,
  `drop_unmatched_rows`) are rejected when loading preprocessing settings.
- `zeroing.method` (`none` or `linear_endpoints`) and
  `min_active_sensors_per_event` are active runtime settings in v1.
- `event_grouping.key_fields` records the fixed v1 grouping contract.
- `export.partition_by` records the fixed v1 aligned-export partitioning.
- `export.format` is active; supports `csv.gz` and `csv`.

## Deferred But Acknowledged

- Temperature normalization
- Cross-correlation lag correction
- OMA-specific concatenation helpers

## Damaged Sensor Override

The organizer warned on April 9, 2026 that one sensor became damaged
between SET3 and SET4 but still emits erroneous data in SET4 and SET5.
The current repository implements that advice as a config-driven
override:

- `OLD_S1_UP_SUP_STR` is excluded for `AQUINAS_SET4_2024_01`
  and `AQUINAS_SET5_2024_06`
- it remains available for `AQUINAS_SET1_2022_07`,
  `AQUINAS_SET2_2023_04`, and `AQUINAS_SET3_2023_08`
- excluded rows remain visible in `sensor_records.csv`
  with `sensor_status = excluded`
- per-event exclusions are surfaced in `event_manifest.csv`
- `sensor_qc_report.csv` records the supporting evidence

What this means in code:

- the exclusion is declared in `configs/default.yaml` under
  `preprocessing.sensor_overrides.exclude`
- preprocessing applies the exclusion before event grouping, reference
  selection, waveform alignment, and aligned export writing
- the damaged sensor cannot be chosen as the reference sensor for
  SET4/SET5
- aligned preprocess artifacts for SET4/SET5 do not include the damaged
  channel as an active sensor
- `summary.json` records exclusion counts and reasons so the policy
  stays auditable inside a run

Local evidence used to validate the override:

- TABLE `Range` for `OLD_S1_UP_SUP_STR` is normal in SET1-SET3
  and collapses to `0.0` across SET4 and SET5
- raw waveform slices in SET4 and SET5 still vary, so the issue is
  not a simple flatline
- the sensor baseline shifts from roughly `-0.28` in SET3 to roughly
  `30` in SET4 and SET5

Important clarification for the team:

- `Range = 0` refers to the TABLE JSON metadata field, not to the raw
  waveform file being all zeros
- the matching SET4 raw files still have visible variation even when
  the TABLE row reports `Range = 0`
- this mismatch is exactly why the channel is treated as unreliable in
  SET4 and SET5
- current dataset validation found `OLD_S1_UP_SUP_STR` to be the only
  sensor with this exact "normal in SET1-SET3, TABLE `Range = 0`
  throughout SET4/SET5" signature; this does not rule out subtler
  anomalies elsewhere

This is intentionally implemented as an auditable config policy, not
as heuristic auto-detection.

## Attribution

The preprocessing API adapts ideas from `AQUINAS_Explorer.R`, shared by
François-Baptiste Cartiaux (OSMOS Group) on April 9, 2026. The Python
implementation is original package code shaped around the same event
selection, synchronization, and zeroing concepts.
