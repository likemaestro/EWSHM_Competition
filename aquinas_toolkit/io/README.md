# io/

## Purpose

Data I/O for the AQUINAS dataset. This package handles reading
index tables and raw waveform files from any AQUINAS SET folder.

## Status

Complete. The `AquinasReader` class is fully functional.

## Key class

**`AquinasReader(dataset_dir)`** -- point it at any `AQUINAS_SET*` folder.

| Method | What it does |
|---|---|
| `summary()` | DataFrame of all 48 sensors and their table files |
| `list_sensor_names()` | Sorted list of sensor name strings |
| `load_index_table(sensor)` | Load one sensor's index table (15 features per record) |
| `load_raw_file(sensor, filename)` | Load a raw waveform JSON as a DataFrame |
| `read_record(sensor, record_uid=, row_index=)` | Return (metadata, waveform) for one event |
| `load_all_index_tables()` | Merge all 48 index tables into one DataFrame |

## Public import

```python
from aquinas_toolkit import AquinasReader
# or
from aquinas_toolkit.io import AquinasReader
```

## Performance

Each sensor directory contains numbered sequential JSON batch files:
`{SENSOR_NAME}_SET{N}_{NUMBER}.json` (1, 2, 3, ...). Each file is roughly
8.8 MB and holds around 247 000 waveform rows. Multiple index-table records
(events) point into the same file via `Start_Row` and `End_Row`.

Measured across the five dataset sets (numbers are per sensor):

| SET | Raw files | Total records | Avg records/file | Max records/file |
|---|---|---|---|---|
| SET1 | 31 | 3 633 | 117 | 226 |
| SET2 | 29 | 1 677 |  57 | 110 |
| SET3 | 31 | 3 493 | 112 | 208 |
| SET4 | 29 |   867 |  29 | 101 |
| SET5 | 30 | 2 577 |  85 | 176 |

Without caching, the same 8.8 MB file would be read and re-parsed from disk
once per event that references it -- up to 226 times for the same file in one
set. `load_raw_file()` caches parsed DataFrames in memory, keyed on
`(sensor_name, raw_filename)`, so each file is read exactly once per reader
instance.

- The cache is scoped to the reader instance. `run_preprocessing()` creates a
  new `AquinasReader` per SET, so memory is released between sets.
- Callers that need to modify the returned DataFrame must call `.copy()` first.
  All existing internal callers (`_load_waveform_from_record`,
  `_load_waveform_slice`, `read_record`) already do this.
- Do not call `_load_json_file` directly; always go through `load_raw_file`.

## Attribution

Original implementation by `Zhenkun Li`.
