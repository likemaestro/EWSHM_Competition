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

## Attribution

Original implementation by Zhenkun Li.
