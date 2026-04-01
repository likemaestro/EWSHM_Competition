# AQUINAS Dataset Reference

**Available QUantities INtended for Analysis and Science**  
Data made available by [OSMOS Group](https://www.osmos-group.com) for research purposes.

---

## Overview

The AQUINAS dataset V1.0 is a set of measurements recorded over a few years on a real, anonymised bridge.

- **24 acceleration sensors** and **24 strain sensors** are installed on the bridge deck.
- The sampling rate is **100 Hz**.
- Recording is **trigger-based**: the dataset is split into individual records corresponding to vehicle passages, and these records are listed in index tables.
- The released dataset is split into **5 monthly datasets**.

The bridge is a **double prestressed concrete deck** with **40 to 50 m long spans**. The upstream and downstream decks are independent and carry traffic in one single direction, so records do not happen at the same time on both decks. The two decks have the same span lengths and very similar geometry, although some cross-section details differ, including concrete thickness and prestress cables. The upstream deck was built more than **10 years before** the downstream deck.

The challenge rules describe the case study as a **prestressed concrete box-girder viaduct located in France**. The handbook keeps the bridge anonymous for research use.

---

## Dataset SETs

| Folder | Period | SET ID |
|---|---|---|
| `AQUINAS_SET1_2022_07` | July 2022 | `SET1` |
| `AQUINAS_SET2_2023_04` | April 2023 | `SET2` |
| `AQUINAS_SET3_2023_08` | August 2023 | `SET3` |
| `AQUINAS_SET4_2024_01` | January 2024 | `SET4` |
| `AQUINAS_SET5_2024_06` | June 2024 | `SET5` |

Each monthly dataset contains:

- `48` index tables in JSON format, one for each sensor
- `48` sensor directories with raw-data JSON files split by calendar day

---

## Filename Convention

Index-table files follow this pattern:

```text
TABLE_{DECK}_{SPAN}_{SIDE}_{SENSOR_CODE}_SET{N}.json
```

Raw waveform directories follow this pattern:

```text
{DECK}_{SPAN}_{SIDE}_{SENSOR_CODE}/
```

Raw waveform files inside each sensor directory follow this pattern:

```text
{DECK}_{SPAN}_{SIDE}_{SENSOR_CODE}_SET{N}_{day}.json
```

### Token definitions

| Token | Possible values | Meaning |
|---|---|---|
| `DECK` | `OLD`, `NEW` | `OLD` = old upstream deck, `NEW` = new downstream deck |
| `SPAN` | `S1`, `S2` | Span 1 or span 2 |
| `SIDE` | `UP`, `DO` | Upstream side or downstream side |
| `SENSOR_CODE` | see below | Location + quantity (+ axis for accelerometers) |
| `N` | `1` to `5` | Dataset SET number |

### Sensor codes

| Code | Location | Quantity | Axis | Description |
|---|---|---|---|---|
| `MID_ACC_Z` | Mid-span (`1/2` span length) | Acceleration | Z | Vertical acceleration at mid-span |
| `MID_ACC_Y` | Mid-span (`1/2` span length) | Acceleration | Y | Transversal acceleration at mid-span |
| `INT_ACC_Z` | Intermediate section | Acceleration | Z | Vertical acceleration at the intermediate section |
| `INT_ACC_Y` | Intermediate section | Acceleration | Y | Transversal acceleration at the intermediate section |
| `INF_STR` | Inferior fibre | Strain | n/a | Longitudinal strain at the inferior fibre |
| `SUP_STR` | Superior fibre | Strain | n/a | Longitudinal strain at the superior fibre |
| `SHE_STR` | Web, set in diagonal | Strain | n/a | Diagonal strain for shear effect |

In the released V1.0 dataset, `ACC_Y` channels appear only on the downstream side (`DO`). No `*_UP_*ACC_Y` channels are present in the files.

---

## Cross-Sections and Sensor Layout

Sensors are installed on the first two spans of each deck, at six cross-sections:

| Section name | Location | Acceleration sensors | Strain sensors |
|---|---|---|---|
| `S1_MID` | `1/2` length of span 1 | `2x Z-axis + 1x Y-axis` | `4x Longitudinal` |
| `S1_INT` | `2/3` length of span 1 | `2x Z-axis + 1x Y-axis` | None |
| `S1_SHE` | Near pier 1 on span 1 | None | `2x Diagonal (shear effect)` |
| `S2_SHE` | Near pier 1 on span 2 | None | `2x Diagonal (shear effect)` |
| `S2_INT` | `1/3` length of span 2 | `2x Z-axis + 1x Y-axis` | None |
| `S2_MID` | `1/2` length of span 2 | `2x Z-axis + 1x Y-axis` | `4x Longitudinal` |

### Location glossary

| Code | Meaning |
|---|---|
| `MID` | Mid-span |
| `INT` | Intermediate section: `2/3` of span 1 or `1/3` of span 2 |
| `INF` | Inferior fibre |
| `SUP` | Superior fibre |
| `SHE` | Sensor on the web, set in diagonal for shear strain |

---

## Units and Time Fields

| Field | Unit / format | Notes |
|---|---|---|
| Acceleration | `g` | `1 g = 9.81 m/s^2` |
| Strain | `permille` | `1 permille = 1 mm/m` |
| Temperature | `deg C` | Ambient temperature at the time of the record |
| `Start_Time`, `End_Time` | UTC datetime string, second precision | Used in TABLE JSON files |
| `timestamp` | UTC datetime string, millisecond precision | Used in raw waveform JSON files |
| Sampling rate | `100 Hz` | `10 ms` between samples |

---

## File Formats

### TABLE JSON

Each TABLE file is a columnar JSON file where every key maps to an array of equal length. Each index position corresponds to one record for that sensor. The handbook lists **15 basic features** per record.

| Key | Type | Description | Example from handbook |
|---|---|---|---|
| `Record_UID` | `int[]` | Unique ID number of the record for that sensor | `16341` |
| `File` | `str[]` | Name of the raw-data JSON file containing the record | `"OLD_S2_UP_SUP_STR_SET1_1.json"` |
| `Start_Row` | `int[]` | First row of the record in the raw-data JSON | `1` |
| `End_Row` | `int[]` | Last row of the record in the raw-data JSON | `1527` |
| `Start_Time` | `str[]` | UTC timestamp of the start of the record | `"2022-07-01 01:47:16"` |
| `End_Time` | `str[]` | UTC timestamp of the end of the record | `"2022-07-01 01:47:31"` |
| `Duration` | `float[]` | Duration of the record in seconds | `15.26` |
| `Start_Value` | `float[]` | First value of the record | `-0.1184` |
| `End_Value` | `float[]` | Last value of the record | `-0.1187` |
| `Diff_Value` | `float[]` | `End_Value - Start_Value` | `-0.0002` |
| `Min_Value` | `float[]` | Minimum value of the record | `-0.1416` |
| `Max_Value` | `float[]` | Maximum value of the record | `-0.116` |
| `Mean_Value` | `float[]` | Mean value of the record | `-0.1197` |
| `Range` | `float[]` | `Max_Value - Min_Value` | `0.0256` |
| `Temperature` | `float[]` | Ambient temperature at the time of the record | `13.3159` |

Important notes from the handbook and the released files:

- `Record_UID` is **specific to each sensor** and should **not** be used as a cross-sensor join key.
- To match records between different sensors, use `Start_Time` and `End_Time`.
- Rounding artefacts may affect `Diff_Value` and `Range`.
- In the released JSON files, `Start_Row` and `End_Row` are **1-based** row numbers into the raw waveform file.

### Event waveform JSON

Raw data is stored in the per-sensor directories inside each SET folder. Files are split by calendar day.

Each raw waveform file is a two-key columnar JSON:

| Key | Type | Description |
|---|---|---|
| `timestamp` | `str[]` | UTC timestamp with millisecond precision |
| `{SENSOR_NAME}` | `float[]` | Raw sensor readings at 100 Hz |

The sensor data key matches the channel name exactly, for example `"OLD_S2_UP_SUP_STR"`.

---

## Released Channel Layout

| Deck | Span | Shear strain | Longitudinal strain | Z acceleration | Y acceleration |
|---|---|---|---|---|---|
| `OLD` or `NEW` | `S1` | `S1_SHE_STR` | `S1_MID_INF_STR`, `S1_MID_SUP_STR` | `S1_MID_ACC_Z`, `S1_INT_ACC_Z` | `S1_MID_ACC_Y`, `S1_INT_ACC_Y` on `DO` only |
| `OLD` or `NEW` | `S2` | `S2_SHE_STR` | `S2_MID_INF_STR`, `S2_MID_SUP_STR` | `S2_MID_ACC_Z`, `S2_INT_ACC_Z` | `S2_MID_ACC_Y`, `S2_INT_ACC_Y` on `DO` only |

---

## Notes From the Challenge Rules

- The objective is to derive structural health scores from **strain and vibration measurements** while considering the effects of traffic.
- The dataset does **not** include labels.
- The challenge rules state that machine-learning or AI methods, if used, should be **unsupervised**.
- The expected computation is **offline**, using the whole dataset and combining all sensors.
