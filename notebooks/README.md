# notebooks/

Jupyter notebooks for exploration, visualisation, and presentation.

## Naming convention

Notebooks are numbered sequentially so they tell a story from
raw data exploration through to the final health score:

| Notebook | Purpose | Status |
|---|---|---|
| `01_sensor_overview` | Dataset structure, raw waveform plots | Done |
| `02_preprocessing` | Event grouping, synchronization, zeroing | Done |
| `03_feature_extraction` | Time- and frequency-domain features | TODO |
| `04_anomaly_detection` | Unsupervised outlier and trend detection | TODO |
| `05_health_scoring` | Final structural health score computation | TODO |

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

- `02_preprocessing.ipynb` is a consumer of the preprocessing API, not
  a second implementation of preprocessing logic.
- In notebook calls to `find_events()`, `timestamp=` means
  "return events whose metadata window contains this timestamp".
  It is not a nearest-event search.
- `sensor_pattern=` uses the preprocessing API's sensor-name filter.
  With `*`, `?`, or `[]`, it uses wildcard matching.
  Without them, it behaves like a case-insensitive substring filter.
- Example: `sensor_pattern="*UP*ACC_Z*"` means upstream
  Z-acceleration sensors such as `OLD_S1_UP_INT_ACC_Z`,
  `OLD_S1_UP_MID_ACC_Z`, `OLD_S2_UP_INT_ACC_Z`, and
  `OLD_S2_UP_MID_ACC_Z`.
- The organizer-style strain example in `02_preprocessing.ipynb`
  uses a real `SET2` event that starts at `2023-04-20 07:04:05`. The
  query timestamp is `07:04:10` (strictly inside the window) because
  `find_events` uses strict containment (`Start_Time < timestamp <
  End_Time`), so the boundary value itself returns nothing.
- The organizer screenshot for acceleration is labeled
  `2022-09-01 17:51:55`, but that timestamp is not present in the
  released competition dataset in this repository.
- The notebook therefore uses the real `SET1` upstream
  `ACC_Z` event at `2022-07-30 18:36:53` as a documented substitute.
  That timestamp is inside the event window under the organizer's
  strict containment rule; the old boundary timestamp
  `2022-07-30 18:36:52` is no longer used.
- The notebook shows before-zeroing and after-zeroing plots in separate
  cells on purpose so the team can compare them without stacked axes.
