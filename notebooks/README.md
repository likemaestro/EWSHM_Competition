# notebooks/

Jupyter notebooks for exploration, visualisation, and presentation.

## Naming convention

Notebooks are numbered sequentially so they tell a story from
raw data exploration through to the final health score:

| Notebook | Purpose | Status |
|---|---|---|
| `01_sensor_overview` | Dataset structure, raw waveform plots | Done |
| `02_preprocessing` | Filtering, normalisation, alignment | TODO |
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
