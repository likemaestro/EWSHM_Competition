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
| `02_preprocessing` | Filtering, normalisation, alignment | TODO |
| `03_feature_extraction` | Time- and frequency-domain features | TODO |
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
- Use the shared dataset helpers from `aquinas_toolkit.utils` instead of
  hardcoding notebook-relative paths.
- For metadata-only notebook workflows, prefer
  `from aquinas_toolkit.io import load_sensor_metadata` over manual
  index-table loops or `read_record(...)`.
- Standard notebook pattern:
  `from aquinas_toolkit.utils import find_dataset_root, list_dataset_dirs`
