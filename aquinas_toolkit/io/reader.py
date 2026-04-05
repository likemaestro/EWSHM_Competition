"""
AQUINAS dataset reader.

Provides the ``AquinasReader`` class for loading index tables and raw
waveform data from any AQUINAS_SET* folder.  Each SET folder contains
48 sensors (24 acceleration + 24 strain) with JSON index tables and
per-day raw-data files.

Original implementation by Zhenkun Li.
Migrated into aquinas_toolkit with minimal formatting changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class AquinasReader:
    """
    Generic reader for any AQUINAS_SET* folder.

    Example folder layout::

        AQUINAS_SET1_2022_07/
            TABLE_*.json
            NEW_S1_DO_INF_STR/
            OLD_S2_UP_SUP_STR/
            ...

    This reader can:

    1. List all sensors in a dataset folder
    2. Load one sensor's index table
    3. Load one raw JSON file
    4. Extract one event record based on File / Start_Row / End_Row
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_dir}")

        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Not a folder: {self.dataset_dir}")

        self.table_files = sorted(self.dataset_dir.glob("TABLE_*.json"))
        self.sensor_dirs = sorted([p for p in self.dataset_dir.iterdir() if p.is_dir()])

        if not self.table_files:
            raise FileNotFoundError(
                f"No TABLE_*.json found in {self.dataset_dir}. "
                "Please confirm this is an AQUINAS_SET folder."
            )

        self.set_name = self.dataset_dir.name

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising every sensor in this dataset folder."""
        rows = []
        for table_file in self.table_files:
            sensor_name = self._sensor_name_from_table(table_file.name)
            sensor_dir = self.dataset_dir / sensor_name
            rows.append(
                {
                    "dataset": self.set_name,
                    "sensor_name": sensor_name,
                    "table_file": table_file.name,
                    "sensor_dir_exists": sensor_dir.exists(),
                    "sensor_dir": str(sensor_dir),
                }
            )
        return pd.DataFrame(rows).sort_values("sensor_name").reset_index(drop=True)

    def list_sensor_names(self) -> List[str]:
        """Return a sorted list of sensor names derived from TABLE files."""
        return sorted(self._sensor_name_from_table(p.name) for p in self.table_files)

    def load_index_table(self, sensor_name: str) -> pd.DataFrame:
        """Load the index table for a single sensor as a DataFrame."""
        table_path = self._find_table_for_sensor(sensor_name)
        data = self._load_json_file(table_path)

        if isinstance(data, list):
            df = pd.DataFrame(data)

        elif isinstance(data, dict):
            # scenario 1: already records/list
            for key in ("data", "records", "rows", "table"):
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                # scenario 2: dict of lists (columnar JSON)
                if all(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)

        else:
            raise ValueError(f"Unsupported JSON structure in {table_path}")

        # if one row and list, expand it
        if len(df) == 1:
            first_row = df.iloc[0]
            if any(isinstance(v, list) for v in first_row):
                row_dict = {}
                max_len = 0
                for col in df.columns:
                    value = first_row[col]
                    if isinstance(value, list):
                        row_dict[col] = value
                        max_len = max(max_len, len(value))
                    else:
                        row_dict[col] = [value]
                        max_len = max(max_len, 1)

                # pad
                for col in row_dict:
                    if len(row_dict[col]) < max_len:
                        row_dict[col] = row_dict[col] + [None] * (max_len - len(row_dict[col]))

                df = pd.DataFrame(row_dict)

        # unwrap single-element lists
        df = df.map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        return df

    def load_raw_file(self, sensor_name: str, raw_filename: str) -> pd.DataFrame:
        """Load a raw waveform JSON file for a sensor as a DataFrame."""
        raw_path = self.dataset_dir / sensor_name / raw_filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        data = self._load_json_file(raw_path)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ("data", "records", "rows"):
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {raw_path}")

        return df

    def read_record(
        self,
        sensor_name: str,
        record_uid: Optional[int] = None,
        row_index: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Read one event record for a given sensor.

        Returns a tuple of (metadata_row, waveform_dataframe).
        Provide either ``record_uid`` or ``row_index`` (defaults to 0).
        """
        index_df = self.load_index_table(sensor_name)

        if record_uid is not None:
            uid_col = self._match_column(index_df, ["Record_UID", "record_uid", "RecordUID"])
            if uid_col is None:
                raise KeyError("Could not find Record_UID column in index table.")
            match = index_df[
                index_df[uid_col].apply(lambda x: self._unwrap_scalar(x)) == record_uid
            ]
            if match.empty:
                raise ValueError(f"Record_UID={record_uid} not found for sensor {sensor_name}")
            meta = match.iloc[0]
        else:
            if row_index is None:
                row_index = 0
            meta = index_df.iloc[row_index]

        file_col = self._match_column(index_df, ["File", "file", "filename"])
        start_col = self._match_column(index_df, ["Start_Row", "start_row", "StartRow"])
        end_col = self._match_column(index_df, ["End_Row", "end_row", "EndRow"])

        if not all([file_col, start_col, end_col]):
            raise KeyError("Index table must contain File / Start_Row / End_Row columns.")

        raw_filename = str(self._unwrap_scalar(meta[file_col]))
        start_row = self._to_int(meta[start_col], "Start_Row")
        end_row = self._to_int(meta[end_col], "End_Row")

        raw_df = self.load_raw_file(sensor_name, raw_filename)

        # AQUINAS handbook: row numbering is 1-based
        sliced = raw_df.iloc[start_row - 1 : end_row].reset_index(drop=True)
        return meta, sliced

    def load_all_index_tables(self) -> pd.DataFrame:
        """
        Load and merge all 48 sensor index tables in this dataset folder.

        Adds ``sensor_name`` and ``dataset`` columns for identification.
        """
        all_tables = []
        for sensor_name in self.list_sensor_names():
            df = self.load_index_table(sensor_name).copy()
            df["sensor_name"] = sensor_name
            df["dataset"] = self.set_name
            all_tables.append(df)

        if not all_tables:
            return pd.DataFrame()

        return pd.concat(all_tables, ignore_index=True)

    def summarize_sensor_records(
        self,
        quantity: str | None = None,
        axis: str | None = None,
    ) -> pd.DataFrame:
        """Return parsed sensor metadata with per-sensor record counts.

        Parameters
        ----------
        quantity:
            Optional measurement type filter such as ``"ACC"`` or ``"STR"``.
        axis:
            Optional axis filter such as ``"Y"`` or ``"Z"``. This only applies
            to acceleration channels.
        """
        rows = []
        quantity_filter = quantity.upper() if quantity is not None else None
        axis_filter = axis.upper() if axis is not None else None

        for sensor_name in self.list_sensor_names():
            sensor_meta = self._parse_sensor_name(sensor_name)

            if quantity_filter is not None and sensor_meta["quantity"] != quantity_filter:
                continue
            if axis_filter is not None and sensor_meta["axis"] != axis_filter:
                continue

            index_df = self.load_index_table(sensor_name)
            rows.append(
                {
                    "dataset": self.set_name,
                    "sensor_name": sensor_name,
                    **sensor_meta,
                    "record_count": len(index_df),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "sensor_name",
                    "deck",
                    "span",
                    "side",
                    "location",
                    "quantity",
                    "axis",
                    "record_count",
                ]
            )

        summary = pd.DataFrame(rows, dtype=object).sort_values(
            ["deck", "span", "side", "location", "quantity", "axis", "sensor_name"]
        ).reset_index(drop=True)
        summary["axis"] = summary["axis"].where(summary["axis"].notna(), None)
        return summary

    def read_event_all_sensors(
        self, row_index: int = 0
    ) -> dict[str, Tuple[pd.Series, pd.DataFrame]]:
        """
        Read the same event index from every sensor.

        Returns a dict mapping sensor_name -> (metadata, waveform).
        Sensors where the event index is out of range are silently skipped.
        """
        results: dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
        for sensor in self.list_sensor_names():
            idx_df = self.load_index_table(sensor)
            if row_index >= len(idx_df):
                continue
            meta, waveform = self.read_record(sensor_name=sensor, row_index=row_index)
            results[sensor] = (meta, waveform)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_table_for_sensor(self, sensor_name: str) -> Path:
        matches = list(self.dataset_dir.glob(f"TABLE_{sensor_name}*.json"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Index table for sensor '{sensor_name}' not found.")

    @staticmethod
    def _sensor_name_from_table(table_filename: str) -> str:
        name = table_filename.replace("TABLE_", "")
        if name.endswith(".json"):
            name = name[:-5]

        # strip trailing _SET1 / _SET2 / ... suffix
        parts = name.split("_")
        if parts[-1].startswith("SET"):
            name = "_".join(parts[:-1])

        return name

    @staticmethod
    def _parse_sensor_name(sensor_name: str) -> dict[str, str | None]:
        parts = sensor_name.split("_")
        if len(parts) < 5:
            raise ValueError(f"Unrecognized sensor name format: {sensor_name}")

        parsed = {
            "deck": parts[0],
            "span": parts[1],
            "side": parts[2],
            "location": parts[3],
            "quantity": parts[4],
            "axis": parts[5] if len(parts) > 5 else None,
        }
        return parsed

    @staticmethod
    def _load_json_file(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _match_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    @staticmethod
    def _unwrap_scalar(x):
        while isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        return x

    @classmethod
    def _to_int(cls, x, field_name="value"):
        x = cls._unwrap_scalar(x)
        if pd.isna(x):
            raise ValueError(f"{field_name} is NaN.")
        return int(float(x))
