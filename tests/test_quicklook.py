import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquinas_toolkit.preprocessing.quicklook import plot_nn_input_event, summarize_nn_inputs


def _write_quicklook_inputs(preprocess_dir: Path) -> None:
    nn_dir = preprocess_dir / "nn_inputs"
    metadata_dir = nn_dir / "metadata"
    report_dir = preprocess_dir / "report"
    nn_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    np.save(nn_dir / "strain_inputs.npy", np.ones((2, 4, 2), dtype=np.float32))
    np.save(nn_dir / "acc_inputs.npy", np.ones((2, 3, 2), dtype=np.float32))
    np.save(nn_dir / "temperature_inputs.npy", np.array([[20.0], [21.0]], dtype=np.float32))
    np.save(nn_dir / "event_ids.npy", np.array(["event_a", "event_b"]))
    np.save(metadata_dir / "frequency_bins.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.save(metadata_dir / "valid_lengths.npy", np.array([8, 8], dtype=np.int32))
    (metadata_dir / "sensor_ids.json").write_text(
        json.dumps(
            {
                "strain": ["OLD_S1_DO_INF_STR", "OLD_S1_DO_SUP_STR"],
                "acc_z": ["OLD_S1_DO_INT_ACC_Z", "OLD_S1_DO_MID_ACC_Z"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "sensor_name": "OLD_S1_DO_INF_STR",
                "include_flag": True,
                "model_channel_id": "STR00",
                "global_model_channel_index": 0,
            }
        ]
    ).to_csv(report_dir / "sensor_map.csv", index=False)


def test_quicklook_summary_and_event_plot(tmp_path: Path) -> None:
    preprocess_dir = tmp_path / "stages" / "preprocess"
    _write_quicklook_inputs(preprocess_dir)

    summary = summarize_nn_inputs(preprocess_dir)
    assert summary["event_count"] == 2
    assert summary["shapes"]["strain_inputs"] == [2, 4, 2]
    assert summary["finite_counts"]["acc_inputs"] == 12

    output_path = plot_nn_input_event(preprocess_dir, event_index=1)
    assert output_path.is_file()
    assert output_path.name == "event_0001.png"
