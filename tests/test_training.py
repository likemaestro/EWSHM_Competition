import json
from pathlib import Path

import numpy as np

from aquinas_toolkit.training import prepare_training_data
from aquinas_toolkit.utils.run_management import RunContext


def _write_nn_inputs(run_dir: Path, *, event_count: int = 10) -> None:
    nn_dir = run_dir / "stages" / "preprocess" / "nn_inputs"
    metadata_dir = nn_dir / "metadata"
    nn_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    strain = np.arange(event_count * 2 * 2, dtype=np.float32).reshape(event_count, 2, 2)
    acc = np.arange(event_count * 3 * 2, dtype=np.float32).reshape(event_count, 3, 2)
    temperature = np.arange(event_count, dtype=np.float32).reshape(event_count, 1)
    np.save(nn_dir / "strain_inputs.npy", strain)
    np.save(nn_dir / "acc_inputs.npy", acc)
    np.save(nn_dir / "temperature_inputs.npy", temperature)
    np.save(nn_dir / "event_ids.npy", np.array([f"event_{i}" for i in range(event_count)]))
    np.save(metadata_dir / "frequency_bins.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.save(metadata_dir / "valid_lengths.npy", np.full(event_count, 8, dtype=np.int32))
    (metadata_dir / "sensor_ids.json").write_text(
        json.dumps(
            {
                "strain": ["OLD_S1_DO_INF_STR", "OLD_S1_DO_SUP_STR"],
                "acc_z": ["OLD_S1_DO_INT_ACC_Z", "OLD_S1_DO_MID_ACC_Z"],
            }
        ),
        encoding="utf-8",
    )


def _run_context(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "results" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  random_seed: 123",
                "  split:",
                "    train: 0.70",
                "    validation: 0.20",
                "    test: 0.10",
                "  standardization:",
                "    enabled: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return RunContext(
        run_id="run-1",
        results_dir=tmp_path / "results",
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=run_dir / "metadata.json",
        debug_log_path=run_dir / "debug.log",
    )


def test_prepare_training_data_writes_deterministic_splits_and_train_only_stats(
    tmp_path: Path,
) -> None:
    run_context = _run_context(tmp_path)
    _write_nn_inputs(run_context.run_dir, event_count=10)

    result = prepare_training_data(run_context)

    train_idx = np.load(result.train_indices_path)
    val_idx = np.load(result.val_indices_path)
    test_idx = np.load(result.test_indices_path)
    assert [len(train_idx), len(val_idx), len(test_idx)] == [7, 2, 1]
    assert sorted(np.concatenate([train_idx, val_idx, test_idx]).tolist()) == list(range(10))
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))

    second = prepare_training_data(run_context)
    assert np.array_equal(train_idx, np.load(second.train_indices_path))
    assert np.array_equal(val_idx, np.load(second.val_indices_path))
    assert np.array_equal(test_idx, np.load(second.test_indices_path))

    nn_dir = run_context.run_dir / "stages" / "preprocess" / "nn_inputs"
    strain = np.load(nn_dir / "strain_inputs.npy")
    acc = np.load(nn_dir / "acc_inputs.npy")
    temperature = np.load(nn_dir / "temperature_inputs.npy")
    stats = np.load(result.normalization_stats_path)

    assert stats["strain_mean"].tolist() == np.nanmean(
        strain[train_idx], axis=(0, 1), keepdims=True
    ).tolist()
    assert stats["acc_mean"].tolist() == np.nanmean(acc[train_idx], axis=(0, 1), keepdims=True).tolist()
    assert stats["temperature_mean"].tolist() == np.nanmean(
        temperature[train_idx], axis=0, keepdims=True
    ).tolist()

    manifest = json.loads(result.split_manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 7, "validation": 2, "test": 1}
    assert manifest["event_count"] == 10
