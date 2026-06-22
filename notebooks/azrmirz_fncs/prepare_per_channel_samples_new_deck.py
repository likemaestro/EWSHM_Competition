"""
Prepare per-channel training samples for Architecture 1 v3, using NEW deck.

This uses Murat's corrected NEW_DECK_ALL_SETS data (channel-duplication bug
fixed -- tensors should already have the correct 8-channel layout, NOT 40).

What this script does:
  1. Loads Murat's NEW deck NPY tensors.
  2. Verifies the channel count is 8 (sanity check the duplication bug is
     actually fixed this time -- if it's still 40, we slice every 5th
     channel as a fallback, same fix as before).
  3. Filters to SET1 events only (matching the team's "train on month 1"
     plan).
  4. Explodes each event into 8 per-channel samples (one strain + one acc +
     temperature per sample).
  5. Splits 70/20/10 train/val/test.
  6. Fits sklearn.preprocessing.StandardScaler on the TRAIN samples only
     (explicit module use, per supervisor's request -- previously we did
     the equivalent computation by hand).
  7. Saves everything to per_channel_data_new_deck/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------------------------------------------------------
# Configuration -- EDIT THIS PATH to match your downloaded folder
# ----------------------------------------------------------------------------

MURAT_NN_DIR = Path(__file__).resolve().parent

OUT_DIR = Path(__file__).resolve().parent / "per_channel_data_new_deck"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.20
SPLIT_TEST = 0.10
RANDOM_SEED = 2026

SET1_PREFIX = "AQUINAS_SET1_2022_07"


def main() -> None:
    print("=" * 70)
    print("STEP 1: Loading Murat's NEW deck NPY tensors")
    print("=" * 70)

    strain_all = np.load(MURAT_NN_DIR / "strain_inputs.npy")
    acc_all    = np.load(MURAT_NN_DIR / "acc_inputs.npy")
    temp_all   = np.load(MURAT_NN_DIR / "temperature_inputs.npy")
    eids_all   = np.load(MURAT_NN_DIR / "event_ids.npy", allow_pickle=True)

    print(f"  strain_all       {strain_all.shape}")
    print(f"  acc_all          {acc_all.shape}")
    print(f"  temperature_all  {temp_all.shape}")
    print(f"  event_ids_all    {eids_all.shape}")

    n_channels_raw = strain_all.shape[-1]

    # ------------------------------------------------------------------
    # STEP 2: Verify channel count / duplication status
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 2: Checking channel layout (found {n_channels_raw} channels)")
    print("=" * 70)

    if n_channels_raw == 8:
        print("  Channel count is 8 -- duplication bug appears FIXED. Using as-is.")
        strain8 = strain_all
        acc8 = acc_all
    elif n_channels_raw == 40:
        print("  Channel count is 40 -- duplication bug STILL PRESENT.")
        print("  Verifying duplication pattern on first event...")
        ev0_ch0 = strain_all[0, :, 0]
        all_match = True
        for ch in range(1, 5):
            diff = np.abs(strain_all[0, :, ch] - ev0_ch0).max()
            if diff > 1e-6:
                all_match = False
            print(f"    strain ev0 ch0 vs ch{ch}: {'OK' if diff < 1e-6 else f'DIFF max={diff:.4g}'}")
        if all_match:
            print("  Duplication confirmed -- applying every-5th-channel fix.")
            strain8 = strain_all[:, :, ::5]
            acc8 = acc_all[:, :, ::5]
        else:
            raise RuntimeError(
                "Channel count is 40 but duplication pattern does NOT match "
                "the previous bug. Manual investigation needed -- do not "
                "proceed blindly."
            )
    else:
        raise RuntimeError(
            f"Unexpected channel count: {n_channels_raw}. Expected 8 or 40. "
            "Inspect the data manually before proceeding."
        )

    print(f"\n  Final shapes after channel handling:")
    print(f"    strain8  {strain8.shape}")
    print(f"    acc8     {acc8.shape}")

    # ------------------------------------------------------------------
    # STEP 3: Filter to SET1 events only
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Filtering to SET1 events only")
    print("=" * 70)

    set1_mask = np.array([eid.startswith(SET1_PREFIX) for eid in eids_all])
    n_set1_events = int(set1_mask.sum())
    print(f"  SET1 events:   {n_set1_events}")
    print(f"  Other events:  {(~set1_mask).sum()}")

    if n_set1_events == 0:
        raise RuntimeError(
            f"No events found starting with '{SET1_PREFIX}'. Check the "
            "event_ids format -- printing first 3 for inspection: "
            f"{eids_all[:3]}"
        )

    strain_set1 = strain8[set1_mask]
    acc_set1    = acc8[set1_mask]
    temp_set1   = temp_all[set1_mask]
    eids_set1   = eids_all[set1_mask]

    print(f"  strain_set1    {strain_set1.shape}")
    print(f"  acc_set1       {acc_set1.shape}")
    print(f"  temperature    {temp_set1.shape}")

    # ------------------------------------------------------------------
    # STEP 4: Explode (event, channel) into per-channel samples
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Exploding events into per-channel samples")
    print("=" * 70)

    n_events, _, n_channels = strain_set1.shape
    n_samples = n_events * n_channels
    print(f"  Events x channels: {n_events} x {n_channels} = {n_samples} samples")

    strain_samples = (
        strain_set1.transpose(0, 2, 1).reshape(-1, strain_set1.shape[1])
    )
    acc_samples = (
        acc_set1.transpose(0, 2, 1).reshape(-1, acc_set1.shape[1])
    )
    temp_samples = (
        np.broadcast_to(temp_set1, (n_events, n_channels))
        .reshape(-1, 1).astype(np.float32)
    )

    sample_event_ids = np.repeat(eids_set1, n_channels)
    sample_channel_indices = np.tile(np.arange(n_channels), n_events)

    print(f"  strain_samples  {strain_samples.shape}")
    print(f"  acc_samples     {acc_samples.shape}")
    print(f"  temp_samples    {temp_samples.shape}")

    # ------------------------------------------------------------------
    # STEP 5: Train/val/test split
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Train/val/test split (70/20/10)")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n_samples)
    n_train = int(round(SPLIT_TRAIN * n_samples))
    n_val = int(round(SPLIT_VAL * n_samples))
    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    print(f"  train: {len(train_idx)} samples ({100*len(train_idx)/n_samples:.1f}%)")
    print(f"  val:   {len(val_idx)} samples ({100*len(val_idx)/n_samples:.1f}%)")
    print(f"  test:  {len(test_idx)} samples ({100*len(test_idx)/n_samples:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 6: Fit sklearn StandardScaler on TRAIN samples only
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Fitting sklearn StandardScaler on train samples")
    print("=" * 70)

    strain_scaler = StandardScaler()
    acc_scaler = StandardScaler()
    temp_scaler = StandardScaler()

    strain_scaler.fit(strain_samples[train_idx])
    acc_scaler.fit(acc_samples[train_idx])
    temp_scaler.fit(temp_samples[train_idx])

    print(f"  strain scaler: mean (first 3) = {strain_scaler.mean_[:3]}")
    print(f"  acc scaler:    mean (first 3) = {acc_scaler.mean_[:3]}")
    print(f"  temp scaler:   mean = {temp_scaler.mean_}")

    # ------------------------------------------------------------------
    # STEP 7: Save everything
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Saving outputs")
    print("=" * 70)

    np.save(OUT_DIR / "strain_samples.npy", strain_samples.astype(np.float32))
    np.save(OUT_DIR / "acc_samples.npy", acc_samples.astype(np.float32))
    np.save(OUT_DIR / "temp_samples.npy", temp_samples)
    np.save(OUT_DIR / "sample_event_ids.npy", sample_event_ids)
    np.save(OUT_DIR / "sample_channel_indices.npy", sample_channel_indices)
    np.save(OUT_DIR / "train_idx.npy", train_idx)
    np.save(OUT_DIR / "val_idx.npy", val_idx)
    np.save(OUT_DIR / "test_idx.npy", test_idx)

    joblib.dump(strain_scaler, OUT_DIR / "strain_scaler.joblib")
    joblib.dump(acc_scaler, OUT_DIR / "acc_scaler.joblib")
    joblib.dump(temp_scaler, OUT_DIR / "temp_scaler.joblib")

    print(f"  All files saved to {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
