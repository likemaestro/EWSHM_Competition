"""
Inspect a specific outlier event in detail.

Targets the most extreme event found in the cross-SET evaluation:
  AQUINAS_SET3_2023_08__NEW__2023-08-22T12-11-46Z__2023-08-22T12-11-59Z
  (total reconstruction error = 40.17, ~130x the SET1 median)

For this event, we:
  1. Locate it in the raw NPY tensors.
  2. Print raw signal statistics for all 8 channels (min/max/mean/std) --
     a sensor glitch (e.g. a stuck value, a single huge spike, clipping)
     usually shows up immediately in these basic stats.
  3. Plot the raw strain and ACC signals for all 8 channels (input only,
     no model needed for this part) so we can visually inspect for
     obvious artifacts.
  4. Run the trained model on this event's 8 channel-samples and plot
     input vs reconstruction, same format as our other reconstruction
     plots, to see specifically how/where the model's reconstruction
     diverges.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------
# Typography per team specification
# font: Helvetica/DejaVu Sans, title 6pt, legend 4.7pt, axis labels 5.5pt
# -----------------------------------------------------------------------
import matplotlib
matplotlib.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             16,
    "axes.titlesize":        16,
    "axes.labelsize":        16,
    "xtick.labelsize":       16,
    "ytick.labelsize":       16,
    "legend.fontsize":       12,
    "legend.title_fontsize": 12,
    "figure.titlesize":      16,
})
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------

import numpy as np
import torch

from architecture_1_v3 import AttentionAutoencoderV3

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

TARGET_EVENT_ID = (
    "AQUINAS_SET3_2023_08__NEW__2023-08-22T12-11-46Z__2023-08-22T12-11-59Z"
)

RAW_NN_DIR = Path(__file__).resolve().parent  # same folder as the npy files
DATA_DIR = Path(__file__).resolve().parent / "per_channel_data_new_deck"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Target event: {TARGET_EVENT_ID}\n")

    print("Loading raw NPY tensors...")
    strain = np.load(RAW_NN_DIR / "strain_inputs.npy")
    acc = np.load(RAW_NN_DIR / "acc_inputs.npy")
    temp = np.load(RAW_NN_DIR / "temperature_inputs.npy")
    eids = np.load(RAW_NN_DIR / "event_ids.npy", allow_pickle=True)

    # Locate the event
    matches = np.where(eids == TARGET_EVENT_ID)[0]
    if len(matches) == 0:
        print("ERROR: event not found. Checking for partial matches...")
        partial = [e for e in eids if TARGET_EVENT_ID[:40] in str(e)]
        print(f"Partial matches: {partial[:5]}")
        return

    idx = int(matches[0])
    print(f"Found at index {idx}\n")

    s_event = strain[idx]   # (200, 8)
    a_event = acc[idx]      # (695, 8)
    t_event = temp[idx, 0]  # scalar

    print(f"Temperature for this event: {t_event:.2f} °C\n")

    # --------------------------------------------------------------------
    # STEP 1: Raw signal diagnostics per channel
    # --------------------------------------------------------------------
    print("=" * 70)
    print("RAW SIGNAL DIAGNOSTICS (before any model/normalization)")
    print("=" * 70)
    print(f"\n{'ch':>3s}  {'strain_min':>11s}  {'strain_max':>11s}  "
          f"{'strain_mean':>12s}  {'strain_std':>11s}")
    for ch in range(8):
        sc = s_event[:, ch]
        print(f"{ch:>3d}  {sc.min():>11.5f}  {sc.max():>11.5f}  "
              f"{sc.mean():>12.5f}  {sc.std():>11.5f}")

    print(f"\n{'ch':>3s}  {'acc_min':>11s}  {'acc_max':>11s}  "
          f"{'acc_mean':>12s}  {'acc_std':>11s}")
    for ch in range(8):
        ac = a_event[:, ch]
        print(f"{ch:>3d}  {ac.min():>11.5f}  {ac.max():>11.5f}  "
              f"{ac.mean():>12.5f}  {ac.std():>11.5f}")

    # Flag any channel with suspiciously extreme values
    print("\nSanity flags:")
    any_flag = False
    for ch in range(8):
        sc = s_event[:, ch]
        ac = a_event[:, ch]
        if np.isnan(sc).any() or np.isnan(ac).any():
            print(f"  channel {ch}: contains NaN values!")
            any_flag = True
        if np.abs(sc).max() > 1.0:  # strain is typically O(0.01-0.1)
            print(f"  channel {ch}: strain max abs value unusually large "
                  f"({np.abs(sc).max():.4f})")
            any_flag = True
        if ac.max() > 10.0:  # acc FFT magnitudes typically well under this
            print(f"  channel {ch}: acc max value unusually large "
                  f"({ac.max():.4f})")
            any_flag = True
    if not any_flag:
        print("  None detected -- raw values look like normal-range signals.")

    # --------------------------------------------------------------------
    # STEP 2: Plot raw signals, all 8 channels
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ch in range(8):
        axes[0].plot(s_event[:, ch], label=f"ch{ch}", alpha=0.7)
        axes[1].plot(a_event[:, ch], label=f"ch{ch}", alpha=0.7)
    axes[0].set_title(f"Raw strain, all 8 channels — {TARGET_EVENT_ID}\nT={t_event:.1f}°C")
    axes[0].set_xlabel("time sample")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Raw ACC FFT magnitude, all 8 channels")
    axes[1].set_xlabel("frequency bin")
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = LOG_DIR / "outlier_event_raw_signals.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved raw signal plot to {out}")

    # --------------------------------------------------------------------
    # STEP 3: Run model, plot input vs reconstruction per channel
    # --------------------------------------------------------------------
    print("\nLoading scalers and model...")
    strain_scaler = joblib.load(DATA_DIR / "strain_scaler.joblib")
    acc_scaler = joblib.load(DATA_DIR / "acc_scaler.joblib")
    temp_scaler = joblib.load(DATA_DIR / "temp_scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_DIR / "architecture_1_v3_best.pt",
                      weights_only=False, map_location=device)
    model = AttentionAutoencoderV3(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Normalize this event's 8 channels
    s_8 = s_event.T  # (8, 200)
    a_8 = a_event.T  # (8, 695)
    t_8 = np.full((8, 1), t_event, dtype=np.float32)

    s_n = strain_scaler.transform(s_8).astype(np.float32)
    a_n = acc_scaler.transform(a_8).astype(np.float32)
    t_n = temp_scaler.transform(t_8).astype(np.float32)

    with torch.no_grad():
        s_t = torch.from_numpy(s_n).to(device)
        a_t = torch.from_numpy(a_n).to(device)
        t_t = torch.from_numpy(t_n).to(device)
        out = model(s_t, a_t, t_t)

        s_err = ((out["strain_hat"] - s_t) ** 2).mean(dim=1).cpu().numpy()
        a_err = ((out["acc_hat"] - a_t) ** 2).mean(dim=1).cpu().numpy()
        total_per_ch = s_err + a_err

    print("\n" + "=" * 70)
    print("PER-CHANNEL RECONSTRUCTION ERROR FOR THIS EVENT")
    print("=" * 70)
    print(f"{'ch':>3s}  {'strain_err':>11s}  {'acc_err':>11s}  {'total':>10s}")
    for ch in range(8):
        print(f"{ch:>3d}  {s_err[ch]:>11.4f}  {a_err[ch]:>11.4f}  "
              f"{total_per_ch[ch]:>10.4f}")
    print(f"\nMean across channels (this is the 'per-event total' reported "
          f"earlier): {total_per_ch.mean():.4f}")

    worst_ch = int(np.argmax(total_per_ch))
    print(f"\nWorst channel: ch{worst_ch} (total error = {total_per_ch[worst_ch]:.4f})")

    # Plot input vs reconstruction for ALL 8 channels in a grid
    fig, axes = plt.subplots(8, 2, figsize=(13, 24))
    for ch in range(8):
        axes[ch, 0].plot(s_n[ch], label="input (normalized)")
        axes[ch, 0].plot(out["strain_hat"][ch].cpu().numpy(),
                         label="reconstruction", alpha=0.7)
        axes[ch, 0].set_title(f"ch{ch} strain (err={s_err[ch]:.3f})", fontsize=9)
        axes[ch, 0].legend(fontsize=7)
        axes[ch, 0].grid(alpha=0.3)

        axes[ch, 1].plot(a_n[ch], label="input (normalized)")
        axes[ch, 1].plot(out["acc_hat"][ch].cpu().numpy(),
                         label="reconstruction", alpha=0.7)
        axes[ch, 1].set_title(f"ch{ch} ACC FFT (err={a_err[ch]:.3f})", fontsize=9)
        axes[ch, 1].legend(fontsize=7)
        axes[ch, 1].grid(alpha=0.3)
    fig.suptitle(f"Outlier event: {TARGET_EVENT_ID}\n"
                 f"T={t_event:.1f}°C, total error={total_per_ch.mean():.2f} "
                 f"(worst channel: ch{worst_ch})", y=1.005)
    plt.tight_layout()
    out_path = LOG_DIR / "outlier_event_reconstruction_all_channels.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved reconstruction comparison to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
