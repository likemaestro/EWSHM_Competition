"""
Evaluate Architecture 1 v3 on the NEW deck SET1 test split.

New in this version (per supervisor's request in the June 18 meeting):
  - Reconstruction-error-vs-TIME plot: shows error chronologically across
    SET1's events, so we can see WHEN within the month an anomaly might
    occur. This is the plot Zhenkun specifically asked for.
  - Log-scale histogram with a candidate 95th-percentile threshold marked,
    to support the "95% healthy, 5% flagged as abnormal" discussion.

Produces:
   - per-sample reconstruction errors (CSV)
   - error histograms by split (PNG)
   - log-scale histogram with threshold (PNG)
   - error vs time plot (PNG)
   - example reconstructions (PNG)
   - per-channel error breakdown (CSV + PNG)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from architecture_1_v3 import AttentionAutoencoderV3

DATA_DIR = Path(__file__).resolve().parent / "per_channel_data_new_deck"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Threshold for "abnormal" -- percentile of the per-event total error
# distribution. Per Zhenkun: "maybe 95% of time bridge is healthy, check
# the rest 5%."
ANOMALY_PERCENTILE = 95


def load_normalized():
    strain = np.load(DATA_DIR / "strain_samples.npy")
    acc = np.load(DATA_DIR / "acc_samples.npy")
    temp = np.load(DATA_DIR / "temp_samples.npy")
    eids = np.load(DATA_DIR / "sample_event_ids.npy", allow_pickle=True)
    chans = np.load(DATA_DIR / "sample_channel_indices.npy")
    train_idx = np.load(DATA_DIR / "train_idx.npy")
    val_idx = np.load(DATA_DIR / "val_idx.npy")
    test_idx = np.load(DATA_DIR / "test_idx.npy")

    strain_scaler = joblib.load(DATA_DIR / "strain_scaler.joblib")
    acc_scaler = joblib.load(DATA_DIR / "acc_scaler.joblib")
    temp_scaler = joblib.load(DATA_DIR / "temp_scaler.joblib")

    s = strain_scaler.transform(strain).astype(np.float32)
    a = acc_scaler.transform(acc).astype(np.float32)
    t = temp_scaler.transform(temp).astype(np.float32)

    return {
        "strain": s, "acc": a, "temp": t,
        "sample_event_ids": eids, "sample_channel_indices": chans,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


def parse_event_timestamp(event_id: str):
    """Extract the start timestamp from an event_id string for time-ordering.

    Event ID format: AQUINAS_SET1_2022_07__OLD__2022-07-01T01-47-16Z__...
    We pull out the first ISO-like timestamp.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", str(event_id))
    if m:
        # Convert HH-MM-SS back to HH:MM:SS so numpy/pandas can parse it
        ts_str = m.group(1).replace("T", " ")
        date_part, time_part = ts_str.split(" ")
        time_part = time_part.replace("-", ":")
        return np.datetime64(f"{date_part}T{time_part}")
    return None


@torch.no_grad()
def per_sample_errors(model, strain, acc, temp, batch_size=256, device="cpu"):
    model.eval()
    N = strain.shape[0]
    s_err = np.zeros(N, dtype=np.float32)
    a_err = np.zeros(N, dtype=np.float32)
    t_err = np.zeros(N, dtype=np.float32)
    total = np.zeros(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        s = torch.from_numpy(strain[start:end]).to(device)
        a = torch.from_numpy(acc[start:end]).to(device)
        t = torch.from_numpy(temp[start:end]).to(device)
        out = model(s, a, t)
        s_err[start:end] = ((out["strain_hat"] - s) ** 2).mean(dim=1).cpu().numpy()
        a_err[start:end] = ((out["acc_hat"] - a) ** 2).mean(dim=1).cpu().numpy()
        t_err[start:end] = ((out["temperature_hat"] - t) ** 2).mean(dim=1).cpu().numpy()
        total[start:end] = s_err[start:end] + a_err[start:end] + 0.1 * t_err[start:end]
    return s_err, a_err, t_err, total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_normalized()
    print(f"Total samples: {data['strain'].shape[0]:,}")

    ckpt_path = CKPT_DIR / "architecture_1_v3_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = AttentionAutoencoderV3(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  epoch={ckpt['epoch']}  best_val_loss={ckpt['val_loss']:.4f}")

    s_err, a_err, t_err, total = per_sample_errors(
        model, data["strain"], data["acc"], data["temp"], device=device,
    )

    split_label = np.array(["?"] * len(s_err), dtype=object)
    split_label[data["train_idx"]] = "train"
    split_label[data["val_idx"]] = "val"
    split_label[data["test_idx"]] = "test"

    print("\nPer-split reconstruction error summary (mean ± std):")
    for label in ["train", "val", "test"]:
        m = split_label == label
        print(
            f"  {label:5s}  strain: {s_err[m].mean():.4f}±{s_err[m].std():.4f}   "
            f"acc: {a_err[m].mean():.4f}±{a_err[m].std():.4f}   "
            f"temp: {t_err[m].mean():.4f}±{t_err[m].std():.4f}   "
            f"total: {total[m].mean():.4f}±{total[m].std():.4f}"
        )

    chans = data["sample_channel_indices"]
    n_channels = int(chans.max()) + 1
    print(f"\nPer-channel reconstruction error (mean over all {len(s_err):,} samples):")
    for ch in range(n_channels):
        m = chans == ch
        print(
            f"  channel {ch}  n={m.sum():5d}  "
            f"strain: {s_err[m].mean():.4f}   "
            f"acc: {a_err[m].mean():.4f}   "
            f"total: {total[m].mean():.4f}"
        )

    # CSV
    out_csv = LOG_DIR / "reconstruction_errors_v3.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "event_id", "channel", "split",
                    "mse_strain", "mse_acc", "mse_temp", "total"])
        for i in range(len(s_err)):
            w.writerow([
                i, str(data["sample_event_ids"][i]), int(chans[i]),
                split_label[i], float(s_err[i]), float(a_err[i]),
                float(t_err[i]), float(total[i]),
            ])
    print(f"\nSaved per-sample errors to {out_csv}")

    # ------------------------------------------------------------------
    # Histograms (same as v2 but kept for continuity)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, arr, name in zip(
        axes, [s_err, a_err, t_err, total],
        ["MSE strain", "MSE acc", "MSE temperature", "Total loss"],
    ):
        for label, color in [("train", "C0"), ("val", "C1"), ("test", "C3")]:
            mask = split_label == label
            ax.hist(arr[mask], bins=60, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(name)
        ax.set_xlabel("error")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = LOG_DIR / "reconstruction_error_histograms_v3.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms to {fig_path}")

    # ------------------------------------------------------------------
    # NEW: Per-event aggregation (mean over channels) + chronological order
    # ------------------------------------------------------------------
    eids = data["sample_event_ids"]
    unique_events, first_idx = np.unique(eids, return_index=True)

    # Aggregate total error per event (mean over its 8 channel-samples)
    event_to_total = {}
    for ev in unique_events:
        m = eids == ev
        event_to_total[ev] = total[m].mean()

    # Parse timestamps and sort chronologically
    event_timestamps = {ev: parse_event_timestamp(ev) for ev in unique_events}
    valid_events = [ev for ev in unique_events if event_timestamps[ev] is not None]
    valid_events_sorted = sorted(valid_events, key=lambda ev: event_timestamps[ev])

    per_event_total_sorted = np.array([event_to_total[ev] for ev in valid_events_sorted])
    per_event_time_sorted = np.array([event_timestamps[ev] for ev in valid_events_sorted])

    print(f"\nParsed timestamps for {len(valid_events_sorted)}/{len(unique_events)} events")

    threshold = np.percentile(per_event_total_sorted, ANOMALY_PERCENTILE)
    n_above = int((per_event_total_sorted > threshold).sum())
    print(f"Threshold at {ANOMALY_PERCENTILE}th percentile: {threshold:.4f}")
    print(f"Events above threshold: {n_above} / {len(per_event_total_sorted)} "
          f"({100*n_above/len(per_event_total_sorted):.1f}%)")

    # ------------------------------------------------------------------
    # PLOT: error vs time (the plot Zhenkun requested)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = np.where(per_event_total_sorted > threshold, "red", "steelblue")
    ax.scatter(per_event_time_sorted, per_event_total_sorted, s=8, c=colors, alpha=0.6)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"{ANOMALY_PERCENTILE}th percentile threshold = {threshold:.3f}")
    ax.set_xlabel("Event time (SET1, July 2022)")
    ax.set_ylabel("Per-event total reconstruction error")
    ax.set_title(
        "Reconstruction error over time within SET1\n"
        f"Red points ({n_above}) exceed the {ANOMALY_PERCENTILE}th percentile threshold"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_path = LOG_DIR / "error_vs_time_v3.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved error-vs-time plot to {fig_path}")

    # ------------------------------------------------------------------
    # PLOT: log-scale histogram with threshold (the "log normal" plot)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per_event_total_sorted, bins=80, color="steelblue", alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"{ANOMALY_PERCENTILE}th percentile = {threshold:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Per-event total reconstruction error")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(
        "Per-event reconstruction error distribution (SET1)\n"
        f"{n_above} events ({100*n_above/len(per_event_total_sorted):.1f}%) "
        f"exceed the {ANOMALY_PERCENTILE}th percentile threshold"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = LOG_DIR / "error_histogram_log_threshold_v3.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved log-scale threshold histogram to {fig_path}")

    # Save per-event chronological CSV too
    out_csv2 = LOG_DIR / "per_event_errors_chronological_v3.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "timestamp", "total_error", "above_threshold"])
        for ev, t_val in zip(valid_events_sorted, per_event_total_sorted):
            w.writerow([ev, str(event_timestamps[ev]), float(t_val),
                       bool(t_val > threshold)])
    print(f"Saved chronological per-event errors to {out_csv2}")

    # ------------------------------------------------------------------
    # Per-channel boxplot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [total[chans == ch] for ch in range(n_channels)]
    ax.boxplot(box_data, tick_labels=[f"ch{i}" for i in range(n_channels)],
               showfliers=False)
    ax.set_title("Per-channel total reconstruction error (all samples)")
    ax.set_ylabel("total error")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = LOG_DIR / "per_channel_error_v3.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved per-channel boxplot to {fig_path}")

    # ------------------------------------------------------------------
    # Example reconstructions
    # ------------------------------------------------------------------
    test_indices = data["test_idx"][:3]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    with torch.no_grad():
        for row, sample_i in enumerate(test_indices):
            s = torch.from_numpy(data["strain"][sample_i:sample_i+1]).to(device)
            a = torch.from_numpy(data["acc"][sample_i:sample_i+1]).to(device)
            t = torch.from_numpy(data["temp"][sample_i:sample_i+1]).to(device)
            out = model(s, a, t)
            axes[row, 0].plot(s[0].cpu().numpy(), label="input")
            axes[row, 0].plot(out["strain_hat"][0].cpu().numpy(),
                              label="reconstruction", alpha=0.7)
            axes[row, 0].set_title(f"Sample {sample_i} (ch{chans[sample_i]}) — strain")
            axes[row, 0].set_xlabel("time sample")
            axes[row, 0].legend(); axes[row, 0].grid(alpha=0.3)
            axes[row, 1].plot(a[0].cpu().numpy(), label="input")
            axes[row, 1].plot(out["acc_hat"][0].cpu().numpy(),
                              label="reconstruction", alpha=0.7)
            axes[row, 1].set_title(f"Sample {sample_i} (ch{chans[sample_i]}) — ACC FFT")
            axes[row, 1].set_xlabel("frequency bin")
            axes[row, 1].legend(); axes[row, 1].grid(alpha=0.3)
    plt.tight_layout()
    fig_path = LOG_DIR / "example_reconstructions_v3.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved example reconstructions to {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
