"""
Cross-SET inference for Architecture 1 v3 on NEW deck data.

We trained the v3 model exclusively on SET1 (July 2022) events. This script
applies the trained model to ALL other SETs present in the same NEW-deck
NPY files, using the SET1-fitted normalization (sklearn StandardScaler
objects saved during training) -- the model and the normalization must
both stay anchored to "what July 2022 looked like" for the comparison
across months to be meaningful.

Unlike earlier scripts, this version does NOT hardcode which SETs exist.
It auto-discovers every distinct SET prefix present in event_ids.npy and
reports on all of them. This avoids silently missing SET4/SET5 if they
are present in this particular Murat upload.

Procedure per SET:
  1. Filter events belonging to that SET.
  2. Explode into per-channel samples (event x 8 channels).
  3. Normalize using the SET1 training scalers (NOT re-fit on this SET).
  4. Run inference with the trained v3 model.
  5. Aggregate per-channel errors into one error per event (mean over the
     8 channel-samples belonging to that event).
  6. Report per-SET summary statistics and produce comparison plots.
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

# ----------------------------------------------------------------------------
# Paths -- EDIT IF NEEDED to match your folder layout
# ----------------------------------------------------------------------------

# Directory containing the raw NEW-deck NPY tensors (strain_inputs.npy etc.)
# -- this is the SAME folder used by prepare_per_channel_samples_new_deck.py
RAW_NN_DIR = Path(__file__).resolve().parent  # adjust if scripts run elsewhere

# Directory containing the SET1-only per-channel data + fitted scalers,
# produced by prepare_per_channel_samples_new_deck.py
DATA_DIR = Path(__file__).resolve().parent / "per_channel_data_new_deck"

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Known SET name -> (short label, human description, plot color)
# Extend this if new SETs appear that aren't listed yet -- the script will
# still work for unknown SETs, just with a generic grey color and the raw
# prefix as the label.
SET_METADATA = {
    "AQUINAS_SET1_2022_07": ("SET1", "July 2022",   "#d62728"),
    "AQUINAS_SET2_2023_04": ("SET2", "April 2023",  "#2ca02c"),
    "AQUINAS_SET3_2023_08": ("SET3", "August 2023", "#ff7f0e"),
    "AQUINAS_SET4_2024_01": ("SET4", "January 2024","#1f77b4"),
    "AQUINAS_SET5_2024_06": ("SET5", "June 2024",   "#9467bd"),
}
FALLBACK_COLOR = "#7f7f7f"


def get_set_meta(prefix: str):
    if prefix in SET_METADATA:
        return SET_METADATA[prefix]
    return (prefix, prefix, FALLBACK_COLOR)


# ----------------------------------------------------------------------------
# Step 1: Load raw tensors + discover which SETs are present
# ----------------------------------------------------------------------------


def load_raw_and_discover_sets():
    print("Loading raw NEW-deck NPY tensors...")
    strain = np.load(RAW_NN_DIR / "strain_inputs.npy")
    acc = np.load(RAW_NN_DIR / "acc_inputs.npy")
    temp = np.load(RAW_NN_DIR / "temperature_inputs.npy")
    eids = np.load(RAW_NN_DIR / "event_ids.npy", allow_pickle=True)
    print(f"  strain   {strain.shape}")
    print(f"  acc      {acc.shape}")
    print(f"  temp     {temp.shape}")
    print(f"  eids     {eids.shape}")

    n_channels = strain.shape[-1]
    print(f"  channels: {n_channels}")
    if n_channels not in (8,):
        print(f"  WARNING: expected 8 channels, found {n_channels}. "
              f"Check whether the duplication bug is present in this file.")

    # Discover SET prefixes by pattern AQUINAS_SETx_YYYY_MM
    prefixes = set()
    pattern = re.compile(r"(AQUINAS_SET\d+_\d{4}_\d{2})")
    for eid in eids:
        m = pattern.match(str(eid))
        if m:
            prefixes.add(m.group(1))
    prefixes = sorted(prefixes)
    print(f"\n  Discovered SET prefixes: {prefixes}")

    counts = {}
    for p in prefixes:
        counts[p] = sum(1 for e in eids if str(e).startswith(p))
    print("  Event counts per SET:")
    for p, c in counts.items():
        label, desc, _ = get_set_meta(p)
        print(f"    {p}  ({label}, {desc}): {c} events")

    return {
        "strain": strain, "acc": acc, "temp": temp, "eids": eids,
        "set_prefixes": prefixes,
    }


# ----------------------------------------------------------------------------
# Step 2: Load SET1-fitted scalers + trained model
# ----------------------------------------------------------------------------


def load_scalers_and_model(device):
    strain_scaler = joblib.load(DATA_DIR / "strain_scaler.joblib")
    acc_scaler = joblib.load(DATA_DIR / "acc_scaler.joblib")
    temp_scaler = joblib.load(DATA_DIR / "temp_scaler.joblib")

    ckpt_path = CKPT_DIR / "architecture_1_v3_best.pt"
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = AttentionAutoencoderV3(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  epoch={ckpt['epoch']}  best_val_loss={ckpt['val_loss']:.4f}")

    return {
        "strain": strain_scaler, "acc": acc_scaler, "temp": temp_scaler,
    }, model


# ----------------------------------------------------------------------------
# Step 3: Process one SET
# ----------------------------------------------------------------------------


def process_set(set_prefix, raw, scalers, model, device, batch_size=256):
    eids = raw["eids"]
    mask = np.array([str(e).startswith(set_prefix) for e in eids])
    n_events = int(mask.sum())
    if n_events == 0:
        return None

    s_set = raw["strain"][mask]   # (n_events, 200, 8)
    a_set = raw["acc"][mask]      # (n_events, 695, 8)
    t_set = raw["temp"][mask]     # (n_events, 1)
    eids_set = eids[mask]
    n_channels = s_set.shape[-1]

    strain_samples = s_set.transpose(0, 2, 1).reshape(-1, s_set.shape[1])
    acc_samples = a_set.transpose(0, 2, 1).reshape(-1, a_set.shape[1])
    temp_samples = (
        np.broadcast_to(t_set, (n_events, n_channels))
        .reshape(-1, 1).astype(np.float32)
    )

    # Normalize using SET1 scalers (fixed reference, NOT re-fit per SET)
    strain_n = scalers["strain"].transform(strain_samples).astype(np.float32)
    acc_n = scalers["acc"].transform(acc_samples).astype(np.float32)
    temp_n = scalers["temp"].transform(temp_samples).astype(np.float32)

    n_samples = strain_n.shape[0]
    s_err = np.zeros(n_samples, dtype=np.float32)
    a_err = np.zeros(n_samples, dtype=np.float32)
    t_err = np.zeros(n_samples, dtype=np.float32)
    total = np.zeros(n_samples, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            s = torch.from_numpy(strain_n[start:end]).to(device)
            a = torch.from_numpy(acc_n[start:end]).to(device)
            t = torch.from_numpy(temp_n[start:end]).to(device)
            out = model(s, a, t)
            s_err[start:end] = ((out["strain_hat"] - s) ** 2).mean(dim=1).cpu().numpy()
            a_err[start:end] = ((out["acc_hat"] - a) ** 2).mean(dim=1).cpu().numpy()
            t_err[start:end] = ((out["temperature_hat"] - t) ** 2).mean(dim=1).cpu().numpy()
            total[start:end] = s_err[start:end] + a_err[start:end] + 0.1 * t_err[start:end]

    per_event_total = total.reshape(n_events, n_channels).mean(axis=1)
    per_event_strain = s_err.reshape(n_events, n_channels).mean(axis=1)
    per_event_acc = a_err.reshape(n_events, n_channels).mean(axis=1)
    per_event_temp_in = t_set[:, 0]

    label, desc, _ = get_set_meta(set_prefix)
    print(f"\n  {label} ({desc}):")
    print(f"    events:                {n_events}")
    print(f"    per-event total error: median={np.median(per_event_total):.4f}  "
          f"mean={per_event_total.mean():.4f}  std={per_event_total.std():.4f}")
    print(f"    temperature range:     [{per_event_temp_in.min():.1f}, "
          f"{per_event_temp_in.max():.1f}] °C")

    return {
        "set_prefix": set_prefix,
        "n_events": n_events,
        "per_event_total": per_event_total,
        "per_event_strain": per_event_strain,
        "per_event_acc": per_event_acc,
        "event_temperatures": per_event_temp_in,
        "event_ids": eids_set,
    }


# ----------------------------------------------------------------------------
# Step 4: Plots
# ----------------------------------------------------------------------------


def make_plots(results: dict, threshold: float):
    available = list(results.keys())

    # ---- Histogram ----
    # Clip the x-axis range for readability -- a small number of extreme
    # outliers (e.g. error > 5) compress the rest of the distribution into
    # an unreadable spike if shown on a linear, unclipped axis. We still
    # report the true min/max in the printed summary; this plot trades a
    # little tail visibility for a readable comparison of the bulk shape.
    all_errors = np.concatenate([results[k]["per_event_total"] for k in available])
    clip_max = float(np.percentile(all_errors, 99.5))

    fig, ax = plt.subplots(figsize=(10, 5))
    for set_prefix in available:
        label, desc, color = get_set_meta(set_prefix)
        per_event = results[set_prefix]["per_event_total"]
        n_clipped = int((per_event > clip_max).sum())
        ax.hist(np.clip(per_event, None, clip_max), bins=60, alpha=0.5,
                label=f"{label} — {desc} (n={len(per_event)}, {n_clipped} clipped)",
                color=color, density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"SET1 95th pct threshold = {threshold:.3f}")
    ax.set_xlim(0, clip_max)
    ax.set_xlabel(f"Per-event total reconstruction error (clipped at 99.5th pct = {clip_max:.2f})")
    ax.set_ylabel("density")
    ax.set_title("Cross-SET reconstruction error\n"
                 "(Model + threshold fixed from SET1 training only; "
                 "x-axis clipped to show bulk shape)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = LOG_DIR / "cross_set_per_event_histogram_v3.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # ---- Boxplot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [results[k]["per_event_total"] for k in available]
    box_labels = [f"{get_set_meta(k)[0]}\n{get_set_meta(k)[1]}" for k in available]
    box_colors = [get_set_meta(k)[2] for k in available]
    bplot = ax.boxplot(box_data, tick_labels=box_labels, showfliers=False,
                       patch_artist=True)
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"SET1 95th pct threshold = {threshold:.3f}")
    ax.set_ylabel("Per-event total reconstruction error")
    ax.set_title("Per-SET error distribution\n"
                 "(Model trained on SET1 only; threshold fixed from SET1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = LOG_DIR / "cross_set_per_event_boxplot_v3.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ---- Error vs temperature ----
    # Same clipping rationale as the histogram above.
    all_errors_for_clip = np.concatenate([results[k]["per_event_total"] for k in available])
    y_clip_max = float(np.percentile(all_errors_for_clip, 99.5))

    fig, ax = plt.subplots(figsize=(10, 5))
    for set_prefix in available:
        label, desc, color = get_set_meta(set_prefix)
        T = results[set_prefix]["event_temperatures"]
        E = results[set_prefix]["per_event_total"]
        ax.scatter(T, E, s=6, alpha=0.5, c=color, label=f"{label} — {desc}")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"threshold = {threshold:.3f}")
    ax.set_ylim(0, y_clip_max)
    ax.set_xlabel("Event temperature (°C)")
    ax.set_ylabel(f"Per-event total reconstruction error (y-axis clipped at {y_clip_max:.2f})")
    ax.set_title("Reconstruction error vs temperature, across SETs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = LOG_DIR / "cross_set_error_vs_temperature_v3.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ---- Summary table figure ----
    fig, ax = plt.subplots(figsize=(12, 1 + 0.4 * len(available)))
    ax.axis("off")
    rows = []
    for set_prefix in available:
        label, desc, _ = get_set_meta(set_prefix)
        r = results[set_prefix]
        pct_above = 100 * (r["per_event_total"] > threshold).mean()
        rows.append([
            label, desc, str(r["n_events"]),
            f"{r['per_event_total'].mean():.4f}",
            f"{np.median(r['per_event_total']):.4f}",
            f"{r['per_event_total'].std():.4f}",
            f"{pct_above:.1f}%",
            f"{r['event_temperatures'].mean():.1f}",
        ])
    column_labels = ["SET", "Period", "Events", "Mean", "Median", "Std",
                     "% above threshold", "Avg T (°C)"]
    table = ax.table(cellText=rows, colLabels=column_labels, cellLoc="center",
                     loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Cross-SET reconstruction error summary "
                 "(SET1-trained model, SET1 threshold)", fontsize=11, pad=10)
    plt.tight_layout()
    out = LOG_DIR / "cross_set_summary_table_v3.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def save_csv(results: dict):
    out_csv = LOG_DIR / "cross_set_per_event_errors_v3.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "event_id", "temperature_C", "mean_strain_mse",
                    "mean_acc_mse", "total_error"])
        for set_prefix, r in results.items():
            label, _, _ = get_set_meta(set_prefix)
            for i in range(r["n_events"]):
                w.writerow([
                    label, str(r["event_ids"][i]),
                    float(r["event_temperatures"][i]),
                    float(r["per_event_strain"][i]),
                    float(r["per_event_acc"][i]),
                    float(r["per_event_total"][i]),
                ])
    print(f"Saved per-event CSV to {out_csv}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Cross-SET inference: v3 model trained on SET1, evaluated on all SETs")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    raw = load_raw_and_discover_sets()
    scalers, model = load_scalers_and_model(device)

    # Determine SET1's threshold by re-deriving it the same way evaluate
    # script did: 95th percentile of SET1's own per-event total error.
    set1_prefix = next((p for p in raw["set_prefixes"] if "SET1" in p), None)
    if set1_prefix is None:
        raise RuntimeError("Could not find a SET1 prefix in this dataset's event_ids.")

    results = {}
    for set_prefix in raw["set_prefixes"]:
        r = process_set(set_prefix, raw, scalers, model, device)
        if r:
            results[set_prefix] = r

    threshold = float(np.percentile(results[set1_prefix]["per_event_total"], 95))
    print(f"\nSET1-derived 95th-percentile threshold: {threshold:.4f}")

    print("\nGenerating plots...")
    make_plots(results, threshold)
    save_csv(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'SET':6s}  {'n_events':>9s}  {'mean':>8s}  {'median':>8s}  "
          f"{'std':>8s}  {'%>thresh':>9s}  {'avg_T':>7s}")
    for set_prefix, r in results.items():
        label, _, _ = get_set_meta(set_prefix)
        pct_above = 100 * (r["per_event_total"] > threshold).mean()
        print(f"{label:6s}  {r['n_events']:>9d}  "
              f"{r['per_event_total'].mean():>8.4f}  "
              f"{np.median(r['per_event_total']):>8.4f}  "
              f"{r['per_event_total'].std():>8.4f}  "
              f"{pct_above:>8.1f}%  "
              f"{r['event_temperatures'].mean():>7.1f}")

    # Relative comparison vs SET1
    if set1_prefix in results:
        s1_med = np.median(results[set1_prefix]["per_event_total"])
        print(f"\nRelative to SET1 baseline (median = {s1_med:.4f}):")
        for set_prefix, r in results.items():
            if set_prefix == set1_prefix:
                continue
            label, _, _ = get_set_meta(set_prefix)
            med = np.median(r["per_event_total"])
            print(f"  {label}: {(med - s1_med) / s1_med * 100:+.1f}%")

    # Flag the most extreme outlier events across ALL sets, for manual
    # investigation -- these are the events worth pulling up individually
    # (could be genuine anomalies, or data artifacts worth checking).
    print("\n" + "=" * 70)
    print("TOP 10 MOST EXTREME EVENTS (across all SETs)")
    print("=" * 70)
    all_records = []
    for set_prefix, r in results.items():
        label, _, _ = get_set_meta(set_prefix)
        for i in range(r["n_events"]):
            all_records.append((
                float(r["per_event_total"][i]), label,
                str(r["event_ids"][i]), float(r["event_temperatures"][i]),
            ))
    all_records.sort(key=lambda x: -x[0])
    print(f"{'error':>10s}  {'SET':6s}  {'temp':>6s}  event_id")
    for err, label, eid, t_val in all_records[:10]:
        print(f"{err:>10.4f}  {label:6s}  {t_val:>5.1f}°C  {eid}")

    print("\nDone.")


if __name__ == "__main__":
    main()