"""
Train Architecture 1 v3 on per-channel NEW deck SET1 samples.

Changes from v2 training script:
  - Uses AttentionAutoencoderV3 (bigger bottleneck=256, deeper 3+3 layers)
  - Uses the sklearn StandardScaler objects saved by
    prepare_per_channel_samples_new_deck.py (instead of manual z-score)
  - Trains on NEW deck data instead of OLD deck

Inputs (produced by prepare_per_channel_samples_new_deck.py):
   per_channel_data_new_deck/strain_samples.npy   (n_samples, 200)
   per_channel_data_new_deck/acc_samples.npy      (n_samples, 820)
   per_channel_data_new_deck/temp_samples.npy     (n_samples, 1)
   per_channel_data_new_deck/{train,val,test}_idx.npy
   per_channel_data_new_deck/{strain,acc,temp}_scaler.joblib

Outputs:
   checkpoints/architecture_1_v3_best.pt
   checkpoints/architecture_1_v3_final.pt
   logs/training_log_v3.csv
   logs/training_curves_v3.png
   logs/test_metrics_v3.json
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from architecture_1_v3 import AttentionAutoencoderV3, reconstruction_loss

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOG_EVERY_N_BATCHES = 100

W_STRAIN = 1.0
W_ACC = 1.0
W_TEMP = 0.1

LATENT_DIM = 256   # per supervisor's request

DATA_DIR = Path(__file__).resolve().parent / "per_channel_data_new_deck"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    strain = np.load(DATA_DIR / "strain_samples.npy")
    acc = np.load(DATA_DIR / "acc_samples.npy")
    temp = np.load(DATA_DIR / "temp_samples.npy")
    train_idx = np.load(DATA_DIR / "train_idx.npy")
    val_idx = np.load(DATA_DIR / "val_idx.npy")
    test_idx = np.load(DATA_DIR / "test_idx.npy")

    strain_scaler = joblib.load(DATA_DIR / "strain_scaler.joblib")
    acc_scaler = joblib.load(DATA_DIR / "acc_scaler.joblib")
    temp_scaler = joblib.load(DATA_DIR / "temp_scaler.joblib")

    return {
        "strain": strain, "acc": acc, "temp": temp,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "scalers": {
            "strain": strain_scaler, "acc": acc_scaler, "temp": temp_scaler,
        },
    }


def normalize(strain, acc, temp, scalers):
    s = scalers["strain"].transform(strain).astype(np.float32)
    a = scalers["acc"].transform(acc).astype(np.float32)
    t = scalers["temp"].transform(temp).astype(np.float32)
    return s, a, t


def make_loader(strain, acc, temp, indices, batch_size, shuffle):
    s = torch.from_numpy(strain[indices])
    a = torch.from_numpy(acc[indices])
    t = torch.from_numpy(temp[indices])
    ds = TensorDataset(s, a, t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    sums = {"loss_total": 0.0, "mse_strain": 0.0, "mse_acc": 0.0, "mse_temp": 0.0}
    n = 0
    for batch_idx, (s, a, t) in enumerate(loader, start=1):
        s, a, t = s.to(device), a.to(device), t.to(device)
        optimizer.zero_grad()
        out = model(s, a, t)
        loss, parts = reconstruction_loss(
            out, s, a, t, w_strain=W_STRAIN, w_acc=W_ACC, w_temp=W_TEMP,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = s.shape[0]
        for k in sums:
            sums[k] += parts[k] * bs
        n += bs
        if batch_idx % LOG_EVERY_N_BATCHES == 0:
            print(f"    batch {batch_idx:4d}/{len(loader)}  loss={parts['loss_total']:.4f}")
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {"loss_total": 0.0, "mse_strain": 0.0, "mse_acc": 0.0, "mse_temp": 0.0}
    n = 0
    for s, a, t in loader:
        s, a, t = s.to(device), a.to(device), t.to(device)
        out = model(s, a, t)
        _, parts = reconstruction_loss(
            out, s, a, t, w_strain=W_STRAIN, w_acc=W_ACC, w_temp=W_TEMP,
        )
        bs = s.shape[0]
        for k in sums:
            sums[k] += parts[k] * bs
        n += bs
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    print("=" * 70)
    print("Architecture 1 v3: training on NEW deck (deeper net, bottleneck=256)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(2026)
    np.random.seed(2026)

    print("\nLoading per-channel samples...")
    data = load_data()
    print(f"  strain {data['strain'].shape}  acc {data['acc'].shape}  temp {data['temp'].shape}")
    print(f"  splits: train={len(data['train_idx'])}  val={len(data['val_idx'])}  test={len(data['test_idx'])}")

    print("\nNormalizing with sklearn StandardScaler...")
    s, a, t = normalize(data["strain"], data["acc"], data["temp"], data["scalers"])

    train_loader = make_loader(s, a, t, data["train_idx"], BATCH_SIZE, shuffle=True)
    val_loader = make_loader(s, a, t, data["val_idx"], BATCH_SIZE, shuffle=False)
    test_loader = make_loader(s, a, t, data["test_idx"], BATCH_SIZE, shuffle=False)
    print(f"  batches per epoch: train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

    print("\nInstantiating model...")
    model = AttentionAutoencoderV3(
        embed_dim=64, num_heads=4, latent_dim=LATENT_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    config = {"embed_dim": 64, "num_heads": 4, "latent_dim": LATENT_DIM}

    print("\nStarting training...")
    log_rows = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        ep_t = time.time() - ep_t0
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_metrics['loss_total']:.4f}  "
            f"val_loss={val_metrics['loss_total']:.4f}  "
            f"({ep_t:.1f}s/epoch  total {elapsed/60:.1f}min)"
        )

        log_rows.append({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": train_metrics[k] for k in train_metrics},
            **{f"val_{k}": val_metrics[k] for k in val_metrics},
        })

        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "config": config,
            }, CKPT_DIR / "architecture_1_v3_best.pt")

    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "val_loss": val_metrics["loss_total"],
        "config": config,
    }, CKPT_DIR / "architecture_1_v3_final.pt")

    print("\nEvaluating on test set with best checkpoint...")
    best = torch.load(CKPT_DIR / "architecture_1_v3_best.pt", weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"  test_loss = {test_metrics['loss_total']:.4f}  (best val = {best_val:.4f})")

    log_path = LOG_DIR / "training_log_v3.csv"
    with open(log_path, "w", newline="") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    print(f"Saved training log to {log_path}")

    with open(LOG_DIR / "test_metrics_v3.json", "w") as f:
        json.dump({"best_val_loss": best_val, **test_metrics}, f, indent=2)

    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        ("loss_total", "Total loss"),
        ("mse_strain", "MSE strain"),
        ("mse_acc", "MSE acc"),
        ("mse_temp", "MSE temperature"),
    ]
    for ax, (key, title) in zip(axes.flat, pairs):
        ax.plot(epochs, [r[f"train_{key}"] for r in log_rows], label="train")
        ax.plot(epochs, [r[f"val_{key}"] for r in log_rows], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Architecture 1 v3 training curves (NEW deck, bottleneck=256)", y=1.01)
    plt.tight_layout()
    out_png = LOG_DIR / "training_curves_v3.png"
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Saved training curves to {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
