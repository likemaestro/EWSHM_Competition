import numpy as np
import torch
import torch.nn as nn
import time

def reconstruction_error(
    original,
    reconstruction
):

    return (
        (original - reconstruction)**2
    ).mean(
        dim=(1,2)
    )

def train_epoch(
    model,
    loader,
    optimizer,
    device="cuda"
):

    model.train()

    total_loss = 0.0

    for batch in loader:

        acc = batch["acc"].to(device)

        strain = batch["strain"].to(device)

        optimizer.zero_grad()

        outputs = model(
            acc,
            strain
        )

        if outputs["type"] == "single":

            x = torch.cat(
                [acc, strain],
                dim=1
            )

            loss = (

                (x - outputs["reconstruction"])**2

            ).mean()

        else:

            loss_acc = (

                (acc - outputs[
                    "acc_reconstruction"
                ])**2

            ).mean()

            loss_strain = (

                (strain - outputs[
                    "strain_reconstruction"
                ])**2

            ).mean()

            loss = (

                0.5 * loss_acc

                +

                0.5 * loss_strain
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += (

            loss.item()

            *

            acc.size(0)
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    device="cuda"
):

    model.eval()

    total_loss = 0.0

    for batch in loader:

        acc = batch["acc"].to(device)

        strain = batch["strain"].to(device)

        outputs = model(
            acc,
            strain
        )

        if outputs["type"] == "single":

            x = torch.cat(
                [acc, strain],
                dim=1
            )

            loss = (

                (x - outputs[
                    "reconstruction"
                ])**2

            ).mean()

        else:

            loss_acc = (

                (acc - outputs[
                    "acc_reconstruction"
                ])**2

            ).mean()

            loss_strain = (

                (strain - outputs[
                    "strain_reconstruction"
                ])**2

            ).mean()

            loss = (

                0.5 * loss_acc

                +

                0.5 * loss_strain
            )

        total_loss += (

            loss.item()

            *

            acc.size(0)
        )

    return total_loss / len(loader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-3,
    device="cuda"
):

    model.to(device)

    t0 = time.time()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss = validate_epoch(
            model,
            val_loader,
            device
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        elapsed = time.time() - t0

        print(

            f"Epoch {epoch+1}/{epochs}"

            f" | train={train_loss:.6f}"

            f" | val={val_loss:.6f}"

            f" | time={elapsed:.2f}s"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(

                model.state_dict(),

                f"{model.__class__.__name__}.pt"
            )

    print("done")

    return train_losses, val_losses