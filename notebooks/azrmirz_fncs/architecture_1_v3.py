"""
Architecture 1 v3 — deeper encoder/decoder, larger bottleneck.

Changes from v2, per supervisor (Zhenkun) feedback in the June 18 meeting:
  1. Bottleneck (latent) size increased from 16 -> 256.
     Rationale: with only 16 neurons the model could only learn the
     low-frequency / DC component of the signal and acted like a crude
     low-pass filter, failing to recover high-frequency structure (visible
     as over-smoothed ACC spectra and strain curves in v2 reconstructions).
  2. Encoder/decoder depth increased from 2+3 layers to 3+3 layers, with a
     GRADUAL taper instead of a sharp drop straight to 16. Going
     384 -> 512 -> 384 -> 256 (encoder) avoids forcing the network through
     an overly narrow channel in one step.
  3. Uses sklearn.preprocessing.StandardScaler explicitly for normalization
     (previously we replicated the same z-score formula manually; this is
     functionally identical but removes any ambiguity raised in the
     meeting).

Tokenization (kept identical to v2 -- this part worked well):
  Each (event, channel) sample has:
    - strain: 200 time-domain samples
    - acc: 820 FFT magnitude bins
    - temperature: 1 scalar
  These are tokenized into 6 attention tokens (1 strain + 4 acc bands + 1
  temperature), same as v2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Constants (same tokenization scheme as v2)
# ----------------------------------------------------------------------------

STRAIN_LEN = 200
ACC_LEN = 695   # NEW deck: 695 frequency bins (OLD deck was 820 -- different
                # "longest event" used for zero-padding between decks)
N_ACC_TOKENS = 4

# ACC_LEN must be exactly divisible by N_ACC_TOKENS for the reshape in
# ChannelTokenizer.forward() to work. 695 / 4 = 173.75, which does NOT
# divide evenly -- if unhandled, .view() throws a shape error. We pad
# ACC_LEN up to the nearest multiple of 4 with a few extra zero bins
# (appended at the high-frequency end) instead of dropping real data.
_ACC_LEN_PADDED = ((ACC_LEN + N_ACC_TOKENS - 1) // N_ACC_TOKENS) * N_ACC_TOKENS  # 696
ACC_PAD = _ACC_LEN_PADDED - ACC_LEN   # 1 extra zero bin appended
ACC_BAND_SIZE = _ACC_LEN_PADDED // N_ACC_TOKENS   # 174
N_TOKENS = 1 + N_ACC_TOKENS + 1            # 1 strain + 4 acc bands + 1 temp = 6


class ChannelTokenizer(nn.Module):
    """Turn one (strain, acc, temperature) sample into a sequence of 6 tokens.

    Identical to v2 -- this part of the design was not flagged as an issue.
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.strain_proj = nn.Linear(STRAIN_LEN, embed_dim)
        self.acc_proj = nn.Linear(ACC_BAND_SIZE, embed_dim)
        self.temp_proj = nn.Linear(1, embed_dim)
        self.type_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)

    def forward(
        self,
        strain: torch.Tensor,       # (B, 200)
        acc: torch.Tensor,          # (B, 820)
        temperature: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        strain_token = self.strain_proj(strain).unsqueeze(1)
        strain_token = strain_token + self.type_embed.weight[0]

        # Pad ACC with zeros so its length divides evenly into N_ACC_TOKENS
        # bands. ACC_PAD is 0 if ACC_LEN already divides evenly (e.g. OLD
        # deck's 820), so this is a no-op in that case.
        if ACC_PAD > 0:
            acc = nn.functional.pad(acc, (0, ACC_PAD))
        acc_bands = acc.view(acc.shape[0], N_ACC_TOKENS, ACC_BAND_SIZE)
        acc_tokens = self.acc_proj(acc_bands)
        acc_tokens = acc_tokens + self.type_embed.weight[1]

        temp_token = self.temp_proj(temperature).unsqueeze(1)
        temp_token = temp_token + self.type_embed.weight[2]

        tokens = torch.cat([strain_token, acc_tokens, temp_token], dim=1)
        return tokens


class AttentionEncoderV3(nn.Module):
    """4-head self-attention over the 6 tokens, then a 3-layer MLP encoder
    with a gradual taper down to the (now much larger) latent bottleneck.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        latent_dim: int = 256,
        n_tokens: int = N_TOKENS,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        flat_dim = n_tokens * embed_dim   # 6 * 64 = 384

        # Gradual taper: 384 -> 512 -> 384 -> latent_dim (256)
        # NOTE: we widen first (384->512) before narrowing, giving the
        # network more capacity to recombine token information before
        # compressing -- this avoids the "too sharp" bottleneck Zhenkun
        # flagged in v2 (384 -> 16 in one step).
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.GELU(),
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Linear(384, latent_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        x = self.attn_norm(tokens + attn_out)
        x = x.flatten(start_dim=1)
        z = self.mlp(x)
        return z


class DecoderV3(nn.Module):
    """3-layer decoder, mirroring the encoder's gradual taper in reverse."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        out_dim = STRAIN_LEN + ACC_LEN + 1   # 1021

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.GELU(),
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(z)
        strain_hat = out[:, :STRAIN_LEN]
        acc_hat = out[:, STRAIN_LEN: STRAIN_LEN + ACC_LEN]
        temp_hat = out[:, -1:]
        return strain_hat, acc_hat, temp_hat


class AttentionAutoencoderV3(nn.Module):
    """Per-channel attention autoencoder, v3: deeper + wider bottleneck."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = ChannelTokenizer(embed_dim=embed_dim)
        self.encoder = AttentionEncoderV3(
            embed_dim=embed_dim, num_heads=num_heads, latent_dim=latent_dim,
        )
        self.decoder = DecoderV3(latent_dim=latent_dim)

    def forward(
        self,
        strain: torch.Tensor,
        acc: torch.Tensor,
        temperature: torch.Tensor,
    ) -> dict:
        tokens = self.tokenizer(strain, acc, temperature)
        z = self.encoder(tokens)
        strain_hat, acc_hat, temp_hat = self.decoder(z)
        return {
            "latent": z,
            "strain_hat": strain_hat,
            "acc_hat": acc_hat,
            "temperature_hat": temp_hat,
        }


def reconstruction_loss(
    out: dict,
    strain: torch.Tensor,
    acc: torch.Tensor,
    temperature: torch.Tensor,
    w_strain: float = 1.0,
    w_acc: float = 1.0,
    w_temp: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    mse_strain = nn.functional.mse_loss(out["strain_hat"], strain)
    mse_acc = nn.functional.mse_loss(out["acc_hat"], acc)
    mse_temp = nn.functional.mse_loss(out["temperature_hat"], temperature)
    total = w_strain * mse_strain + w_acc * mse_acc + w_temp * mse_temp
    return total, {
        "loss_total": total.item(),
        "mse_strain": mse_strain.item(),
        "mse_acc": mse_acc.item(),
        "mse_temp": mse_temp.item(),
    }


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    model = AttentionAutoencoderV3(latent_dim=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Print per-layer hidden sizes for the report/slides
    print("\nEncoder MLP:")
    for layer in model.encoder.mlp:
        if isinstance(layer, nn.Linear):
            print(f"  Linear({layer.in_features} -> {layer.out_features})")
    print("Decoder MLP:")
    for layer in model.decoder.net:
        if isinstance(layer, nn.Linear):
            print(f"  Linear({layer.in_features} -> {layer.out_features})")

    B = 16
    strain = torch.randn(B, STRAIN_LEN)
    acc = torch.relu(torch.randn(B, ACC_LEN))
    temp = torch.randn(B, 1)

    out = model(strain, acc, temp)
    print("\nShape check:")
    print(f"  latent          {out['latent'].shape}")
    print(f"  strain_hat      {out['strain_hat'].shape}    (target: {strain.shape})")
    print(f"  acc_hat         {out['acc_hat'].shape}   (target: {acc.shape})")
    print(f"  temperature_hat {out['temperature_hat'].shape}        (target: {temp.shape})")
    loss, parts = reconstruction_loss(out, strain, acc, temp)
    print(f"  loss            {loss.item():.4f}")