import torch
import torch.nn as nn
import numpy as np

class SequenceAttentionAE(nn.Module):

    def __init__(
        self,
        acc_seq=695,
        strain_seq=200,
        channels=40,

        heads=4,
        latent_dim=128
    ):
        super().__init__()

        self.acc_seq = acc_seq
        self.strain_seq = strain_seq
        self.channels = channels

        #Attention block

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(
            channels
        )

        flattened = (acc_seq + strain_seq) * channels

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(flattened, latent_dim*16),
            nn.ReLU(),
            nn.Linear(latent_dim*16, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim*16),
            nn.ReLU(),
            nn.Linear(
                latent_dim*16,
                flattened
            )
        )

    def forward(self, acc, strain):
        """
        x:
        [batch,895,40]
        """

        x = torch.cat(
            [acc, strain],
            dim=1
        )

        attn_out, weights = self.attn(
            x,
            x,
            x
        )

        x = self.norm(
            x + attn_out
        )

        batch = x.shape[0]

        flat = x.reshape(
            batch,
            -1
        )

        latent = self.encoder(
            flat
        )

        recon = self.decoder(
            latent
        )

        recon = recon.view(
            batch,
            (self.acc_seq + self.strain_seq),
            self.channels
        )

        return {

            "reconstruction": recon,

            "latent": latent,

            "attention": weights,

            "type": "single"
        }
    


class DualBranchFusionAE(nn.Module):

    def __init__(
        self,
        acc_seq=695,
        strain_seq=200,
        channels=40,

        acc_latent=64,
        strain_latent=32,

        fused_latent=64,

        heads=4
    ):
        super().__init__()

        self.channels = channels
        self.acc_seq = acc_seq
        self.strain_seq = strain_seq

        # -------------------
        # Attention blocks
        # -------------------

        self.acc_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.strain_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.norm_acc = nn.LayerNorm(
            channels
        )

        self.norm_strain = nn.LayerNorm(
            channels
        )

        # -------------------
        # Encoders
        # -------------------

        acc_flat = acc_seq * channels
        strain_flat = strain_seq * channels

        self.acc_encoder = nn.Sequential(

            nn.Linear(
                acc_flat,
                acc_latent * 8
            ),

            nn.ReLU(),

            nn.Linear(
                acc_latent * 8,
                acc_latent
            )
        )

        self.strain_encoder = nn.Sequential(

            nn.Linear(
                strain_flat,
                strain_latent * 8
            ),

            nn.ReLU(),

            nn.Linear(
                strain_latent * 8,
                strain_latent
            )
        )

        # -------------------
        # Fusion bottleneck
        # -------------------

        fusion_in = (
            acc_latent
            + strain_latent
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_in,
                fused_latent
            ),

            nn.ReLU()
        )

        # split back

        self.unfusion = nn.Sequential(

            nn.Linear(
                fused_latent,
                fusion_in
            ),

            nn.ReLU()
        )

        # -------------------
        # Decoders
        # -------------------

        self.acc_decoder = nn.Sequential(

            nn.Linear(
                acc_latent,
                acc_latent * 8
            ),

            nn.ReLU(),

            nn.Linear(
                acc_latent * 8,
                acc_flat
            )
        )

        self.strain_decoder = nn.Sequential(

            nn.Linear(
                strain_latent,
                strain_latent * 8
            ),

            nn.ReLU(),

            nn.Linear(
                strain_latent * 8,
                strain_flat
            )
        )

        self.acc_latent = acc_latent
        self.strain_latent = strain_latent

    def forward(
        self,
        acc,
        strain
    ):

        batch = acc.shape[0]

        # ------------------
        # Attention branches
        # ------------------

        acc_attn, acc_w = self.acc_attn(
            acc,
            acc,
            acc
        )

        strain_attn, strain_w = self.strain_attn(
            strain,
            strain,
            strain
        )

        acc = self.norm_acc(
            acc + acc_attn
        )

        strain = self.norm_strain(
            strain + strain_attn
        )

        # flatten

        acc_flat = acc.reshape(
            batch,
            -1
        )

        strain_flat = strain.reshape(
            batch,
            -1
        )

        # modality latents

        z_acc = self.acc_encoder(
            acc_flat
        )

        z_strain = self.strain_encoder(
            strain_flat
        )

        # fuse

        fused = torch.cat(
            [z_acc, z_strain],
            dim=1
        )

        z_shared = self.fusion(
            fused
        )

        split = self.unfusion(
            z_shared
        )

        z_acc_dec = split[
            :,
            :self.acc_latent
        ]

        z_strain_dec = split[
            :,
            self.acc_latent:
        ]

        # decode

        acc_recon = self.acc_decoder(
            z_acc_dec
        )

        strain_recon = self.strain_decoder(
            z_strain_dec
        )

        acc_recon = acc_recon.view(
            batch,
            self.acc_seq,
            self.channels
        )

        strain_recon = strain_recon.view(
            batch,
            self.strain_seq,
            self.channels
        )

        return {

            "acc_reconstruction": acc_recon,

            "strain_reconstruction": strain_recon,

            "z_acc": z_acc,

            "z_strain": z_strain,

            "latent": z_shared,

            "type": "dual"
        }