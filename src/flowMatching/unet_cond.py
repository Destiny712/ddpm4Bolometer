"""
1D U-Net with scale conditioning for flow matching velocity prediction.

Extends FlowUNet1D by adding a scale embedding: log(max|noisy|) is encoded
via sinusoidal embedding + MLP, then added to the timestep embedding so
every ResBlock sees both t and scale information.

Usage:
    from src.flowMatching.unet_cond import FlowUNet1DScaleCond
    model = FlowUNet1DScaleCond()
    v_pred = model(x_t, x_tilde, t, scale)
"""

import math
import torch
import torch.nn as nn

from src.flowMatching.unet import (
    ContinuousTimestepEmbedding, ResBlock1D, SelfAttention1D,
    Downsample1D, Upsample1D,
)


class ScaleMLP(nn.Module):
    """log(scale) -> sinusoidal embedding -> MLP -> conditioning vector."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.embed = ContinuousTimestepEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scale : (B,) — max(|noisy|) before normalization

        Returns
        -------
        emb : (B, emb_dim)
        """
        log_s = torch.log(scale.clamp(min=1e-8)) * 10.0
        return self.mlp(self.embed(log_s))


class FlowUNet1DScaleCond(nn.Module):
    """1D U-Net with timestep + scale conditioning for flow matching.

    Identical architecture to FlowUNet1D, but the ResBlocks receive
    t_emb + scale_emb instead of just t_emb.

    Parameters
    ----------
    in_channels : int
        Input channels (2 = x_t + x_tilde concatenated).
    out_channels : int
        Output channels (1 = predicted velocity v).
    base_channels : int
        Channels at level 0.
    channel_mults : tuple
        Channel multipliers per level.
    emb_dim : int
        Embedding dimension for both timestep and scale.
    attn_levels : tuple
        Which encoder levels get self-attention (0-indexed).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        emb_dim: int = 256,
        attn_levels: tuple = (),
    ):
        super().__init__()
        self.n_levels = len(channel_mults)
        channels = [base_channels * m for m in channel_mults]

        # Timestep embedding
        self.time_embed = ContinuousTimestepEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # Scale embedding
        self.scale_mlp = ScaleMLP(emb_dim)

        # Initial projection
        self.conv_in = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        # ---- Encoder ----
        self.enc_blocks = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        self.enc_attns = nn.ModuleList()

        for i in range(self.n_levels):
            ch_in = channels[i - 1] if i > 0 else channels[0]
            ch_out = channels[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim),
                ResBlock1D(ch_out, ch_out, emb_dim),
            ]))
            if i in attn_levels:
                self.enc_attns.append(SelfAttention1D(ch_out))
            else:
                self.enc_attns.append(nn.Identity())

            if i < self.n_levels - 1:
                self.enc_downsamples.append(Downsample1D(ch_out))
            else:
                self.enc_downsamples.append(nn.Identity())

        # ---- Bottleneck ----
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock1D(mid_ch, mid_ch, emb_dim)
        self.mid_attn = SelfAttention1D(mid_ch)
        self.mid_block2 = ResBlock1D(mid_ch, mid_ch, emb_dim)

        # ---- Decoder ----
        self.dec_blocks = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        for i in reversed(range(self.n_levels)):
            ch_out = channels[i]
            ch_in = ch_out + channels[i]  # concat with encoder skip

            if i > 0:
                ch_target = channels[i - 1]
            else:
                ch_target = channels[0]

            self.dec_blocks.append(nn.ModuleList([
                ResBlock1D(ch_in, ch_out, emb_dim),
                ResBlock1D(ch_out, ch_target, emb_dim),
            ]))

            if i in attn_levels:
                self.dec_attns.append(SelfAttention1D(ch_out))
            else:
                self.dec_attns.append(nn.Identity())

            if i > 0:
                self.dec_upsamples.append(Upsample1D(ch_target))
            else:
                self.dec_upsamples.append(nn.Identity())

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_tilde: torch.Tensor,
        t: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, L) — interpolated state at time t
        x_tilde : (B, 1, L) — noisy observation (conditioning)
        t : (B,) — continuous time in [0, 1]
        scale : (B,) — max(|noisy|) before normalization

        Returns
        -------
        v_pred : (B, 1, L) — predicted velocity
        """
        # Concatenate input and condition
        x = torch.cat([x_t, x_tilde], dim=1)  # (B, 2, L)

        # Combined embedding: timestep + scale
        t_emb = self.time_mlp(self.time_embed(t))  # (B, emb_dim)
        s_emb = self.scale_mlp(scale)               # (B, emb_dim)
        cond = t_emb + s_emb                         # (B, emb_dim)

        # Initial conv
        h = self.conv_in(x)

        # Encoder
        skips = []
        for i in range(self.n_levels):
            for block in self.enc_blocks[i]:
                h = block(h, cond)
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < self.n_levels - 1:
                h = self.enc_downsamples[i](h)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder
        for j, i in enumerate(reversed(range(self.n_levels))):
            skip = skips[i]
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)

            res1, res2 = self.dec_blocks[j]
            h = res1(h, cond)
            h = self.dec_attns[j](h)
            h = res2(h, cond)

            if i > 0:
                h = self.dec_upsamples[j](h)

        return self.conv_out(h)
