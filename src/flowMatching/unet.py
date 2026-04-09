"""
1D U-Net for flow matching velocity prediction on bolometer pulse signals.

Architecture:
    Input: [x_t, x_tilde] concatenated -> 2 channels, 10000 samples
    Encoder: 4 levels (64 -> 128 -> 256 -> 512), stride-2 Conv1d
    Bottleneck: 512 channels with self-attention
    Decoder: mirror with skip connections
    Timestep: continuous t in [0,1] -> sinusoidal embedding -> MLP
    Output: 1 channel (predicted velocity v)

Usage:
    from src.flowMatching.unet import FlowUNet1D
    model = FlowUNet1D()
    v_pred = model(x_t, x_tilde, t)  # t: (B,) float in [0, 1]
"""

import math
import torch
import torch.nn as nn


class ContinuousTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) float tensor in [0, 1]

        Returns
        -------
        emb : (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class TimestepMLP(nn.Module):
    """Sinusoidal embedding -> MLP -> per-block conditioning vector."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.embed = ContinuousTimestepEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))  # (B, emb_dim)


class ResBlock1D(nn.Module):
    """Residual block: Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU + skip.

    Timestep conditioning is added after the first GroupNorm+SiLU.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 256, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

        # Timestep projection
        self.t_proj = nn.Linear(emb_dim, out_ch)

        # Residual connection (1x1 conv if channels change)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L)
        t_emb : (B, emb_dim)
        """
        h = self.act(self.norm1(self.conv1(x)))
        # Add timestep embedding
        h = h + self.t_proj(t_emb).unsqueeze(-1)  # (B, out_ch, 1) broadcast
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the sequence dimension."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, L) -> (B, C, L)"""
        B, C, L = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # (B, C, L)
        return x + h


class Downsample1D(nn.Module):
    """Stride-2 convolution for downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Nearest-neighbor upsample + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class FlowUNet1D(nn.Module):
    """1D U-Net for conditional flow matching velocity prediction.

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
        Timestep embedding dimension.
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
        channels = [base_channels * m for m in channel_mults]  # [64, 128, 256, 512]

        # Timestep embedding
        self.time_mlp = TimestepMLP(emb_dim)

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
                self.enc_downsamples.append(nn.Identity())  # no downsample at last level

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
            # Skip connection doubles input channels
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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, L) — interpolated state at time t
        x_tilde : (B, 1, L) — noisy observation (conditioning)
        t : (B,) — continuous time in [0, 1]

        Returns
        -------
        v_pred : (B, 1, L) — predicted velocity
        """
        # Concatenate input and condition
        x = torch.cat([x_t, x_tilde], dim=1)  # (B, 2, L)
        t_emb = self.time_mlp(t)  # (B, emb_dim)

        # Initial conv
        h = self.conv_in(x)

        # Encoder — collect skip connections
        skips = []
        for i in range(self.n_levels):
            for block in self.enc_blocks[i]:
                h = block(h, t_emb)
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < self.n_levels - 1:
                h = self.enc_downsamples[i](h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for j, i in enumerate(reversed(range(self.n_levels))):
            skip = skips[i]
            # Match lengths if needed (from downsampling rounding)
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)

            res1, res2 = self.dec_blocks[j]
            h = res1(h, t_emb)
            h = self.dec_attns[j](h)
            h = res2(h, t_emb)

            if i > 0:
                h = self.dec_upsamples[j](h)

        return self.conv_out(h)
