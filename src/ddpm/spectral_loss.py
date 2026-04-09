"""
Spectral loss functions for DDPM training.

Three frequency-domain losses to complement L1/L2:

    SC   — Spectral Convergence (multi-resolution STFT)
    LSD  — Log-Spectral Distance
    J_asd — Amplitude Spectral Density ratio (DeepClean/LIGO-style)

Usage:
    from src.ddpm.spectral_loss import SpectralLoss
    spec_loss = SpectralLoss(w_l1=1.0, w_sc=0.1, w_lsd=0.0, w_asd=0.1)
    loss = spec_loss(x_pred, x_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Combined time-domain + spectral loss with configurable weights.

    Parameters
    ----------
    w_l1 : float
        Weight for L1 loss (time domain).
    w_sc : float
        Weight for spectral convergence loss.
    w_lsd : float
        Weight for log-spectral distance.
    w_asd : float
        Weight for ASD ratio loss (J_asd).
    stft_sizes : list of int
        Window sizes for multi-resolution STFT (SC and LSD).
    asd_win : int
        Window size for Welch PSD estimation (J_asd).
    asd_overlap : int
        Overlap for Welch segments. Defaults to asd_win // 2.
    """

    def __init__(self, w_l1: float = 1.0, w_sc: float = 0.0,
                 w_lsd: float = 0.0, w_asd: float = 0.0,
                 stft_sizes: list = None,
                 asd_win: int = 1024, asd_overlap: int = None):
        super().__init__()
        self.w_l1 = w_l1
        self.w_sc = w_sc
        self.w_lsd = w_lsd
        self.w_asd = w_asd
        self.stft_sizes = stft_sizes or [256, 512, 1024]
        self.asd_win = asd_win
        self.asd_overlap = asd_overlap if asd_overlap is not None else asd_win // 2

    def _stft_mag(self, x, n_fft):
        """Compute STFT magnitude. x: (B, L)."""
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=x.device)
        stft = torch.stft(x, n_fft=n_fft, hop_length=hop,
                          win_length=n_fft, window=window,
                          return_complex=True)
        return stft.abs()  # (B, n_fft//2+1, T_frames)

    def spectral_convergence(self, x_pred, x_target):
        """SC = ||STFT(pred) - STFT(target)||_F / ||STFT(target)||_F

        Multi-resolution: average over multiple STFT window sizes.
        """
        # Flatten to (B, L)
        pred = x_pred.reshape(x_pred.shape[0], -1)
        target = x_target.reshape(x_target.shape[0], -1)

        sc_total = 0.0
        for n_fft in self.stft_sizes:
            mag_pred = self._stft_mag(pred, n_fft)
            mag_target = self._stft_mag(target, n_fft)
            sc = torch.norm(mag_target - mag_pred, p='fro') / (
                torch.norm(mag_target, p='fro') + 1e-8)
            sc_total += sc

        return sc_total / len(self.stft_sizes)

    def log_spectral_distance(self, x_pred, x_target):
        """LSD = mean of sqrt(mean over f of (log S_pred - log S_target)^2).

        Uses the largest STFT window for best frequency resolution.
        """
        pred = x_pred.reshape(x_pred.shape[0], -1)
        target = x_target.reshape(x_target.shape[0], -1)

        n_fft = max(self.stft_sizes)
        mag_pred = self._stft_mag(pred, n_fft)
        mag_target = self._stft_mag(target, n_fft)

        # Log power spectra (clamp for numerical stability)
        log_pred = torch.log10(mag_pred.clamp(min=1e-10))
        log_target = torch.log10(mag_target.clamp(min=1e-10))

        # LSD per frame, averaged
        lsd = torch.sqrt(((log_pred - log_target) ** 2).mean(dim=1) + 1e-10)
        return lsd.mean()

    def _welch_psd(self, x, win_size, overlap):
        """Welch PSD estimate. x: (B, L). Returns (B, n_freq)."""
        B, L = x.shape
        step = win_size - overlap
        window = torch.hann_window(win_size, device=x.device)

        # Unfold into segments
        segments = x.unfold(dimension=1, size=win_size, step=step)  # (B, n_seg, win)
        segments = segments * window.unsqueeze(0).unsqueeze(0)

        # FFT and power
        fft = torch.fft.rfft(segments, dim=-1)
        psd = (fft.abs() ** 2).mean(dim=1)  # average over segments: (B, n_freq)

        return psd

    def asd_ratio_loss(self, x_pred, x_target):
        """J_asd = mean over f of sqrt(PSD_residual / PSD_target).

        Measures relative spectral error, as used in DeepClean/LIGO.
        """
        pred = x_pred.reshape(x_pred.shape[0], -1)
        target = x_target.reshape(x_target.shape[0], -1)

        residual = pred - target

        psd_res = self._welch_psd(residual, self.asd_win, self.asd_overlap)
        psd_tgt = self._welch_psd(target, self.asd_win, self.asd_overlap)

        # ASD ratio per frequency bin, averaged
        ratio = torch.sqrt(psd_res / (psd_tgt + 1e-10))
        return ratio.mean()

    def forward(self, x_pred, x_target):
        """Compute combined loss.

        Parameters
        ----------
        x_pred, x_target : (B, 1, L) or (B, L)

        Returns
        -------
        total_loss : scalar
        loss_dict : dict of individual loss components
        """
        loss_dict = {}
        total = 0.0

        if self.w_l1 > 0:
            l1 = F.l1_loss(x_pred, x_target)
            loss_dict['l1'] = l1.item()
            total = total + self.w_l1 * l1

        if self.w_sc > 0:
            sc = self.spectral_convergence(x_pred, x_target)
            loss_dict['sc'] = sc.item()
            total = total + self.w_sc * sc

        if self.w_lsd > 0:
            lsd = self.log_spectral_distance(x_pred, x_target)
            loss_dict['lsd'] = lsd.item()
            total = total + self.w_lsd * lsd

        if self.w_asd > 0:
            asd = self.asd_ratio_loss(x_pred, x_target)
            loss_dict['asd'] = asd.item()
            total = total + self.w_asd * asd

        loss_dict['total'] = total.item()
        return total, loss_dict
