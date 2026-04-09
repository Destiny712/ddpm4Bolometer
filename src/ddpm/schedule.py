"""
Quadratic noise schedule for DDPM.

Precomputes all diffusion constants (beta, alpha, alpha_bar) for T steps.

Usage:
    from src.ddpm.schedule import DiffusionSchedule
    schedule = DiffusionSchedule(T=50)
    alpha_bar_t = schedule.alpha_bar[t]
"""

import torch


class DiffusionSchedule:
    """Quadratic noise schedule: beta_t increases quadratically from beta_1 to beta_T.

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    beta_1 : float
        Starting noise level.
    beta_T : float
        Ending noise level.
    """

    def __init__(self, T: int = 50, beta_1: float = 1e-4, beta_T: float = 0.05):
        self.T = T

        # Quadratic schedule: beta_t = (interp(sqrt(beta_1), sqrt(beta_T)))^2
        t = torch.arange(1, T + 1, dtype=torch.float64)
        sqrt_beta = (
            (T - t) / (T - 1) * beta_1 ** 0.5
            + (t - 1) / (T - 1) * beta_T ** 0.5
        )
        beta = sqrt_beta ** 2
        beta = beta.clamp(min=1e-8, max=0.999)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # Store as float32 for use in training
        self.beta = beta.float()             # (T,)
        self.alpha = alpha.float()           # (T,)
        self.alpha_bar = alpha_bar.float()   # (T,)

        # Precompute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_alpha = torch.sqrt(self.alpha)

    def to(self, device):
        """Move all tensors to device."""
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_alpha = self.sqrt_alpha.to(device)
        return self
