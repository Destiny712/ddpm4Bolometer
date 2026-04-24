"""
DDPM forward and reverse diffusion processes.

Handles:
  - Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
  - Training loss: MSE between true noise and predicted noise
  - Reverse sampling: iterative denoising from x_T ~ N(0, I)

Usage:
    from src.ddpm.diffusion import GaussianDiffusion
    diffusion = GaussianDiffusion(model, schedule)
    loss = diffusion.training_loss(x_0, x_tilde)
    x_denoised = diffusion.sample(x_tilde)
"""

import torch
import torch.nn as nn

from src.ddpm.schedule import DiffusionSchedule


class GaussianDiffusion(nn.Module):
    """Conditional Gaussian diffusion for signal denoising.

    cond_mode:
        'step' — condition the denoiser on the discrete step index 1..T
                 (original DDPM behavior).
        'sqrt_ab' — condition on the continuous noise level sqrt(alpha_bar).
                 During training, sqrt_ab is sampled uniformly in the bin
                 [S_t, S_{t-1}] of the grid S = [1, sqrt_alpha_bar[0], ...,
                 sqrt_alpha_bar[T-1]] (WaveGrad / DeScoD-ECG recipe).
    """

    def __init__(self, model: nn.Module, schedule: DiffusionSchedule,
                 loss_type: str = 'l2', spectral_loss=None,
                 cond_mode: str = 'step'):
        super().__init__()
        assert cond_mode in ('step', 'sqrt_ab'), f"unknown cond_mode: {cond_mode}"
        self.model = model
        self.schedule = schedule
        self.T = schedule.T
        self.cond_mode = cond_mode
        self.loss_fn = nn.functional.l1_loss if loss_type == 'l1' else nn.functional.mse_loss
        self.spectral_loss = spectral_loss

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward process: sample x_t given x_0.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Parameters
        ----------
        x_0 : (B, 1, L) — clean signal
        t : (B,) — timestep indices (0-indexed into schedule arrays)
        noise : (B, 1, L) — optional pre-sampled noise

        Returns
        -------
        x_t : (B, 1, L)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1_ab = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        return sqrt_ab * x_0 + sqrt_1_ab * noise

    def _predict_noise(self, x_t, x_tilde, cond, scale=None):
        """Call model with or without scale conditioning.

        `cond` is either a 1-indexed step tensor (cond_mode='step') or
        a continuous sqrt(alpha_bar) tensor (cond_mode='sqrt_ab').
        """
        if scale is not None:
            return self.model(x_t, x_tilde, cond, scale)
        return self.model(x_t, x_tilde, cond)

    def _sample_sqrt_ab(self, t: torch.Tensor) -> torch.Tensor:
        """For cond_mode='sqrt_ab': sample continuous sqrt(alpha_bar) ~ Uniform(low, high).

        Bin edges for 0-indexed step `t`:
            high = sqrt_alpha_bar[t-1] if t > 0 else 1.0
            low  = sqrt_alpha_bar[t]
        (The grid S = [1, sqrt_ab[0], ..., sqrt_ab[T-1]] is decreasing.)
        """
        device = t.device
        sab = self.schedule.sqrt_alpha_bar  # (T,)
        # Build per-sample upper/lower bin edges
        prev_sab = torch.cat([sab.new_ones(1), sab[:-1]])  # S_{t-1} in 0-indexed
        high = prev_sab[t]
        low = sab[t]
        u = torch.rand(t.shape[0], device=device)
        return low + u * (high - low)

    def training_loss(self, x_0: torch.Tensor,
                      x_tilde: torch.Tensor,
                      scale: torch.Tensor = None) -> torch.Tensor:
        """Compute training loss (noise prediction + optional spectral).

        Parameters
        ----------
        x_0 : (B, 1, L) — clean signal
        x_tilde : (B, 1, L) — noisy observation (conditioning)
        scale : (B,) — optional max(|noisy|) for scale-conditioned models

        Returns
        -------
        loss : scalar tensor
        loss_dict : dict (only if spectral_loss is set, else None)
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps (0-indexed)
        t = torch.randint(0, self.T, (B,), device=device)

        noise = torch.randn_like(x_0)

        # Determine sqrt(alpha_bar) and conditioning value for this mode
        if self.cond_mode == 'sqrt_ab':
            sqrt_ab_vec = self._sample_sqrt_ab(t)            # (B,) continuous
            sqrt_1_ab_vec = torch.sqrt((1.0 - sqrt_ab_vec ** 2).clamp(min=0.0))
            cond = sqrt_ab_vec                                # model sees float
        else:
            sqrt_ab_vec = self.schedule.sqrt_alpha_bar[t]
            sqrt_1_ab_vec = self.schedule.sqrt_one_minus_alpha_bar[t]
            cond = (t + 1).long()                             # 1-indexed step

        # Forward process
        sqrt_ab = sqrt_ab_vec.view(-1, 1, 1)
        sqrt_1_ab = sqrt_1_ab_vec.view(-1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise

        eps_pred = self._predict_noise(x_t, x_tilde, cond, scale)

        noise_loss = self.loss_fn(eps_pred, noise)

        if self.spectral_loss is None:
            return noise_loss

        # Compute implied x̂_0 from eps_pred (Tweedie estimate)
        x0_hat = (x_t - sqrt_1_ab * eps_pred) / sqrt_ab

        spec_loss, loss_dict = self.spectral_loss(x0_hat, x_0)
        loss_dict['noise'] = noise_loss.item()

        total = noise_loss + spec_loss
        loss_dict['total'] = total.item()

        return total, loss_dict

    @torch.no_grad()
    def sample(self, x_tilde: torch.Tensor,
               scale: torch.Tensor = None,
               return_trajectory: bool = False,
               stochastic: bool = True) -> torch.Tensor:
        """Reverse process: denoise x_tilde by running T steps.

        Parameters
        ----------
        x_tilde : (B, 1, L) — noisy observation
        scale : (B,) — optional max(|noisy|) for scale-conditioned models
        return_trajectory : bool
            If True, return all intermediate x_t.
        stochastic : bool
            If False, skip noise addition at each reverse step (deterministic DDPM).

        Returns
        -------
        x_0 : (B, 1, L) — denoised signal
        trajectory : list of (B, 1, L) — only if return_trajectory=True
        """
        device = x_tilde.device
        B, C, L = x_tilde.shape

        # Start from pure noise
        x_t = torch.randn(B, 1, L, device=device)

        trajectory = [x_t] if return_trajectory else None

        for i in reversed(range(self.T)):
            if self.cond_mode == 'sqrt_ab':
                cond = self.schedule.sqrt_alpha_bar[i].expand(B)
            else:
                cond = torch.full((B,), i + 1, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = self._predict_noise(x_t, x_tilde, cond, scale)

            # DDPM reverse step
            alpha_t = self.schedule.alpha[i]
            alpha_bar_t = self.schedule.alpha_bar[i]
            beta_t = self.schedule.beta[i]

            # Mean: (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred)
            coeff = beta_t / self.schedule.sqrt_one_minus_alpha_bar[i]
            mean = (1.0 / self.schedule.sqrt_alpha[i]) * (x_t - coeff * eps_pred)

            if i > 0 and stochastic:
                # Add noise (not at final step)
                sigma = torch.sqrt(beta_t)
                x_t = mean + sigma * torch.randn_like(x_t)
            else:
                x_t = mean

            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return x_t, trajectory
        return x_t

    @torch.no_grad()
    def ddim_sample(self, x_tilde: torch.Tensor,
                    scale: torch.Tensor = None,
                    steps: int = None,
                    eta: float = 0.0) -> torch.Tensor:
        """DDIM sampling (Song et al. 2020). Deterministic when eta=0.

        Uses the same trained model — only the sampling rule changes.
        Supports step skipping: e.g. steps=10 with T=50 uses a
        uniform subsequence of 10 timesteps.

        Parameters
        ----------
        x_tilde : (B, 1, L) — noisy observation
        scale : (B,) — optional scale conditioning
        steps : int — number of sampling steps (default: self.T)
        eta : float — 0 = deterministic DDIM, 1 = DDPM-equivalent

        Returns
        -------
        x_0 : (B, 1, L) — denoised signal
        """
        if steps is None:
            steps = self.T
        device = x_tilde.device
        B, C, L = x_tilde.shape

        # Build subsequence of timesteps (0-indexed into schedule)
        # e.g. steps=10, T=50 → [49, 44, 39, 34, 29, 24, 19, 14, 9, 4]
        tau = torch.linspace(self.T - 1, 0, steps, device=device).long()

        x_t = torch.randn(B, 1, L, device=device)

        for i in range(len(tau)):
            t_cur = tau[i]
            if self.cond_mode == 'sqrt_ab':
                cond = self.schedule.sqrt_alpha_bar[t_cur].expand(B)
            else:
                cond = torch.full((B,), t_cur + 1, device=device, dtype=torch.long)

            eps_pred = self._predict_noise(x_t, x_tilde, cond, scale)

            alpha_bar_t = self.schedule.alpha_bar[t_cur]
            sqrt_ab_t = self.schedule.sqrt_alpha_bar[t_cur]
            sqrt_1_ab_t = self.schedule.sqrt_one_minus_alpha_bar[t_cur]

            # Predicted x_0
            x0_pred = (x_t - sqrt_1_ab_t * eps_pred) / sqrt_ab_t

            if i < len(tau) - 1:
                t_next = tau[i + 1]
                alpha_bar_next = self.schedule.alpha_bar[t_next]
                sqrt_ab_next = self.schedule.sqrt_alpha_bar[t_next]
                sqrt_1_ab_next = self.schedule.sqrt_one_minus_alpha_bar[t_next]

                # DDIM sigma (eta=0 → deterministic)
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_next) / (1 - alpha_bar_t)
                    * (1 - alpha_bar_t / alpha_bar_next)
                )

                # Direction pointing to x_t
                dir_xt = torch.sqrt(
                    torch.clamp(1 - alpha_bar_next - sigma ** 2, min=0.0)
                ) * eps_pred

                x_t = sqrt_ab_next * x0_pred + dir_xt
                if eta > 0:
                    x_t = x_t + sigma * torch.randn_like(x_t)
            else:
                x_t = x0_pred

        return x_t

    @torch.no_grad()
    def sample_multi_shot(self, x_tilde: torch.Tensor,
                          scale: torch.Tensor = None,
                          M: int = 10,
                          aggregation: str = 'mean',
                          sampler: str = 'ddpm',
                          ddim_steps: int = None,
                          eta: float = 0.0,
                          stochastic: bool = True) -> torch.Tensor:
        """Multi-shot inference: run M independent reverse processes and aggregate.

        Parameters
        ----------
        x_tilde : (B, 1, L)
        scale : (B,) — optional max(|noisy|) for scale-conditioned models
        M : int — number of shots
        aggregation : 'mean' or 'median'
        sampler : 'ddpm' or 'ddim'
        ddim_steps : int — number of DDIM steps (only used if sampler='ddim')
        eta : float — DDIM eta (only used if sampler='ddim')
        stochastic : bool — if False, skip noise in DDPM reverse steps

        Returns
        -------
        x_0_agg : (B, 1, L) — aggregated denoised signal
        """
        def _single():
            if sampler == 'ddim':
                return self.ddim_sample(x_tilde, scale=scale, steps=ddim_steps, eta=eta)
            return self.sample(x_tilde, scale=scale, stochastic=stochastic)

        samples = torch.stack([_single() for _ in range(M)])
        if aggregation == 'median':
            return samples.median(dim=0).values
        return samples.mean(dim=0)
