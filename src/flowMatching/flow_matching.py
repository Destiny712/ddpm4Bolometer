"""
Conditional flow matching for 1D signal denoising.

Forward interpolation: x_t = (1-t)*x_0 + t*eps, t in [0, 1]
Velocity target: v = eps - x_0
Inference: ODE integration from t=1 (noise) to t=0 (data)

Usage:
    from src.flowMatching.flow_matching import FlowMatching
    flow = FlowMatching(model, loss_type='l1')
    loss = flow.training_loss(x_0, x_tilde)
    x_denoised = flow.sample(x_tilde, steps=50)
"""

import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    """Conditional flow matching for signal denoising."""

    def __init__(self, model: nn.Module, loss_type: str = 'l2',
                 spectral_loss=None):
        super().__init__()
        self.model = model
        self.loss_fn = nn.functional.l1_loss if loss_type == 'l1' else nn.functional.mse_loss
        self.spectral_loss = spectral_loss

    def interpolate(self, x_0: torch.Tensor, t: torch.Tensor,
                    noise: torch.Tensor = None):
        """Forward interpolation: x_t = (1-t)*x_0 + t*noise.

        Parameters
        ----------
        x_0 : (B, 1, L) — clean signal
        t : (B,) — time in [0, 1]
        noise : (B, 1, L) — optional pre-sampled noise

        Returns
        -------
        x_t : (B, 1, L)
        noise : (B, 1, L)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        t_broad = t.view(-1, 1, 1)  # (B, 1, 1)
        x_t = (1.0 - t_broad) * x_0 + t_broad * noise
        return x_t, noise

    def _predict_velocity(self, x_t, x_tilde, t, scale=None):
        """Call model with or without scale conditioning."""
        if scale is not None:
            return self.model(x_t, x_tilde, t, scale)
        return self.model(x_t, x_tilde, t)

    def training_loss(self, x_0: torch.Tensor,
                      x_tilde: torch.Tensor,
                      scale: torch.Tensor = None) -> torch.Tensor:
        """Compute flow matching training loss.

        Parameters
        ----------
        x_0 : (B, 1, L) — clean signal
        x_tilde : (B, 1, L) — noisy observation (conditioning)
        scale : (B,) — optional max(|noisy|) for scale-conditioned models

        Returns
        -------
        loss : scalar tensor
        (loss, loss_dict) : if spectral_loss is set
        """
        B = x_0.shape[0]
        device = x_0.device

        # Uniform t in (0, 1), avoid exact boundaries
        t = torch.rand(B, device=device) * (1.0 - 2e-5) + 1e-5

        # Sample noise and interpolate
        noise = torch.randn_like(x_0)
        x_t, _ = self.interpolate(x_0, t, noise)

        # Target velocity: direction from data to noise
        v_target = noise - x_0

        # Predict velocity
        v_pred = self._predict_velocity(x_t, x_tilde, t, scale)

        velocity_loss = self.loss_fn(v_pred, v_target)

        if self.spectral_loss is None:
            return velocity_loss

        # Compute implied x0_hat from v_pred for spectral loss
        # x_t = (1-t)*x_0 + t*noise, v = noise - x_0
        # => x_0 = x_t - t*v  (since x_t = x_0 + t*v)
        t_broad = t.view(-1, 1, 1)
        x0_hat = x_t - t_broad * v_pred

        spec_loss, loss_dict = self.spectral_loss(x0_hat, x_0)
        loss_dict['velocity'] = velocity_loss.item()

        total = velocity_loss + spec_loss
        loss_dict['total'] = total.item()

        return total, loss_dict

    @torch.no_grad()
    def sample(self, x_tilde: torch.Tensor,
               scale: torch.Tensor = None,
               steps: int = 50,
               solver: str = 'euler',
               return_trajectory: bool = False) -> torch.Tensor:
        """ODE integration from t=1 (noise) to t=0 (data).

        Parameters
        ----------
        x_tilde : (B, 1, L) — noisy observation
        scale : (B,) — optional max(|noisy|) for scale-conditioned models
        steps : int — number of ODE integration steps
        solver : 'euler' or 'midpoint'
        return_trajectory : bool

        Returns
        -------
        x_0 : (B, 1, L) — denoised signal
        trajectory : list of (B, 1, L) — only if return_trajectory=True
        """
        device = x_tilde.device
        B, C, L = x_tilde.shape

        # Start from pure noise at t=1
        x_t = torch.randn(B, 1, L, device=device)
        dt = 1.0 / steps

        trajectory = [x_t] if return_trajectory else None

        for i in range(steps):
            t_cur = 1.0 - i * dt  # from 1.0 down toward 0
            t_batch = torch.full((B,), t_cur, device=device)

            if solver == 'euler':
                v = self._predict_velocity(x_t, x_tilde, t_batch, scale)
                x_t = x_t - dt * v

            elif solver == 'midpoint':
                # Evaluate at current point
                v1 = self._predict_velocity(x_t, x_tilde, t_batch, scale)
                # Step to midpoint
                x_mid = x_t - (dt / 2) * v1
                t_mid = torch.full((B,), t_cur - dt / 2, device=device)
                # Re-evaluate at midpoint
                v2 = self._predict_velocity(x_mid, x_tilde, t_mid, scale)
                # Full step with midpoint velocity
                x_t = x_t - dt * v2

            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return x_t, trajectory
        return x_t

    @torch.no_grad()
    def sample_multi_shot(self, x_tilde: torch.Tensor,
                          scale: torch.Tensor = None,
                          M: int = 10,
                          aggregation: str = 'mean',
                          steps: int = 50,
                          solver: str = 'euler') -> torch.Tensor:
        """Multi-shot inference: M independent ODE integrations, aggregated.

        Parameters
        ----------
        x_tilde : (B, 1, L)
        scale : (B,) — optional max(|noisy|) for scale-conditioned models
        M : int — number of shots
        aggregation : 'mean' or 'median'
        steps : int — number of ODE steps per shot
        solver : 'euler' or 'midpoint'

        Returns
        -------
        x_0_agg : (B, 1, L) — aggregated denoised signal
        """
        def _single():
            return self.sample(x_tilde, scale=scale, steps=steps, solver=solver)

        samples = torch.stack([_single() for _ in range(M)])
        if aggregation == 'median':
            return samples.median(dim=0).values
        return samples.mean(dim=0)
