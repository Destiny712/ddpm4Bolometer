"""
Slow amplitude envelope for non-stationarity within a noise window.

Generates a smooth, slowly varying function around 1.0 that modulates
the noise amplitude — modeling the fact that noise power changes over
time within a window.
"""

import numpy as np


def generate_slow_envelope(rng: np.random.Generator,
                           n_samples: int,
                           fs: float,
                           max_variation: float = 0.2,
                           f_cutoff: float = 0.2) -> np.ndarray:
    """Generate a slowly varying amplitude envelope.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of output samples.
    fs : float
        Sample rate [Hz].
    max_variation : float
        Maximum fractional deviation from 1.0 (e.g. 0.2 = ±20%).
    f_cutoff : float
        Maximum frequency of envelope variation [Hz].

    Returns
    -------
    envelope : ndarray, shape (n_samples,)
        Values centered around 1.0.
    """
    if max_variation <= 0:
        return np.ones(n_samples)

    n_freqs = n_samples // 2 + 1
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Only keep frequencies below cutoff
    env_fft = np.zeros(n_freqs, dtype=complex)
    mask = (freqs > 0) & (freqs < f_cutoff)
    n_active = mask.sum()
    if n_active == 0:
        return np.ones(n_samples)

    env_fft[mask] = (rng.standard_normal(n_active)
                     + 1j * rng.standard_normal(n_active))

    envelope = np.fft.irfft(env_fft, n=n_samples)

    # Normalize to [-1, 1], then scale
    peak = np.max(np.abs(envelope))
    if peak > 0:
        envelope = envelope / peak

    return 1.0 + max_variation * envelope
