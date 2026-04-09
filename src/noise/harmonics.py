"""
Generate harmonic series (AC power line, PT cooler) with randomized amplitudes.

Each harmonic has:
  - Mean amplitude following power-law decay: A_base / k^p
  - Log-uniform random factor per harmonic: 10^uniform(-0.5, 0.5)
  - Random phase: uniform [0, 2π)
"""

import numpy as np


def generate_harmonics(rng: np.random.Generator,
                       f_fundamental: float,
                       n_harmonics: int,
                       a_base: float,
                       p_decay: float,
                       n_samples: int,
                       fs: float) -> np.ndarray:
    """Generate a sum of harmonics with power-law amplitude decay.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    f_fundamental : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Maximum number of harmonics (including fundamental).
    a_base : float
        Amplitude of the fundamental [V].
    p_decay : float
        Power-law exponent for harmonic amplitude decay.
    n_samples : int
        Number of output samples.
    fs : float
        Sample rate [Hz].

    Returns
    -------
    signal : ndarray, shape (n_samples,)
    """
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    for k in range(1, n_harmonics + 1):
        f_k = k * f_fundamental
        if f_k >= fs / 2:
            break
        # Power-law decay with log-uniform randomness
        log_rand = rng.uniform(-0.5, 0.5)
        A_k = a_base / k ** p_decay * 10 ** log_rand
        phase_k = rng.uniform(0, 2 * np.pi)
        signal += A_k * np.sin(2 * np.pi * f_k * t + phase_k)

    return signal
