"""
Generate colored noise with 1/f^α, linear-f, and white components.

The power spectral density is shaped as:
  S(f) = a_pink / f^α  +  a_lin * f  +  a_white

Generated in the frequency domain with random phase, then inverse FFT'd.

References:
  - CUORE electro-thermal model, arXiv:2205.04549, Eq. 5.9
    (i²_ext = C₁·f + C₂/f)
"""

import numpy as np


def generate_colored_noise(rng: np.random.Generator,
                           n_samples: int,
                           fs: float,
                           alpha: float,
                           a_pink: float,
                           a_lin: float,
                           a_white: float) -> np.ndarray:
    """Generate colored noise from a parametric PSD.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of output samples.
    fs : float
        Sample rate [Hz].
    alpha : float
        Exponent for 1/f noise (typically 0.8–1.2).
    a_pink : float
        Amplitude coefficient for 1/f^α component.
    a_lin : float
        Amplitude coefficient for linear-f component.
    a_white : float
        Amplitude coefficient for white noise floor.

    Returns
    -------
    noise : ndarray, shape (n_samples,)
    """
    n_freqs = n_samples // 2 + 1
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Build PSD shape (skip DC to avoid divergence)
    psd = np.zeros(n_freqs)
    psd[1:] = (a_pink / freqs[1:] ** alpha
               + a_lin * freqs[1:]
               + a_white)
    psd[0] = 0.0

    # Convert PSD to amplitude spectrum
    # Factor ensures correct variance after IFFT
    amp = np.sqrt(psd * n_samples / (2 * fs))

    # Random-phase complex noise in frequency domain
    noise_fft = (rng.standard_normal(n_freqs)
                 + 1j * rng.standard_normal(n_freqs))
    noise_fft *= amp

    # DC and Nyquist must be real
    noise_fft[0] = noise_fft[0].real
    if n_samples % 2 == 0:
        noise_fft[-1] = noise_fft[-1].real

    return np.fft.irfft(noise_fft, n=n_samples)
