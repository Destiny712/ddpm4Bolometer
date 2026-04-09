"""
Generate broadband mechanical resonance noise.

Models the cryostat's structural resonances excited by environmental
disturbances (seismic, equipment vibrations, etc.). Each resonance is
a bandpass-filtered white noise centered at a resonant frequency with
a finite Q factor.

Reference: Vetter et al., Eur. Phys. J. C (2024) 84:243, Figure 2
"""

import numpy as np
from scipy.signal import butter, sosfilt


def generate_resonances(rng: np.random.Generator,
                        n_samples: int,
                        fs: float,
                        resonances: list[dict]) -> np.ndarray:
    """Generate noise from multiple mechanical resonances.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of output samples.
    fs : float
        Sample rate [Hz].
    resonances : list of dict
        Each dict has keys:
          'f_center': center frequency [Hz]
          'Q': quality factor (controls peak width)
          'amplitude': peak amplitude [V]

    Returns
    -------
    signal : ndarray, shape (n_samples,)
    """
    signal = np.zeros(n_samples)

    for res in resonances:
        f_c = res['f_center']
        Q = res['Q']
        amp = res['amplitude']

        # Bandwidth from Q factor
        bw = f_c / Q
        f_low = max(f_c - bw / 2, 0.1)
        f_high = min(f_c + bw / 2, fs / 2 - 1)

        if f_low >= f_high:
            continue

        # Bandpass filter white noise
        sos = butter(2, [f_low, f_high], btype='band', fs=fs, output='sos')
        white = rng.standard_normal(n_samples)
        filtered = sosfilt(sos, white)

        # Scale to desired amplitude
        std = np.std(filtered)
        if std > 0:
            filtered = filtered / std * amp

        signal += filtered

    return signal


def sample_resonance_params(rng: np.random.Generator,
                            target_rms: float) -> list[dict]:
    """Sample random mechanical resonance parameters.

    Generates 3–6 resonances with center frequencies roughly matching
    the broad bumps seen in CUORE accelerometer data (Figure 2):
    ~10, ~25, ~40, ~70 Hz range.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    target_rms : float
        Overall noise RMS [V], used to scale resonance amplitudes.

    Returns
    -------
    resonances : list of dict
    """
    # Typical resonance center frequencies [Hz]
    base_freqs = [10.0, 25.0, 40.0, 70.0]

    n_res = rng.integers(3, 7)  # 3 to 6 resonances
    resonances = []

    for i in range(n_res):
        if i < len(base_freqs):
            f_c = base_freqs[i] * rng.uniform(0.7, 1.3)
        else:
            f_c = rng.uniform(5.0, 90.0)

        Q = rng.uniform(5.0, 20.0)
        amp = target_rms * rng.uniform(0.1, 0.5)

        resonances.append({
            'f_center': f_c,
            'Q': Q,
            'amplitude': amp,
        })

    return resonances
