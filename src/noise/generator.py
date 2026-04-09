"""
CUORE-like noise generator — combines all noise components.

Orchestrates harmonics, colored noise, and envelope modules to produce
realistic detector noise windows for DDPM training.

Usage:
    from src.noise.generator import generate_noise, sample_noise_params
    rng = np.random.default_rng(42)
    params = sample_noise_params(rng)
    noise = generate_noise(rng, params)
"""

import numpy as np

from src.noise.harmonics import generate_harmonics
from src.noise.colored import generate_colored_noise
from src.noise.envelope import generate_slow_envelope
from src.noise.resonances import generate_resonances, sample_resonance_params
from src.basics.filters import apply_bessel_decimate

# Internal sample rate — must match pulse simulator
F_INTERNAL = 10000


def sample_noise_params(rng: np.random.Generator,
                        noise_rms: float = 7.5e-3) -> dict:
    """Sample randomized noise parameters for one window.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    noise_rms : float
        Base noise RMS [V]. Randomized by ×[0.5, 1.5].

    Returns
    -------
    params : dict
        All noise parameters for one window, suitable for logging/QA.
    """
    rms_scale = rng.uniform(0.5, 1.5)
    target_rms = noise_rms * rms_scale

    # 1/f^alpha parameters
    alpha = rng.uniform(0.8, 1.2)
    f_cross = rng.uniform(0.5, 3.0)

    # PSD shape: a_pink/f^α + a_lin*f + a_white
    # At crossover: a_pink / f_cross^α = a_lin * f_cross
    a_pink = 1.0
    a_lin = 1.0 / f_cross ** (alpha + 1)
    # White floor ≈ pink level at 50 Hz, randomized
    a_white = a_pink / 50.0 ** alpha * rng.uniform(0.5, 2.0)

    return {
        # Overall
        'target_rms': target_rms,
        'rms_scale': rms_scale,
        # AC harmonics (50 Hz)
        'ac_a_base': target_rms * rng.uniform(0.3, 1.0),
        'ac_p_decay': rng.uniform(1.0, 2.0),
        'ac_n_harmonics': 20,
        # PT cooler harmonics (1.4 Hz)
        'pt_freq': 1.4 + rng.uniform(-0.05, 0.05),
        'pt_a_base': target_rms * rng.uniform(0.2, 0.8),
        'pt_p_decay': rng.uniform(1.0, 2.0),
        'pt_n_harmonics': 50,
        # Colored noise
        'alpha': alpha,
        'f_cross': f_cross,
        'a_pink': a_pink,
        'a_lin': a_lin,
        'a_white': a_white,
        # White noise floor (separate from colored noise)
        'white_rms': target_rms * rng.uniform(0.1, 0.4),
        # Mechanical resonances
        'resonances': sample_resonance_params(rng, target_rms),
        # Envelope
        'envelope_variation': rng.uniform(0.0, 0.2),
    }


def generate_noise(rng: np.random.Generator,
                   params: dict,
                   duration: float = 10.0,
                   f_sample: float = 1000.0,
                   f_bessel: float = 120.0,
                   bessel_order: int = 6) -> np.ndarray:
    """Generate a single noise window from pre-sampled parameters.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    params : dict
        Noise parameters from sample_noise_params().
    duration : float
        Window duration [s].
    f_sample : float
        Output sample rate [Hz].
    f_bessel : float
        Bessel filter cutoff [Hz].
    bessel_order : int
        Bessel filter order.

    Returns
    -------
    noise : ndarray, shape (int(duration * f_sample),)
        Noise waveform [V].
    """
    n_internal = int(duration * F_INTERNAL)

    # --- 1. AC harmonics (50 Hz) ---
    ac = generate_harmonics(
        rng,
        f_fundamental=50.0,
        n_harmonics=params['ac_n_harmonics'],
        a_base=params['ac_a_base'],
        p_decay=params['ac_p_decay'],
        n_samples=n_internal,
        fs=F_INTERNAL,
    )

    # --- 2. PT cooler harmonics (1.4 Hz) ---
    pt = generate_harmonics(
        rng,
        f_fundamental=params['pt_freq'],
        n_harmonics=params['pt_n_harmonics'],
        a_base=params['pt_a_base'],
        p_decay=params['pt_p_decay'],
        n_samples=n_internal,
        fs=F_INTERNAL,
    )

    # --- 3. Colored noise (1/f^α + linear-f + white) ---
    colored = generate_colored_noise(
        rng, n_internal, F_INTERNAL,
        alpha=params['alpha'],
        a_pink=params['a_pink'],
        a_lin=params['a_lin'],
        a_white=params['a_white'],
    )
    # Scale colored noise to desired contribution
    colored_target = params['target_rms'] * rng.uniform(0.3, 1.0)
    std_colored = np.std(colored)
    if std_colored > 0:
        colored = colored / std_colored * colored_target

    # --- 4. White noise floor ---
    white = rng.standard_normal(n_internal) * params['white_rms']

    # --- 5. Mechanical resonances ---
    resonance = generate_resonances(
        rng, n_internal, F_INTERNAL,
        resonances=params['resonances'],
    )

    # --- 6. Sum all components ---
    noise = ac + pt + colored + white + resonance

    # --- 7. Slow envelope ---
    envelope = generate_slow_envelope(
        rng, n_internal, F_INTERNAL,
        max_variation=params['envelope_variation'],
    )
    noise *= envelope

    # --- 8. Bessel filter + decimate ---
    n_out = int(duration * f_sample)
    noise = apply_bessel_decimate(
        noise, F_INTERNAL, f_sample, f_bessel, bessel_order,
    )
    noise = noise[:n_out]

    # --- 9. Final RMS scaling ---
    std_noise = np.std(noise)
    if std_noise > 0:
        noise = noise / std_noise * params['target_rms']

    return noise
