"""
Common signal processing filters shared by pulse and noise generators.

Implements the CUORE readout chain: 6th-order Bessel low-pass at 120 Hz.
"""

import numpy as np
from scipy.signal import bessel, sosfilt


def make_bessel_sos(f_cutoff: float = 120.0, order: int = 6,
                    fs: float = 10000.0):
    """Create Bessel low-pass filter as second-order sections.

    Parameters
    ----------
    f_cutoff : float
        Cutoff frequency [Hz].
    order : int
        Filter order.
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    sos : ndarray
        Second-order sections representation.
    """
    return bessel(order, f_cutoff, btype='low', analog=False,
                  output='sos', fs=fs, norm='phase')


def apply_bessel_decimate(signal: np.ndarray,
                          fs_internal: float = 10000.0,
                          fs_output: float = 1000.0,
                          f_cutoff: float = 120.0,
                          order: int = 6) -> np.ndarray:
    """Apply Bessel filter and decimate to output sample rate.

    Parameters
    ----------
    signal : ndarray
        Input signal at fs_internal.
    fs_internal : float
        Internal (high) sample rate [Hz].
    fs_output : float
        Output (decimated) sample rate [Hz].
    f_cutoff : float
        Bessel filter cutoff frequency [Hz].
    order : int
        Bessel filter order.

    Returns
    -------
    out : ndarray
        Filtered and decimated signal.
    """
    sos = make_bessel_sos(f_cutoff, order, fs_internal)
    filtered = sosfilt(sos, signal)
    dec = int(fs_internal / fs_output)
    return filtered[::dec]
