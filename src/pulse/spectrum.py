"""
CUORE energy spectrum model for sampling training energies.

Uses the measured spectrum from figS2_spectrum.root (FinalSpectrum histogram),
loaded from data/cuore_spectrum.npz. Falls back to a parametric approximation
if the data file is not found.
"""

import os
import numpy as np


_SPECTRUM_DATA = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'cuore_spectrum.npz'
)


class EnergySpectrum:
    """Precomputed CDF for fast inverse-CDF energy sampling.

    Loads the measured CUORE spectrum from data/cuore_spectrum.npz
    (extracted from figS2_spectrum.root / FinalSpectrum__3).

    Parameters
    ----------
    E_min : float
        Minimum energy [keV]. Must be > 0 (exclusive lower bound).
    E_max : float
        Maximum energy [keV].
    """

    def __init__(self, E_min: float = 1.0, E_max: float = 5407.0):
        if os.path.exists(_SPECTRUM_DATA):
            data = np.load(_SPECTRUM_DATA)
            centers = data['bin_centers_keV']
            counts = data['counts']

            # Restrict to [E_min, E_max]
            mask = (centers >= E_min) & (centers <= E_max)
            self.E_grid = centers[mask]
            self.rate = counts[mask]
        else:
            print(f"Warning: {_SPECTRUM_DATA} not found, using parametric fallback")
            self.E_grid = np.linspace(E_min, E_max, 50000)
            self.rate = _parametric_rate(self.E_grid)

        # Ensure no negative rates
        self.rate = np.maximum(self.rate, 0.0)

        # Cumulative distribution (trapezoidal integration)
        dE = np.diff(self.E_grid, prepend=self.E_grid[0])
        self.cdf = np.cumsum(self.rate * dE)
        self.cdf = self.cdf / self.cdf[-1]

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Sample n energies from the spectrum via inverse CDF.

        Parameters
        ----------
        rng : np.random.Generator
        n : int

        Returns
        -------
        energies : ndarray, shape (n,)
            Sampled energies in keV.
        """
        u = rng.uniform(0, 1, n)
        return np.interp(u, self.cdf, self.E_grid)


def _parametric_rate(E):
    """Parametric fallback spectrum (approximate)."""
    E = np.asarray(E, dtype=float)
    continuum = 30.0 * np.exp(-E / 400.0) + 0.3 * np.exp(-E / 1500.0) + 0.03
    peaks = [
        (609, 5, 3.0), (1173, 5, 5.0), (1332, 5, 5.0), (1461, 5, 10.0),
        (2528, 5, 0.1), (2615, 5, 3.0), (3200, 20, 2.0), (4012, 20, 1.0),
        (4198, 20, 0.5), (4687, 20, 3.0), (4784, 20, 5.0), (5304, 20, 50.0),
    ]
    rate = continuum.copy()
    for E0, sigma, height in peaks:
        rate += height * np.exp(-0.5 * ((E - E0) / sigma) ** 2)
    return rate
