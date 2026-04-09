"""
QA check: plot the parametric CUORE energy spectrum model.

Generates two plots:
  1. Spectrum rate vs energy (log scale, to compare with Moriond slide)
  2. Histogram of sampled energies (verifies sampling matches the model)

Usage:
    python -u -m src.pulse.qa_spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pulse.spectrum import spectrum_rate, EnergySpectrum, ALL_PEAKS


def main():
    E = np.linspace(1, 5500, 10000)
    rate = spectrum_rate(E)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # --- Plot 1: Spectrum model ---
    ax = axes[0]
    ax.semilogy(E, rate, 'b-', lw=1.0, label='Model')

    # Mark peaks
    for E0, sigma, height in ALL_PEAKS:
        peak_rate = spectrum_rate(E0)
        ax.annotate(f'{E0:.0f}', xy=(E0, peak_rate),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=7, ha='center', color='red')

    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Rate [arb. units]')
    ax.set_title('CUORE Spectrum Model (compare with Moriond slide 13)')
    ax.set_xlim(0, 5500)
    ax.set_ylim(1e-2, 200)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Plot 2: Sampled histogram ---
    ax = axes[1]
    spectrum = EnergySpectrum()
    rng = np.random.default_rng(0)
    samples = spectrum.sample(rng, 100000)

    ax.hist(samples, bins=500, density=True, alpha=0.7, label='Sampled')

    # Overlay normalized model
    norm = np.trapz(rate, E)
    ax.plot(E, rate / norm, 'r-', lw=1.0, label='Model (normalized)')

    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Probability density')
    ax.set_title('Sampled Energy Distribution (100k samples)')
    ax.set_xlim(0, 5500)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('qa_spectrum.png', dpi=150)
    print("Saved qa_spectrum.png")
    plt.show()


if __name__ == '__main__':
    main()
