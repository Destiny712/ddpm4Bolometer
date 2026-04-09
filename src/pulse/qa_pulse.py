"""
QA check: randomly select N pulses from a generated dataset and plot.

Each subplot shows V_bol vs time with annotations:
  - Energy (keV)
  - Single or pileup
  - Key detector parameters

Usage:
    python -u -m src.pulse.qa_pulse --input data/clean_pulses.h5 --n 8
"""

import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="QA: plot random pulses from generated dataset."
    )
    parser.add_argument('--input', type=str, default='data/clean_pulses.h5',
                        help='Input HDF5 file')
    parser.add_argument('--n', type=int, default=8,
                        help='Number of pulses to plot')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for selection')
    parser.add_argument('--output', type=str, default='qa_pulse.png',
                        help='Output image file')
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as f:
        n_total = f.attrs['n_pulses']
        f_sample = f.attrs['f_sample']
        duration = f.attrs['duration']

        rng = np.random.default_rng(args.seed)
        indices = rng.choice(n_total, size=min(args.n, n_total), replace=False)
        indices = np.sort(indices)

        # Read selected pulses
        waveforms = f['waveforms'][indices]
        E1 = f['energies_1'][indices]
        E2 = f['energies_2'][indices]
        t1 = f['onsets_1'][indices]
        t2 = f['onsets_2'][indices]
        pileup = f['is_pileup'][indices]

        R0 = f['params/R0'][indices]
        T0 = f['params/T0'][indices]
        T_base = f['params/T_base'][indices]
        q = f['params/q'][indices]
        a_ec = f['params/a_ec'][indices]

    n_plot = len(indices)
    ncols = 2
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    t = np.arange(waveforms.shape[1]) / f_sample

    for k in range(n_plot):
        row, col = divmod(k, ncols)
        ax = axes[row, col]

        ax.plot(t, waveforms[k], 'b-', lw=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('V_bol [V]')
        ax.grid(True, alpha=0.3)

        # Build title
        idx = indices[k]
        if pileup[k]:
            title = (f'#{idx}  PILEUP  '
                     f'E1={E1[k]:.0f} keV @ {t1[k]:.3f}s, '
                     f'E2={E2[k]:.0f} keV @ {t2[k]:.3f}s')
        else:
            title = f'#{idx}  SINGLE  E={E1[k]:.0f} keV @ {t1[k]:.3f}s'
        ax.set_title(title, fontsize=9)

        # Parameter annotation
        text = (f'R0={R0[k]:.3f} Ω  T0={T0[k]:.2f} K\n'
                f'T_base={T_base[k]*1e3:.2f} mK  q={q[k]:.1f}\n'
                f'a_ec={a_ec[k]:.2f}')
        ax.text(0.98, 0.95, text, transform=ax.transAxes,
                fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for k in range(n_plot, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'QA Pulse Check — {args.input}', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
