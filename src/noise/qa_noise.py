"""
QA check: generate and plot noise windows with time-series and PSD.

Each row shows one noise window:
  - Left: V(t) time series
  - Right: PSD (log-log) with markers at 1.4 Hz, 50 Hz, and f_cross

Annotated with key parameters: RMS, α, f_cross, AC amp, PT amp,
envelope variation.

Usage:
    python -u -m src.noise.qa_noise --n 10 --output qa_noise.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.noise.generator import generate_noise, sample_noise_params


def main():
    parser = argparse.ArgumentParser(
        description="QA: generate and plot noise windows."
    )
    parser.add_argument('--n', type=int, default=10,
                        help='Number of noise windows to generate')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Window duration [s]')
    parser.add_argument('--f_sample', type=float, default=1000.0,
                        help='Output sample rate [Hz]')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='qa_noise.png',
                        help='Output image file')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(args.n, 2, figsize=(16, 3 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    t = np.arange(int(args.duration * args.f_sample)) / args.f_sample

    for i in range(args.n):
        params = sample_noise_params(rng)
        noise = generate_noise(rng, params,
                               duration=args.duration,
                               f_sample=args.f_sample)

        actual_rms = np.std(noise)

        # --- Left: time series ---
        ax_t = axes[i, 0]
        ax_t.plot(t, noise * 1e3, 'b-', lw=0.4)
        ax_t.set_xlabel('Time [s]')
        ax_t.set_ylabel('Voltage [mV]')
        ax_t.set_title(f'Noise #{i}', fontsize=9)
        ax_t.grid(True, alpha=0.3)

        # Annotation
        text_t = (f'RMS={actual_rms*1e3:.2f} mV\n'
                  f'env={params["envelope_variation"]:.2f}')
        ax_t.text(0.98, 0.95, text_t, transform=ax_t.transAxes,
                  fontsize=7, va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # --- Right: PSD ---
        ax_f = axes[i, 1]
        freqs, psd = welch(noise, fs=args.f_sample,
                           nperseg=min(len(noise), 4096))
        ax_f.loglog(freqs[1:], psd[1:], 'b-', lw=0.6)
        ax_f.set_xlabel('Frequency [Hz]')
        ax_f.set_ylabel('PSD [V²/Hz]')
        ax_f.set_title(f'Noise #{i} — PSD', fontsize=9)
        ax_f.grid(True, alpha=0.3, which='both')

        # Mark key frequencies
        for f_mark, label, color in [
            (params['pt_freq'], 'PT', 'red'),
            (50.0, 'AC', 'green'),
            (params['f_cross'], 'f_cross', 'orange'),
        ]:
            ax_f.axvline(f_mark, color=color, ls='--', lw=0.8, alpha=0.7)
            ax_f.text(f_mark * 1.1, ax_f.get_ylim()[1] * 0.5, label,
                      fontsize=6, color=color, va='top')

        # Mark resonance frequencies
        for res in params['resonances']:
            ax_f.axvline(res['f_center'], color='purple', ls=':',
                         lw=0.6, alpha=0.5)

        # Annotation
        n_res = len(params['resonances'])
        res_freqs = ', '.join(f'{r["f_center"]:.0f}'
                              for r in params['resonances'])
        text_f = (f'α={params["alpha"]:.2f}\n'
                  f'f_cross={params["f_cross"]:.2f} Hz\n'
                  f'AC_amp={params["ac_a_base"]*1e3:.2f} mV\n'
                  f'PT_amp={params["pt_a_base"]*1e3:.2f} mV\n'
                  f'white={params["white_rms"]*1e3:.2f} mV\n'
                  f'res({n_res}): {res_freqs} Hz')
        ax_f.text(0.98, 0.95, text_f, transform=ax_f.transAxes,
                  fontsize=7, va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('QA Noise Check', fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
