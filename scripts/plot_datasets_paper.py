"""Paper figure: simulation datasets overview.

4 rows x 2 columns. Rows:
  1. single pulse from clean_low
  2. pileup      from clean_low
  3. single pulse from clean_high
  4. pileup      from clean_high
Each row: (time-domain, PSD) with clean, noisy, pure-noise overlaid.
Units: mV (time) and mV^2/Hz (PSD). PSD axes are log-log.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


CLEAN_LOW = '/media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/test/clean_000.h5'
CLEAN_HIGH = '/media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_high/train/clean_000.h5'
NOISE_H5 = '/media/Disk_YIN/yunshancheng/cuore/noise/test/noise_000.h5'


def pick_index(h5_path, pileup, offset=0, min_amp_mV=None):
    """Pick the offset-th index matching `pileup`; optionally require
    peak-minus-baseline amplitude (in mV) above min_amp_mV."""
    with h5py.File(h5_path, 'r') as f:
        is_pu = f['is_pileup'][:]
        pool = np.where(is_pu == pileup)[0]
        if min_amp_mV is None:
            return int(pool[offset])
        wfs = f['waveforms']
        kept = []
        for i in pool:
            w = wfs[int(i)]
            amp_mV = (np.max(w) - np.median(w[:200])) * 1000.0
            if amp_mV >= min_amp_mV:
                kept.append(int(i))
                if len(kept) > offset:
                    return kept[offset]
    raise ValueError(f'not enough pulses above {min_amp_mV} mV in {h5_path}')


def load_waveform(h5_path, idx):
    with h5py.File(h5_path, 'r') as f:
        return f['waveforms'][idx].astype(np.float64), float(f.attrs['f_sample'])


def compute_psd(x, fs):
    nperseg = min(2048, len(x))
    f, p = welch(x, fs=fs, nperseg=nperseg, detrend='constant')
    # drop DC for log-log plotting
    return f[1:], p[1:]


def plot_row(ax_time, ax_psd, clean_mV, noise_mV, fs, title, show_legend=False):
    t = np.arange(clean_mV.size) / fs
    noisy_mV = clean_mV + noise_mV

    # time domain
    ax_time.plot(t, noisy_mV, color='tab:red', lw=0.4, label='observation', alpha=0.8)
    ax_time.plot(t, noise_mV, color='tab:gray', lw=0.15, label='noise', alpha=0.7)
    ax_time.plot(t, clean_mV, color='tab:blue', lw=0.8, label='clean')
    ax_time.set_xlabel('time (s)')
    ax_time.set_ylabel('amplitude (mV)')
    ax_time.set_title(title)
    if show_legend:
        ax_time.legend(loc='upper right', fontsize=8, frameon=False)

    # PSD (both axes log). PSD is in (mV)^2/Hz.
    f_c, p_c = compute_psd(clean_mV, fs)
    f_n, p_n = compute_psd(noise_mV, fs)
    f_o, p_o = compute_psd(noisy_mV, fs)
    ax_psd.loglog(f_o, p_o, color='tab:red', lw=1.3, label='observation', alpha=0.8)
    ax_psd.loglog(f_n, p_n, color='tab:gray', lw=0.8, label='noise', alpha=0.8)
    ax_psd.loglog(f_c, p_c, color='tab:blue', lw=0.8, label='clean')
    ax_psd.set_xlabel('frequency (Hz)')
    ax_psd.set_ylabel('PSD (mV$^2$/Hz)')
    ax_psd.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='/home/yunshan/cuore/plots/paper/datasets.png')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # (path, is_pileup, offset, min_amp_mV, title)
    rows = [
        (CLEAN_LOW,  0, 0, None,  'example 1'),
        (CLEAN_HIGH, 0, 0, 200.0, 'example 2'),
        (CLEAN_LOW,  1, 1, None,  'example 3'),
        (CLEAN_HIGH, 1, 0, 200.0, 'example 4'),
    ]

    # pre-pick clean and noise indices
    configs = []
    with h5py.File(NOISE_H5, 'r') as nf:
        n_noise = nf['waveforms'].shape[0]
        noise_idxs = rng.choice(n_noise, size=len(rows), replace=False)

    for (path, pu, off, min_amp, title), n_idx in zip(rows, noise_idxs):
        c_idx = pick_index(path, pu, off, min_amp_mV=min_amp)
        configs.append((path, c_idx, int(n_idx), title))

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.2 * len(rows)))
    fs = None
    with h5py.File(NOISE_H5, 'r') as nf:
        noise_all = nf['waveforms']
        for row, (path, c_idx, n_idx, title) in enumerate(configs):
            clean_V, fs = load_waveform(path, c_idx)
            noise_V = noise_all[n_idx].astype(np.float64)
            clean_mV = clean_V * 1000.0
            noise_mV = noise_V * 1000.0
            print(f'row {row}: {title} clean_idx={c_idx} noise_idx={n_idx}')
            plot_row(axes[row, 0], axes[row, 1], clean_mV, noise_mV, fs, title,
                     show_legend=(row == 0))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
