"""Paper figure: denoising-algorithm examples.

Same 4 datasets as scripts/plot_datasets_paper.py (identical clean + noise
indices via seed=0). 4 rows x 2 columns:
  - left  : time domain, noisy + 1-shot denoised  (mV)
  - right : PSD, noisy + clean + noise + 1-shot denoised  (mV^2/Hz, log-log)

Denoiser: DDPM deterministic (stochastic=False) with the ddpm_l1_low model.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import torch

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion


CLEAN_LOW = '/media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/test/clean_000.h5'
CLEAN_HIGH = '/media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_high/train/clean_000.h5'
NOISE_H5 = '/media/Disk_YIN/yunshancheng/cuore/noise/test/noise_000.h5'
MODEL_PATH = '/media/Disk_YIN/yunshancheng/cuore/ddpm_l1_low/best_model.pt'


def pick_index(h5_path, pileup, offset=0, min_amp_mV=None):
    """Mirror of plot_datasets_paper.pick_index — returns the offset-th
    index matching `pileup`, optionally with peak amplitude above min_amp_mV."""
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
    return f[1:], p[1:]  # drop DC for log-log


def denoise(diffusion, device, noisy_V):
    """Run deterministic DDPM on one waveform. Returns denoised V array."""
    scale = float(np.max(np.abs(noisy_V)))
    noisy_n = noisy_V / scale
    x = torch.from_numpy(noisy_n.astype(np.float32))[None, None, :].to(device)
    with torch.no_grad():
        y = diffusion.sample(x, stochastic=False)
    return y.squeeze().cpu().numpy() * scale


def plot_row(ax_time, ax_psd, clean_mV, noise_mV, den_mV, fs, title,
             show_legend=False):
    t = np.arange(clean_mV.size) / fs
    noisy_mV = clean_mV + noise_mV

    # time domain: noisy + 1-shot denoised
    ax_time.plot(t, noisy_mV, color='tab:red', lw=0.4, label='observation', alpha=0.8)
    ax_time.plot(t, den_mV, color='tab:purple', lw=1.0, label='denoised')
    ax_time.set_xlabel('time (s)')
    ax_time.set_ylabel('amplitude (mV)')
    ax_time.set_title(title)
    if show_legend:
        ax_time.legend(loc='upper right', fontsize=8, frameon=False)

    # PSD: noisy, clean, noise, denoised
    f_c, p_c = compute_psd(clean_mV, fs)
    f_n, p_n = compute_psd(noise_mV, fs)
    f_o, p_o = compute_psd(noisy_mV, fs)
    f_d, p_d = compute_psd(den_mV, fs)
    ax_psd.loglog(f_o, p_o, color='tab:red', lw=1.3, label='observation', alpha=0.8)
    ax_psd.loglog(f_n, p_n, color='tab:gray', lw=0.8, label='noise', alpha=0.8)
    ax_psd.loglog(f_c, p_c, color='tab:blue', lw=0.8, label='clean')
    ax_psd.loglog(f_d, p_d, color='tab:purple', lw=0.8, label='denoised')
    ax_psd.set_xlabel('frequency (Hz)')
    ax_psd.set_ylabel('PSD (mV$^2$/Hz)')
    if show_legend:
        ax_psd.legend(loc='lower left', fontsize=7, frameon=False)
    ax_psd.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--out', default='/home/yunshan/cuore/plots/paper/denoising_examples.pdf')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model once
    schedule = DiffusionSchedule(T=args.T).to(device)
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule)

    # Same config as plot_datasets_paper.py
    rows = [
        (CLEAN_LOW,  0, 0, None,  'example 1'),
        (CLEAN_HIGH, 0, 0, 200.0, 'example 2'),
        (CLEAN_LOW,  1, 1, None,  'example 3'),
        (CLEAN_HIGH, 1, 0, 200.0, 'example 4'),
    ]
    with h5py.File(NOISE_H5, 'r') as nf:
        n_noise = nf['waveforms'].shape[0]
        noise_idxs = rng.choice(n_noise, size=len(rows), replace=False)

    configs = []
    for (path, pu, off, min_amp, title), n_idx in zip(rows, noise_idxs):
        c_idx = pick_index(path, pu, off, min_amp_mV=min_amp)
        configs.append((path, c_idx, int(n_idx), title))

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.2 * len(rows)))

    with h5py.File(NOISE_H5, 'r') as nf:
        noise_all = nf['waveforms']
        for row, (path, c_idx, n_idx, title) in enumerate(configs):
            clean_V, fs = load_waveform(path, c_idx)
            noise_V = noise_all[n_idx].astype(np.float64)
            noisy_V = clean_V + noise_V
            den_V = denoise(diffusion, device, noisy_V)
            clean_mV = clean_V * 1000.0
            noise_mV = noise_V * 1000.0
            den_mV = den_V * 1000.0
            print(f'row {row}: {title} clean_idx={c_idx} noise_idx={n_idx}')
            plot_row(axes[row, 0], axes[row, 1], clean_mV, noise_mV,
                     den_mV, fs, title, show_legend=(row == 0))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
