"""Plot DDPM reverse-diffusion trajectory snapshots for a single ~40 mV
non-pileup pulse. Produces 5 bare-axes PNGs (noisy + T=50, 30, 29, 0) for
paper figures.
"""
import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion


def bare_plot(y, color, out_path):
    y = y[::2]  # decimate to 5000 points (even indices)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(y, color=color, linewidth=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='/media/Disk_YIN/yunshancheng/cuore/ddpm_l1_low/best_model.pt')
    p.add_argument('--clean_h5', default='/media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/test/clean_000.h5')
    p.add_argument('--noise_h5', default='/media/Disk_YIN/yunshancheng/cuore/noise/test/noise_000.h5')
    p.add_argument('--clean_idx', type=int, default=1364)
    p.add_argument('--noise_idx', type=int, default=0)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='/home/yunshan/cuore/plots/paper')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File(args.clean_h5, 'r') as f:
        clean = f['waveforms'][args.clean_idx].astype(np.float32)
        is_pu = int(f['is_pileup'][args.clean_idx])
    with h5py.File(args.noise_h5, 'r') as f:
        noise = f['waveforms'][args.noise_idx].astype(np.float32)

    assert is_pu == 0, f"pulse {args.clean_idx} is pileup"

    noisy = clean + noise
    scale = float(np.max(np.abs(noisy)))
    clean_n = clean / scale
    noisy_n = noisy / scale

    schedule = DiffusionSchedule(T=args.T).to(device)
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule)

    x_tilde = torch.from_numpy(noisy_n)[None, None, :].to(device)

    _, traj = diffusion.sample(x_tilde, return_trajectory=True, stochastic=True)
    # traj length = T+1. traj[k] = x_{T-k}. So x_T = traj[0], x_0 = traj[T].
    def x_at(t):
        k = args.T - t
        return traj[k].squeeze().cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)

    # Noisy observation (clean + noise) — red
    bare_plot(noisy_n, color='red', out_path=os.path.join(args.out_dir, 'noisy.png'))

    # Reverse trajectory snapshots — blue
    for t in (50, 20, 12, 0):
        y = x_at(t)
        bare_plot(y, color='blue',
                  out_path=os.path.join(args.out_dir, f'xt_{t:02d}.png'))

    print('amplitudes (physical, mV):', (np.max(clean) - np.median(clean[:200])) * 1000)
    print(f"wrote to {args.out_dir}: noisy.png, xt_50.png, xt_20.png, xt_12.png, xt_00.png")


if __name__ == '__main__':
    main()
