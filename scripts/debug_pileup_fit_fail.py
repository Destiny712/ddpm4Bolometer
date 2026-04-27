"""
TEMP debug script — investigate pileup windows where biexp fitting fails.

Runs DDPM inference on a handful of pileup samples, fits clean/noisy/1-shot/
10-shot with biexp_double, flags any fit that (a) raised in curve_fit or
(b) rail-pinned a tau bound, and plots data+fit overlay for the bad windows.
Prints best-fit params, the parameter bounds, and chi2/ndf to stdout.

Usage:
    python3 -u scripts/debug_pileup_fit_fail.py \
        --model_path /media/Disk_YIN/yunshancheng/cuore/ddpm_l1_low/best_model.pt \
        --clean_dir /media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/test/clean_000.h5 \
        --noise_dir /media/Disk_YIN/yunshancheng/cuore/noise/test/noise_000.h5 \
        --output /tmp/pileup_fit_fail.png \
        --n_pileup 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import PulseNoiseDataset
from src.basics.fit import (
    fit_pulse, biexp_single, biexp_double,
    _TAU_R_MIN, _TAU_R_MAX, _TAU_D_MIN, _TAU_D_MAX,
)


def rail_hit(fr, tol_frac=0.01):
    """Return (is_bad, reason). Bad = curve_fit failed or tau pinned at bound."""
    if not fr.success:
        return True, "curve_fit failed"
    tol_r = tol_frac * (_TAU_R_MAX - _TAU_R_MIN)
    tol_d = tol_frac * (_TAU_D_MAX - _TAU_D_MIN)
    for i, (tr, td) in enumerate(zip(fr.tau_r, fr.tau_d)):
        if tr - _TAU_R_MIN < tol_r:
            return True, f"P{i+1} tau_r at lower bound"
        if _TAU_R_MAX - tr < tol_r:
            return True, f"P{i+1} tau_r at upper bound"
        if td - _TAU_D_MIN < tol_d:
            return True, f"P{i+1} tau_d at lower bound"
        if _TAU_D_MAX - td < tol_d:
            return True, f"P{i+1} tau_d at upper bound"
    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--noise_dir', required=True)
    p.add_argument('--output', default='/tmp/pileup_fit_fail.png')
    p.add_argument('--n_pileup', type=int, default=30)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    schedule = DiffusionSchedule(T=args.T).to(device)
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule)

    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)
    with h5py.File(args.clean_dir, 'r') as f:
        is_pileup_all = f['is_pileup'][:]

    pool = np.where(is_pileup_all)[0]
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(pool, size=min(args.n_pileup, len(pool)), replace=False)
    print(f"Selected {len(indices)} pileup samples from pool of {len(pool)}")

    # Parameter bounds (print once)
    print(f"\nFit parameter bounds (9 params for pileup):")
    print(f"  B     in (-inf, +inf)")
    print(f"  A_i   in [0, +inf)")
    print(f"  t0_i  in [0, 10] s")
    print(f"  tau_r in [{_TAU_R_MIN*1e3:.1f}, {_TAU_R_MAX*1e3:.1f}] ms")
    print(f"  tau_d in [{_TAU_D_MIN*1e3:.1f}, {_TAU_D_MAX*1e3:.1f}] ms")

    # Denoise + fit each pileup sample
    entries = []  # list of dicts with fits and signals
    for idx in indices:
        x_clean, x_noisy, scale = dataset[idx]
        scale = scale.item()
        x_noisy_dev = x_noisy.unsqueeze(0).to(device)

        x_single_norm = diffusion.sample(x_noisy_dev).squeeze().cpu().numpy()
        x_multi_norm = diffusion.sample_multi_shot(
            x_noisy_dev, M=10, aggregation='mean').squeeze().cpu().numpy()

        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale
        x_single = x_single_norm * scale
        x_multi = x_multi_norm * scale

        fits = {
            'clean':  fit_pulse(x_clean_np, 2),
            'noisy':  fit_pulse(x_noisy_np, 2),
            'single': fit_pulse(x_single, 2),
            'multi':  fit_pulse(x_multi, 2),
        }
        signals = {'clean': x_clean_np, 'noisy': x_noisy_np,
                   'single': x_single, 'multi': x_multi}

        bad = {}
        for src in ['noisy', 'single', 'multi']:
            is_bad, why = rail_hit(fits[src])
            bad[src] = (is_bad, why)

        any_bad = any(v[0] for v in bad.values())
        entries.append(dict(idx=int(idx), scale=scale, fits=fits,
                            signals=signals, bad=bad, any_bad=any_bad))
        print(f"  idx={idx:5d}  any_bad={any_bad}  "
              f"noisy={bad['noisy'][1] or 'ok'}  "
              f"1shot={bad['single'][1] or 'ok'}  "
              f"10shot={bad['multi'][1] or 'ok'}")

    failed = [e for e in entries if e['any_bad']]
    print(f"\n{len(failed)}/{len(entries)} pileup windows have at least one bad fit")

    if not failed:
        print("No failures — skipping plot.")
        return

    # Detailed report
    print(f"\n{'='*80}\nDetails of failing windows\n{'='*80}")
    for e in failed:
        print(f"\nWindow {e['idx']}  (scale={e['scale']*1e3:.1f} mV)")
        for src in ['clean', 'noisy', 'single', 'multi']:
            fr = e['fits'][src]
            flag = ''
            if src in e['bad']:
                flag = ' ❌ ' + e['bad'][src][1] if e['bad'][src][0] else ' ok'
            print(f"  [{src:6s}]{flag}  chi2/ndf={fr.chi2_per_ndf:.4e}  "
                  f"success={fr.success}")
            print(f"      B = {fr.baseline*1e3:+.4f} mV")
            for i, (A, t0, tr, td, tp) in enumerate(zip(
                    fr.peak_amps, fr.onset_times, fr.tau_r,
                    fr.tau_d, fr.peak_times)):
                print(f"      P{i+1}: A={A*1e3:+8.3f} mV  t0={t0:7.3f} s  "
                      f"tau_r={tr*1e3:6.1f} ms  tau_d={td*1e3:6.1f} ms  "
                      f"t_peak={tp:7.3f} s")

    # Plot: rows = failing windows, cols = (clean, noisy, 1-shot, 10-shot)
    n_rows = len(failed)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.0 * n_rows), squeeze=False)
    col_labels = ['clean', 'noisy', 'single', 'multi']
    col_names = {'clean': 'Clean', 'noisy': 'Noisy',
                 'single': '1-shot', 'multi': '10-shot'}
    fs = 1000.0

    for row, e in enumerate(failed):
        t = np.arange(len(e['signals']['clean'])) / fs
        for col, src in enumerate(col_labels):
            ax = axes[row, col]
            sig = e['signals'][src]
            fr = e['fits'][src]
            ax.plot(t, sig * 1e3, 'k-', lw=0.5, alpha=0.7, label='data')
            fit_curve = biexp_double(t, *fr.params)
            is_bad, why = (False, '')
            if src in e['bad']:
                is_bad, why = e['bad'][src]
            fit_color = 'red' if is_bad else 'tab:blue'
            ax.plot(t, fit_curve * 1e3, color=fit_color, lw=0.9, label='fit')
            status = f'❌ {why}' if is_bad else 'ok'
            title_color = 'red' if is_bad else 'black'
            ax.set_title(f"{col_names[src]}  chi2/ndf={fr.chi2_per_ndf:.2e}\n{status}",
                         fontsize=8, color=title_color)
            ax.set_xlabel('Time [s]', fontsize=7)
            ax.set_ylabel('mV', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Best-fit param text box
            param_lines = [f"B = {fr.baseline*1e3:+.2f} mV"]
            for i, (A, t0, tr, td) in enumerate(zip(
                    fr.peak_amps, fr.onset_times, fr.tau_r, fr.tau_d)):
                param_lines.append(
                    f"P{i+1}: A={A*1e3:+.2f} mV  t0={t0:.3f} s\n"
                    f"    tau_r={tr*1e3:.1f} ms  tau_d={td*1e3:.0f} ms"
                )
            ax.text(0.98, 0.02, '\n'.join(param_lines),
                    transform=ax.transAxes, fontsize=6,
                    ha='right', va='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.75,
                              edgecolor='gray', lw=0.3))
            if col == 0:
                ax.text(-0.25, 0.5, f"idx={e['idx']}",
                        transform=ax.transAxes, fontsize=9,
                        rotation=90, va='center', ha='center',
                        fontweight='bold')

    fig.suptitle(f'Pileup fit failures — {len(failed)}/{len(entries)} windows',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
