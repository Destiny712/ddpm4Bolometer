"""
Inference and evaluation for trained DDPM pulse denoiser.

Supports single-shot and multi-shot (M=10) denoising with visualization.
Evaluates 4 metrics from DeScoD-ECG: MSE, MAD, PRD, Cosine Similarity.

Usage:
    python -u -m src.ddpm.inference \
        --model_path /media/AVFD/yunshancheng/cuore/ddpm_test/best_model.pt \
        --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_000.h5 \
        --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_000.h5 \
        --output qa_inference.png \
        --n 5
"""

import argparse

import h5py
import numpy as np
from scipy.signal import welch, find_peaks
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.unet import UNet1D
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.dataset import PulseNoiseDataset


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((clean - denoised) ** 2))


def compute_mad(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Maximum absolute distance."""
    return float(np.max(np.abs(clean - denoised)))


def compute_prd(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Percentage root-mean-square difference (Eq. 20 in DeScoD-ECG)."""
    num = np.sum((clean - denoised) ** 2)
    denom = np.sum((clean - np.mean(clean)) ** 2)
    if denom == 0:
        return float('inf')
    return float(np.sqrt(num / denom) * 100.0)


def compute_cosine_sim(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Cosine similarity."""
    dot = np.dot(clean, denoised)
    norm_c = np.linalg.norm(clean)
    norm_d = np.linalg.norm(denoised)
    if norm_c == 0 or norm_d == 0:
        return 0.0
    return float(dot / (norm_c * norm_d))


def compute_cc(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Normalized cross-correlation (mean-subtracted)."""
    c = clean - np.mean(clean)
    d = denoised - np.mean(denoised)
    norm = np.linalg.norm(c) * np.linalg.norm(d)
    if norm == 0:
        return 0.0
    return float(np.dot(c, d) / norm)


def compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """SNR in dB: 10*log10(signal_power / error_power)."""
    signal_power = np.sum(clean ** 2)
    error_power = np.sum((clean - denoised) ** 2)
    if error_power == 0:
        return float('inf')
    return float(10.0 * np.log10(signal_power / error_power))


def compute_sc(clean: np.ndarray, denoised: np.ndarray,
               n_fft: int = 1024) -> float:
    """Spectral Convergence: ||STFT(clean)-STFT(den)||_F / ||STFT(clean)||_F."""
    hop = n_fft // 4
    S_c = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        clean, n_fft)[::hop] * np.hanning(n_fft)))
    S_d = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        denoised, n_fft)[::hop] * np.hanning(n_fft)))
    return float(np.linalg.norm(S_c - S_d) / (np.linalg.norm(S_c) + 1e-10))


def compute_lsd(clean: np.ndarray, denoised: np.ndarray,
                n_fft: int = 1024) -> float:
    """Log-Spectral Distance: RMS of log-power difference across frames."""
    hop = n_fft // 4
    S_c = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        clean, n_fft)[::hop] * np.hanning(n_fft)))
    S_d = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        denoised, n_fft)[::hop] * np.hanning(n_fft)))
    log_c = np.log10(np.maximum(S_c, 1e-10))
    log_d = np.log10(np.maximum(S_d, 1e-10))
    lsd_per_frame = np.sqrt(np.mean((log_c - log_d) ** 2, axis=1))
    return float(np.mean(lsd_per_frame))


def compute_jasd(clean: np.ndarray, denoised: np.ndarray,
                 fs: float = 1000.0, nperseg: int = 1024) -> float:
    """J_asd: mean over f of sqrt(PSD_residual / PSD_clean)."""
    residual = denoised - clean
    f, psd_res = welch(residual, fs=fs, nperseg=nperseg)
    _, psd_cln = welch(clean, fs=fs, nperseg=nperseg)
    ratio = np.sqrt(psd_res / (psd_cln + 1e-20))
    return float(np.mean(ratio))


def parabolic_peak(signal: np.ndarray, idx: int) -> tuple:
    """Parabolic interpolation around a discrete peak for sub-sample precision.

    Returns (interpolated_position, interpolated_amplitude).
    """
    if idx <= 0 or idx >= len(signal) - 1:
        return float(idx), float(signal[idx])
    y0, y1, y2 = signal[idx - 1], signal[idx], signal[idx + 1]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-20:
        return float(idx), float(y1)
    delta = (y0 - y2) / denom
    amp = y1 - 0.25 * (y0 - y2) * delta
    return float(idx + delta), float(amp)


def detect_peaks(signal: np.ndarray, n_expected: int) -> list:
    """Detect peaks using find_peaks, return list of (position, amplitude).

    Uses parabolic interpolation for sub-sample precision.
    n_expected: 1 for single pulse, 2 for pileup.
    Returns sorted by position (time order).
    """
    peaks, props = find_peaks(signal, prominence=0.01 * np.max(signal),
                              distance=50)
    if len(peaks) == 0:
        # Fallback: use global max
        idx = int(np.argmax(signal))
        pos, amp = parabolic_peak(signal, idx)
        return [(pos, amp)]

    # Sort by prominence descending, take top n_expected
    order = np.argsort(-props['prominences'])
    selected = sorted(peaks[order[:n_expected]])

    result = []
    for idx in selected:
        pos, amp = parabolic_peak(signal, idx)
        result.append((pos, amp))
    return result


def compute_all_metrics(clean: np.ndarray, signal: np.ndarray) -> dict:
    """Compute all evaluation metrics (without amplitude — added separately)."""
    return {
        'mse': compute_mse(clean, signal),
        'mad': compute_mad(clean, signal),
        'prd': compute_prd(clean, signal),
        'cosine_sim': compute_cosine_sim(clean, signal),
        'cc': compute_cc(clean, signal),
        'snr': compute_snr(clean, signal),
        'sc': compute_sc(clean, signal),
        'lsd': compute_lsd(clean, signal),
        'jasd': compute_jasd(clean, signal),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDPM inference and QA.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_inference.png')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of examples to visualize')
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated sample indices (overrides --n and --seed)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter samples by category. Built-in: pileup, single, '
                             'low_pileup (=pileup_low_100). With index file: also '
                             'low_100, low_200, pileup_low_200, or any custom category.')
    parser.add_argument('--scale_cond', action='store_true',
                        help='Use scale-conditioned U-Net (UNet1DScaleCond)')
    parser.add_argument('--aggregation', choices=['mean', 'median'], default='mean',
                        help='Multi-shot aggregation: mean (default) or median')
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm',
                        help='Sampling method: ddpm (stochastic) or ddim (deterministic)')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='Number of DDIM steps (default: same as T)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta: 0=deterministic, 1=DDPM-equivalent')
    parser.add_argument('--no_noise', action='store_true',
                        help='Deterministic DDPM: skip noise in reverse steps')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    schedule = DiffusionSchedule(T=args.T).to(device)
    if args.scale_cond:
        from src.ddpm.unet_cond import UNet1DScaleCond
        model = UNet1DScaleCond().to(device)
        print("Using scale-conditioned U-Net")
    else:
        model = UNet1D().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule)

    # Load test samples
    dataset = PulseNoiseDataset(args.clean_dir, args.noise_dir)

    # Load pileup flags from HDF5
    clean_path = args.clean_dir
    with h5py.File(clean_path, 'r') as f:
        is_pileup_all = f['is_pileup'][:]

    # Try loading precomputed index
    from src.ddpm.dataset import load_index
    precomputed_index = load_index(clean_path)

    # Map filter names to index categories
    _filter_to_category = {
        'pileup': 'pileup',
        'single': 'single',
        'low_pileup': 'pileup_low_100',
        'low_200': 'low_200',
        'low_100': 'low_100',
        'pileup_low_200': 'pileup_low_200',
    }

    # Select indices
    if args.indices:
        indices = [int(x) for x in args.indices.split(',')]
    elif args.filter:
        category = _filter_to_category.get(args.filter, args.filter)
        if precomputed_index and category in precomputed_index:
            pool = np.array(precomputed_index[category])
        else:
            # Fallback: compute on the fly
            print(f"Warning: no precomputed index for '{category}', computing from HDF5...")
            with h5py.File(clean_path, 'r') as f:
                waveforms_max = f['waveforms'][:].max(axis=1)
            if args.filter == 'pileup':
                pool = np.where(is_pileup_all)[0]
            elif args.filter == 'single':
                pool = np.where(~is_pileup_all)[0]
            elif args.filter == 'low_pileup':
                pool = np.where(is_pileup_all & (waveforms_max < 0.1))[0]
            else:
                raise ValueError(f"Unknown filter '{args.filter}' and no index file found. "
                                 f"Run preprocess_index.py first.")
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(pool, size=min(args.n, len(pool)), replace=False).tolist()
        print(f"Filter '{args.filter}': {len(pool)} candidates, selected {len(indices)}")
    else:
        torch.manual_seed(args.seed)
        indices = torch.randperm(len(dataset))[:args.n].tolist()

    # ── Run both single-shot and 10-shot ──
    results = []
    for idx in indices:
        x_clean, x_noisy, scale = dataset[idx]
        scale = scale.item()
        pileup = bool(is_pileup_all[idx])
        n_peaks = 2 if pileup else 1

        # Denoise in normalized space
        x_noisy_dev = x_noisy.unsqueeze(0).to(device)
        scale_dev = torch.tensor([scale], device=device) if args.scale_cond else None
        tag = "pileup" if pileup else "single"
        sampler_label = args.sampler.upper()
        if args.sampler == 'ddim':
            ddim_s = args.ddim_steps or args.T
            sampler_label += f"({ddim_s})"
        print(f"Sample {idx} ({tag}, scale={scale*1e3:.1f} mV): {sampler_label} (single-shot)...", end='', flush=True)
        if args.sampler == 'ddim':
            x_single_norm = diffusion.ddim_sample(x_noisy_dev, scale=scale_dev, steps=args.ddim_steps, eta=args.eta).squeeze().cpu().numpy()
        else:
            x_single_norm = diffusion.sample(x_noisy_dev, scale=scale_dev, stochastic=not args.no_noise).squeeze().cpu().numpy()
        print(f" (10-shot {args.aggregation})...", end='', flush=True)
        x_multi_norm = diffusion.sample_multi_shot(
            x_noisy_dev, scale=scale_dev, M=10, aggregation=args.aggregation,
            sampler=args.sampler, ddim_steps=args.ddim_steps, eta=args.eta,
            stochastic=not args.no_noise,
        ).squeeze().cpu().numpy()
        print(" done")

        # Rescale back to physical units
        x_clean_np = x_clean.squeeze().numpy() * scale
        x_noisy_np = x_noisy.squeeze().numpy() * scale
        x_single = x_single_norm * scale
        x_multi = x_multi_norm * scale

        m_noisy = compute_all_metrics(x_clean_np, x_noisy_np)
        m_single = compute_all_metrics(x_clean_np, x_single)
        m_multi = compute_all_metrics(x_clean_np, x_multi)

        # Peak amplitudes via parabolic interpolation
        # Convention: peaks ordered by position (time), first = earliest
        clean_peaks = detect_peaks(x_clean_np, n_peaks)
        for label, signal, m in [('noisy', x_noisy_np, m_noisy),
                                  ('single', x_single, m_single),
                                  ('multi', x_multi, m_multi)]:
            sig_peaks = detect_peaks(signal, n_peaks)
            m['peaks'] = []
            # Match signal peaks to clean peaks by nearest position
            used = set()
            for i, (c_pos, c_amp) in enumerate(clean_peaks):
                best_j, best_dist = None, float('inf')
                for j, (s_pos, s_amp) in enumerate(sig_peaks):
                    if j in used:
                        continue
                    dist = abs(s_pos - c_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j is not None:
                    used.add(best_j)
                    s_pos, s_amp = sig_peaks[best_j]
                else:
                    s_pos, s_amp = 0.0, 0.0
                err_pct = (s_amp - c_amp) / c_amp * 100.0 if c_amp != 0 else 0.0
                # Timing error: position in ms (samples / 1 kHz)
                time_err_ms = s_pos - c_pos  # at 1 kHz, sample index = ms
                m['peaks'].append({
                    'clean_amp': c_amp, 'signal_amp': s_amp, 'err_pct': err_pct,
                    'clean_pos': c_pos, 'signal_pos': s_pos,
                    'time_err_ms': time_err_ms,
                })
        # Baseline RMS: t in [0, 1.5s) — before pulse onset
        n_baseline = int(1.5 * 1000.0)  # 1500 samples at 1 kHz
        for label, signal, m in [('noisy', x_noisy_np, m_noisy),
                                  ('single', x_single, m_single),
                                  ('multi', x_multi, m_multi)]:
            m['baseline_rms'] = float(np.sqrt(np.mean(signal[:n_baseline] ** 2)))

        m_noisy['is_pileup'] = pileup
        m_single['is_pileup'] = pileup
        m_multi['is_pileup'] = pileup

        results.append({
            'idx': idx,
            'clean': x_clean_np,
            'noisy': x_noisy_np,
            'single': x_single,
            'multi': x_multi,
            'm_noisy': m_noisy,
            'm_single': m_single,
            'm_multi': m_multi,
        })

    # ── Print metrics table ──
    all_keys = ['mse', 'mad', 'prd', 'cosine_sim', 'cc', 'snr',
                'sc', 'lsd', 'jasd']
    avg = {'noisy': {}, 'single': {}, 'multi': {}}
    for r in results:
        for key in all_keys:
            for label, mdict in [('noisy', r['m_noisy']),
                                  ('single', r['m_single']),
                                  ('multi', r['m_multi'])]:
                avg[label][key] = avg[label].get(key, 0) + mdict[key]

    n = len(results)

    # Average baseline RMS
    for label, mkey in [('noisy', 'm_noisy'), ('single', 'm_single'), ('multi', 'm_multi')]:
        avg[label]['baseline_rms'] = sum(r[mkey]['baseline_rms'] for r in results) / n

    # Average amplitude error across all peaks (1 per single, 2 per pileup)
    for label, mkey in [('noisy', 'm_noisy'), ('single', 'm_single'), ('multi', 'm_multi')]:
        total_err, count = 0.0, 0
        for r in results:
            for pk in r[mkey]['peaks']:
                total_err += abs(pk['err_pct'])
                count += 1
        avg[label]['amp_abs_err_pct'] = total_err / max(count, 1)
    for label in ['noisy', 'single', 'multi']:
        for key in all_keys:
            avg[label][key] /= n

    print(f"\n{'=' * 150}")
    print(f"{'':>8} | {'MSE':>10} {'MAD':>10} {'BL RMS':>10} {'PRD%':>8} {'Cosine':>8} {'CC':>8} "
          f"{'SNR dB':>8} {'SC':>8} {'LSD':>8} {'J_asd':>8} {'|Amp|%':>8}")
    print(f"{'-' * 150}")
    for label, name in [('noisy', 'Noisy'), ('single', '1-shot'), ('multi', '10-shot')]:
        m = avg[label]
        print(f"{name:>8} | {m['mse']:10.6f} {m['mad']:10.6f} {m['baseline_rms']*1e3:9.4f}m {m['prd']:8.2f} "
              f"{m['cosine_sim']:8.5f} {m['cc']:8.5f} {m['snr']:8.2f} "
              f"{m['sc']:8.4f} {m['lsd']:8.4f} {m['jasd']:8.4f} "
              f"{m['amp_abs_err_pct']:8.2f}")
    print(f"{'=' * 150}")

    # ── Plot ──
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(28, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        t = np.arange(len(r['clean'])) / 1000.0

        # Col 0: waveforms overlay
        ax = axes[row, 0]
        ax.plot(t, r['single'] * 1e3, 'g-', lw=0.5, alpha=0.7, label='1-shot')
        ax.plot(t, r['multi'] * 1e3, 'm-', lw=0.5, alpha=0.7, label='10-shot')
        ax.plot(t, r['clean'] * 1e3, 'b-', lw=0.5, label='Clean')
        ax.plot(t, r['noisy'] * 1e3, 'r-', lw=0.3, alpha=0.5, label='Noisy')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_title(f"Sample {r['idx']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Col 1: residuals
        ax2 = axes[row, 1]
        ax2.plot(t, (r['clean'] - r['single']) * 1e3, 'g-', lw=0.4, alpha=0.7, label='1-shot')
        ax2.plot(t, (r['clean'] - r['multi']) * 1e3, 'm-', lw=0.4, alpha=0.7, label='10-shot')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Residual [mV]')
        ax2.set_title('Clean - Denoised', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        # Col 2: PSD comparison
        ax_psd = axes[row, 2]
        fs = 1000.0
        nperseg = min(1024, len(r['clean']))
        f_clean, psd_clean = welch(r['clean'], fs=fs, nperseg=nperseg)
        f_noisy, psd_noisy = welch(r['noisy'], fs=fs, nperseg=nperseg)
        f_multi, psd_multi = welch(r['multi'], fs=fs, nperseg=nperseg)
        ax_psd.semilogy(f_clean, psd_clean, 'b-', lw=0.8, label='Clean')
        ax_psd.semilogy(f_noisy, psd_noisy, 'r-', lw=0.5, alpha=0.6, label='Noisy')
        ax_psd.semilogy(f_multi, psd_multi, 'm-', lw=0.8, label='10-shot')
        ax_psd.set_xlabel('Frequency [Hz]')
        ax_psd.set_ylabel('PSD [V²/Hz]')
        ax_psd.set_title('Power Spectral Density', fontsize=9)
        ax_psd.legend(fontsize=6)
        ax_psd.grid(True, alpha=0.3)

        # Col 3: metrics text
        ax3 = axes[row, 3]
        ax3.axis('off')
        mn, ms, mm = r['m_noisy'], r['m_single'], r['m_multi']
        pileup = mn['is_pileup']
        tag = "PILEUP" if pileup else "SINGLE"
        lines = [
            f"  {tag}  Sample {r['idx']}",
            f"{'Metric':>12}  {'Noisy':>10}  {'1-shot':>10}  {'10-shot':>10}",
            f"{'MSE':>12}  {mn['mse']:10.6f}  {ms['mse']:10.6f}  {mm['mse']:10.6f}",
            f"{'MAD':>12}  {mn['mad']*1e3:9.3f}m  {ms['mad']*1e3:9.3f}m  {mm['mad']*1e3:9.3f}m",
            f"{'BL RMS':>12}  {mn['baseline_rms']*1e3:9.4f}m  {ms['baseline_rms']*1e3:9.4f}m  {mm['baseline_rms']*1e3:9.4f}m",
            f"{'PRD (%)':>12}  {mn['prd']:10.2f}  {ms['prd']:10.2f}  {mm['prd']:10.2f}",
            f"{'Cosine':>12}  {mn['cosine_sim']:10.6f}  {ms['cosine_sim']:10.6f}  {mm['cosine_sim']:10.6f}",
            f"{'CC':>12}  {mn['cc']:10.6f}  {ms['cc']:10.6f}  {mm['cc']:10.6f}",
            f"{'SNR (dB)':>12}  {mn['snr']:10.2f}  {ms['snr']:10.2f}  {mm['snr']:10.2f}",
            f"{'SC':>12}  {mn['sc']:10.4f}  {ms['sc']:10.4f}  {mm['sc']:10.4f}",
            f"{'LSD':>12}  {mn['lsd']:10.4f}  {ms['lsd']:10.4f}  {mm['lsd']:10.4f}",
            f"{'J_asd':>12}  {mn['jasd']:10.4f}  {ms['jasd']:10.4f}  {mm['jasd']:10.4f}",
        ]
        # Peak amplitude and timing rows
        n_pk = len(mn['peaks'])
        for i in range(n_pk):
            pk_label = f"Pk{i+1}" if n_pk > 1 else "Pk"
            c_amp = mn['peaks'][i]['clean_amp']
            c_pos = mn['peaks'][i]['clean_pos']
            lines.append(f"  {pk_label} clean: {c_amp*1e3:.3f} mV @ {c_pos:.1f} ms")
            lines.append(
                f"{pk_label+' amp mV':>12}  "
                f"{mn['peaks'][i]['signal_amp']*1e3:10.3f}  "
                f"{ms['peaks'][i]['signal_amp']*1e3:10.3f}  "
                f"{mm['peaks'][i]['signal_amp']*1e3:10.3f}"
            )
            lines.append(
                f"{pk_label+' amp%':>12}  "
                f"{mn['peaks'][i]['err_pct']:+10.2f}  "
                f"{ms['peaks'][i]['err_pct']:+10.2f}  "
                f"{mm['peaks'][i]['err_pct']:+10.2f}"
            )
            lines.append(
                f"{pk_label+' dt ms':>12}  "
                f"{mn['peaks'][i]['time_err_ms']:+10.2f}  "
                f"{ms['peaks'][i]['time_err_ms']:+10.2f}  "
                f"{mm['peaks'][i]['time_err_ms']:+10.2f}"
            )
        text = '\n'.join(lines)
        ax3.text(0.02, 0.5, text, transform=ax3.transAxes, fontsize=7,
                 verticalalignment='center', fontfamily='monospace')

    sampler_info = args.sampler.upper()
    if args.sampler == 'ddim':
        sampler_info += f" (steps={args.ddim_steps or args.T}, eta={args.eta})"
    fig.suptitle(f'{sampler_info}: 1-shot vs 10-shot ({args.aggregation})', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
