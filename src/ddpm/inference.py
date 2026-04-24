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
import time

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


def compute_mae(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(clean - denoised)))


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
                n_fft: int = 1024, fs: float = 1000.0,
                f_max: float = 100.0, floor_rel: float = 1e-4) -> float:
    """Log-Spectral Distance: RMS of log-power difference across frames.

    Only frequency bins f <= f_max are included (signal band). Both spectra
    are floored at floor_rel * max(|S_clean|) to avoid dominance by empty
    bins where |S| is near zero.
    """
    hop = n_fft // 4
    S_c = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        clean, n_fft)[::hop] * np.hanning(n_fft)))
    S_d = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        denoised, n_fft)[::hop] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    band = freqs <= f_max
    S_c = S_c[:, band]
    S_d = S_d[:, band]
    floor = floor_rel * max(np.max(S_c), 1e-30)
    log_c = np.log10(np.maximum(S_c, floor))
    log_d = np.log10(np.maximum(S_d, floor))
    lsd_per_frame = np.sqrt(np.mean((log_c - log_d) ** 2, axis=1))
    return float(np.mean(lsd_per_frame))


def compute_jasd(clean: np.ndarray, denoised: np.ndarray,
                 fs: float = 1000.0, nperseg: int = 1024,
                 f_max: float = 100.0, floor_rel: float = 1e-4) -> float:
    """J_asd: mean over f of sqrt(PSD_residual / PSD_clean).

    Restricted to f <= f_max (signal band). The clean PSD denominator is
    floored at floor_rel * max(PSD_clean) so empty high-frequency bins do
    not dominate the average.
    """
    residual = denoised - clean
    f, psd_res = welch(residual, fs=fs, nperseg=nperseg)
    _, psd_cln = welch(clean, fs=fs, nperseg=nperseg)
    band = f <= f_max
    psd_res = psd_res[band]
    psd_cln = psd_cln[band]
    floor = floor_rel * max(np.max(psd_cln), 1e-30)
    ratio = np.sqrt(psd_res / np.maximum(psd_cln, floor))
    return float(np.mean(ratio))


def parabolic_peak(signal: np.ndarray, idx: int, half_win: int = 7) -> tuple:
    """Least-squares parabolic fit around a discrete peak for sub-sample precision.

    Fits y = a*(x-idx)^2 + b*(x-idx) + c over 2*half_win+1 points centered
    on idx. The vertex gives sub-sample position and interpolated amplitude.

    Returns (interpolated_position, interpolated_amplitude).
    """
    lo = max(0, idx - half_win)
    hi = min(len(signal), idx + half_win + 1)
    if hi - lo < 3:
        return float(idx), float(signal[idx])
    x = np.arange(lo, hi, dtype=np.float64) - idx
    y = signal[lo:hi].astype(np.float64)
    # Fit y = a*x^2 + b*x + c
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    if abs(a) < 1e-20 or a > 0:
        # Not concave down — just return discrete peak
        return float(idx), float(signal[idx])
    delta = -b / (2.0 * a)
    amp = c - b * b / (4.0 * a)
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


def compute_all_metrics(clean: np.ndarray, signal: np.ndarray,
                        clean_norm: np.ndarray,
                        signal_norm: np.ndarray) -> dict:
    """Compute all evaluation metrics (without amplitude — added separately).

    Parameters
    ----------
    clean, signal : physical-unit arrays
    clean_norm, signal_norm : normalized arrays (divided by scale). Used for
        scale-invariant NMSE/NMAE and normalized-space MSE/MAE/MAD.
    """
    mean_clean_sq = float(np.mean(clean ** 2))
    mean_abs_clean = float(np.mean(np.abs(clean)))
    mse = compute_mse(clean, signal)
    mae = compute_mae(clean, signal)

    return {
        'mse': mse,
        'mae': mae,
        'mad': compute_mad(clean, signal),
        'nmse': mse / mean_clean_sq if mean_clean_sq > 0 else float('nan'),
        'nmae': mae / mean_abs_clean if mean_abs_clean > 0 else float('nan'),
        'norm_mse': compute_mse(clean_norm, signal_norm),
        'norm_mae': compute_mae(clean_norm, signal_norm),
        'norm_mad': compute_mad(clean_norm, signal_norm),
        'prd': compute_prd(clean, signal),
        'cosine_sim': compute_cosine_sim(clean, signal),
        'cc': compute_cc(clean, signal),
        'snr': compute_snr(clean, signal),
        'sc': compute_sc(clean, signal),
        'lsd': compute_lsd(clean, signal),
        'jasd': compute_jasd(clean, signal),
    }


# ── Scatter plots ────────────────────────────────────────────────────────────

# Metric order for scatter grid. Each entry is a dict:
#   key: metric name in m_* dicts (or 'peak_amp_err'/'peak_time_err')
#   label: y-axis label
#   is_peak: True → pileup contributes 2 points (one per peak)
#   transform: optional fn applied to y-values (e.g. 1 - CC)
#   ymax: optional upper y-axis clip
_SCATTER_METRICS = [
    dict(key='mse',           label='MSE'),
    dict(key='mae',           label='MAE'),
    dict(key='mad',           label='MAD'),
    dict(key='nmse',          label='NMSE'),
    dict(key='nmae',          label='NMAE'),
    dict(key='norm_mse',      label='Norm MSE'),
    dict(key='norm_mae',      label='Norm MAE'),
    dict(key='norm_mad',      label='Norm MAD'),
    dict(key='baseline_rms',  label='BL RMS'),
    dict(key='baseline_rms_norm', label='BL RMS (norm)'),
    dict(key='chi2_per_ndf',  label='chi2/ndf'),
    dict(key='prd',           label='PRD (%)'),
    dict(key='cosine_sim',    label='1 - Cosine',  transform=lambda v: 1.0 - v),
    dict(key='cc',            label='1 - CC',      transform=lambda v: 1.0 - v),
    dict(key='snr',           label='SNR (dB)'),
    dict(key='sc',            label='SC'),
    dict(key='lsd',           label='LSD'),
    dict(key='jasd',          label='J_asd'),
    dict(key='peak_amp_err',  label='|Amp| err %', is_peak=True, ymax=100.0),
    dict(key='peak_time_err', label='|dt| [ms]',   is_peak=True),
]


def _collect_series(results, mkey, metric_key, is_peak):
    """Return (x_snr, y_metric) arrays for one signal source.

    mkey: 'm_noisy', 'm_single', 'm_multi' — which signal the metric is for.
    Pileup events contribute 2 points for peak metrics, 1 for others.
    x-axis is always NOISY SNR (so pileup peak metrics repeat the x value).
    """
    xs, ys = [], []
    for r in results:
        snr_x = r['m_noisy']['snr']
        m = r[mkey]
        if is_peak:
            for pk in m['peaks']:
                if metric_key == 'peak_amp_err':
                    ys.append(abs(pk['err_pct']))
                elif metric_key == 'peak_time_err':
                    ys.append(abs(pk['time_err_ms']))
                xs.append(snr_x)
        else:
            v = m[metric_key]
            if np.isfinite(v):
                xs.append(snr_x)
                ys.append(v)
    return np.array(xs), np.array(ys)


def plot_metric_scatter(results, output_path, title_suffix=''):
    """4xN scatter grid: each metric vs noisy SNR, overlaid for
    noisy / 1-shot / 10-shot."""
    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    sources = [
        ('m_noisy',  'Noisy',   'r'),
        ('m_single', '1-shot',  'g'),
        ('m_multi',  '10-shot', 'm'),
    ]
    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        ymax = mdef.get('ymax', None)
        for mkey, name, color in sources:
            xs, ys = _collect_series(results, mkey, mdef['key'], is_peak)
            if tf is not None and len(ys):
                ys = tf(ys)
            ax.scatter(xs, ys, s=6, c=color, alpha=0.5, label=name)
        ax.set_xlabel('Noisy SNR [dB]', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    # Hide unused axes
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(f'Metrics vs noisy SNR{title_suffix}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_pileup_scatter(results, output_path, title_suffix=''):
    """4xN scatter grid comparing non-pileup vs pileup, 1-shot vs 10-shot."""
    non_pileup = [r for r in results if not r['m_noisy']['is_pileup']]
    pileup = [r for r in results if r['m_noisy']['is_pileup']]

    series = [
        (non_pileup, 'm_single', 'non-pileup 1-shot',  'g', 'o'),
        (non_pileup, 'm_multi',  'non-pileup 10-shot', 'b', 'o'),
        (pileup,     'm_single', 'pileup 1-shot',      'orange', '^'),
        (pileup,     'm_multi',  'pileup 10-shot',     'r', '^'),
    ]

    n = len(_SCATTER_METRICS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        ymax = mdef.get('ymax', None)
        for subset, mkey, name, color, marker in series:
            if not subset:
                continue
            xs, ys = _collect_series(subset, mkey, mdef['key'], is_peak)
            if tf is not None and len(ys):
                ys = tf(ys)
            ax.scatter(xs, ys, s=10, c=color, marker=marker, alpha=0.6, label=name)
        ax.set_xlabel('Noisy SNR [dB]', fontsize=8)
        ax.set_ylabel(mdef['label'], fontsize=8)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(f'Non-pileup vs pileup{title_suffix}  '
                 f'(n_nonpu={len(non_pileup)}, n_pu={len(pileup)})',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDPM inference and QA.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noise_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='qa_inference.png')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of examples to process (scatter stats)')
    parser.add_argument('--n_plot', type=int, default=10,
                        help='Number of examples in per-window QA figure (<= n)')
    parser.add_argument('--scatter_output', type=str, default=None,
                        help='Path for metric-vs-SNR scatter figure')
    parser.add_argument('--scatter_pileup_output', type=str, default=None,
                        help='Path for pileup-vs-non-pileup scatter figure')
    parser.add_argument('--results_pkl', type=str, default=None,
                        help='Pickle path for stripped per-window metric dicts. '
                             'Saved after inference; load with scripts/replot_scatter.py.')
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--beta_1', type=float, default=1e-4,
                        help='Must match training value (default: 1e-4)')
    parser.add_argument('--beta_T', type=float, default=0.05,
                        help='Must match training value (default: 0.05)')
    parser.add_argument('--cond_mode', type=str, default='step',
                        choices=['step', 'sqrt_ab'],
                        help='Must match training value (default: step)')
    parser.add_argument('--cond_scale', type=float, default=1000.0,
                        help="Must match training value; ignored in 'step' mode")
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
    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1, beta_T=args.beta_T).to(device)
    print(f"Schedule: T={args.T}, beta_1={args.beta_1}, beta_T={args.beta_T}, "
          f"alpha_bar_T={schedule.alpha_bar[-1].item():.4g}")
    if args.scale_cond:
        from src.ddpm.unet_cond import UNet1DScaleCond
        model = UNet1DScaleCond(cond_mode=args.cond_mode,
                                cond_scale=args.cond_scale).to(device)
        print(f"Using scale-conditioned U-Net, cond_mode={args.cond_mode}")
    else:
        model = UNet1D(cond_mode=args.cond_mode,
                       cond_scale=args.cond_scale).to(device)
        print(f"Using UNet1D, cond_mode={args.cond_mode}")
    model.load_state_dict(torch.load(args.model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    diffusion = GaussianDiffusion(model, schedule, cond_mode=args.cond_mode)

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
    t_single_list = []
    t_multi_list = []
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
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.sampler == 'ddim':
            x_single_out = diffusion.ddim_sample(x_noisy_dev, scale=scale_dev, steps=args.ddim_steps, eta=args.eta)
        else:
            x_single_out = diffusion.sample(x_noisy_dev, scale=scale_dev, stochastic=not args.no_noise)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_single = time.perf_counter() - t0
        t_single_list.append(t_single)
        x_single_norm = x_single_out.squeeze().cpu().numpy()

        print(f" ({t_single*1e3:.1f} ms) (10-shot {args.aggregation})...", end='', flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_multi_out = diffusion.sample_multi_shot(
            x_noisy_dev, scale=scale_dev, M=10, aggregation=args.aggregation,
            sampler=args.sampler, ddim_steps=args.ddim_steps, eta=args.eta,
            stochastic=not args.no_noise,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_multi = time.perf_counter() - t0
        t_multi_list.append(t_multi)
        x_multi_norm = x_multi_out.squeeze().cpu().numpy()
        print(f" ({t_multi*1e3:.1f} ms) done")

        # Normalized arrays (before rescaling)
        x_clean_norm = x_clean.squeeze().numpy()
        x_noisy_norm = x_noisy.squeeze().numpy()

        # Rescale back to physical units
        x_clean_np = x_clean_norm * scale
        x_noisy_np = x_noisy_norm * scale
        x_single = x_single_norm * scale
        x_multi = x_multi_norm * scale

        m_noisy = compute_all_metrics(x_clean_np, x_noisy_np,
                                      x_clean_norm, x_noisy_norm)
        m_single = compute_all_metrics(x_clean_np, x_single,
                                       x_clean_norm, x_single_norm)
        m_multi = compute_all_metrics(x_clean_np, x_multi,
                                      x_clean_norm, x_multi_norm)

        # Peak amplitudes and peak times via biexponential fit (paper eq. 2).
        # Convention: pulses ordered by peak time (earliest first).
        from src.basics.fit import fit_pulse
        fit_clean = fit_pulse(x_clean_np, n_peaks)
        fit_noisy = fit_pulse(x_noisy_np, n_peaks)
        fit_single = fit_pulse(x_single, n_peaks)
        fit_multi = fit_pulse(x_multi, n_peaks)
        for label, signal, m, fr in [('noisy', x_noisy_np, m_noisy, fit_noisy),
                                      ('single', x_single, m_single, fit_single),
                                      ('multi', x_multi, m_multi, fit_multi)]:
            m['peaks'] = []
            m['chi2_per_ndf'] = fr.chi2_per_ndf
            m['fit_success'] = fr.success
            for i in range(n_peaks):
                c_amp = fit_clean.peak_amps[i]
                c_pos_s = fit_clean.peak_times[i]
                s_amp = fr.peak_amps[i]
                s_pos_s = fr.peak_times[i]
                err_pct = (s_amp - c_amp) / c_amp * 100.0 if c_amp != 0 else 0.0
                time_err_ms = (s_pos_s - c_pos_s) * 1000.0
                m['peaks'].append({
                    'clean_amp': c_amp,
                    'signal_amp': s_amp,
                    'err_pct': err_pct,
                    'clean_pos': c_pos_s * 1000.0,  # store in ms
                    'signal_pos': s_pos_s * 1000.0,
                    'time_err_ms': time_err_ms,
                })
        # Also compute clean-fit chi2/ndf for reference and store on all dicts
        for m in (m_noisy, m_single, m_multi):
            m['chi2_per_ndf_clean_ref'] = fit_clean.chi2_per_ndf
        # Baseline RMS: t in [0, 1.5s) — before pulse onset.
        # Reported in both physical units and normalized units (divided by
        # the per-window scale = max(|noisy|)), so low-amplitude events can
        # be compared on a common footing.
        n_baseline = int(1.5 * 1000.0)  # 1500 samples at 1 kHz
        for label, signal, norm_signal, m in [
                ('noisy',  x_noisy_np, x_noisy_norm,  m_noisy),
                ('single', x_single,   x_single_norm, m_single),
                ('multi',  x_multi,    x_multi_norm,  m_multi)]:
            m['baseline_rms'] = float(np.sqrt(np.mean(signal[:n_baseline] ** 2)))
            m['baseline_rms_norm'] = float(np.sqrt(
                np.mean(norm_signal[:n_baseline] ** 2)))

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
    all_keys = ['mse', 'mae', 'mad', 'prd', 'cosine_sim', 'cc', 'snr',
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

    # Average chi2/ndf (skip NaN from failed fits)
    for label, mkey in [('noisy', 'm_noisy'), ('single', 'm_single'), ('multi', 'm_multi')]:
        vals = [r[mkey]['chi2_per_ndf'] for r in results
                if np.isfinite(r[mkey]['chi2_per_ndf'])]
        avg[label]['chi2_per_ndf'] = float(np.mean(vals)) if vals else float('nan')

    # Average amplitude error across all peaks (1 per single, 2 per pileup)
    for label, mkey in [('noisy', 'm_noisy'), ('single', 'm_single'), ('multi', 'm_multi')]:
        total_err, count = 0.0, 0
        total_t_err, t_count = 0.0, 0
        for r in results:
            for pk in r[mkey]['peaks']:
                total_err += abs(pk['err_pct'])
                count += 1
                total_t_err += abs(pk['time_err_ms'])
                t_count += 1
        avg[label]['amp_abs_err_pct'] = total_err / max(count, 1)
        avg[label]['time_abs_err_ms'] = total_t_err / max(t_count, 1)
    for label in ['noisy', 'single', 'multi']:
        for key in all_keys:
            avg[label][key] /= n

    print(f"\n{'=' * 170}")
    print(f"{'':>8} | {'MSE':>10} {'MAE':>10} {'MAD':>10} {'BL RMS':>10} {'chi2/ndf':>11} {'PRD%':>8} {'Cosine':>8} {'CC':>8} "
          f"{'SNR dB':>8} {'SC':>8} {'LSD':>8} {'J_asd':>8} {'|Amp|%':>8} {'|dt|ms':>8}")
    print(f"{'-' * 150}")
    for label, name in [('noisy', 'Noisy'), ('single', '1-shot'), ('multi', '10-shot')]:
        m = avg[label]
        print(f"{name:>8} | {m['mse']:10.6f} {m['mae']:10.6f} {m['mad']:10.6f} "
              f"{m['baseline_rms']*1e3:9.4f}m {m['chi2_per_ndf']:11.3f} {m['prd']:8.2f} "
              f"{m['cosine_sim']:8.5f} {m['cc']:8.5f} {m['snr']:8.2f} "
              f"{m['sc']:8.4f} {m['lsd']:8.4f} {m['jasd']:8.4f} "
              f"{m['amp_abs_err_pct']:8.2f} {m['time_abs_err_ms']:8.2f}")
    print(f"{'=' * 170}")

    # ── Inference-speed benchmark ──
    ts = np.array(t_single_list)
    tm = np.array(t_multi_list)
    print(f"\n{'─' * 60}")
    print(f"Inference speed benchmark (n={n} windows, device={device.type})")
    print(f"  Sampler: {args.sampler}, T={args.T}"
          f"{', ddim_steps='+str(args.ddim_steps) if args.sampler=='ddim' else ''}"
          f", no_noise={args.no_noise}")
    print(f"  1-shot:  mean={ts.mean()*1e3:7.2f} ms  "
          f"std={ts.std()*1e3:6.2f}  "
          f"min={ts.min()*1e3:6.2f}  max={ts.max()*1e3:6.2f}  "
          f"throughput={1.0/ts.mean():6.2f} win/s")
    print(f"  10-shot: mean={tm.mean()*1e3:7.2f} ms  "
          f"std={tm.std()*1e3:6.2f}  "
          f"min={tm.min()*1e3:6.2f}  max={tm.max()*1e3:6.2f}  "
          f"throughput={1.0/tm.mean():6.2f} win/s")
    print(f"  total wall: 1-shot={ts.sum():.1f}s  10-shot={tm.sum():.1f}s")
    print(f"{'─' * 60}")

    # ── Plot ──
    plot_results = results[:args.n_plot]
    n_rows = len(plot_results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(28, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(plot_results):
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
            f"{'MAE':>12}  {mn['mae']:10.6f}  {ms['mae']:10.6f}  {mm['mae']:10.6f}",
            f"{'MAD':>12}  {mn['mad']*1e3:9.3f}m  {ms['mad']*1e3:9.3f}m  {mm['mad']*1e3:9.3f}m",
            f"{'BL RMS':>12}  {mn['baseline_rms']*1e3:9.4f}m  {ms['baseline_rms']*1e3:9.4f}m  {mm['baseline_rms']*1e3:9.4f}m",
            f"{'chi2/ndf':>12}  {mn['chi2_per_ndf']:10.3f}  {ms['chi2_per_ndf']:10.3f}  {mm['chi2_per_ndf']:10.3f}",
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

    # ── Scatter plots ──
    title_suffix = ''
    if args.no_noise:
        title_suffix = ' (no-noise / deterministic)'
    if args.scatter_output:
        plot_metric_scatter(results, args.scatter_output,
                            title_suffix=title_suffix)
    if args.scatter_pileup_output:
        plot_pileup_scatter(results, args.scatter_pileup_output,
                            title_suffix=title_suffix)

    if args.results_pkl:
        import pickle
        payload = {
            'meta': {
                'model_path': args.model_path,
                'clean_dir': args.clean_dir,
                'noise_dir': args.noise_dir,
                'sampler': args.sampler,
                'T': args.T,
                'no_noise': args.no_noise,
                'n': len(results),
                'title_suffix': title_suffix,
            },
            'results': [{
                'idx': r['idx'],
                'm_noisy': r['m_noisy'],
                'm_single': r['m_single'],
                'm_multi': r['m_multi'],
            } for r in results],
        }
        with open(args.results_pkl, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Saved results pickle: {args.results_pkl}")


if __name__ == '__main__':
    main()
