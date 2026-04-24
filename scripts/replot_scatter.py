"""
Replot scatter figures from a saved inference results pickle.

No GPU, no model load — just metric dicts. Iterate on plot styling in ~1s.

Y-axis top is set to 1.2 * max(95th-percentile of each colored source)
per subplot, so outliers don't dominate the frame.

Y-axis bottom: metrics with a natural floor (MSE/MAE/MAD-family,
spectral distances, percentages) are pinned at 0. Signed peak errors
(peak_amp_err %, peak_time_err ms) are pinned at -10. Other signed
metrics (SNR dB) use min * 1.2 if the data min is negative, else
min * 0.8.

Peak amplitude and time errors are plotted SIGNED: negative means the
reconstruction under-shoots the clean amplitude / arrives earlier than
the clean peak.

X-axis: either noisy SNR (default) or clean peak amplitude (mV). The
latter is selected per-output-file via the --*_vs_amp flags.

Usage:
    python3 -u scripts/replot_scatter.py \
        --results_pkl /path/to/results.pkl \
        --scatter_output          /path/to/scatter_metrics.png \
        --scatter_pileup_output   /path/to/scatter_pileup.png \
        --scatter_output_vs_amp   /path/to/scatter_metrics_vs_amp.png \
        --scatter_pileup_output_vs_amp /path/to/scatter_pileup_vs_amp.png
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ddpm.inference import _SCATTER_METRICS


# Natural lower bounds. For absent keys => no floor (falls back to min rule).
# All MSE/MAE/MAD-family, percentages, baseline RMS, spectral distances, and
# (1 - cosine)/(1 - cc) transforms are non-negative.
_NATURAL_YMIN = {
    'mse':           0.0,
    'mae':           0.0,
    'mad':           0.0,
    'nmse':          0.0,
    'nmae':          0.0,
    'norm_mse':      0.0,
    'norm_mae':      0.0,
    'norm_mad':      0.0,
    'baseline_rms':  0.0,
    'baseline_rms_norm': 0.0,
    'chi2_per_ndf':  0.0,
    'prd':           0.0,
    'cosine_sim':    0.0,   # plotted as 1 - cos, in [0, 2]
    'cc':            0.0,   # plotted as 1 - cc,  in [0, 2]
    'sc':            0.0,
    'lsd':           0.0,
    'jasd':          0.0,
    # 'snr' omitted — dB can be negative, use min-rule.
}

# Hard lower-bound overrides, regardless of data min.
_FIXED_YMIN = {
    'peak_amp_err':  -10.0,
    'peak_time_err': -10.0,
}

# Override labels so they don't advertise absolute value for signed errors.
_LABEL_OVERRIDES = {
    'peak_amp_err':  'Amp err %',
    'peak_time_err': 'dt [ms]',
}


def _window_clean_amp(r):
    """Representative clean pulse amplitude (mV) for a window: max across peaks.
    Stored pk['clean_amp'] is in volts; convert to mV here.
    Returns None if the window has no peaks."""
    peaks = r['m_noisy']['peaks']
    if not peaks:
        return None
    return max(pk['clean_amp'] for pk in peaks) * 1e3


def _collect_series(results, mkey, metric_key, is_peak, x_mode='snr'):
    """Flatten per-window results into parallel (xs, ys) arrays.

    x_mode:
      'snr'        -> x = noisy-signal SNR (one value per window)
      'clean_amp'  -> x = clean peak amplitude (mV); per-peak for peak
                      metrics, else max across the window's peaks

    Peak metrics (is_peak=True) are kept SIGNED (no abs()).
    """
    xs, ys = [], []
    for r in results:
        if x_mode == 'snr':
            x_window = r['m_noisy']['snr']
        elif x_mode == 'clean_amp':
            x_window = _window_clean_amp(r)
            if x_window is None:
                continue
        else:
            raise ValueError(f"unknown x_mode: {x_mode}")
        m = r[mkey]
        if is_peak:
            for pk in m['peaks']:
                if x_mode == 'clean_amp':
                    x_val = pk['clean_amp'] * 1e3  # V → mV
                else:
                    x_val = x_window
                if metric_key == 'peak_amp_err':
                    ys.append(pk['err_pct'])
                elif metric_key == 'peak_time_err':
                    ys.append(pk['time_err_ms'])
                xs.append(x_val)
        else:
            v = m[metric_key]
            if np.isfinite(v):
                xs.append(x_window)
                ys.append(v)
    return np.array(xs), np.array(ys)


def _ylim_from_series(series_ys, metric_key, top_factor=1.2):
    """Return (bottom, top) for a subplot. Either may be None if no data."""
    finite_all = []
    p95s = []
    for ys in series_ys:
        if ys is None or len(ys) == 0:
            continue
        ys_finite = ys[np.isfinite(ys)]
        if len(ys_finite):
            finite_all.append(ys_finite)
            p95s.append(np.percentile(ys_finite, 95))
    if not p95s:
        return None, None

    top = top_factor * max(p95s)

    if metric_key in _FIXED_YMIN:
        bottom = _FIXED_YMIN[metric_key]
    elif metric_key in _NATURAL_YMIN:
        bottom = _NATURAL_YMIN[metric_key]
    else:
        overall_min = float(np.min(np.concatenate(finite_all)))
        bottom = overall_min * 1.2 if overall_min < 0 else overall_min * 0.8

    return bottom, top


def _xlabel(x_mode):
    return 'Noisy SNR [dB]' if x_mode == 'snr' else 'Clean peak amplitude [mV]'


def _x_suffix(x_mode):
    return 'noisy SNR' if x_mode == 'snr' else 'clean peak amplitude'


def plot_metric_scatter(results, output_path, title_suffix='', x_mode='snr'):
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
    xlab = _xlabel(x_mode)
    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        series_ys = []
        for mkey, name, color in sources:
            xs, ys = _collect_series(results, mkey, mdef['key'], is_peak,
                                     x_mode=x_mode)
            if tf is not None and len(ys):
                ys = tf(ys)
            ax.scatter(xs, ys, s=6, c=color, alpha=0.5, label=name)
            series_ys.append(ys)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(_LABEL_OVERRIDES.get(mdef['key'], mdef['label']), fontsize=8)
        bottom, top = _ylim_from_series(series_ys, mdef['key'])
        if top is not None:
            ax.set_ylim(bottom=bottom, top=top)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(f'Metrics vs {_x_suffix(x_mode)}{title_suffix}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_pileup_scatter(results, output_path, title_suffix='', x_mode='snr'):
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

    xlab = _xlabel(x_mode)
    for k, mdef in enumerate(_SCATTER_METRICS):
        ax = axes[k // ncols, k % ncols]
        is_peak = mdef.get('is_peak', False)
        tf = mdef.get('transform', None)
        series_ys = []
        for subset, mkey, name, color, marker in series:
            if not subset:
                continue
            xs, ys = _collect_series(subset, mkey, mdef['key'], is_peak,
                                     x_mode=x_mode)
            if tf is not None and len(ys):
                ys = tf(ys)
            ax.scatter(xs, ys, s=10, c=color, marker=marker, alpha=0.6, label=name)
            series_ys.append(ys)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(_LABEL_OVERRIDES.get(mdef['key'], mdef['label']), fontsize=8)
        bottom, top = _ylim_from_series(series_ys, mdef['key'])
        if top is not None:
            ax.set_ylim(bottom=bottom, top=top)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6, loc='best')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(f'Non-pileup vs pileup ({_x_suffix(x_mode)}){title_suffix}  '
                 f'(n_nonpu={len(non_pileup)}, n_pu={len(pileup)})',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_pkl', required=True)
    p.add_argument('--scatter_output', default=None,
                   help='Metric scatter grid with noisy-SNR x-axis')
    p.add_argument('--scatter_pileup_output', default=None,
                   help='Pileup scatter grid with noisy-SNR x-axis')
    p.add_argument('--scatter_output_vs_amp', default=None,
                   help='Metric scatter grid with clean-peak-amplitude x-axis')
    p.add_argument('--scatter_pileup_output_vs_amp', default=None,
                   help='Pileup scatter grid with clean-peak-amplitude x-axis')
    args = p.parse_args()

    with open(args.results_pkl, 'rb') as f:
        payload = pickle.load(f)

    meta = payload['meta']
    results = payload['results']
    print(f"Loaded {len(results)} windows from {args.results_pkl}")
    print(f"  sampler={meta['sampler']}, T={meta['T']}, "
          f"no_noise={meta['no_noise']}, n={meta['n']}")

    title_suffix = meta.get('title_suffix', '')

    if args.scatter_output:
        plot_metric_scatter(results, args.scatter_output,
                            title_suffix=title_suffix, x_mode='snr')
    if args.scatter_pileup_output:
        plot_pileup_scatter(results, args.scatter_pileup_output,
                            title_suffix=title_suffix, x_mode='snr')
    if args.scatter_output_vs_amp:
        plot_metric_scatter(results, args.scatter_output_vs_amp,
                            title_suffix=title_suffix, x_mode='clean_amp')
    if args.scatter_pileup_output_vs_amp:
        plot_pileup_scatter(results, args.scatter_pileup_output_vs_amp,
                            title_suffix=title_suffix, x_mode='clean_amp')


if __name__ == '__main__':
    main()
