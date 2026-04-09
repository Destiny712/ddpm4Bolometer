"""
QA comparison of easytrigger (fixed baseline) vs trigger (adaptive rolling)
on a single IETI .bin file. Generates two PNGs, each with 10 triggered windows.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ieti.convert import process_adc_file
from src.ieti.easytrigger import (
    calculate_baseline_metrics as easy_baseline,
    find_trigger_points as easy_find,
    extract_signal_segment as easy_extract,
)
from src.ieti.trigger import (
    find_triggers_adaptive,
    extract_signal_segment as adaptive_extract,
)


BINFILE = "data/ccvr3_run501345/501345_20251017T140000_002_000.bin"

# Window parameters
PRE_SEC = 2.0
POST_SEC = 8.0
NUM_WINDOWS = 10


def plot_windows(windows, title, outfile):
    """Plot up to 10 windows in a 2x5 grid."""
    n = min(len(windows), NUM_WINDOWS)
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    axes = axes.flatten()

    for i in range(n):
        t, v = windows[i]
        axes[i].plot(t, v * 1e3, 'b-', lw=0.5)
        axes[i].set_xlabel('Time [s]', fontsize=7)
        axes[i].set_ylabel('Voltage [mV]', fontsize=7)
        axes[i].set_title(f'Window {i+1} (t={t[0]:.4f}s)', fontsize=8)
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, alpha=0.3)

    for i in range(n, 10):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile} ({n} windows)")


def main():
    print(f"Reading {BINFILE}...")
    times, voltages, header = process_adc_file(BINFILE)
    fs = header["fs"]
    print(f"  {len(voltages)} samples, fs={fs} Hz, "
          f"duration={len(voltages)/fs:.1f}s")

    # ── Method 1: easytrigger (fixed baseline) ──
    print("\n--- EasyTrigger (fixed baseline) ---")
    mean_val, rms = easy_baseline(voltages)
    threshold_mult = 9
    threshold_upper = mean_val + threshold_mult * rms
    threshold_lower = mean_val - threshold_mult * rms
    print(f"  Baseline mean={mean_val:.6f} V, RMS={rms*1e3:.4f} mV")
    print(f"  Threshold: [{threshold_lower:.6f}, {threshold_upper:.6f}] V")

    easy_indices = easy_find(
        voltages, threshold_upper, threshold_lower, fs,
        PRE_SEC, POST_SEC, num_triggers=NUM_WINDOWS,
    )
    print(f"  Found {len(easy_indices)} triggers")

    easy_windows = []
    for idx in easy_indices:
        t_seg, v_seg = easy_extract(times, voltages, idx, PRE_SEC, POST_SEC, fs)
        easy_windows.append((t_seg, v_seg))

    plot_windows(
        easy_windows,
        f"EasyTrigger (fixed baseline, {threshold_mult}σ) — {BINFILE.split('/')[-1]}",
        "qa_easytrigger.png",
    )

    # ── Method 2: adaptive trigger (rolling median + MAD) ──
    print("\n--- Adaptive Trigger (rolling median + MAD) ---")
    adaptive_indices, rolling_med, rolling_mad = find_triggers_adaptive(
        voltages, fs, window_sec=300.0, n_sigma=6.0, refractory_sec=10.0,
    )
    print(f"  Found {len(adaptive_indices)} triggers")

    adaptive_windows = []
    for idx in adaptive_indices[:NUM_WINDOWS]:
        t_seg, v_seg = adaptive_extract(
            times, voltages, int(idx), PRE_SEC, POST_SEC, fs,
        )
        adaptive_windows.append((t_seg, v_seg))

    plot_windows(
        adaptive_windows,
        f"Adaptive Trigger (rolling 300s, 6σ) — {BINFILE.split('/')[-1]}",
        "qa_adaptive_trigger.png",
    )


if __name__ == "__main__":
    main()
