"""
Preprocess clean HDF5 shards to build pulse index files.

For each clean_XXX.h5, saves clean_XXX_index.json with sample indices
grouped by category, avoiding repeated amplitude computation during
training and inference.

Categories:
    all             — all sample indices
    single          — single-pulse events
    pileup          — pileup events (two pulses)
    low_100         — max amplitude < 100 mV
    low_200         — max amplitude < 200 mV
    pileup_low_100  — pileup with max amplitude < 100 mV
    pileup_low_200  — pileup with max amplitude < 200 mV

Usage:
    python3 -u -m src.ddpm.preprocess_index \
        --clean_dir /path/to/clean

    # Single shard:
    python3 -u -m src.ddpm.preprocess_index \
        --clean_dir /path/to/clean/clean_000.h5
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np


def build_index(h5_path: str) -> dict:
    """Build category index for a single clean HDF5 shard.

    Returns dict of category -> list of int indices.
    """
    with h5py.File(h5_path, 'r') as f:
        waveforms = f['waveforms'][:]
        is_pileup = f['is_pileup'][:]

    max_amp = waveforms.max(axis=1)  # V
    max_amp_mv = max_amp * 1e3       # mV

    n = len(waveforms)
    all_idx = np.arange(n)

    index = {
        'all': all_idx.tolist(),
        'single': all_idx[~is_pileup].tolist(),
        'pileup': all_idx[is_pileup].tolist(),
        'low_100': all_idx[max_amp_mv < 100].tolist(),
        'low_200': all_idx[max_amp_mv < 200].tolist(),
        'pileup_low_100': all_idx[is_pileup & (max_amp_mv < 100)].tolist(),
        'pileup_low_200': all_idx[is_pileup & (max_amp_mv < 200)].tolist(),
    }

    return index


def save_index(h5_path: str, index: dict):
    """Save index JSON alongside the HDF5 file."""
    base = os.path.splitext(h5_path)[0]
    out_path = base + '_index.json'

    # Add summary counts
    meta = {
        'source': os.path.basename(h5_path),
        'counts': {k: len(v) for k, v in index.items()},
    }
    output = {'meta': meta, 'indices': index}

    with open(out_path, 'w') as f:
        json.dump(output, f)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build pulse index files for clean HDF5 shards.")
    parser.add_argument('--clean_dir', type=str, required=True,
                        help='Single .h5 file or directory of clean_*.h5 shards')
    args = parser.parse_args()

    if os.path.isfile(args.clean_dir):
        files = [args.clean_dir]
    else:
        files = sorted(glob.glob(os.path.join(args.clean_dir, 'clean_*.h5')))

    if not files:
        print(f"No clean HDF5 files found at {args.clean_dir}")
        return

    for h5_path in files:
        print(f"Processing {h5_path}...")
        index = build_index(h5_path)
        out_path = save_index(h5_path, index)
        counts = {k: len(v) for k, v in index.items()}
        print(f"  -> {out_path}")
        for k, v in counts.items():
            print(f"     {k:>20}: {v}")


if __name__ == '__main__':
    main()
