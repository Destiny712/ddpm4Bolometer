"""
Generate 100K noise windows in 10 shards of 10K each.

Usage:
    python -u scripts/generate_noise_shards.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import h5py
import numpy as np

from src.noise.generator import generate_noise, sample_noise_params

OUTPUT_DIR = '/media/AVFD/yunshancheng/cuore/noise'
N_SHARDS = 10
WINDOWS_PER_SHARD = 10000
BASE_SEED = 7777
DURATION = 10.0
F_SAMPLE = 1000.0


def generate_noise_shard(n_windows, output_file, seed):
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION * F_SAMPLE)

    waveforms = np.zeros((n_windows, n_samples), dtype=np.float64)

    # Store key noise parameters per window
    par_target_rms = np.zeros(n_windows)
    par_alpha = np.zeros(n_windows)
    par_f_cross = np.zeros(n_windows)
    par_ac_a_base = np.zeros(n_windows)
    par_pt_a_base = np.zeros(n_windows)
    par_white_rms = np.zeros(n_windows)
    par_envelope_var = np.zeros(n_windows)
    par_n_resonances = np.zeros(n_windows, dtype=int)

    t_start = time.time()

    for i in range(n_windows):
        params = sample_noise_params(rng)
        noise = generate_noise(rng, params,
                               duration=DURATION, f_sample=F_SAMPLE)
        waveforms[i] = noise

        par_target_rms[i] = params['target_rms']
        par_alpha[i] = params['alpha']
        par_f_cross[i] = params['f_cross']
        par_ac_a_base[i] = params['ac_a_base']
        par_pt_a_base[i] = params['pt_a_base']
        par_white_rms[i] = params['white_rms']
        par_envelope_var[i] = params['envelope_variation']
        par_n_resonances[i] = len(params['resonances'])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_windows - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_windows}] "
                  f"({rate:.1f} windows/s, ETA {eta:.0f}s)")

    # Save
    print(f"Saving to {output_file} ...")
    with h5py.File(output_file, 'w') as f:
        f.attrs['f_sample'] = F_SAMPLE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = n_samples
        f.attrs['n_windows'] = n_windows
        f.attrs['seed'] = seed

        f.create_dataset('waveforms', data=waveforms, compression='gzip')

        grp = f.create_group('params')
        grp.create_dataset('target_rms', data=par_target_rms)
        grp.create_dataset('alpha', data=par_alpha)
        grp.create_dataset('f_cross', data=par_f_cross)
        grp.create_dataset('ac_a_base', data=par_ac_a_base)
        grp.create_dataset('pt_a_base', data=par_pt_a_base)
        grp.create_dataset('white_rms', data=par_white_rms)
        grp.create_dataset('envelope_variation', data=par_envelope_var)
        grp.create_dataset('n_resonances', data=par_n_resonances)

    elapsed = time.time() - t_start
    print(f"Done. {n_windows} windows in {elapsed:.1f}s "
          f"({n_windows/elapsed:.1f} windows/s)")


if __name__ == '__main__':
    for shard in range(N_SHARDS):
        output_file = os.path.join(OUTPUT_DIR, f'noise_{shard:03d}.h5')
        if os.path.exists(output_file):
            print(f"Shard {shard} already exists, skipping: {output_file}")
            continue

        seed = BASE_SEED + shard * 1000
        print(f"\n{'='*60}")
        print(f"Generating shard {shard}/{N_SHARDS}: {output_file}")
        print(f"  Windows: {WINDOWS_PER_SHARD}, seed: {seed}")
        print(f"{'='*60}")

        generate_noise_shard(WINDOWS_PER_SHARD, output_file, seed)

    print(f"\nAll {N_SHARDS} shards complete.")
