"""
Generate clean pulse dataset for DDPM training.

Outputs an HDF5 file with simulated CUORE-like pulse waveforms,
including single pulses and pileup events.

Usage:
    python -u -m src.pulse.generate --n_pulses 100 --output data/clean_pulses.h5
"""

import argparse
import time

import h5py
import numpy as np

from src.pulse.simulator import (
    DetectorParams, sample_params, find_equilibrium, is_valid_equilibrium,
    simulate_pulse, simulate_pileup,
)
from src.pulse.spectrum import EnergySpectrum


def generate_dataset(n_pulses: int, output_file: str,
                     duration: float = 10.0,
                     f_sample: float = 1000.0,
                     pileup_fraction: float = 0.3,
                     pileup_dt_min: float = 0.01,
                     pileup_dt_max: float = 2.0,
                     onset: float = 1.5,
                     seed: int = 42,
                     E_min: float = 1.0,
                     E_max: float = 5407.0):
    """Generate and save clean pulse waveforms.

    Parameters
    ----------
    n_pulses : int
        Number of pulse windows to generate.
    output_file : str
        Path to output HDF5 file.
    duration : float
        Window duration [s].
    f_sample : float
        Output sampling rate [Hz].
    pileup_fraction : float
        Fraction of windows that are pileup (two pulses).
    pileup_dt_min, pileup_dt_max : float
        Range of time separation between pileup pulses [s].
    onset : float
        Fixed onset time for the first pulse [s].
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    spectrum = EnergySpectrum(E_min=E_min, E_max=E_max)
    n_samples = int(duration * f_sample)

    # Pre-allocate arrays
    waveforms = np.zeros((n_pulses, n_samples), dtype=np.float64)
    energies_1 = np.zeros(n_pulses, dtype=np.float64)
    energies_2 = np.full(n_pulses, np.nan, dtype=np.float64)
    onsets_1 = np.zeros(n_pulses, dtype=np.float64)
    onsets_2 = np.full(n_pulses, np.nan, dtype=np.float64)
    is_pileup = np.zeros(n_pulses, dtype=bool)

    # Per-pulse detector parameters
    par_R0 = np.zeros(n_pulses)
    par_T0 = np.zeros(n_pulses)
    par_lambda0 = np.zeros(n_pulses)
    par_q = np.zeros(n_pulses)
    par_g_ec = np.zeros(n_pulses)
    par_a_ec = np.zeros(n_pulses)
    par_C_p = np.zeros(n_pulses)
    par_T_base = np.zeros(n_pulses)

    # Equilibrium state
    eq_V = np.zeros(n_pulses)
    eq_Te = np.zeros(n_pulses)
    eq_Tc = np.zeros(n_pulses)
    eq_Tt = np.zeros(n_pulses)

    t_start = time.time()

    for i in range(n_pulses):
        # Sample detector parameters, rejecting unphysical equilibria
        max_attempts = 200
        for attempt in range(max_attempts):
            params = sample_params(rng)
            try:
                eq = find_equilibrium(params)
            except Exception:
                continue
            if is_valid_equilibrium(params, eq):
                break
        else:
            raise RuntimeError(
                f"Pulse {i}: could not find valid parameters in "
                f"{max_attempts} attempts"
            )

        # Decide pileup
        do_pileup = rng.random() < pileup_fraction

        # Fixed onset
        t1 = onset

        # Sample energy
        E1 = spectrum.sample(rng, 1)[0]

        if do_pileup:
            E2 = spectrum.sample(rng, 1)[0]
            dt = rng.uniform(pileup_dt_min, pileup_dt_max)
            t2 = t1 + dt

            try:
                t, v = simulate_pileup(
                    [E1, E2], [t1, t2], params, eq,
                    duration=duration, f_sample=f_sample,
                )
            except RuntimeError as e:
                print(f"  Pulse {i}: ODE failed ({e}), skipping pileup")
                do_pileup = False

        if not do_pileup:
            try:
                t, v = simulate_pulse(
                    E1, params, eq,
                    duration=duration, t_onset=t1, f_sample=f_sample,
                )
            except RuntimeError as e:
                print(f"  Pulse {i}: ODE failed ({e}), writing zeros")
                v = np.zeros(n_samples)

        # Store
        waveforms[i] = v[:n_samples]
        energies_1[i] = E1
        onsets_1[i] = t1
        is_pileup[i] = do_pileup
        if do_pileup:
            energies_2[i] = E2
            onsets_2[i] = t2

        # Parameters
        par_R0[i] = params.R0
        par_T0[i] = params.T0
        par_lambda0[i] = params.lambda0
        par_q[i] = params.q
        par_g_ec[i] = params.g_ec
        par_a_ec[i] = params.a_ec
        par_C_p[i] = params.C_p
        par_T_base[i] = params.T_base

        # Equilibrium
        eq_V[i] = eq.V_bol
        eq_Te[i] = eq.T_e
        eq_Tc[i] = eq.T_c
        eq_Tt[i] = eq.T_t

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (n_pulses - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{n_pulses}] E1={E1:.1f} keV"
              f"{f', E2={E2:.1f} keV' if do_pileup else ''}"
              f"  ({rate:.1f} pulses/s, ETA {eta:.0f}s)")

    # Save to HDF5
    print(f"\nSaving to {output_file} ...")
    with h5py.File(output_file, 'w') as f:
        # Global attributes
        f.attrs['f_sample'] = f_sample
        f.attrs['duration'] = duration
        f.attrs['n_samples'] = n_samples
        f.attrs['n_pulses'] = n_pulses
        f.attrs['pileup_fraction'] = pileup_fraction
        f.attrs['seed'] = seed

        # Waveforms
        f.create_dataset('waveforms', data=waveforms, compression='gzip')

        # Pulse metadata
        f.create_dataset('energies_1', data=energies_1)
        f.create_dataset('energies_2', data=energies_2)
        f.create_dataset('onsets_1', data=onsets_1)
        f.create_dataset('onsets_2', data=onsets_2)
        f.create_dataset('is_pileup', data=is_pileup)

        # Detector parameters
        grp = f.create_group('params')
        grp.create_dataset('R0', data=par_R0)
        grp.create_dataset('T0', data=par_T0)
        grp.create_dataset('lambda0', data=par_lambda0)
        grp.create_dataset('q', data=par_q)
        grp.create_dataset('g_ec', data=par_g_ec)
        grp.create_dataset('a_ec', data=par_a_ec)
        grp.create_dataset('C_p', data=par_C_p)
        grp.create_dataset('T_base', data=par_T_base)

        # Equilibrium state
        grp = f.create_group('equilibrium')
        grp.create_dataset('V_bol', data=eq_V)
        grp.create_dataset('T_e', data=eq_Te)
        grp.create_dataset('T_c', data=eq_Tc)
        grp.create_dataset('T_t', data=eq_Tt)

    elapsed = time.time() - t_start
    print(f"Done. {n_pulses} pulses in {elapsed:.1f}s "
          f"({n_pulses/elapsed:.1f} pulses/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean CUORE-like pulse dataset for DDPM training."
    )
    parser.add_argument('--n_pulses', type=int, default=100,
                        help='Number of pulse windows to generate')
    parser.add_argument('--output', type=str, default='data/clean_pulses.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Window duration [s]')
    parser.add_argument('--f_sample', type=float, default=1000.0,
                        help='Output sampling rate [Hz]')
    parser.add_argument('--pileup_fraction', type=float, default=0.3,
                        help='Fraction of pileup windows')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    generate_dataset(
        n_pulses=args.n_pulses,
        output_file=args.output,
        duration=args.duration,
        f_sample=args.f_sample,
        pileup_fraction=args.pileup_fraction,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
