# Pulse Simulator

Physics-based simulation of CUORE cryogenic bolometer pulses for generating clean training data.

## Physics Model

Implements the nonlinear electro-thermal detector model from [arXiv:2205.04549](https://arxiv.org/abs/2205.04549). A CUORE bolometer is a TeO2 crystal coupled to an NTD-Ge thermistor operating at ~10 mK. When a particle deposits energy, the crystal heats up, changing the thermistor resistance, which is read out as a voltage pulse.

The model solves a 4-ODE perturbation system describing coupled thermal and electrical dynamics:

```
Thermal chain:  Energy deposit → Electrons → Crystal → PTFE → Heat sink
Electrical:     V_bol(t) from thermistor resistance change under constant bias
```

Key physics:
- **Thermal**: Three coupled heat capacities (electron, crystal, PTFE) with thermal conductances between them
- **Electrical**: NTD-Ge thermistor with temperature-dependent resistance R(T) = R0 * exp((T0/T)^0.5)
- **Signal chain**: ODE solved at 10 kHz internal rate, Bessel low-pass filtered at 120 Hz, decimated to output rate (1 kHz)

## Detector Parameters

Each simulated pulse uses a randomly sampled set of detector parameters drawn from truncated Gaussians. This captures detector-to-detector variation across the 988 CUORE channels.

Randomized per pulse:
- `R0`, `T0` — thermistor resistance model parameters
- `lambda0`, `q` — thermal conductance model parameters
- `g_ec`, `a_ec` — electron-crystal coupling
- `C_p` — parasitic heat capacity
- `T_base` — base temperature

Fixed:
- `gamma`, `g_ct`, `g_ts` — thermal chain conductances
- `c_e`, `c_c`, `c_t` — heat capacities
- `R_bias`, `V_bias`, `gain` — readout circuit
- `W`, `f_bessel`, `bessel_order` — signal conditioning

Unphysical parameter draws are rejected via equilibrium validation (resistance ratio < 0.02, loop gain < 0.8).

## Energy Spectrum

Uses the measured CUORE spectrum from `figS2_spectrum.root` (FinalSpectrum histogram), extracted to `data/cuore_spectrum.npz`. The spectrum covers 0–10000 keV in 5 keV bins and is used directly for inverse-CDF energy sampling — no parametric fitting. A parametric fallback (exponential continuum + Gaussian peaks) is used if the data file is missing.

Energy range can be restricted via `E_min`/`E_max` parameters to generate subsets (e.g., low-energy E < 150 keV vs high-energy E >= 150 keV).

## Dataset Generation

Each pulse in the dataset is generated as:

1. Sample detector parameters (rejection sampling for physical validity)
2. Find self-consistent thermal equilibrium via fixed-point iteration
3. Decide single pulse (70%) or pileup (30%)
4. Sample energy from spectrum, fixed onset at 1.5 s
5. For pileup: sample second energy and onset (0.01–2.0 s after first)
6. Solve ODE, filter, decimate to 10-second windows at 1 kHz (10000 samples)

Output HDF5 contains:
- `waveforms` — (N, 10000) float64, voltage in Volts
- `energies_1`, `energies_2` — deposited energies [keV]
- `onsets_1`, `onsets_2` — pulse onset times [s]
- `is_pileup` — boolean flag
- `params` — detector parameters per pulse
- `equilibrium` — equilibrium state per pulse

## Structure

```
src/pulse/
  simulator.py    - ODE solver, detector model, equilibrium finder
  spectrum.py     - Energy spectrum model and inverse-CDF sampler
  generate.py     - HDF5 dataset generator (CLI entry point)
  qa_pulse.py     - QA plots: waveforms with metadata
  qa_spectrum.py  - QA plots: spectrum model vs sampled histogram
```

## Usage

### Generate dataset

Single shard:

```bash
python3 -u -m src.pulse.generate \
    --n_pulses 10000 \
    --output /path/to/clean_000.h5 \
    --duration 10.0 --f_sample 1000.0 \
    --pileup_fraction 0.3 --seed 0
```

Multi-shard generation with energy range:

```bash
# Low energy (E < 150 keV)
python3 -u scripts/generate_clean_shards.py \
    --output_dir /path/to/clean_low --E_min 1 --E_max 150

# High energy (E >= 150 keV)
python3 -u scripts/generate_clean_shards.py \
    --output_dir /path/to/clean_high --E_min 150 --E_max 5407
```

### QA visualization

```bash
python3 -u -m src.pulse.qa_pulse --input clean_000.h5 --n 10 --output qa_pulse.png
python3 -u -m src.pulse.qa_spectrum --output qa_spectrum.png
```
