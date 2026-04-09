# CUORE Pulse Denoiser

Generative-model-based denoising of cryogenic bolometer pulse signals from the [CUORE experiment](https://cuore.lngs.infn.it/). CUORE searches for neutrinoless double-beta decay using 988 TeO2 bolometers at ~10 mK. Each channel records voltage pulses proportional to deposited energy, contaminated by thermal and electronic noise. This project trains conditional generative models to recover clean pulse waveforms from noisy observations to improve energy resolution.

## Data Pipeline

### Pulse Simulation (`src/pulse/`)

Physics-based simulation of clean bolometer pulses. A nonlinear electro-thermal ODE model ([arXiv:2205.04549](https://arxiv.org/abs/2205.04549)) describes the coupled thermal-electrical response of the TeO2 crystal + NTD-Ge thermistor system:

- 4-ODE perturbation system: energy deposit propagates through electron → crystal → PTFE → heat sink
- Thermistor readout: voltage pulse from temperature-dependent resistance R(T) = R0 * exp((T0/T)^0.5)
- Signal chain: 10 kHz ODE → 120 Hz Bessel low-pass → 1 kHz output (10-second windows, 10000 samples)
- Detector parameters randomized per pulse to capture channel-to-channel variation across 988 channels
- Energy sampled from measured CUORE spectrum (`figS2_spectrum.root`), range (0, 5407] keV
- 70% single pulses (onset at 1.5 s), 30% pileup (second pulse 0.01–2.0 s after first)

Output: HDF5 shards with `waveforms` (N, 10000), `energies`, `onsets`, `is_pileup`, `params`.

### Noise (`src/noise/`)

Realistic synthetic noise generator combining five sources observed in real CUORE data:

1. **AC power line harmonics** — 50 Hz + 20 harmonics, power-law decay
2. **Pulse tube cooler harmonics** — 1.4 Hz + 50 harmonics
3. **Colored noise** — 1/f^alpha + linear + white, with randomized crossover frequency
4. **White noise floor** — independent Gaussian
5. **Mechanical resonances** — 3–6 narrowband peaks from seismic/equipment vibrations

All components summed, modulated by a slow amplitude envelope (<0.2 Hz), Bessel-filtered at 120 Hz, and decimated to 1 kHz. RMS scaled to ~7.5 mV. For training, real noise windows extracted from LUCE data are used instead.

### Common Signal Processing (`src/basics/`)

Shared Bessel low-pass filter (6th order, 120 Hz) matching the CUORE DAQ anti-aliasing chain, used by both pulse and noise generators.

### Training Data

Training pairs are formed by adding noise to clean pulses: `x_noisy = x_clean + noise`. Clean-noise pairings are reshuffled each epoch for combinatorial augmentation. Per-window normalization by max(|noisy|) maps inputs to ~[-1, 1].

```
clean/clean_000.h5 ... clean_009.h5   (simulated clean pulses)
noise/noise_000.h5 ... noise_009.h5   (real noise from LUCE)
```

## Denoising Models

### DDPM (`src/ddpm/`)

Denoising Diffusion Probabilistic Model — the primary denoising framework.

- **Forward process**: Gradual Gaussian noise addition over T=50 steps (quadratic beta schedule)
- **Reverse process**: Iterative denoising conditioned on noisy observation x_tilde
- **Architecture**: 1D U-Net (~15.1M params), 4-level encoder-decoder with skip connections, self-attention at bottleneck. Input: [x_t, x_tilde] concatenated (2 channels). Output: predicted noise.
- **Scale conditioning** (optional): encodes max(|noisy|) so the model knows the SNR regime
- **Loss**: L1 or L2 on noise prediction, plus optional spectral losses (SC, LSD, J_asd) on the implied clean estimate
- **Inference**: Stochastic reverse process with multi-shot (M=10) averaging. DDIM sampler also available.

See [`src/ddpm/README.md`](src/ddpm/README.md) for full details.

### Flow Matching (`src/flowMatching/`)

Conditional Flow Matching — an alternative generative framework for direct comparison with DDPM.

- **Forward**: Linear interpolation x_t = (1-t)*x_0 + t*eps, continuous t in [0, 1]
- **Reverse**: Deterministic ODE integration from t=1 (noise) to t=0 (data)
- **Model predicts**: velocity v = eps - x_0 (vs noise prediction in DDPM)
- **No noise schedule**: interpolation path is a straight line, no beta/alpha parameters
- **Solvers**: Euler (default) or midpoint (2nd order)
- Same U-Net architecture and dataset as DDPM

See [`src/flowMatching/README.md`](src/flowMatching/README.md) for full details.

## Project Structure

```
src/
  basics/             - Shared signal processing (Bessel filter)
  pulse/              - Bolometer pulse simulator (ODE model, spectrum, HDF5 generation)
  noise/              - Synthetic noise generator (5-component model)
  ddpm/               - DDPM denoiser (model, training, inference, metrics)
  flowMatching/       - Flow matching denoiser (model, training, inference)

scripts/
  generate_clean_shards.py   - Multi-shard pulse generation
  generate_noise_shards.py   - Multi-shard noise generation

data/
  cuore_spectrum.npz         - Measured CUORE energy spectrum for sampling
```

## Quick Start

### Generate training data

```bash
# Clean pulses (10 shards x 10000 pulses)
python3 -u scripts/generate_clean_shards.py \
    --output_dir /path/to/clean --n_shards 10

# Or use real noise from LUCE (extracted separately)
```

### Train DDPM

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

### Train Flow Matching

```bash
nohup stdbuf -oL python3 -u -m src.flowMatching.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

### Inference

```bash
# DDPM
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10

# Flow Matching
python3 -u -m src.flowMatching.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10
```

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Li et al., "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal" (arXiv:2208.00542)
- Ormiston et al., "Noise reduction in gravitational-wave data via deep learning" (Phys. Rev. Research, 2020)
- Adams et al., "CUORE detector model" (arXiv:2205.04549)
