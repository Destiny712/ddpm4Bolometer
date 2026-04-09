# Noise Generator

Realistic synthetic noise generator for CUORE cryogenic bolometer signals, modeling the dominant noise sources observed in real detector data.

## Noise Components

The generator combines five independent noise sources into a single time-domain waveform:

### 1. AC Power Line Harmonics (50 Hz)

Electromagnetic pickup from the AC mains at 50 Hz fundamental with 20 harmonics. Amplitudes follow a power-law decay A_k = a_base / k^p_decay with per-harmonic randomization (factor of ~3 up or down). Random phases.

### 2. Pulse Tube Cooler Harmonics (1.4 Hz)

Mechanical vibrations from the cryostat's pulse tube cooler at ~1.4 Hz fundamental with 50 harmonics (covering up to ~70 Hz). Same power-law decay structure as AC harmonics but with independent parameters.

### 3. Colored Noise (1/f + linear + white)

Broadband noise with a parametric power spectral density:

```
PSD(f) = a_pink / f^alpha + a_lin * f + a_white
```

- `alpha` in [0.8, 1.2] — 1/f slope (classic flicker noise ~1.0)
- `f_cross` in [0.5, 3.0] Hz — crossover frequency where pink meets white
- Amplitudes derived from `f_cross` and target noise RMS

Generated in frequency domain with random phases, then inverse-FFT.

### 4. White Noise Floor

Independent Gaussian white noise added separately from the colored noise component.

### 5. Mechanical Resonances

Narrowband noise peaks from seismic and equipment vibrations. 3–6 resonances per window with:

- Base frequencies near [10, 25, 40, 70] Hz (jittered +-30%)
- Quality factors Q in [5, 20]
- Amplitudes: [0.1, 0.5] x target RMS

Each resonance is bandpass-filtered white noise (Butterworth filter).

## Amplitude Envelope

All summed noise is multiplied by a slowly varying amplitude envelope (< 0.2 Hz) modeling non-stationary gain variations, bounded to +/-20% modulation around unity.

## Signal Chain

Matches the pulse simulator signal chain:

1. All components generated at 10 kHz internal sampling rate
2. Summed and multiplied by envelope
3. Bessel low-pass filtered at 120 Hz (6th order), matching CUORE DAQ anti-aliasing
4. Decimated to output rate (1 kHz)
5. RMS-scaled to target noise level (default 7.5 mV)

## Parameter Sampling

Each noise window gets independently sampled parameters via `sample_noise_params()`, capturing window-to-window variation in the noise characteristics. Key randomized quantities:

| Parameter | Range | Description |
|-----------|-------|-------------|
| alpha | 0.8 – 1.2 | 1/f slope |
| f_cross | 0.5 – 3.0 Hz | Pink/white crossover |
| AC a_base | varies | 50 Hz harmonic amplitude |
| PT a_base | varies | 1.4 Hz harmonic amplitude |
| max_variation | 0 – 0.2 | Envelope modulation depth |
| n_resonances | 3 – 6 | Number of mechanical peaks |

## Structure

```
src/noise/
  generator.py    - Orchestrator: parameter sampling + noise assembly
  colored.py      - 1/f^alpha + linear + white noise (frequency domain)
  harmonics.py    - Harmonic series with power-law decay
  resonances.py   - Bandpass-filtered mechanical resonance noise
  envelope.py     - Slow amplitude modulation envelope
  qa_noise.py     - QA plots: waveforms + PSD with annotations
```

## Usage

### Generate noise (used by pulse dataset pipeline)

```python
from src.noise.generator import sample_noise_params, generate_noise
import numpy as np

rng = np.random.default_rng(42)
params = sample_noise_params(rng)
noise = generate_noise(rng, params, duration=10.0, f_sample=1000.0)
# noise.shape == (10000,)
```

### QA visualization

```bash
python3 -u -m src.noise.qa_noise --n 5 --output qa_noise.png
```

Plots time-domain waveforms and log-log PSDs with annotated frequency markers (PT fundamental, AC fundamental, crossover, resonance centers).
