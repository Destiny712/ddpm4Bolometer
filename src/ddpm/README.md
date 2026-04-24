# DDPM Pulse Denoiser

Denoising Diffusion Probabilistic Model (DDPM) for denoising cryogenic bolometer pulse signals from the CUORE experiment.

## Task

CUORE measures rare nuclear decays using TeO2 bolometers operating at ~10 mK. Each detector channel records voltage pulses proportional to deposited energy, contaminated by thermal and electronic noise. This DDPM denoiser recovers clean pulse waveforms from noisy observations to improve energy resolution.

## Approach

A conditional DDPM trained on paired (clean, noisy) data:

- **Clean signals**: Simulated via an ODE electro-thermal model. 10-second windows at 1000 Hz (10000 samples). Energy range (0, 5407] keV with 70/30 single-pulse/pileup ratio.
- **Noise**: Real noise extracted from quiet (no-pulse) windows in LUCE data.
- **Training pairs**: Clean + real noise = noisy observation. Noise-clean pairings are reshuffled each epoch for combinatorial augmentation.

### Diffusion Process

- **Forward**: Gradually adds Gaussian noise to clean signal x_0 over T steps using a quadratic beta schedule. Defaults are `T=50`, `beta_1=1e-4`, `beta_T=0.05`, but all three are CLI options (`--T`, `--beta_1`, `--beta_T`). Larger `beta_T` drives `alpha_bar_T` closer to zero so that x_T is a true standard Gaussian. With the default 0.05, `alpha_bar_T ≈ 0.41` — x_T retains ~40% signal, so there is a train/inference mismatch at step T (training sees signal+noise, sampling starts from pure N(0, I)). DeScoD-ECG uses `beta_T=0.5` to eliminate this gap.
- **Reverse**: Starting from pure Gaussian noise, iteratively denoises conditioned on the noisy observation x_tilde (concatenated as a second input channel).
- **Training objective**: Predict the noise epsilon added at each timestep. Base loss is L1 or L2 on the noise prediction, with optional spectral losses on the implied clean estimate x̂_0 (see Loss Functions below).
- **Timestep / noise-level conditioning** (`--cond_mode`):
  - `step` (default): the denoiser is conditioned on the integer diffusion step index `t ∈ {1, …, T}` via a sinusoidal embedding. This is the original DDPM behavior and is what all existing checkpoints were trained with.
  - `sqrt_ab`: the denoiser is conditioned on the continuous noise level `√ᾱ` (WaveGrad / DeScoD-ECG recipe). During training, we first pick an integer step `t`, then sample `√ᾱ ~ Uniform(S_t, S_{t-1})` inside that bin of the grid `S = [1, √ᾱ_1, …, √ᾱ_T]`, and feed that scalar into the U-Net's sinusoidal embedding. Because the network sees a continuum of noise levels rather than `T` discrete points, it generalizes across schedules and tends to work better with fewer reverse steps at inference. The scalar is multiplied by `--cond_scale` (default `1000`) before the sinusoidal encoding so that values in `(0, 1]` span the full frequency grid.
- **Inference**: Two samplers available:
  - **DDPM** (default): Full T-step stochastic reverse process. Multi-shot (M=10) aggregation of independent runs reduces variance. Supports mean or median aggregation (`--aggregation`).
  - **DDIM** (`--sampler ddim`): Deterministic reverse process (eta=0). Same trained model, no retraining needed. Supports step skipping (`--ddim_steps`) for faster inference. Partial stochasticity via `--eta` (0=deterministic, 1=DDPM-equivalent). In practice, DDPM with multi-shot averaging outperforms DDIM for this conditional denoising task — the stochasticity helps explore better solutions for low-SNR inputs.

### Normalization

Per-window normalization: both clean and noisy signals are divided by max(|noisy|) to map inputs to ~[-1, 1]. The scale factor is saved and applied to the output for physical-unit reconstruction.

## Architecture

Two U-Net variants are available:

### UNet1D (default)

1D U-Net with timestep conditioning:

```
Input: [x_t, x_tilde] concatenated (2 channels, 10000 samples)

Encoder:  4 levels (64 -> 128 -> 256 -> 512), stride-2 Conv1d downsample
          2 ResBlocks per level, GroupNorm, SiLU activation
Bottleneck: 512 channels, self-attention
Decoder:  Mirror of encoder with skip connections
Attention: At level 3 (1250 samples) and bottleneck (625 samples)
Timestep: Sinusoidal embedding -> MLP -> added into each ResBlock

Output: 1 channel (predicted noise epsilon)
```

~15.1M parameters.

### UNet1DScaleCond (`--scale_cond`)

Same architecture as UNet1D, plus scale conditioning. The normalization scale `max(|noisy|)` is encoded and injected alongside the timestep embedding so the model knows the SNR regime of each input.

```
scale -> log(scale)*10 -> SinusoidalEmbedding -> MLP -> scale_emb
cond = timestep_emb + scale_emb  (fed to every ResBlock)
```

~15.6M parameters (+525K for the scale MLP).

**Motivation**: Per-window normalization maps all inputs to [-1, 1], but a low-amplitude pileup (pulse peak ~30 mV, noise ~80 mV) looks very different from a high-amplitude pulse (peak ~5000 mV) in normalized space. Without scale information, the model cannot distinguish noise-dominated inputs from pulse-dominated ones. Since the noise level is roughly constant across windows, `max(|noisy|)` directly indicates pulse amplitude and thus SNR.

## Loss Functions

The training loss consists of a base noise-prediction loss plus optional spectral losses computed on the Tweedie estimate x̂_0:

```
Total = base_loss(eps_pred, eps) + w_l1*L1(x̂_0, x_0) + w_sc*SC + w_lsd*LSD + w_asd*J_asd
```

### Base loss (`--loss`)

- **L1**: Mean absolute error on noise prediction. Preserves sharp features (rising edges).
- **L2**: Mean squared error on noise prediction. Smoother gradients but tends to blur edges.

### Spectral losses (on x̂_0 vs x_0)

All spectral losses are optional and controlled by `--w_*` flags (default 0 = disabled).

- **SC** (`--w_sc`): Spectral Convergence. Multi-resolution STFT magnitude error normalized by target magnitude. Partially phase-aware through time-frequency localization. Catches rising edge shifts and pileup shape errors. Common in speech synthesis (HiFi-GAN, MelGAN).

- **LSD** (`--w_lsd`): Log-Spectral Distance. RMS of log-PSD difference between predicted and target. Treats all frequency decades equally on a log scale. Phase-blind.

- **J_asd** (`--w_asd`): Amplitude Spectral Density ratio. `mean over f of sqrt(PSD_residual / PSD_target)`. Measures relative spectral error per frequency bin. Standard evaluation metric in CUORE and LIGO (DeepClean). Used as training loss in gravitational-wave denoising.

- **L1 on x̂_0** (`--w_l1`): Additional L1 loss directly on the reconstructed signal (not the noise prediction). Anchors time-domain amplitude and phase.

### Choosing weights

Start with `--w_sc 0.1 --w_asd 0.1` and adjust. The spectral losses have different magnitudes; use the training log (which prints per-component losses) to balance them. Setting all `w_*` to 0 recovers the original noise-only loss.

## Project Structure

```
src/ddpm/
  unet.py           - 1D U-Net model (UNet1D)
  unet_cond.py      - Scale-conditioned U-Net (UNet1DScaleCond)
  schedule.py       - Quadratic noise schedule (beta, alpha, alpha_bar)
  diffusion.py      - Forward/reverse diffusion, training loss, sampling
  dataset.py        - HDF5 dataset with on-the-fly clean+noise pairing
  spectral_loss.py  - SC, LSD, J_asd spectral loss functions
  train.py          - Training loop with checkpointing and resume support
  inference.py      - Single/multi-shot inference with metrics and plots
```

## Data Format

HDF5 shards in separate directories:

```
clean/clean_000.h5 ... clean_009.h5   (simulated clean pulses)
noise/noise_000.h5 ... noise_009.h5   (real noise windows from LUCE)
```

Each shard contains a `waveforms` dataset of shape (N, 10000) and an `n_windows` attribute.

## Usage

### Training

Basic (L1 noise loss only):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean/clean_000.h5 \
    --noise_dir /path/to/noise/noise_000.h5 \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

L2 noise loss with AMP (example used for `ddpm_l2_low`, mirrors `ddpm_l1_low` except `--loss l2`):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /media/Disk_YIN/yunshancheng/cuore/clean_v2/clean_low/train \
    --noise_dir /media/Disk_YIN/yunshancheng/cuore/noise/train \
    --output_dir /media/Disk_YIN/yunshancheng/cuore/ddpm_l2_low \
    --loss l2 --amp \
    --epochs 100 --batch_size 32 --lr 2e-4 --T 50 \
    --val_fraction 0.1 --save_every 10 --seed 42 --num_workers 4 \
    > /media/Disk_YIN/yunshancheng/cuore/ddpm_l2_low/train.log 2>&1 & echo "PID: $!"
```

With spectral losses:

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --w_sc 0.1 --w_asd 0.1 \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

With scale conditioning:

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --scale_cond \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

DeScoD-ECG-style schedule + continuous `√ᾱ` conditioning (larger beta_T so x_T really is Gaussian, continuous noise-level embedding):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --beta_1 1e-4 --beta_T 0.5 --cond_mode sqrt_ab \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

Inference must pass the same `--beta_1`, `--beta_T`, `--cond_mode` (and `--cond_scale`, if non-default) as training.

With mixed precision (AMP):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --amp \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

Resume from checkpoint (restores optimizer and best_val_loss; scheduler is rebuilt for remaining epochs with no warm-up):

```bash
nohup stdbuf -oL python3 -u -m src.ddpm.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --w_sc 0.1 --w_asd 0.1 \
    --epochs 300 --batch_size 32 --lr 1e-5 --num_workers 4 \
    --resume /path/to/output/checkpoint_100.pt \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

On resume, `--lr` sets the starting LR and a fresh cosine schedule decays it to 1e-7 over the remaining epochs. No warm-up or cosine restart.

Use full dataset (all shards) by passing the directory:

```bash
--clean_dir /path/to/clean --noise_dir /path/to/noise
```

### Inference

```bash
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10
```

With scale-conditioned model:

```bash
python3 -u -m src.ddpm.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10 --scale_cond
```

Filter by sample type:

```bash
# Only low-amplitude pileup (pileup with max < 100 mV)
python3 -u -m src.ddpm.inference ... --filter low_pileup --n 10

# Only pileup events
python3 -u -m src.ddpm.inference ... --filter pileup --n 10

# Only single pulses
python3 -u -m src.ddpm.inference ... --filter single --n 10

# Specific sample indices
python3 -u -m src.ddpm.inference ... --indices 23,31,33,59,65
```

Multi-shot aggregation (default: mean):

```bash
# 10-shot with median aggregation (more robust to outlier samples)
python3 -u -m src.ddpm.inference ... --aggregation median
```

DDIM sampler (deterministic, supports step skipping):

```bash
# DDIM with full 50 steps
python3 -u -m src.ddpm.inference ... --sampler ddim

# DDIM with 10 steps (5x faster)
python3 -u -m src.ddpm.inference ... --sampler ddim --ddim_steps 10

# DDIM with partial stochasticity
python3 -u -m src.ddpm.inference ... --sampler ddim --eta 0.5

# deterministic inference
   nohup stdbuf -oL python3 -u -m src.ddpm.inference \                                                                                                            
     --model_path /media/AVFD/yunshancheng/cuore/ddpm_l1/best_model.pt \                                                                                          
     --clean_dir /media/AVFD/yunshancheng/cuore/clean/clean_001.h5 \                                                                                              
     --noise_dir /media/AVFD/yunshancheng/cuore/noise/noise_001.h5 \                                                                                              
     --T 50 --n 10 --filter low_pileup \                                                                                                                          
     --output /media/AVFD/yunshancheng/cuore/ddpm_l1/qa_low_pileup_deterministic.png \                                                                            
     --no_noise \                                                                                                                                                 
     > /media/AVFD/yunshancheng/cuore/ddpm_l1/inference_deterministic.log 2>&1 &                                                                                  
   echo $!  
```

### Evaluation Metrics

Time-domain:
- **MSE**: Mean squared error
- **MAD**: Maximum absolute distance
- **BL RMS**: Baseline RMS in the pre-pulse region t in [0, 1.5s) — measures residual noise level after denoising. Compared across noisy, 1-shot, and 10-shot.
- **PRD**: Percentage root-mean-square difference
- **Cosine Similarity**: Waveform shape agreement
- **CC**: Normalized cross-correlation (mean-subtracted)
- **SNR**: Signal-to-noise ratio in dB

Spectral:
- **SC**: Spectral Convergence — STFT magnitude error (partially phase-aware)
- **LSD**: Log-Spectral Distance — RMS log-PSD difference
- **J_asd**: ASD ratio — mean sqrt(PSD_residual / PSD_target)

Peak analysis:
- Peak selection: `scipy.signal.find_peaks` detects candidates, ranked by **prominence** (not absolute height) to correctly identify real peaks over noise ripples on decay slopes. Top N peaks are selected (N=1 for single, N=2 for pileup).
- Peak measurement: parabolic interpolation on the 3 samples around each selected peak for sub-sample precision in both position and amplitude.
- **Pk amp**: Reconstructed peak amplitude in mV
- **Pk amp%**: Relative amplitude error vs clean signal
- **Pk dt ms**: Peak timing error in ms (signal - clean)
- For pileup events, both peaks (Pk1, Pk2) are reported separately. Peaks are ordered by time (first = earliest) and matched between clean and denoised signals by nearest position.

### QA Plot Layout

The inference QA plot has 4 columns per sample:
1. **Waveforms**: Clean, noisy, 1-shot, and 10-shot overlay
2. **Residuals**: Clean minus denoised (1-shot and 10-shot)
3. **PSD**: Power spectral density comparing clean, noisy, and 10-shot denoised
4. **Metrics**: Per-sample metrics table with all numeric results

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Li et al., "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal" (arXiv:2208.00542)
- Ormiston et al., "Noise reduction in gravitational-wave data via deep learning" (Phys. Rev. Research, 2020) — DeepClean ASD loss
- Stevens et al., "Removing Structured Noise with Diffusion Models" (TMLR, 2025) — Joint diffusion framework
