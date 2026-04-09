# Flow Matching Pulse Denoiser

Conditional Flow Matching (CFM) for denoising cryogenic bolometer pulse signals from the CUORE experiment.

## Task

Same as the DDPM denoiser: recover clean pulse waveforms from noisy observations to improve energy resolution. This module provides an alternative generative framework for the same conditional denoising task, enabling direct comparison with DDPM.

## Approach

A conditional flow matching model trained on paired (clean, noisy) data:

- **Clean signals**: Simulated via an ODE electro-thermal model. 10-second windows at 1000 Hz (10000 samples).
- **Noise**: Real noise extracted from quiet (no-pulse) windows in LUCE data.
- **Training pairs**: Clean + real noise = noisy observation. Noise-clean pairings are reshuffled each epoch for combinatorial augmentation.
- **Dataset**: Reuses `PulseNoiseDataset` from the DDPM module — no data preprocessing changes needed.

### Flow Matching vs DDPM

| | DDPM | Flow Matching |
|---|---|---|
| Forward process | Gradual noise addition via alpha schedule | Linear interpolation: x_t = (1-t)*x_0 + t*eps |
| Model predicts | noise eps | velocity v = eps - x_0 |
| Time | discrete t in {1,...,T} | continuous t in [0, 1] |
| Inference | T reverse steps with noise schedule | ODE integration (Euler or midpoint) |
| Schedule | quadratic beta schedule required | none — interpolation is linear |
| Stochasticity | inherent in reverse steps | deterministic (ODE) |

### Training

- **Forward interpolation**: x_t = (1-t)*x_0 + t*eps, where t ~ Uniform(0, 1) and eps ~ N(0, I).
- **Velocity target**: v_target = eps - x_0 (direction from data to noise).
- **Training loss**: L1 or L2 between predicted velocity and target velocity. Optional spectral losses on implied x0_hat = x_t - t*v_pred.
- **No noise schedule**: Unlike DDPM, there are no beta/alpha parameters to tune. The interpolation path is a straight line in data space.

### Inference

ODE integration from t=1 (pure noise) to t=0 (data), conditioned on the noisy observation x_tilde:

- **Euler solver** (default): x_{t-dt} = x_t - dt * v(x_t, x_tilde, t). One model evaluation per step.
- **Midpoint solver**: Evaluate at current point, half-step, re-evaluate at midpoint, full step. Two model evaluations per step but more accurate — can use fewer steps for comparable quality.
- **Multi-shot**: Multiple independent ODE integrations from different initial noise samples, aggregated via mean or median.

### Normalization

Same as DDPM: per-window normalization by max(|noisy|). Scale factor saved for physical-unit reconstruction.

## Architecture

Two U-Net variants, mirroring the DDPM module:

### FlowUNet1D (default)

1D U-Net with continuous timestep conditioning:

```
Input: [x_t, x_tilde] concatenated (2 channels, 10000 samples)

Encoder:  4 levels (64 -> 128 -> 256 -> 512), stride-2 Conv1d downsample
          2 ResBlocks per level, GroupNorm, SiLU activation
Bottleneck: 512 channels, self-attention
Decoder:  Mirror of encoder with skip connections
Timestep: Continuous t in [0,1] -> sinusoidal embedding -> MLP -> added into each ResBlock

Output: 1 channel (predicted velocity v)
```

~15.1M parameters.

### FlowUNet1DScaleCond (`--scale_cond`)

Same architecture plus scale conditioning. The normalization scale max(|noisy|) is encoded and added to the timestep embedding.

```
scale -> log(scale)*10 -> SinusoidalEmbedding -> MLP -> scale_emb
cond = timestep_emb + scale_emb  (fed to every ResBlock)
```

~15.6M parameters.

## Loss Functions

Same structure as DDPM, but the base loss is on velocity prediction instead of noise prediction:

```
Total = base_loss(v_pred, v_target) + w_l1*L1(x0_hat, x_0) + w_sc*SC + w_lsd*LSD + w_asd*J_asd
```

where x0_hat = x_t - t*v_pred is the implied clean estimate.

Spectral losses (SC, LSD, J_asd, L1 on x0_hat) are imported from `src.ddpm.spectral_loss` and controlled by `--w_*` flags.

## Project Structure

```
src/flowMatching/
  unet.py              - FlowUNet1D with continuous time embedding
  unet_cond.py         - FlowUNet1DScaleCond (+ scale conditioning)
  flow_matching.py     - Forward interpolation, training loss, ODE sampling
  train.py             - Training loop with checkpointing and resume
  inference.py         - Single/multi-shot inference with metrics and plots
```

Reused from DDPM (via import):
- `src.ddpm.dataset` — PulseNoiseDataset
- `src.ddpm.spectral_loss` — SpectralLoss
- `src.ddpm.inference` — metric functions (MSE, SNR, SC, etc.)

## Usage

### Training

Basic (L1 velocity loss):

```bash
nohup stdbuf -oL python3 -u -m src.flowMatching.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

With spectral losses:

```bash
nohup stdbuf -oL python3 -u -m src.flowMatching.train \
    --clean_dir /path/to/clean \
    --noise_dir /path/to/noise \
    --output_dir /path/to/output \
    --loss l1 --w_sc 0.1 --w_asd 0.1 \
    --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 4 \
    > /path/to/output/train.log 2>&1 & echo "PID: $!"
```

With scale conditioning:

```bash
python3 -u -m src.flowMatching.train ... --scale_cond
```

With AMP and torch.compile:

```bash
python3 -u -m src.flowMatching.train ... --amp --compile
```

Resume from checkpoint:

```bash
python3 -u -m src.flowMatching.train ... --resume /path/to/checkpoint_050.pt --lr 1e-5
```

### Inference

```bash
python3 -u -m src.flowMatching.inference \
    --model_path /path/to/best_model.pt \
    --clean_dir /path/to/clean/clean_001.h5 \
    --noise_dir /path/to/noise/noise_001.h5 \
    --output qa_inference.png --n 10
```

With midpoint solver:

```bash
python3 -u -m src.flowMatching.inference ... --solver midpoint --steps 25
```

Filter by sample type:

```bash
python3 -u -m src.flowMatching.inference ... --filter low_pileup --n 10
python3 -u -m src.flowMatching.inference ... --filter pileup --n 10
python3 -u -m src.flowMatching.inference ... --indices 23,31,33,59,65
```

Multi-shot aggregation:

```bash
python3 -u -m src.flowMatching.inference ... --aggregation median
```

### Evaluation Metrics

Same as the DDPM module. See `src/ddpm/README.md` for full metric descriptions.

## References

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Li et al., "DeScoD-ECG" (arXiv:2208.00542)
