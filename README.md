# H3MLX
### High-Performance MiniMax H3 Inference Engine on Apple Silicon (C & Metal 4)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M1--M5)-black.svg)]()
[![Backend: Metal 4 / AMX](https://img.shields.io/badge/Backend-Metal%204%20%2F%20AMX-orange.svg)]()
[![Status: Active Development](https://img.shields.io/badge/Status-v3.2%20Active-green.svg)]()

H3MLX is an open-source, low-overhead inference engine for the **MiniMax H3** (Hailuo 01) video generation architecture, engineered natively for Apple Silicon (M1–M5 Max and Ultra) using pure C, Metal 4, and the Apple Matrix Coprocessor (AMX).

The project builds upon the foundational architecture created by Salvatore Sanfilippo ([@antirez](https://github.com/antirez/h3.c)), extending it with custom Metal compute shaders, second-order symplectic flow matching, high-frequency spectral shaping, and hardware-accelerated 10-bit video mastering via Apple VideoToolbox.

---

## 🎯 Project Goals

1. **Zero Framework Bloat**: No PyTorch, no CUDA abstractions, no heavy Python runtime dependencies during inference. The numerical core is written in C99 and compiled with `clang -O3`.
2. **Unified Memory Exploitation**: Direct allocation and zero-copy tensor sharing between CPU and Metal GPU across macOS Unified Memory Architecture (UMA).
3. **Full Model Capacity**: Evaluates all **50 dense transformer layers** without layer-dropping or structural truncation.
4. **Reproducible Numerical Stability**: Mathematical solvers designed to minimize truncation error in few-step distillation regimes.

---

## ⚠️ Current Status of Audio Generation (Call for Community Help)

> [!IMPORTANT]
> **Audio Status: Currently Broken / Experimental**  
> While the MiniMax H3 model weights include an audio generation and decoding branch (Audio VAE), the current local implementation produces audio artifacts, desynchronization, or silence.  
> 
> As a result, **audio generation is bypassed/muted by default** in standard video outputs.  
> 
> **Call for Contributions (RFC / Help Wanted)**:  
> If you have experience with causal audio latent decoders, Mel-spectrogram synthesis, or C-based audio DSP, we welcome assistance in debugging and fixing the audio pipeline. Please see the open issues or submit a pull request against `h3-lora-lab/h3_audio_vae.c`.

---

## ⚡ Measured System Benchmarks

*Hardware: Apple Silicon M5 Max (16-inch, 128 GB Unified Memory, >400 GB/s bandwidth).*  
*Configuration: 50 dense DiT layers, 8-step PDD flow matching, dynamic AMX INT8 FC2.*

*Note: Aesthetic and visual quality evaluation is intentionally left to the community to judge independently across diverse prompts and styles. The table below reports solely reproducible hardware execution metrics.*

| Preset | Aspect Ratio | Canvas Resolution | Frames ($T$) | Duration | GPU Denoise Time | Total Wall Time | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Champion (3:2)** | 3:2 | 768 × 512 | 56 | 2.33s @ 24fps | 28.31s | 46.50s | 1.20 FPS |
| **Champion 4s (3:2)** | 3:2 | 768 × 512 | 90 | 3.75s @ 24fps | 64.12s | 83.11s | 1.08 FPS |
| **Cinema Widescreen (16:9)** | 16:9 | 960 × 544 | 56 | 2.33s @ 24fps | 37.40s | 58.20s | 0.96 FPS |
| **Square (1:1)** | 1:1 | 640 × 640 | 56 | 2.33s @ 24fps | 36.80s | 57.10s | 0.98 FPS |
| **Vertical Reel (9:16)** | 9:16 | 576 × 1024 | 56 | 2.33s @ 24fps | 59.20s | 88.40s | 0.63 FPS |

---

## 🏗️ Architectural Foundations

### 1. DPM++ 2M Symplectic Trajectory Solver
In standard first-order Euler discretization ($x_{k+1} = x_k + \Delta t \cdot v_k$), the truncation error is $O(\Delta t^2)$. In few-step regimes (8 steps, $\Delta t \approx 0.125$), this causes trajectory drift. H3MLX implements an on-chip Adams-Bashforth second-order solver directly within Metal GPU shaders (`h3_shaders.metal`):
$$r_k = \frac{\sigma_k - \sigma_{k+1}}{\sigma_{k-1} - \sigma_k}, \quad v_k^{\text{curved}} = \text{fma}(0.5 \cdot r_k, v_k - v_{k-1}, v_k)$$
$$x_{k+1} = \text{mix}\left(x_k + \sigma_k \cdot v_k^{\text{curved}}, \; x_k, \; \frac{\sigma_{k+1}}{\sigma_k}\right)$$
This reduces global truncation error to $O(\Delta t^3)$, computed via fused multiply-add (FMA) register instructions without global memory round-trips.

### 2. TVD Minmod Temporal Pre-Emphasis
Causal 3D Video VAE decoders apply $4\times$ temporal pooling. For translating features, this induces temporal sinc attenuation ($\omega > \pi / \|\vec{d}\|$), producing motion blur. Prior to VAE decompression, the engine evaluates the second-order discrete temporal Laplacian $\nabla_t^2 x_t$ bounded by a non-linear Total Variation Diminishing (TVD) Minmod slope limiter from computational fluid dynamics. This prevents boundary ringing and preserves edges during camera or subject motion.

### 3. FreqFlow Dynamic Late-Step Velocity Boost
In late diffusion steps ($\sigma \le 0.35$), velocity gradients $v_t$ undergo a localized high-frequency spatial boost:
$$\alpha = \text{strength} \times \left(1.0 - \frac{\sigma}{0.35}\right)$$
The operator is strictly bounded by adjacent spatial gradients to avoid ringing or high-frequency flicker.

### 4. 2D Spatial Super-Nyquist Pre-VAE Phase Alignment
To pre-compensate the spatial transfer function and low-pass softening inherent in $8\times$ convolutional spatial upsampling within the 3D VAE decoder, the engine applies a non-linear phase correction to the latent tensor immediately before passing it to the VAE.

### 5. Hardware 10-Bit VideoToolbox Mastering
When `--4k` or `--smart-filter master-optics` is enabled, the decoded frames are processed through Apple VideoToolbox using the hardware Media Engine:
* Contrast Adaptive Sharpening (AMD FidelityFX CAS).
* Film grain emulation (Kodak Vision3 5219 sensitometric profile) to eliminate digital macroblocking.
* HEVC Main 10-bit (`p010le`) encoding at 60 Mbps.

---

## 🚀 Quickstart

### Prerequisites
* macOS 14.0 or later (macOS 15+ recommended).
* Apple Silicon Mac (M1, M2, M3, M4, M5 — Pro/Max/Ultra recommended).
* Command Line Tools (`xcode-select --install`).
* `ffmpeg` (`brew install ffmpeg`).

### 1. Clone and Setup
```bash
git clone https://github.com/RobZombAI/H3MLX.git
cd H3MLX
./setup.sh
```

### 2. Download Model Weights
Download the official MiniMax H3 checkpoint:
```bash
./download_models.sh
```

### 3. Interactive Studio (English TUI)
Launch the interactive studio to configure Text-to-Video or Image-to-Video generation:
```bash
./h3mlx studio
```

### 4. Command Line Interface (CLI)
Generate video directly from the command line:

```bash
# Text-to-Video with Champion preset:
./h3mlx -p "Cinematic close-up portrait of a person, soft evening lighting, sharp focus" --preset h3mlx_champion_gold --4k

# Image-to-Video (animate a starting photo):
./h3mlx -p "Gentle camera zoom in, wind blowing softly" --first-frame input_photo.jpg --4k

# First and Last Frame Interpolation:
./h3mlx -p "Smooth morph transition" --first-frame start.jpg --last-frame end.jpg --4k

# Enable Frontier 7 features (FreqFlow + Pre-VAE Phase Alignment + Master Optics):
./h3mlx -p "Dynamic motion scene of a dancer, sharp focus" --frontier 7 --4k
```

### 5. Python API
```python
import h3mlx_engine_core

result = h3mlx_engine_core.execute_h3_generation(
    prompt="Cinematic landscape at sunset, 35mm film look",
    width=768,
    height=512,
    frames=56,
    steps=8,
    frontier="7",
    upscale_4k=True,
    output_path="outputs/output_video.mp4"
)

if result.success:
    print(f"Generated in {result.wall_time_s:.2f}s: {result.output_path}")
```

---

## 📁 Repository Structure

```
H3MLX/
├── bin/
│   ├── h3mlx                     # CLI shortcut
│   ├── h3mlx-studio              # Interactive TUI Studio shortcut
│   └── fanctl                    # Apple Silicon fan control helper
├── h3-lora-lab/                  # Pure C and Metal 4 engine core
│   ├── Makefile                  # clang -O3 compilation configuration
│   ├── h3.c, h3.h                # Top-level inference coordination
│   ├── h3_dit.c, h3_dit.h        # DiT block execution & ODE solver
│   ├── h3_host.c, h3_host.h      # Host memory, FreqFlow & Phase Alignment
│   ├── h3_gpu.m, h3_gpu.h        # Metal 4 AMX matrix kernel dispatch
│   ├── h3_metal.m, h3_metal.h    # Metal device management
│   ├── h3_shaders.metal          # Metal shaders (DPM++ 2M, fused attention)
│   ├── h3_video_vae.c/.h         # 3D Video VAE causal decoder
│   ├── h3_audio_vae.c/.h         # Audio VAE decoder (experimental / WIP)
│   └── h3_max_suite/             # LoRA utilities and trainer tooling
├── h3mlx_engine_core.py          # Unified Python engine wrapper
├── h3mlx_cli.py                  # Command-line argument parser
├── h3mlx_studio.py               # Interactive English TUI Studio
├── h3mlx_presets.py              # Canonical aspect ratio presets
├── h3mlx_smart_filters.py        # Post-processing filters
├── h3_cinema_upscaler.py         # VideoToolbox 10-bit HEVC & AMD CAS upscaler
├── prompts_library/              # Curated prompt collection
├── setup.sh                      # Environment setup script
├── download_models.sh            # Weight downloader script
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🤝 Contributing

We welcome contributions from the community, especially regarding:
* **Audio VAE stabilization**: Debugging the audio latent decoder and audio/video muxing (`h3_audio_vae.c`).
* **Metal kernel optimization**: Improving threadgroup occupancy and tile utilization on M-series GPUs.
* **Quantization precision**: Exploring FP8 / mixed-precision AMX matrix strategies.

Please open an issue or submit a PR.

---

## 📜 License & Acknowledgments

Released under the [MIT License](LICENSE).

* Built upon the foundational pure C code by **Salvatore Sanfilippo ([@antirez](https://github.com/antirez/h3.c))**.
* Model architecture and weights by **MiniMax AI / Hailuo Team**.
* Metal 4, AMX, and VideoToolbox optimizations by **RobZomb AI**.
