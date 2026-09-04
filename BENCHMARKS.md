# H3MLX Reference Benchmarks
### Reproducible Hardware Performance on Apple Silicon (M5 Max, 128GB Unified Memory, >400 GB/s)
#### Metal 4 NAX Fused Attention · DPM++ 2M Second-Order Solver · FreqFlow & Super-Nyquist Phase Alignment · Latent TVD Minmod Pre-Emphasis

This document records local, reproducible hardware benchmarks conducted on **Apple Silicon M5 Max (128GB Unified Memory)** using the **H3MLX v3.2 Frontier Engine**.

*Aesthetic evaluation is intentionally left to community assessment. The tables below record objective hardware execution metrics.*

---

## ⚡ 1. Short Benchmark Suite (56 Frames, 2.33s @ 24fps)

All presets evaluate the exact causal temporal lattice $T = 17 \times 3 + 5 = 56$ frames across **50 dense DiT layers (100% spatial capacity)**, Metal DPM++ 2M solver, TVD Minmod pre-emphasis, and dynamic AMX INT8 FC2 quantization:

| Preset | Aspect & Canvas Resolution (RAW → Master) | Total Wall Time (56 fr / 2.33s) | Throughput | Smart Filter Profile | File Size (RAW / Master) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Champion (3:2)** | `3:2 (768x512 → 3072x2048)` | **`51.59 s`** | **`1.09 FPS`** | Portrait & Bilateral | `2.1 MB` / `12.2 MB` |
| **Cinema Widescreen (16:9)** | `16:9 (960x544 → 3840x2176)` | **`68.47 s`** | **`0.82 FPS`** | Cinema Action | `3.6 MB` / `16.9 MB` |
| **Square High-Density (1:1)** | `1:1 (640x640 → 2560x2560)` | **`68.43 s`** | **`0.82 FPS`** | Speed & Detail | `3.3 MB` / `13.9 MB` |
| **Vertical Cinema Reel (9:16)** | `9:16 (576x1024 → 2304x4096)` | **`103.62 s`** | **`0.54 FPS`** | Vertical Beauty | `2.7 MB` / `9.9 MB` |
| **Stylized Anime (3:2)** | `3:2 (768x512 → 3072x2048)` | **`66.77 s`** | **`0.84 FPS`** | Anime & Stylized | `2.3 MB` / `13.3 MB` |

---

## 🎬 2. Cinema Master Suite (90 Frames, 3.75s @ 24fps)

Extended sequence duration on the causal lattice $T = 17 \times 5 + 5 = 90$ frames across 50 dense DiT layers:

| Preset | Aspect & Canvas Resolution (RAW → Master) | Total Wall Time (90 fr / 3.75s) | Throughput | Smart Filter Profile | File Size (RAW / Master) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Champion (3:2)** | `3:2 (768x512 → 3072x2048)` | **`85.12 s`** | **`1.06 FPS`** | Portrait & Beauty | `2.2 MB` / `14.8 MB` |
| **Cinema Widescreen (16:9)** | `16:9 (960x544 → 3840x2176)` | **`122.53 s`** | **`0.73 FPS`** | Cinema Action | `4.3 MB` / `26.1 MB` |
| **Square High-Density (1:1)** | `1:1 (640x640 → 2560x2560)` | **`93.93 s`** | **`0.96 FPS`** | Speed & Detail | `5.1 MB` / `19.7 MB` |
| **Vertical Cinema Reel (9:16)** | `9:16 (576x1024 → 2304x4096)` | **`157.11 s`** | **`0.57 FPS`** | Vertical Beauty | `4.2 MB` / `19.8 MB` |
| **Stylized Anime (3:2)** | `3:2 (768x512 → 3072x2048)` | **`95.74 s`** | **`0.94 FPS`** | Anime & Stylized | `2.7 MB` / `16.3 MB` |

---

## 📋 3. Detailed Specifications per Preset

### 1. Champion Master (3:2)
* **Prompt**: *"Cinematic close-up portrait of a person smiling, natural soft lighting, highly detailed"*
* **Native Canvas**: $768\times512$ | **Mastering Canvas**: $3072\times2048$ (3:2 4K UHD Master)
* **Post-Processing Chain**: Bilateral de-gridding ($\sigma_S=2, \sigma_R=0.06$) + Lanczos upscaling + AMD FidelityFX CAS (0.22).
* **Execution Metrics**: `85.09 s` total wall time (`1.06 FPS`).

### 2. Cinema Widescreen (16:9)
* **Prompt**: *"Cinematic wide shot of a futuristic neon city at sunset with rain reflections, highly detailed"*
* **Native Canvas**: $960\times544$ | **Mastering Canvas**: $3840\times2176$ (16:9 Widescreen Master)
* **Post-Processing Chain**: Bilateral de-gridding + Lanczos upscaling + AMD FidelityFX CAS (0.30).
* **Execution Metrics**: `122.53 s` total wall time (`0.73 FPS`).

### 3. Square High-Density (1:1)
* **Prompt**: *"A sleek sports car driving through a scenic mountain road in autumn, realistic, 4k"*
* **Native Canvas**: $640\times640$ | **Mastering Canvas**: $2560\times2560$ (1:1 Square Master)
* **Post-Processing Chain**: 3D stabilization + Lanczos upscaling + AMD FidelityFX CAS (0.35).
* **Execution Metrics**: `93.93 s` total wall time (`0.96 FPS`).

### 4. Vertical Cinema Reel (9:16)
* **Prompt**: *"Cinematic vertical portrait of a model walking down a sunlit avenue, expressive eyes and warm smile"*
* **Native Canvas**: $576\times1024$ | **Mastering Canvas**: $2304\times4096$ (9:16 Vertical Master)
* **Post-Processing Chain**: Bilateral de-gridding + Lanczos upscaling + AMD FidelityFX CAS (0.22).
* **Execution Metrics**: `157.11 s` total wall time (`0.57 FPS`).

### 5. Stylized Anime (3:2)
* **Prompt**: *"Lush green valley with rolling hills, giant wind turbine, fluffy clouds, anime aesthetic"*
* **Native Canvas**: $768\times512$ | **Mastering Canvas**: $3072\times2048$ (3:2 Anime Master)
* **Post-Processing Chain**: Debanding filter + Spline upscaling + AMD FidelityFX CAS (0.42).
* **Execution Metrics**: `95.74 s` total wall time (`0.94 FPS`).

---

## 💾 4. Dual Output Workflow (RAW + MASTER 4K)

By default, every generation produces both output tiers:
1. **RAW File**: Uncompressed video sampled at native canvas resolution directly from the Metal GPU (optimal for inspection and downstream editing).
2. **MASTER 4K File**: Mastered video processed with Lanczos scaling, bilateral de-gridding, AMD FidelityFX CAS adaptive contrast sharpening, and Apple VideoToolbox Main 10-bit HEVC encoding.

---

## ⚠️ 5. Thermal Considerations on Apple Silicon

Long batch video generation drives high GPU and memory controller load (>400 GB/s continuous unified memory throughput). On MacBook Pro laptops, configuring fan control to maximum or using macOS High Power Mode is recommended to maintain peak clock frequencies without thermal throttling.
