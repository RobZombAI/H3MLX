---
name: h3xml-engine
description: "High-performance video and image generation toolkit for H3XML (MiniMax H3 Metal 4 NAX accelerated engine). Features calibrated aspect ratio presets, fast iteration modes, text-to-image and text-to-video pipelines on Apple Silicon (M-series with Unified Memory)."
---

# H3XML Engine & Presets Guide

H3XML is the native Metal 4 NAX inference engine for MiniMax H3 (33B parameters) on Apple Silicon (M-series with Unified Memory). It provides full mathematical compatibility with `antirez/h3.c` while delivering accelerated GPU denoise passes and zero-overhead model residency.

---

## Calibrated Resolution & Aspect Ratio Configurations

### 1. 768x512 Balanced Widescreen (3:2)
* **Command Flags**: `--width 768 --height 512 --frames 90 --steps 20 --layers 45 --reuse 2 --use-int8-row-fc2`
* **Passes**: 10 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `113.69s`, Total = `144.91s`
* **Characteristics**: Balanced token density for 3:2 landscape compositions.

### 2. 864x480 Standard Wide 16:9 (Panavision)
* **Command Flags**: `--width 864 --height 480 --frames 90 --steps 40 --layers 50 --reuse 6 --use-int8-row-fc2`
* **Passes**: 8 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `104.43s`, Total = `137.82s`
* **Characteristics**: Standard 16:9 cinematic aspect ratio with full 50 layers.

### 3. 864x480 Standard Wide Balanced 16:9
* **Command Flags**: `--width 864 --height 480 --frames 90 --steps 20 --layers 45 --reuse 2 --use-int8-row-fc2`
* **Passes**: 11 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `126.03s`, Total = `158.20s`
* **Characteristics**: 45 layers with step reuse 2 for 16:9 widescreen.

### 4. 512x512 Master Square (1:1) — Fast Portrait
* **Command Flags**: `--width 512 --height 512 --frames 90 --steps 40 --layers 50 --reuse 6 --use-int8-row-fc2`
* **Passes**: 8 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `48.82s`, Total = `75.73s`
* **Characteristics**: Square 1:1 format focusing token budget on close-up facial features.

### 5. 512x512 Balanced Square (1:1)
* **Command Flags**: `--width 512 --height 512 --frames 90 --steps 20 --layers 45 --reuse 2 --use-int8-row-fc2`
* **Passes**: 11 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `59.86s`, Total = `86.08s`
* **Characteristics**: Fast square generation with 45 layers.

### 6. 768x768 High-Res Square (1:1)
* **Command Flags**: `--width 768 --height 768 --frames 90 --steps 20 --layers 45 --reuse 2 --use-int8-row-fc2`
* **Passes**: 11 GPU passes calculated
* **Performance (4.0s / 90F)**: Denoise = `237.12s`, Total = `276.90s`
* **Characteristics**: 2304 spatial tokens for high spatial detail in 1:1 framing.

---

## Fast Turnaround Configurations (1.0s / 22 Frames)

For rapid prompt exploration and lighting checks:

### 7. Fast Mode: 512x512 (22 Frames / ~1s)
* **Command Flags**: `--width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 --use-int8-row-fc2`
* **Performance**: Denoise = `8.79s`, Total = `28.38s`

### 8. Fast Mode: 768x512 Widescreen (22 Frames / ~1s)
* **Command Flags**: `--width 768 --height 512 --frames 22 --steps 40 --layers 50 --reuse 6 --use-int8-row-fc2`
* **Performance**: Denoise = `15.01s`, Total = `36.65s`

---

## Text-to-Image (T2I) Snapshot Mode

For 1-frame image generation:
* **Command Flags**: `--width 768 --height 512 --frames 5 --steps 20 --layers 45 --reuse 1 --use-int8-row-fc2`
* Generates an instant high-resolution frame via causal frame slicing ($T=5 \to t_0$).

---

## Execution & Environment Setup

Native Metal 4 NAX environment flags:
```bash
export H3_PROFILE=1
export H3_NAX=1
export H3_CPU_SAMPLER=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_DIT_COMMAND_BLOCKS=0
export H3_SOLVER=euler
export OMP_NUM_THREADS=18
```
