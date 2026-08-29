# 🚀 MiniMax H3-Max Complete Ecosystem Suite

This suite provides the complete, open-weights reproducible implementation of the **MiniMax H3-Max** video generation ecosystem, spanning training, distillation, serving, and local native inference on Apple Silicon.

---

## 🏗️ Architecture

```
h3_max_suite/
├── trainer/
│   └── miles_sglang_trainer.py   # SGLang miles RL + LoRA SFT Trainer & PDD Exporter
├── inference/
│   ├── sgl_diffusion_serving.py  # SGL-Diffusion Serving with Sol-Attention & Sol-Engine
│   └── h3_max_engine.sh          # Native C/Metal 4 Engine Runner (M5 Max UMA Zero-Copy)
├── weights/
│   └── (Symlinked/Cached .safetensors LoRA adapters: 768p 8-step & 4-step)
└── scripts/
    └── run_h3_max.sh             # Quick execution script
```

---

## 👩🍳 1. Training & Post-Training (SGLang miles)
- **RL Algorithm**: Group Relative Policy Optimization (GRPO) / Direct Preference Optimization (DPO).
- **Aesthetic Reward Model**: Enforces strict prompt understanding, lighting coherence, and physical anatomy.
- **LoRA Targets**: Attention QKV, Feed-Forward MLP projections, and AdaLN modulation layers.
- **PDD Multi-Head Distillation**: Distills velocity trajectory into 4–8 sampling steps.

```bash
/Users/robzomb/hunyuan3d-studio/.venv/bin/python3 h3_max_suite/trainer/miles_sglang_trainer.py
```

---

## ⚡ 2. High-Throughput Serving (sgl-diffusion)
- **Sol-Attention**: Block-sparse dynamic self-attention providing 3.95x acceleration.
- **Sol-Engine**: Adaptive velocity caching skipping redundant DiT evaluations.

```bash
/Users/robzomb/hunyuan3d-studio/.venv/bin/python3 h3_max_suite/inference/sgl_diffusion_serving.py
```

---

## 🍏 3. Native Apple Silicon M5 Max Execution (`h3.c` + Metal 4 NAX)
- **128 GB UMA Zero-Copy**: Zero serialization latency, 800 GB/s bandwidth.
- **Mastering**: 1080p 10-Bit Main10 Apple Native (`hvc1` FourCC + FastStart + 48 kHz AAC EBU R128).

```bash
./h3_max_suite/inference/h3_max_engine.sh "Your prompt here" "outputs/my_video.mp4" 8 960 544 50
```
