# 👩🍳 MiniMax H3-Max: Complete End-to-End Replication Recipe

This guide details the step-by-step process to reproduce **H3-Max performance** for training, distillation, serving, and local native execution.

---

## 🗺️ Complete Reproduction Pipeline

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  1. Data Pipeline       │     │  2. SGLang miles        │     │  3. PDD Distillation    │
│  - Video & Prompt Data  │ ──► │  - LoRA SFT (Rank 64)   │ ──► │  - Multi-Head Output    │
│  - VAE Latents Cache    │     │  - RL (GRPO Aesthetic)  │     │  - 4-8 Step Velocity    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  6. Master 10-Bit Video │     │  5. Sol-Attention Engine│     │  4. Weight Packaging    │
│  - Apple `hvc1` + Audio │ ◄── │  - h3.c Metal 4 NAX     │ ◄── │  - Safetensors Export   │
│  - EBU R128 (-14 LUFS)  │     │  - Sol-Engine Step Cache│     │  - Zero-Overhead Fold   │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 👩🍳 Step 1: Data Preparation & Latent Caching
Pre-encode your video datasets to eliminate GPU bottlenecks during reinforcement learning:
```bash
/Users/robzomb/hunyuan3d-studio/.venv/bin/python3 h3_max_suite/trainer/prepare_dataset.py
```

## 🧠 Step 2: Post-Training (LoRA SFT + GRPO Reinforcement Learning)
Run the SGLang miles post-training recipe to train the rank-64 adapter and PDD multi-head velocity projections:
```bash
/Users/robzomb/hunyuan3d-studio/.venv/bin/python3 h3_max_suite/trainer/miles_sglang_trainer.py
```

## 📦 Step 3: LoRA Weight Folding (Zero Overhead)
Fuse your trained LoRA adapter directly into the base weights so that inference runs with 0% extra memory latency:
```bash
/Users/robzomb/hunyuan3d-studio/.venv/bin/python3 h3_max_suite/scripts/fold_lora_into_h3.py \
  /Users/robzomb/h3-models/MiniMax-H3 \
  /Users/robzomb/h3-models/loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors \
  /Users/robzomb/h3-models/MiniMax-H3-Max
```

## ⚡ Step 4: High-Throughput Serving & Execution
Launch generation on local Apple Silicon M5 Max with unified memory zero-copy and 48 kHz broadcast audio:
```bash
./h3_max_suite/inference/h3_max_engine.sh "Your prompt" "outputs/my_master.mp4" 8 960 544 50
```
