---
name: minimax-h3-turbo
description: Ultra-fast ComfyUI co-designed execution toolkit for MiniMax H3 / H3-Max Turbo video generation. Features SLA-Attention (Sparse Local Attention in middle DiT blocks 14-36), Predictive Euler Step Reuse (reuse 2), 4-step PDD distillation, Motion Context temporal video chaining, and sub-90s generation on Apple Silicon M5 Max. Activate whenever the user asks for fast H3 generation, ComfyUI H3 workflows, SLA-attention, motion context, or sub-90s / turbo video generation.
---

# MiniMax H3-Turbo (ComfyUI Co-Design & Fast Suite)

This skill provides ultra-fast inference recipes and continuous sequence chaining based on the ComfyUI Cloud graph architecture and Sol-Engine optimizations.

---

## 1. Fast Turbo Engine (4 Steps / 45 Layers / Reuse 2 / SLA-Attention)

```bash
cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab

# SLA-Attention Token Sparsity (Blocks 14-36)
export H3_TOKEN_REDUCTION=1
export H3_TOKEN_REDUCTION_BLOCKS="14:36"
export H3_TOKEN_REDUCTION_SCALE="1.0"
export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1

caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" \
  -p "YOUR_PROMPT" \
  --width 960 --height 544 \
  --frames 107 \
  --steps 4 \
  --layers 45 \
  --reuse 2 \
  --token-reduction 1 \
  --use-int8-row-fc2 \
  -o "outputs/turbo_raw.mp4"
```

## 2. Motion Context (Temporal Sequence Chaining)

To extend a video continuously across multiple shots without visual jumping:
```bash
# Pass the last frame of Clip 1 as the first frame of Clip 2
./h3 --profile \
  -d "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" \
  -p "Continuation prompt..." \
  --first "outputs/clip1_last_frame.jpg" \
  --frames 107 --steps 4 --layers 45 --reuse 2 \
  -o "outputs/clip2_raw.mp4"
```

## 3. Automated Unified Runner

You can also run directly via:
```bash
./h3_max_suite/inference/h3_max_engine.sh "YOUR_PROMPT" "outputs/my_video.mp4" [STEPS=4] [WIDTH=960] [HEIGHT=544] [LAYERS=45] [FRAMES=107] [REUSE=2]
```
