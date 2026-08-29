---
name: minimax-h3-max
description: Baseline high-fidelity guide and execution toolkit for MiniMax H3-Max video generation, SGLang miles RL + LoRA SFT training, PDD 6-step/8-step high-fidelity inference, and Apple Silicon Metal 4 NAX native acceleration. Activate whenever the user mentions MiniMax H3, H3-Max, Hailuo 3, SGLang miles, or asks for high-fidelity 6-step/8-step cinematic video generation with synchronized 48 kHz audio.
---

# MiniMax H3-Max (Golden High-Fidelity Standard Suite)

This skill provides full technical instructions, training recipes, and native Apple Silicon M5 Max execution commands for **MiniMax H3-Max** with 6-step and 8-step high-fidelity DiT denoise.

---

## 1. Golden Standard Execution Baseline (6-8 Steps / 50 Layers / 73 Frames)

```bash
cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab

export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" \
  -p "YOUR_CINEMATIC_PROMPT" \
  --width 960 --height 544 \
  --frames 73 \
  --steps 8 \
  --layers 50 \
  --reuse 2 \
  --use-int8-row-fc2 \
  --seed 333 \
  -o "outputs/raw_video.mp4"
```

### Golden Rules:
1. **Canvas Geometry**: Always use $960 \times 544$ (Width $\ge$ Height) to preserve 3D-RoPE rotary coordinate alignment and prevent body/limb stretching.
2. **Causal Chunking**: Always use $73$ frames ($17 \times 4 + 5$) for exactly 32 VAE tiles (cuts VAE decode to ~40s with zero padding).
3. **Social Vertical Reels (9:16)**: Generate natively at $960 \times 544$ and master to $1080 \times 1920$ via FFmpeg hardware center crop.
4. **Forbidden Flags**: Never use `--use-reference-rope` (causes ghosting) or `--sol-attn`/`--sol-cache` (causes CPU locks).

---

## 2. 10-Bit Main10 Apple Native Mastering

```bash
# 16:9 Landscape Master (1920x1080)
ffmpeg -y -i raw.mp4 \
  -filter_complex "[0:v]scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]" \
  -map "[v]" -map "[a]" \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -tag:v hvc1 -r 24 \
  -c:a aac -b:a 320k -ar 48000 -movflags +faststart master_1080p.mp4

# 9:16 Vertical Reel Master (1080x1920)
ffmpeg -y -i raw.mp4 \
  -filter_complex "[0:v]crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]" \
  -map "[v]" -map "[a]" \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -tag:v hvc1 -r 24 \
  -c:a aac -b:a 320k -ar 48000 -movflags +faststart master_reel_9x16.mp4
```

