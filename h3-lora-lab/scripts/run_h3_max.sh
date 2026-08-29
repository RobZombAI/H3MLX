#!/bin/bash
set -e

# === MINIMAX H3-MAX PRODUCTION RUNNER ===
export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab
mkdir -p outputs/h3_max

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

# High-adherence photorealistic prompt
PROMPT="MiniMax H3-Max cinematic demo: An elegant futuristic electric hypercar speeding along a neon-lit Tokyo highway at night, rain-slicked asphalt reflections, glowing cybernetic billboards, dynamic tracking low-angle camera, flawless motion blur, ultra-crisp 8K details, cinematic synthwave bass soundtrack with authentic engine roar."

echo "=========================================================="
echo "⚡ EXECUTING MINIMAX H3-MAX PIPELINE ON M5 MAX (960x544, 8-STEP)"
echo "=========================================================="

START_TIME=$(date +%s)

caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width 960 --height 544 \
  --frames 56 \
  --steps 8 \
  --layers 48 \
  --use-int8-row-fc2 \
  -o outputs/h3_max/h3_max_raw.mp4

echo "=== MASTERING TO 1080P 10-BIT MAIN10 (HVC1 + FASTSTART + 48K AAC) ==="
ffmpeg -y -i outputs/h3_max/h3_max_raw.mp4 \
  -filter_complex "[0:v]scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]" \
  -map "[v]" -map "[a]" \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -tag:v hvc1 -r 24 \
  -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart \
  outputs/h3_max/h3_max_master_24fps_1080p.mp4

ffmpeg -y -ss 00:00:01.000 -i outputs/h3_max/h3_max_raw.mp4 \
  -vframes 1 -update 1 \
  /Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/h3_max_preview.jpg

END_TIME=$(date +%s)
DIFF=$((END_TIME - START_TIME))

echo "=========================================================="
echo "⚡ H3-MAX DEMO COMPLETE IN ${DIFF} SECONDS TOTAL!"
echo "=========================================================="
