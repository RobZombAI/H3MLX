#!/bin/bash
set -e

# ==============================================================================
# 🚀 MiniMax H3-Max Unified Native Engine Runner (Apple Silicon M5 Max)
# Incorporating ComfyUI Graph Strengths: SLA-Attention, Motion Context & PDD Turbo
# ==============================================================================

PROMPT="${1:-"Slow-motion cinematic video. A terrifying horror monster slowly emerges from a thick, bubbling, pitch-black slimy sea. Mud and glowing toxic green slime drip from its distorted, wet body. The camera starts with a close-up on the bubbling dark water, then slowly pans up to reveal the creature's sunken, glowing yellow eyes cutting through a dense coastal fog."}"
OUTPUT="${2:-"outputs/h3_max_suite_output.mp4"}"
STEPS="${3:-4}"
WIDTH="${4:-960}"
HEIGHT="${5:-544}"
LAYERS="${6:-45}"
FRAMES="${7:-107}"
REUSE="${8:-2}"
FIRST_FRAME="${9:-""}"

# === HARDWARE & NEURAL ACCELERATION (METAL 4 NAX) ===
export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

# === COMFYUI SLA-ATTENTION EQUIVALENT (SPARSE LOCAL ATTENTION BLOCKS 14-36) ===
export H3_TOKEN_REDUCTION=1
export H3_TOKEN_REDUCTION_BLOCKS="14:36"
export H3_TOKEN_REDUCTION_SCALE="1.0"

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

ROOT_DIR="/Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab"
MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================================="
echo "⚡ RUNNING H3-MAX PRODUCTION ENGINE (COMFYUI CO-DESIGN)"
echo "  Prompt: $PROMPT"
echo "  Canvas: ${WIDTH}x${HEIGHT} | Frames: $FRAMES"
echo "  Steps: $STEPS | Layers: $LAYERS | Reuse: $REUSE"
echo "  SLA-Attention: Blocks 14-36 Enabled"
if [ -n "$FIRST_FRAME" ]; then
  echo "  Motion Context (First Frame): $FIRST_FRAME"
fi
echo "=========================================================="

START_TIME=$(date +%s)

FIRST_FRAME_ARG=""
if [ -n "$FIRST_FRAME" ] && [ -f "$FIRST_FRAME" ]; then
  FIRST_FRAME_ARG="--first $FIRST_FRAME"
fi

caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width "$WIDTH" --height "$HEIGHT" \
  --frames "$FRAMES" \
  --steps "$STEPS" \
  --layers "$LAYERS" \
  --reuse "$REUSE" \
  --token-reduction 1 \
  --use-int8-row-fc2 \
  $FIRST_FRAME_ARG \
  -o "$OUTPUT"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

DIRNAME=$(dirname "$OUTPUT")
BASENAME=$(basename "$OUTPUT" .mp4)
FINAL_NAME="${BASENAME}_${WIDTH}x${HEIGHT}_${STEPS}step_${LAYERS}L_reuse${REUSE}_${TOTAL_TIME}s"
MASTER_PATH="${DIRNAME}/${FINAL_NAME}_master_24fps_1080p.mp4"
PREVIEW_PATH="/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/${FINAL_NAME}_preview.jpg"

echo "=== MASTERING TO 1080P 10-BIT MAIN10 (HVC1 + FASTSTART + 48K AAC) ==="
ffmpeg -y -i "$OUTPUT" \
  -filter_complex "[0:v]scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]" \
  -map "[v]" -map "[a]" \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -tag:v hvc1 -r 24 \
  -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart \
  "$MASTER_PATH"

ffmpeg -y -ss 00:00:02.000 -i "$OUTPUT" \
  -vframes 1 -update 1 \
  "$PREVIEW_PATH"

echo "=========================================================="
echo "⚡ H3-MAX PIPELINE COMPLETE IN ${TOTAL_TIME} SECONDS!"
echo "  Master File: $MASTER_PATH"
echo "  Preview: $PREVIEW_PATH"
echo "=========================================================="
