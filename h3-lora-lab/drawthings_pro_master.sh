#!/bin/bash
set -e

# ==============================================================================
# 👑 DRAW THINGS PRO GOLDEN PROFILE FOR MINIMAX H3-MAX (APPLE SILICON M3/M5 MAX)
# ==============================================================================

# 1. Driver & Hardware Kernel Settings (Zero Overhead, Pure AMX/Metal Execution)
export H3_PROFILE=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab
mkdir -p outputs/drawthings_golden_master

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

# 2. Configurable Arguments with Draw Things Pro Defaults
WIDTH=${1:-960}
HEIGHT=${2:-544}
FRAMES=${3:-73}
STEPS=${4:-8}
LAYERS=${5:-50}
REUSE=${6:-2}
SEED=${7:-314}
PROMPT=${8:-"Cinematic realistic 4k video. A group of elegant couples passionately dancing authentic Argentine tango in a candlelit vintage Buenos Aires milonga ballroom. Dancers in fitted black suits and flowing deep red silk dresses performing graceful ochos, tight pivots, and dramatic leg wraps across a glossy parquet wooden floor. Warm ambient chandelier glow, atmospheric golden haze, rich romantic mood. Fluid steadycam tracking shot moving gently around the couples. Authentic acoustic tango music, rich bandoneon accordion melody, passionate violin solo, deep upright bass, and crisp rhythmic tapping of leather dance heels in 48kHz stereo audio."}

echo "=========================================================="
echo "👑 LAUNCHING DRAW THINGS PRO GOLDEN PROFILE RUN"
echo "  Resolution: ${WIDTH}x${HEIGHT} | Frames: ${FRAMES} (Causal 17n+5)"
echo "  Sampler: 2nd-Order Multistep Flow | Steps: ${STEPS} | Layers: ${LAYERS}"
echo "  Schedule: Trailing Shifted | Reuse: ${REUSE} | Seed: ${SEED}"
echo "=========================================================="

START_TIME=$(date +%s)

caffeinate -dimsu ./h3 --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width "$WIDTH" --height "$HEIGHT" \
  --frames "$FRAMES" \
  --steps "$STEPS" \
  --layers "$LAYERS" \
  --reuse "$REUSE" \
  --use-int8-row-fc2 \
  --seed "$SEED" \
  -o outputs/drawthings_golden_master/temp_raw.mp4

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

FINAL_NAME="drawthings_golden_${WIDTH}x${HEIGHT}_${STEPS}step_${LAYERS}L_${FRAMES}f_seed${SEED}_${TOTAL_TIME}s"
FINAL_VIDEO="outputs/drawthings_golden_master/${FINAL_NAME}.mp4"
PREVIEW_FINAL="/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/${FINAL_NAME}_preview.jpg"

mv outputs/drawthings_golden_master/temp_raw.mp4 "$FINAL_VIDEO"

ffmpeg -y -ss 00:00:01.500 -i "$FINAL_VIDEO" -vframes 1 -update 1 "$PREVIEW_FINAL" 2>/dev/null || true

echo "=========================================================="
echo "⚡ RUN COMPLETED IN ${TOTAL_TIME} SECONDS TOTAL!"
echo "  Video File: $FINAL_VIDEO"
echo "  Preview:    $PREVIEW_FINAL"
echo "=========================================================="
