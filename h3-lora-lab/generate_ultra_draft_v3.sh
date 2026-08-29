#!/bin/bash
set -e

# ==============================================================================
# 🏆 ULTRA-DRAFT MASTER V3 (OFFICIAL GOLD PRESET)
# Champion Configuration on Apple Silicon M5 Max
# ==============================================================================

PROMPT="${1:-Cinematic realistic 4k close-up portrait. A gorgeous young Italian woman with sparkling hazel eyes, detailed eyelashes, natural radiant smile, soft chestnut wavy hair sitting at a sunlit Roman cafe terrace. Warm morning Mediterranean light, crisp facial skin texture, subtle bokeh background. Natural 48kHz audio.}"
DURATION_SECONDS="${2:-4}"
OUTPUT_PATH="${3:-}"

# Formula reticolo temporale: T = 17n + 5 (24 fps)
case "$DURATION_SECONDS" in
  1) FRAMES=22 ;;  # n=1 (0.92s)
  2) FRAMES=39 ;;  # n=2 (1.63s)
  3) FRAMES=73 ;;  # n=4 (3.04s)
  4) FRAMES=90 ;;  # n=5 (3.75s)
  5) FRAMES=124 ;; # n=7 (5.17s)
  6) FRAMES=141 ;; # n=8 (5.88s)
  *) FRAMES=90 ;;
esac

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
mkdir -p outputs/gold_champions

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH="outputs/gold_champions/ultra_draft_master_v3_640x640_5step_50L_${FRAMES}f_seed333.mp4"
fi

echo "=========================================================="
echo "🏆 RUNNING ULTRA-DRAFT MASTER V3 (GOLD PRESET)"
echo "  Canvas: 640x640 | Frames: $FRAMES (${DURATION_SECONDS}s) | Layers: 50"
echo "  Solver: 5-Step DPM++ 2M Trailing Flow (3 GPU Evaluations)"
echo "  Quant: --use-int8-row-fc2 | Seed: 333"
echo "=========================================================="

caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width 640 --height 640 \
  --frames "$FRAMES" \
  --steps 5 \
  --layers 50 \
  --reuse 2 \
  --use-int8-row-fc2 \
  --seed 333 \
  -o "$OUTPUT_PATH"

echo "=========================================================="
echo "🎉 GENERATION COMPLETE: $OUTPUT_PATH"
echo "=========================================================="
