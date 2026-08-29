#!/bin/bash
set -e

# ==============================================================================
# 💎 MINIMAX H3 FAST MASTER CHAMPION (APPLE SILICON M5 MAX NATIVE OPTIMIZED)
# ==============================================================================

export H3_PROFILE=1
export H3_NAX="qkv-attn"
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=18

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab
mkdir -p outputs/fast_master_champion

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"
DEFAULT_PROMPT="Masterpiece award-winning cinematic close-up portrait of a breathtakingly beautiful young Italian woman. Crisp 8k optical definition, sparkling realistic hazel-green iris with detailed radial fibers and specular sunlight reflections, natural porcelain skin texture with delicate pores, soft genuine warm smile with immaculate separated teeth, loose chestnut wavy hair strands catching rim lighting. Soft blurred Roman cafe background, authentic 35mm f/1.4 lens bokeh, 48kHz spatial audio."

PROMPT="${1:-$DEFAULT_PROMPT}"
FRAMES="${2:-22}"
STEPS="${3:-8}"
REUSE="${4:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="outputs/fast_master_champion/fast_master_${FRAMES}f_${STEPS}step_${TIMESTAMP}.mp4"

echo "=========================================================="
echo "💎 RUNNING MINIMAX H3 FAST MASTER CHAMPION"
echo "  Prompt: $PROMPT"
echo "  Frames: $FRAMES | Steps: $STEPS | Reuse: $REUSE | Layers: 50"
echo "  Canvas: 640x640 | Hardware: Apple Silicon M5 Max"
echo "=========================================================="

START=$(date +%s)
echo ""
echo "🚀 [FASE 1/4] Inizializzazione Text Encoder Qwen 3-VL & Pesi UMA..."

caffeinate -dimsu nice -n 0 ./h3 --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width 640 --height 640 \
  --frames "$FRAMES" \
  --steps "$STEPS" \
  --layers 50 \
  --reuse "$REUSE" \
  --use-int8-row-fc2 \
  --seed 333 \
  -o "$OUTPUT"

END=$(date +%s)

TOTAL=$((END - START))

# 🎬 AUTOMATIC ANIMATED GIF & THUMBNAIL GENERATION FOR INSTANT IN-CHAT PLAYBACK:
GIF_OUTPUT="${OUTPUT%.mp4}_animated.gif"
THUMB_OUTPUT="${OUTPUT%.mp4}_thumb.jpg"
ffmpeg -y -i "$OUTPUT" -vf "fps=20,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" "$GIF_OUTPUT" 2>/dev/null || true
ffmpeg -y -ss 00:00:00.600 -i "$OUTPUT" -vframes 1 -update 1 "$THUMB_OUTPUT" 2>/dev/null || true

# Copy to artifacts directory if available:
ART_DIR="/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df"
if [ -d "$ART_DIR" ]; then
  cp "$OUTPUT" "$ART_DIR/latest_fast_master.mp4" 2>/dev/null || true
  cp "$GIF_OUTPUT" "$ART_DIR/latest_fast_master_animated.gif" 2>/dev/null || true
  cp "$THUMB_OUTPUT" "$ART_DIR/latest_fast_master_thumb.jpg" 2>/dev/null || true
fi

echo ""
echo "=========================================================="
echo "🎉 FAST MASTER RUN COMPLETED IN ${TOTAL}s TOTAL!"
echo "  Video MP4:    $OUTPUT"
echo "  Animated GIF: $GIF_OUTPUT"
echo "  Thumbnail:    $THUMB_OUTPUT"
echo "=========================================================="
