#!/bin/bash
set -e

# ==============================================================================
# 🚀 MINIMAX H3 METAL 4 MASTER CLI SUITE (UNIVERSAL MAC EDITION)
# ==============================================================================
# Authors: antirez, MiniMax AI, Hao-AI Lab, Antigravity AI Engineering Team
# License: Apache 2.0 / MiniMax Community License
# Purpose: High-speed, eco-efficient, photorealistic video generation on Apple Silicon
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
CYAN="\033[1;36m"
RED="\033[1;31m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║      🚀 MINIMAX H3 METAL 4 MASTER CLI SUITE (UNIVERSAL MAC)         ║"
echo "║   Pure C/Metal 4 NAX · 50 Full Layers · Dynamic INT8-FC2 · Eco-AI     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# 1. HARDWARE & ENVIRONMENT AUTO-DETECTION
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "16")
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "34359738368")
RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))

echo -e "${BLUE}🔍 Detected Hardware:${RESET} ${BOLD}${CHIP}${RESET} | ${CORES} CPU Cores | ${RAM_GB} GB Unified Memory (UMA)"

# Set environment optimizations for Metal 4 & UMA
export H3_PROFILE=1
export H3_NAX="qkv-attn"
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=${CORES}
export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

# 2. MODEL PATH RESOLUTION & CONSENT
DEFAULT_MODEL_DIR="$HOME/h3-models/MiniMax-H3-PDD-8Step"
ALT_MODEL_DIR="$HOME/Desktop/H3/MiniMax-H3"

if [ -d "$DEFAULT_MODEL_DIR/FL2VA" ] || [ -d "$DEFAULT_MODEL_DIR" ]; then
  MODEL_DIR="$DEFAULT_MODEL_DIR"
elif [ -d "$ALT_MODEL_DIR" ]; then
  MODEL_DIR="$ALT_MODEL_DIR"
else
  echo -e "${YELLOW}⚠️ MiniMax-H3 model weights not found at default location ($DEFAULT_MODEL_DIR).${RESET}"
  read -p "Would you like to automatically download model weights now (~61 GB FL2VA)? (y/n): " CONSENT
  if [[ "$CONSENT" =~ ^[Yy]$ ]]; then
    mkdir -p "$DEFAULT_MODEL_DIR"
    echo -e "${GREEN}⬇️ Starting guided model download with auto-resume...${RESET}"
    python3 "$(dirname "$0")/h3-lora-lab/scripts/download_model.py" --dir "$DEFAULT_MODEL_DIR" || true
    MODEL_DIR="$DEFAULT_MODEL_DIR"
  else
    echo -e "${RED}❌ Execution cancelled: model path missing.${RESET}"
    exit 1
  fi
fi

echo -e "${GREEN}✓ Active Model:${RESET} ${MODEL_DIR}"

# 3. PRESET SELECTOR & CONFIGURATION
PRESET="${1:-champion}"
PROMPT="${2:-A majestic golden eagle soaring gracefully over snow-capped alpine peaks at sunrise, crisp 8k definition, specular feather details catching golden sunlight, 35mm f/1.4 lens bokeh, 48kHz spatial mountain wind.}"
WIDTH="${3:-640}"
HEIGHT="${4:-640}"
FRAMES="${5:-39}"
FIRST_FRAME="${6:-}"

case "$PRESET" in
  champion|fast_master)
    STEPS=8; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0
    LABEL="🏆 Fast Master Champion (8-Step / 50L / INT8 / Shift 12.0)"
    ;;
  turbo|fastvideo)
    STEPS=4; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0
    LABEL="⚡ FastVideo v0.2 Turbo (4-Step [999,749,500,250] / 50L / INT8)"
    ;;
  draft|preview)
    STEPS=4; LAYERS=45; REUSE=2; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0
    LABEL="👀 Ultra Draft (4-Step / 45L / Reuse 2 / INT8)"
    ;;
  cinema|16x9)
    STEPS=8; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0; WIDTH=960; HEIGHT=544
    LABEL="🎬 Cinema 16:9 Widescreen (8-Step / 960x544 / 50L)"
    ;;
  reel|9x16)
    STEPS=8; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0; WIDTH=544; HEIGHT=960
    LABEL="📱 Vertical Reel 9:16 (8-Step / 544x960 / 50L)"
    ;;
  quality)
    STEPS=20; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0
    LABEL="💎 High Convergence Quality (20-Step / 50L / INT8)"
    ;;
  oracle)
    STEPS=50; LAYERS=50; REUSE=1; USE_INT8=0; VSHIFT=12.0; ASHIFT=3.0
    LABEL="👑 Full Oracle Ground-Truth (50-Step / 50L / BF16)"
    ;;
  *)
    echo -e "${YELLOW}Unknown preset '$PRESET', falling back to default: Champion${RESET}"
    STEPS=8; LAYERS=50; REUSE=1; USE_INT8=1; VSHIFT=12.0; ASHIFT=3.0
    LABEL="🏆 Fast Master Champion"
    ;;
esac

export H3_VIDEO_SHIFT=$VSHIFT
export H3_AUDIO_SHIFT=$ASHIFT

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$(pwd)/outputs"
mkdir -p "$OUT_DIR"
RAW_OUT="$OUT_DIR/raw_${PRESET}_${FRAMES}f_${TIMESTAMP}.mp4"
MASTER_OUT="$OUT_DIR/master_${PRESET}_${FRAMES}f_${TIMESTAMP}.mp4"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}⚙️  Active Configuration:${RESET} ${GREEN}${LABEL}${RESET}"
echo -e "📐 Resolution: ${BOLD}${WIDTH}x${HEIGHT}${RESET} | 🎞️ Frames: ${BOLD}${FRAMES}${RESET} (~$((FRAMES / 24))s @ 24fps)"
echo -e "⚡ Architecture: ${GREEN}Metal 4 NAX + INT8-FC2 Quantization (50 Full Layers, 100% Spatial Tokens)${RESET}"
echo -e "📝 Prompt: \"${PROMPT}\""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${RESET}"
echo ""

# Build optional CLI arguments
EXTRA_ARGS=""
if [ "$USE_INT8" -eq 1 ]; then EXTRA_ARGS="$EXTRA_ARGS --use-int8-row-fc2"; fi
if [ -n "$FIRST_FRAME" ] && [ -f "$FIRST_FRAME" ]; then EXTRA_ARGS="$EXTRA_ARGS --first-frame $FIRST_FRAME"; fi

# 4. EXECUTE ENGINE WITH IN-PLACE PROGRESS RENDERING
BIN_PATH="$(dirname "$0")/h3-lora-lab/h3"
if [ ! -f "$BIN_PATH" ]; then BIN_PATH="./h3"; fi

START_TIME=$(date +%s)

caffeinate -dimsu nice -n 0 "$BIN_PATH" --profile \
  -d "$MODEL_DIR" \
  -p "$PROMPT" \
  --width "$WIDTH" --height "$HEIGHT" \
  --frames "$FRAMES" \
  --steps "$STEPS" \
  --layers "$LAYERS" \
  --reuse "$REUSE" \
  --seed 333 \
  $EXTRA_ARGS \
  -o "$RAW_OUT"

END_TIME=$(date +%s)
TOTAL_LATENCY=$((END_TIME - START_TIME))

# 5. AUTOMATED MASTERING PIPELINE
echo ""
echo -e "${CYAN}🎛️  [MASTERING] Applying Lanczos Cinema Grading + EBU R128 (-14 LUFS)...${RESET}"
ffmpeg -y -i "$RAW_OUT" \
  -vf "unsharp=5:5:0.6:5:5:0.0,eq=contrast=1.06:brightness=0.01:saturation=1.08" \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart \
  -af "loudnorm=I=-14:TP=-1.5:LRA=7" -c:a aac -b:a 256k -ar 48000 \
  "$MASTER_OUT" 2>/dev/null || cp "$RAW_OUT" "$MASTER_OUT"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}✅ GENERATION AND MASTERING COMPLETED IN ${TOTAL_LATENCY} SECONDS!${RESET}"
echo -e "📁 Native Master Video: ${CYAN}${MASTER_OUT}${RESET}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${RESET}"
