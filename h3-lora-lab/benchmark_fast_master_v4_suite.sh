#!/bin/bash
set -e

# ==============================================================================
# 🚀 FAST MASTER V4 AUTOMATED BENCHMARK SUITE (1s, 2s, 4s)
# Optimized VAE Tile (640px) + 50L Full DiT + 5-Step DPM++ 2M (3 Evals)
# ==============================================================================

export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

# 🚀 VAE OPTIMIZATION FOR 128GB UNIFIED MEMORY (SINGLE TILE DECODE)
export H3_VAE_TILE_PIXELS=640

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

mkdir -p outputs/v4_benchmark_suite

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"
PROMPT="Cinematic realistic 4k master shot. A beautiful young Italian woman with sparkling hazel eyes, detailed eyelashes, natural radiant smile, soft chestnut wavy hair sitting at a sunlit Roman cafe terrace. Warm morning Mediterranean light, crisp facial skin texture, subtle bokeh background. Natural 48kHz audio."

echo "=================================================================="
echo "🚀 STARTING FAST MASTER V4 BENCHMARK SUITE ON APPLE SILICON M5 MAX"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. TEST 1: 1 SECOND (22 FRAMES)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/3] RUNNING 1 SECOND (22 FRAMES)"
T1_START=$(date +%s)
caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" -p "$PROMPT" \
  --width 640 --height 640 --frames 22 \
  --steps 5 --layers 50 --reuse 2 --use-int8-row-fc2 --seed 333 \
  -o outputs/v4_benchmark_suite/temp_1s.mp4
T1_TIME=$(($(date +%s) - T1_START))
FINAL_1="outputs/v4_benchmark_suite/v4_fast_master_640x640_5step_50L_22f_${T1_TIME}s.mp4"
mv outputs/v4_benchmark_suite/temp_1s.mp4 "$FINAL_1"
ffmpeg -y -ss 00:00:00.400 -i "$FINAL_1" -vframes 1 -update 1 "/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/v4_1s_preview_${T1_TIME}s.jpg" 2>/dev/null || true

# ------------------------------------------------------------------------------
# 2. TEST 2: 2 SECONDS (39 FRAMES)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/3] RUNNING 2 SECONDS (39 FRAMES)"
T2_START=$(date +%s)
caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" -p "$PROMPT" \
  --width 640 --height 640 --frames 39 \
  --steps 5 --layers 50 --reuse 2 --use-int8-row-fc2 --seed 333 \
  -o outputs/v4_benchmark_suite/temp_2s.mp4
T2_TIME=$(($(date +%s) - T2_START))
FINAL_2="outputs/v4_benchmark_suite/v4_fast_master_640x640_5step_50L_39f_${T2_TIME}s.mp4"
mv outputs/v4_benchmark_suite/temp_2s.mp4 "$FINAL_2"
ffmpeg -y -ss 00:00:00.600 -i "$FINAL_2" -vframes 1 -update 1 "/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/v4_2s_preview_${T2_TIME}s.jpg" 2>/dev/null || true

# ------------------------------------------------------------------------------
# 3. TEST 3: 4 SECONDS (90 FRAMES)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/3] RUNNING 4 SECONDS (90 FRAMES)"
T3_START=$(date +%s)
caffeinate -dimsu nice -n -20 ./h3 --profile \
  -d "$MODEL_DIR" -p "$PROMPT" \
  --width 640 --height 640 --frames 90 \
  --steps 5 --layers 50 --reuse 2 --use-int8-row-fc2 --seed 333 \
  -o outputs/v4_benchmark_suite/temp_4s.mp4
T3_TIME=$(($(date +%s) - T3_START))
FINAL_3="outputs/v4_benchmark_suite/v4_fast_master_640x640_5step_50L_90f_${T3_TIME}s.mp4"
mv outputs/v4_benchmark_suite/temp_4s.mp4 "$FINAL_3"
ffmpeg -y -ss 00:00:01.500 -i "$FINAL_3" -vframes 1 -update 1 "/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/v4_4s_preview_${T3_TIME}s.jpg" 2>/dev/null || true

echo "=================================================================="
echo "🎉 FAST MASTER V4 BENCHMARK SUITE COMPLETED!"
echo "  1s (22f): ${T1_TIME}s"
echo "  2s (39f): ${T2_TIME}s"
echo "  4s (90f): ${T3_TIME}s"
echo "=================================================================="
