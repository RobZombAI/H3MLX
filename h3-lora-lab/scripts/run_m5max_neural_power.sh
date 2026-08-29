#!/bin/bash
set -e

# === UNLEASH ALL M5 MAX NEURAL, GPU & CPU ACCELERATORS ===
export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_GPU_SAMPLER=1
export OMP_NUM_THREADS=12

# Disable debug validation for pure raw silicon throughput
export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab
mkdir -p outputs/neural_max_speed

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

# High-fidelity prompt with scientific anatomical accuracy and commercial lighting
PROMPT="Scientific photorealistic commercial: an elite female Olympic diver leaps from the 10-meter platform in explosive athletic form. Biomechanically perfect somersault trajectory, micro-droplets of water dispersing in dynamic arcs, focused ocular intensity, shimmering skin subsurface scattering under intense arena halogen floodlights, turquoise crystal water reflections, 35mm cinema prime, IMAX 8k broadcast master."

echo "=========================================================="
echo "🚀 EXECUTING AT MAXIMUM M5 MAX HARDWARE THROUGHPUT"
echo "=========================================================="

START_TIME=$(date +%s)

# Run with maximum CPU Quality of Service (QoS) and highest priority
caffeinate -dimsu nice -n -20 ./h3 \
  -d "$MODEL_DIR" \
  --steps 6 \
  --layers 48 \
  --use-int8-row-fc2 \
  --width 960 --height 544 \
  --frames 56 \
  -p "$PROMPT" \
  -o outputs/neural_max_speed/clip_neural_max_raw.mp4

# Hardware VideoToolbox dual-engine mastering
ffmpeg -y -i outputs/neural_max_speed/clip_neural_max_raw.mp4 \
  -filter_complex "[0:v]scale=1920:1080:flags=spline+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v]" \
  -map "[v]" -map 0:a \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -r 30 \
  -c:a aac -b:a 320k \
  outputs/neural_max_speed/clip_neural_max_master_30fps_1080p.mp4

ffmpeg -y -i outputs/neural_max_speed/clip_neural_max_raw.mp4 \
  -filter_complex "[0:v]scale=1920:1080:flags=spline+accurate_rnd+full_chroma_int+full_chroma_inp,cas=0.30,format=yuv420p10le[v]" \
  -map "[v]" -map 0:a \
  -c:v hevc_videotoolbox -profile:v main10 -pix_fmt p010le -b:v 80M -r 24 \
  -c:a aac -b:a 320k \
  outputs/neural_max_speed/clip_neural_max_master_24fps_1080p.mp4

ffmpeg -y -ss 00:00:01.000 -i outputs/neural_max_speed/clip_neural_max_raw.mp4 -vframes 1 /Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/neural_max_preview.jpg

END_TIME=$(date +%s)
DIFF=$((END_TIME - START_TIME))

echo "=========================================================="
echo "⚡ COMPLETE IN ${DIFF} SECONDS TOTAL!"
echo "=========================================================="
