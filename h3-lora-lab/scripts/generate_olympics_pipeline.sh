#!/bin/bash
set -e

export H3_PROFILE=1
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1

export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0

cd /Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab
mkdir -p outputs/olympics

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

echo "=========================================================="
echo "🎬 STARTING OLYMPIC DIVING MULTI-SHOT PRODUCTION (M5 MAX)"
echo "=========================================================="

START_TOTAL=$(date +%s)

# --- SHOT 1: ESTABLISHING SHOT ---
echo ""
echo ">>> [1/4] Generating Shot 1: Olympic Diving Venue & Crowd..."
START_SHOT1=$(date +%s)
PROMPT1="Olympic diving venue, crystal clear turquoise pool, 10-meter diving tower, packed stadium with cheering crowd, bright golden sunlight, cinematic photorealistic wide shot, 8k resolution"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 56 \
  -p "$PROMPT1" \
  -o outputs/olympics/clip1_venue_raw.mp4
END_SHOT1=$(date +%s)
TIME_SHOT1=$((END_SHOT1 - START_SHOT1))
echo ">>> Shot 1 finished in ${TIME_SHOT1}s"

# --- SHOT 2: DIVER PREPARATION ---
echo ""
echo ">>> [2/4] Generating Shot 2: Diver Poised on 10m Platform..."
START_SHOT2=$(date +%s)
PROMPT2="Female Olympic diver standing poised on 10-meter diving platform, arms raised in starting position, intense focused expression, athletic Olympic swimsuit, stadium background, cinematic photorealistic medium shot"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 56 \
  -p "$PROMPT2" \
  -o outputs/olympics/clip2_diver_prep_raw.mp4
END_SHOT2=$(date +%s)
TIME_SHOT2=$((END_SHOT2 - START_SHOT2))
echo ">>> Shot 2 finished in ${TIME_SHOT2}s"

# --- SHOT 3: DIVE ROTATION & ACTION ---
echo ""
echo ">>> [3/4] Generating Shot 3: Mid-air Somervault & Athletic Dive..."
START_SHOT3=$(date +%s)
PROMPT3="Olympic diver performing spectacular graceful dive somersault from 10-meter platform, dynamic rotation in mid-air, athletic form, splash anticipation, natural bright sunlight, cinematic photorealistic slow-motion"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 56 \
  -p "$PROMPT3" \
  -o outputs/olympics/clip3_dive_action_raw.mp4
END_SHOT3=$(date +%s)
TIME_SHOT3=$((END_SHOT3 - START_SHOT3))
echo ">>> Shot 3 finished in ${TIME_SHOT3}s"

# --- SHOT 4: UNDERWATER IMMERSION ---
echo ""
echo ">>> [4/4] Generating Shot 4: Underwater Entry & Bubbles..."
START_SHOT4=$(date +%s)
PROMPT4="Underwater camera shot of Olympic diver smoothly piercing crystal clear swimming pool water, majestic bubble trail, turquoise water with shimmering light rays, serene cinematic slow-motion photorealistic"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 56 \
  -p "$PROMPT4" \
  -o outputs/olympics/clip4_underwater_raw.mp4
END_SHOT4=$(date +%s)
TIME_SHOT4=$((END_SHOT4 - START_SHOT4))
echo ">>> Shot 4 finished in ${TIME_SHOT4}s"

END_TOTAL=$(date +%s)
TIME_TOTAL=$((END_TOTAL - START_TOTAL))

echo ""
echo "=========================================================="
echo "🎬 ALL 4 CLIPS GENERATED IN ${TIME_TOTAL}s!"
echo "Shot 1: ${TIME_SHOT1}s | Shot 2: ${TIME_SHOT2}s | Shot 3: ${TIME_SHOT3}s | Shot 4: ${TIME_SHOT4}s"
echo "=========================================================="
