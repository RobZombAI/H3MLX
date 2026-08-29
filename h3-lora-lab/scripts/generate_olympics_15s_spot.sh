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
mkdir -p outputs/olympics_15s

MODEL_DIR="/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"

echo "=========================================================="
echo "🎬 STARTING OLYMPIC 15-SECOND COMMERCIAL SPOT (M5 MAX)"
echo "Target: 4 High-Density Neural Clips (107 / 90 / 107 / 107 frames)"
echo "=========================================================="

START_TOTAL=$(date +%s)

# --- NODE 1: ARENA & MAJESTY (Opening) ---
echo ""
echo ">>> [1/4] Node 1 (Arena): Olympic Aquatic Centre & Crowd..."
START_N1=$(date +%s)
PROMPT1="Cinematic wide establishing shot of an ultra-modern Olympic diving arena, glowing turquoise pool, towering 10-meter platform, thousands of cheering spectators in tiered stadium, dramatic golden cinematic beams of sunlight, photorealistic 8k commercial"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 107 \
  -p "$PROMPT1" \
  -o outputs/olympics_15s/node1_arena_raw.mp4
END_N1=$(date +%s)
TIME_N1=$((END_N1 - START_N1))
echo ">>> Node 1 finished in ${TIME_N1}s"

# --- NODE 2: ATHLETE FOCUS & TENSION ---
echo ""
echo ">>> [2/4] Node 2 (Focus): Diver Ritual on 10m Platform..."
START_N2=$(date +%s)
PROMPT2="Intense cinematic close-up and profile of a female Olympic diver standing at the edge of the 10-meter diving platform, water droplets on skin, hyper-focused eyes, athletic form, stadium background out of focus, photorealistic 8k commercial"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 90 \
  -p "$PROMPT2" \
  -o outputs/olympics_15s/node2_diver_focus_raw.mp4
END_N2=$(date +%s)
TIME_N2=$((END_N2 - START_N2))
echo ">>> Node 2 finished in ${TIME_N2}s"

# --- NODE 3: THE FLIGHT & ROTATION (Climax) ---
echo ""
echo ">>> [3/4] Node 3 (Flight): Explosive 10m Somersault Takeoff..."
START_N3=$(date +%s)
PROMPT3="Breathtaking slow-motion shot of an Olympic diver leaping off the 10-meter tower, executing a flawless acrobatic triple twist somersault in mid-air, body silhouette against dramatic stadium spotlights, cinematic photorealistic"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 107 \
  -p "$PROMPT3" \
  -o outputs/olympics_15s/node3_flight_raw.mp4
END_N3=$(date +%s)
TIME_N3=$((END_N3 - START_N3))
echo ">>> Node 3 finished in ${TIME_N3}s"

# --- NODE 4: WATER IMMERSION & TRIUMPH (Outro) ---
echo ""
echo ">>> [4/4] Node 4 (Immersion): Needle Entry & Underwater Glide..."
START_N4=$(date +%s)
PROMPT4="Spectacular underwater tracking shot of Olympic diver slicing smoothly into crystal turquoise pool water with zero splash, cascading bioluminescent bubble vortex, golden light shafts piercing deep water, serene triumph, cinematic 8k"
./h3 \
  -d "$MODEL_DIR" \
  --steps 8 \
  --layers 50 \
  --use-int8-row-fc2 \
  --width 864 --height 480 \
  --frames 107 \
  -p "$PROMPT4" \
  -o outputs/olympics_15s/node4_immersion_raw.mp4
END_N4=$(date +%s)
TIME_N4=$((END_N4 - START_N4))
echo ">>> Node 4 finished in ${TIME_N4}s"

END_TOTAL=$(date +%s)
TIME_TOTAL=$((END_TOTAL - START_TOTAL))

echo ""
echo "=========================================================="
echo "🎬 ALL 4 HIGH-DENSITY CLIPS GENERATED IN ${TIME_TOTAL}s!"
echo "Node 1: ${TIME_N1}s | Node 2: ${TIME_N2}s | Node 3: ${TIME_N3}s | Node 4: ${TIME_N4}s"
echo "=========================================================="
