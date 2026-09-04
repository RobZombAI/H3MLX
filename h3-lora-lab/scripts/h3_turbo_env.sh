#!/bin/bash
# === H3 Engine Maximum Throughput — Zero Quality Loss ===

# ACTIVE (verified operational)
export H3_NAX=1
export H3_ZERO_COPY_WEIGHTS=1
export H3_REUSE_MPS_COMMAND=1
export H3_PROFILE=1

# NEW — Zero quality loss
export H3_SOL_ATTN=1
export H3_SOL_CACHE=1
export H3_SOL_ATTN_THRESHOLD=10.0
export H3_VAE_TILE_PIXELS=544
export H3_SOL_STATS=1

# Metal performance
export METAL_DEVICE_WRAPPER_TYPE=0
export MTL_DEBUG_LAYER=0
export MTL_SHADER_VALIDATION=0
export METAL_CAPTURE_ENABLED=0
