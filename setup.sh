#!/usr/bin/env bash
# H3MLX Setup and Verification Script for Apple Silicon
set -e

echo "========================================================================"
echo "H3MLX Environment Setup & Apple Silicon Verification"
echo "========================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Architecture
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Error: H3MLX requires Apple Silicon hardware (arm64). Detected: $ARCH"
    exit 1
fi
echo "Apple Silicon (arm64) architecture verified."

# 2. Compile Native C/Metal Engine
if [ ! -f "h3-lora-lab/h3" ]; then
    echo "Compiling native C/Metal engine (h3)..."
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
    make -C h3-lora-lab -j"$NCPU"
fi
echo "Native h3 binary verified."

# 3. Setup Virtual Environment via uv or python3
if command -v uv >/dev/null 2>&1; then
    echo "Configuring Python environment using 'uv'..."
    if [ ! -d ".venv" ]; then
        uv venv .venv
    fi
    uv pip install --python .venv/bin/python numpy matplotlib pillow huggingface_hub
else
    echo "Configuring Python environment using standard python3..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip numpy matplotlib pillow huggingface_hub
fi

chmod +x h3mlx h3mlx-studio bin/h3mlx bin/h3mlx-studio download_models.sh download_models.py 2>/dev/null || true

# 4. Check Model Weights
echo "Checking MiniMax H3 checkpoint directory..."
MODEL_FOUND=0
for DIR in \
    "$HOME/h3-models/MiniMax-H3-PDD-8Step" \
    "$HOME/h3-models/MiniMax-H3" \
    "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" \
    "/Users/robzomb/h3-models/MiniMax-H3" \
    "$HOME/Desktop/H3/MiniMax-H3"; do
    if [ -d "$DIR" ]; then
        echo "Found MiniMax H3 weights at: $DIR"
        MODEL_FOUND=1
        break
    fi
done

if [ "$MODEL_FOUND" -eq 0 ]; then
    echo "Notice: No local weights found."
    if [ -t 0 ]; then
        read -p "Download model weights now via download_models.sh (~24 GB)? (y/n): " ANSWER
        if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
            ./download_models.sh
        else
            echo "You can download weights at any time by running: ./download_models.sh"
        fi
    else
        echo "Run './download_models.sh' to download model weights."
    fi
fi

echo "========================================================================"
echo "H3MLX Setup Complete"
echo "   • Interactive Studio : ./h3mlx studio"
echo "   • Command Line (CLI) : ./h3mlx --help"
echo "========================================================================"
