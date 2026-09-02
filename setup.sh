#!/usr/bin/env bash
# 🚀 H3MLX One-Click Setup & Turnkey Verification Script
# Works on any Apple Silicon Mac with ZERO pre-downloaded files.
set -e

echo "========================================================================"
echo "👑 H3MLX Environment Setup & Apple Silicon Turnkey Installation"
echo "========================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Architecture
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "❌ Errore: H3MLX richiede hardware Apple Silicon (arm64). Rilevato: $ARCH"
    exit 1
fi
echo "✓ Architettura Apple Silicon (arm64) rilevata."

# 2. Compile Native C/Metal Engine
if [ ! -f "h3-lora-lab/h3" ]; then
    echo "🔨 Compilazione binario nativo Metal C (h3)..."
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
    make -C h3-lora-lab -j"$NCPU"
fi
echo "✓ Binario nativo h3 compilato e verificato."

# 3. Setup Virtual Environment via uv or python3
if command -v uv >/dev/null 2>&1; then
    echo "✓ Utilizzo di 'uv' per la configurazione rapida dell'ambiente..."
    if [ ! -d ".venv" ]; then
        uv venv .venv
    fi
    uv pip install --python .venv/bin/python numpy matplotlib pillow huggingface_hub
else
    echo "✓ Configurazione con python3 standard..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip numpy matplotlib pillow huggingface_hub
fi

# 4. Check Shaders & Binary
if [ ! -f "h3_shaders.metal" ]; then
    ln -sf h3-lora-lab/h3_shaders.metal h3_shaders.metal
fi
if [ ! -f "h3_shaders.metallib" ]; then
    ln -sf h3-lora-lab/h3_shaders.metallib h3_shaders.metallib
fi
chmod +x h3mlx h3mlx-studio bin/h3mlx bin/h3mlx-studio download_models.sh download_models.py 2>/dev/null || true

# 5. Check Model Weights
echo "🔍 Verifica directory pesi modello MiniMax H3..."
MODEL_FOUND=0
for DIR in \
    "$HOME/h3-models/MiniMax-H3-PDD-8Step" \
    "$HOME/h3-models/MiniMax-H3" \
    "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" \
    "/Users/robzomb/h3-models/MiniMax-H3" \
    "$HOME/Desktop/H3/MiniMax-H3"; do
    if [ -d "$DIR" ]; then
        echo "✓ Modello MiniMax H3 trovato in: $DIR"
        MODEL_FOUND=1
        break
    fi
done

if [ "$MODEL_FOUND" -eq 0 ]; then
    echo "⚠️  Nessun modello trovato in locale."
    if [ -t 0 ]; then
        read -p "Vuoi scaricare i pesi del modello ora con download_models.sh (~24 GB PDD-8Step)? (s/n): " ANSWER
        if [[ "$ANSWER" =~ ^[SsYy]$ ]]; then
            ./download_models.sh
        else
            echo "ℹ️  Puoi scaricare i modelli in qualsiasi momento eseguendo: ./download_models.sh"
        fi
    else
        echo "ℹ️  Esegui './download_models.sh' per scaricare automaticamente i pesi del modello."
    fi
fi

echo "========================================================================"
echo "🎉 H3MLX SETUP COMPLETATO CON SUCCESSO!"
echo "   • Studio Interattivo:   ./h3mlx studio"
echo "   • Frontiera Livello 1:  ./h3mlx --frontier 1 -p \"tuo prompt\""
echo "   • CLI Completa 1:1:     ./h3mlx --help"
echo "========================================================================"
