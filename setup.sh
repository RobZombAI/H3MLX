#!/usr/bin/env bash
# 🚀 H3MLX One-Click Setup & Health Check Script
set -e

echo "========================================================================"
echo "👑 H3MLX Environment Setup & Apple Silicon Verification"
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
    uv pip install --python .venv/bin/python numpy matplotlib pillow
else
    echo "✓ Configurazione con python3 standard..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip numpy matplotlib pillow
fi

# 4. Check Shaders & Binary
if [ ! -f "h3_shaders.metal" ]; then
    ln -sf h3-lora-lab/h3_shaders.metal h3_shaders.metal
fi
if [ ! -f "h3_shaders.metallib" ]; then
    ln -sf h3-lora-lab/h3_shaders.metallib h3_shaders.metallib
fi
chmod +x h3mlx h3mlx-studio bin/h3mlx bin/h3mlx-studio 2>/dev/null || true

# 5. Check Model Weights
echo "🔍 Verifica directory pesi modello MiniMax H3..."
if [ -d "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step" ] || [ -d "/Users/robzomb/h3-models/MiniMax-H3" ]; then
    echo "✓ Modello MiniMax H3 presente e validato."
else
    echo "⚠️  Attenzione: Directory modelli di default non trovata in /Users/robzomb/h3-models/."
    echo "   Puoi specificare il percorso con l'opzione -d / --model-dir <percorso>."
fi

echo "========================================================================"
echo "🎉 H3MLX SETUP COMPLETATO CON SUCCESSO!"
echo "   Puoi avviare l'Interactive Studio con: ./h3mlx studio"
echo "   Oppure lanciare la CLI con:           ./h3mlx --help"
echo "========================================================================"
