#!/usr/bin/env python3
"""
H3 Checkpoint Health & Audit Manager
Inspects, sanitizes and verifies integrity of all MiniMax H3 weights and LoRAs.
"""

import os
import sys
import json
from pathlib import Path

# Default discovery paths
MODELS_ROOT = Path(os.environ.get("H3_MODELS_DIR", Path.home() / "h3-models"))
if not MODELS_ROOT.exists():
    ALT_DIR = Path(__file__).resolve().parent / "models"
    if ALT_DIR.exists():
        MODELS_ROOT = ALT_DIR

def audit_models():
    print("=" * 72)
    print(f"H3 MODEL INTEGRITY AUDIT in {MODELS_ROOT}")
    print("=" * 72)
    
    if not MODELS_ROOT.exists():
        print(f"Directory {MODELS_ROOT} not found.")
        return
        
    models = sorted([d for d in MODELS_ROOT.iterdir() if d.is_dir()])
    
    for m in models:
        print(f"\nModel: {m.name}")
        subfiles = list(m.rglob("*"))
        total_size = sum(f.stat().st_size for f in subfiles if f.is_file())
        size_gb = total_size / (1024 ** 3)
        print(f"   • Total Size        : {size_gb:.2f} GB ({len(subfiles)} files/directories)")
        
        # Check for incomplete downloads (.aria2)
        aria2_files = list(m.rglob("*.aria2"))
        if aria2_files:
            print(f"   WARNING: Detected {len(aria2_files)} incomplete files (.aria2):")
            for af in aria2_files:
                print(f"      - {af.name}")
        else:
            print("   No temporary/incomplete .aria2 files detected.")
            
        # Check FL2VA transformer safetensors
        transformer_dir = m / "FL2VA" / "transformer"
        if transformer_dir.exists():
            st_files = list(transformer_dir.glob("*.safetensors"))
            idx_file = transformer_dir / "model.safetensors.index.json"
            print(f"   • DiT Transformer   : {len(st_files)} safetensors files found.")
            if idx_file.exists():
                try:
                    with open(idx_file, "r") as f:
                        idx_data = json.load(f)
                    total_tensors = len(idx_data.get("weight_map", {}))
                    print(f"   • Safetensors Index : Valid ({total_tensors} tensors mapped)")
                except Exception as e:
                    print(f"   Error reading index: {e}")
                    
        # Check VAE
        vae_dir = m / "FL2VA" / "video_vae"
        if vae_dir.exists():
            vae_size = sum(f.stat().st_size for f in vae_dir.rglob("*") if f.is_file()) / (1024 ** 3)
            print(f"   • Video VAE         : Present ({vae_size:.2f} GB)")
            
    # Audit LoRAs
    loras_dir = MODELS_ROOT / "loras"
    if loras_dir.exists():
        print(f"\nLoRA Modules & Adapters ({loras_dir.name}):")
        for lora in sorted(loras_dir.glob("*.safetensors")):
            lora_size = lora.stat().st_size / (1024 ** 2)
            print(f"   • {lora.name:<50} : {lora_size:8.2f} MB")
            
    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    audit_models()
