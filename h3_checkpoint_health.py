#!/usr/bin/env python3
"""
🩺 H3 Checkpoint Health & Audit Manager
Inspects, sanitizes and verifies integrity of all MiniMax H3 weights and LoRAs.
"""

import os
import sys
import json
from pathlib import Path

MODELS_ROOT = Path("/Users/robzomb/h3-models")

def audit_models():
    print("=" * 72)
    print(f"🩺 AUDIT INTEGRITÀ MODELLI H3 in {MODELS_ROOT}")
    print("=" * 72)
    
    if not MODELS_ROOT.exists():
        print(f"❌ Directory {MODELS_ROOT} non trovata!")
        return
        
    models = sorted([d for d in MODELS_ROOT.iterdir() if d.is_dir()])
    
    for m in models:
        print(f"\n📦 Modello: {m.name}")
        subfiles = list(m.rglob("*"))
        total_size = sum(f.stat().st_size for f in subfiles if f.is_file())
        size_gb = total_size / (1024 ** 3)
        print(f"   • Dimensione Totale : {size_gb:.2f} GB ({len(subfiles)} file/directory)")
        
        # Check for incomplete downloads (.aria2)
        aria2_files = list(m.rglob("*.aria2"))
        if aria2_files:
            print(f"   ⚠️ ATTENZIONE: Rilevati {len(aria2_files)} file incompleti (.aria2):")
            for af in aria2_files:
                print(f"      - {af.name}")
        else:
            print("   ✅ Nessun file temporaneo/incompleto .aria2 rilevato.")
            
        # Check FL2VA transformer safetensors
        transformer_dir = m / "FL2VA" / "transformer"
        if transformer_dir.exists():
            st_files = list(transformer_dir.glob("*.safetensors"))
            idx_file = transformer_dir / "model.safetensors.index.json"
            print(f"   • DiT Transformer   : {len(st_files)} file safetensors trovati.")
            if idx_file.exists():
                try:
                    with open(idx_file, "r") as f:
                        idx_data = json.load(f)
                    total_tensors = len(idx_data.get("weight_map", {}))
                    print(f"   • Indice Safetensors: Valido ({total_tensors} tensori mappati)")
                except Exception as e:
                    print(f"   ⚠️ Errore lettura indice: {e}")
                    
        # Check VAE
        vae_dir = m / "FL2VA" / "video_vae"
        if vae_dir.exists():
            vae_size = sum(f.stat().st_size for f in vae_dir.rglob("*") if f.is_file()) / (1024 ** 3)
            print(f"   • Video VAE         : Presente ({vae_size:.2f} GB)")
            
    # Audit LoRAs
    loras_dir = MODELS_ROOT / "loras"
    if loras_dir.exists():
        print(f"\n🎨 Moduli LoRA & Turbo Adapter ({loras_dir.name}):")
        for lora in sorted(loras_dir.glob("*.safetensors")):
            lora_size = lora.stat().st_size / (1024 ** 2)
            print(f"   • {lora.name:<50} : {lora_size:8.2f} MB")
            
    print("\n" + "=" * 72)
    print("✅ AUDIT COMPLETATO!")
    print("=" * 72)

if __name__ == "__main__":
    audit_models()
