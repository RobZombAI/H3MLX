#!/usr/bin/env python3
"""
👑 H3MLX CLI (Universal Apple Silicon Edition)
Official Compatible Engine: Livello 1 (Metal 4 NAX Fused Attention + Native GPU Trajectory Sampler)
50 Full Dense Layers · Pure Spatial Sampling · UMA Zero-Copy · Row-Major INT8 FC2
"""

import os
import sys
import argparse
from pathlib import Path

from h3mlx_presets import calculate_canonical_frames
from h3mlx_engine_core import execute_h3_generation, resolve_model_path

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["studio", "interactive", "ui", "tui", "-i", "--interactive"]:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    parser = argparse.ArgumentParser(
        description="👑 H3MLX Universal CLI - Livello 1 Ufficiale Compatibile (Metal 4 NAX + GPU Sampler)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. IO & Prompt
    parser.add_argument("-p", "--prompt", type=str,
                        default="Shot on Arri Alexa LF with Cooke Anamorphic S4i Prime 50mm T2.3 lens, MTF optical sub-pixel phase coherence, John Wick in crisp tailored black wool suit with white shirt and black tie facing 3/4 frontally with razor-sharp Keanu Reeves likeness executing a rapid tactical Gun-Fu double-tap in torrential night rain, brilliant golden muzzle flash illuminating facial skin pores, brass shell casing ejecting in mid-air, 4k 24fps master",
                        help="Prompt testuale per la generazione video")
    parser.add_argument("-o", "--output", type=str, default="outputs/h3mlx_livello1_output.mp4",
                        help="Percorso del file MP4 di output")
    parser.add_argument("-d", "--model-dir", type=str, default="",
                        help="Directory personalizzata per i pesi di MiniMax H3 (auto-rilevata se vuota)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Avvia lo Studio TUI interattivo")
    
    # 2. Dimensions & Duration
    parser.add_argument("--seconds", type=float, default=3.75,
                        help="Durata del video in secondi (a 24 fps)")
    parser.add_argument("--width", type=int, default=768,
                        help="Larghezza in pixel")
    parser.add_argument("--height", type=int, default=512,
                        help="Altezza in pixel")
    parser.add_argument("--frames", type=int, default=0,
                        help="Numero esatto di frame (calcolato automaticamente da --seconds se 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed del generatore casuale")
    
    # 3. Optional Diagnostics & Baseline Switch
    parser.add_argument("--canonical", action="store_true",
                        help="Esegui in modalità canonica pura antirez BF16 per confronto")
    parser.add_argument("--no-profile", dest="profile", action="store_false", default=True,
                        help="Disabilita il profiling dettagliato delle fasi Metal")
    
    args = parser.parse_args()
    
    if args.interactive:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    # Frame calculation
    if args.frames > 0:
        total_frames = args.frames
    else:
        total_frames = calculate_canonical_frames(args.seconds, args.width, args.height)

    mode = "canonical" if args.canonical else "boosted"

    print("=" * 72)
    print("👑 H3MLX ENGINE · LIVELLO 1: NAX + GPU SAMPLER (SCELTA UFFICIALE)")
    print("=" * 72)
    print(f"📐 Canvas:        {args.width}x{args.height} | Frames: {total_frames} ({total_frames/24:.2f}s @ 24fps)")
    print(f"⚡ Architettura:  Metal 4 NAX={'ON' if mode=='boosted' else 'OFF'} | GPU Sampler={'ON' if mode=='boosted' else 'OFF'}")
    print(f"🛡️ Qualità:       50 Layer Densi Completi (100% Densità Spaziale, Pure Sampling)")
    print(f"⏱️ Stima Tempo:   ~82.7s su Apple Silicon M5 Max (Throughput: 1.10 FPS)")
    print(f"💾 File Output:   {args.output}")
    print(f"📝 Prompt:        \"{args.prompt[:75]}...\"")
    print("=" * 72 + "\n")

    res = execute_h3_generation(
        prompt=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        frames=total_frames,
        steps=14,
        seed=args.seed,
        engine_mode=mode,
        solver="euler",
        reuse=1,
        layers=50,
        token_reduction=False,  # Level 1: 100% dense spatial sampling
        int8=(mode == "boosted"),
        model_dir=args.model_dir if args.model_dir else None,
        profile=args.profile
    )

    if res.success:
        print("\n" + "=" * 72)
        print(f"✅ Generazione Livello 1 completata con successo in {res.wall_time_s:.2f}s!")
        print(f"🎥 Video finale salvato in: {res.output_path}")
        if res.profile_data:
            print("📊 Profiling Fasi Metal:")
            for k, v in res.profile_data.items():
                print(f"   • {k:<25}: {v:.2f}s")
        print("=" * 72 + "\n")
    else:
        print(f"\n❌ Errore durante l'esecuzione:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
