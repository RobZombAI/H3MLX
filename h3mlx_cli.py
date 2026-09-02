#!/usr/bin/env python3
"""
👑 H3MLX Unified Universal CLI (v2.5 Master Edition)
1:1 Complete & Faithful Drop-in Replacement for Salvatore Sanfilippo (antirez) h3.c CLI
with RobZomb H3MLX Metal 4 NAX Acceleration, INT8 FC2, 3D VAE Zero-Stitch, & 4K Cinema Upscaler.
"""

import os
import sys
import argparse
from pathlib import Path

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames, get_preset
from h3mlx_engine_core import execute_h3_generation, resolve_model_path

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["studio", "interactive", "ui", "tui", "-i", "--interactive"]:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    parser = argparse.ArgumentParser(
        description="👑 H3MLX Universal CLI - 1:1 Antirez h3.c Compatible with Metal 4 NAX Acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Antirez Model & IO Options
    parser.add_argument("-d", "--model-dir", type=str, default="", help="Path to MiniMax H3 checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, default="A graceful flamenco dancer in red dress spinning energetically, studio lighting, highly detailed", help="Text generation prompt")
    parser.add_argument("-o", "--output", type=str, default="outputs/h3mlx_output.mp4", help="Output MP4 file path")
    parser.add_argument("--preset", type=str, default="", choices=list(PRESETS.keys()), help="Load a pre-configured video preset")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive studio director")
    
    # 2. Dimensions & Temporal Grid
    parser.add_argument("--width", type=int, default=768, help="Internal render width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Internal render height in pixels")
    parser.add_argument("--render-width", type=int, default=0, help="Optional lower internal model canvas width")
    parser.add_argument("--render-height", type=int, default=0, help="Optional lower internal model canvas height")
    parser.add_argument("--frames", type=int, default=0, help="Total video frames (e.g. 48 for 2s, 73 for 3s, 90 for 4s)")
    parser.add_argument("--seconds", type=float, default=3.0, help="Duration in seconds (calculated at 24 fps)")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed")
    
    # 3. Denoising & DiT Controls
    parser.add_argument("--steps", type=int, default=8, help="Denoising steps (8 for PDD, 14-20 for full fidelity)")
    parser.add_argument("--reuse", type=int, default=1, help="Denoiser step reuse (1 = exact, 2 = fast, 4 = aggressive)")
    parser.add_argument("--layers", type=int, default=50, help="Number of DiT residual blocks to retain (35-50)")
    parser.add_argument("--core-reuse", type=int, default=1, help="Recompute transformer core every N steps")
    parser.add_argument("--token-reduction", action="store_true", help="Enable adaptive spatial token reduction in middle blocks 4-34")
    parser.add_argument("--ssd-streaming", action="store_true", help="Keep only 2 DiT blocks in memory and stream from SSD")
    
    # 4. Conditioning & Multimodal References
    parser.add_argument("--first-frame", "--i2v", dest="first_frame", type=str, default="", help="First conditioning frame (Image-to-Video)")
    parser.add_argument("--last-frame", type=str, default="", help="Last conditioning frame (Interpolation)")
    parser.add_argument("--ref-image", type=str, default="", help="Reference image conditioning")
    parser.add_argument("--ref-video", type=str, default="", help="Reference video conditioning")
    parser.add_argument("--ref-audio", type=str, default="", help="Reference audio conditioning")
    parser.add_argument("--speech-audio", type=str, default="", help="Overlay speech/dialogue audio track natively")
    
    # 5. Engine Selection & Accelerations
    parser.add_argument("--engine", "--mode", dest="engine_mode", type=str, default="h3mlx",
                        choices=["h3mlx", "boosted", "canonical", "antirez", "pure"],
                        help="Inference engine: 'h3mlx' (Metal 4 NAX + INT8 + 3D VAE) or 'canonical' (antirez pure baseline)")
    parser.add_argument("--canonical", action="store_true", help="Enforce pure antirez baseline execution")
    parser.add_argument("--boosted", action="store_true", help="Enforce H3MLX boosted execution")
    parser.add_argument("--int8", action="store_true", default=True, help="Enable Metal 4 NAX Row-Major INT8 dynamic FC2 quantization")
    parser.add_argument("--no-int8", dest="int8", action="store_false", help="Disable INT8 and run in BF16")
    parser.add_argument("--solver", type=str, default="auto", choices=["auto", "dpm3m", "ab3", "euler"],
                        help="ODE flow solver: 'dpm3m' (3rd order), 'ab3', or 'euler'")
    parser.add_argument("--4k", "--upscale", dest="upscale_4k", action="store_true", help="Upscale generated output to 4K UHD Master (3840x2160)")
    parser.add_argument("--profile", action="store_true", default=True, help="Print per-phase Metal profiling metrics")
    parser.add_argument("--frames-dir", type=str, default="", help="Directory to dump individual decoded RGB frames (.ppm)")
    
    args = parser.parse_args()
    
    if args.interactive:
        import h3mlx_studio
        h3mlx_studio.main()
        return
        
    # Apply Preset if specified
    if args.preset:
        p_cfg = get_preset(args.preset)
        print(f"🎬 Caricamento Preset: {p_cfg['name']} ({p_cfg['description']})")
        args.width = p_cfg.get("width", args.width)
        args.height = p_cfg.get("height", args.height)
        args.seconds = p_cfg.get("seconds", args.seconds)
        args.frames = p_cfg.get("frames", args.frames)
        args.steps = p_cfg.get("steps", args.steps)
        args.reuse = p_cfg.get("reuse", args.reuse)
        args.layers = p_cfg.get("layers", args.layers)
        args.prompt = p_cfg.get("prompt", args.prompt)
        if "token_reduction" in p_cfg:
            args.token_reduction = p_cfg["token_reduction"]
        if "int8" in p_cfg:
            args.int8 = p_cfg["int8"]
        if "solver" in p_cfg:
            args.solver = p_cfg["solver"]
        if "mode" in p_cfg:
            args.engine_mode = p_cfg["mode"]
        if p_cfg.get("upscale_4k"):
            args.upscale_4k = True
            
    # Resolve Canonical vs Boosted flags
    if args.canonical:
        engine_mode = "canonical"
    elif args.boosted or args.engine_mode in ["h3mlx", "boosted"]:
        engine_mode = "boosted"
    else:
        engine_mode = args.engine_mode
        
    # Resolve Frame Count
    frames = args.frames if args.frames > 0 else calculate_canonical_frames(args.seconds)
    
    print("\n" + "="*70)
    print(f"🚀 H3MLX Master Pipeline | Engine: {engine_mode.upper()} | Model: {args.steps} Steps")
    print(f"📐 Canvas: {args.width}x{args.height} | Frames: {frames} ({args.seconds}s @ 24fps) | Seed: {args.seed}")
    print(f"⚡ Accelerations: NAX={'ON' if engine_mode=='boosted' else 'OFF'}, INT8={'ON' if (engine_mode=='boosted' and args.int8) else 'OFF'}, TokenReduction={'ON' if args.token_reduction else 'OFF'}")
    print(f"📝 Prompt: \"{args.prompt}\"")
    print(f"💾 Output: {args.output}")
    print("="*70 + "\n")
    
    res = execute_h3_generation(
        prompt=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        frames=frames,
        steps=args.steps,
        seed=args.seed,
        engine_mode=engine_mode,
        solver=args.solver,
        reuse=args.reuse,
        layers=args.layers,
        token_reduction=args.token_reduction,
        int8=args.int8,
        first_frame=args.first_frame if args.first_frame else None,
        last_frame=args.last_frame if args.last_frame else None,
        ref_image=args.ref_image if args.ref_image else None,
        ref_video=args.ref_video if args.ref_video else None,
        ref_audio=args.ref_audio if args.ref_audio else None,
        speech_audio=args.speech_audio if args.speech_audio else None,
        ssd_streaming=args.ssd_streaming,
        upscale_4k=args.upscale_4k,
        model_dir=args.model_dir if args.model_dir else None,
        profile=args.profile
    )
    
    if res.success:
        print("\n" + "="*70)
        print(f"✅ Generazione completata con successo in {res.wall_time_s:.2f}s!")
        print(f"🎥 Video finale salvato in: {res.output_path}")
        if res.profile_data:
            print("📊 Profiling Fasi:")
            for phase, duration in res.profile_data.items():
                print(f"   • {phase:25s}: {duration:.2f}s")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print(f"\n❌ Errore durante la generazione:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
