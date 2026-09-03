#!/usr/bin/env python3
"""
👑 H3MLX Unified Universal CLI (v2.5 Master Edition)
1:1 Complete & Faithful Drop-in Replacement for Salvatore Sanfilippo (antirez) h3.c CLI
with RobZomb H3MLX Metal 4 NAX Acceleration, All 5 Frontier Levels, INT8 FC2, 3D VAE Zero-Stitch, & 4K Cinema Upscaler.
"""

import os
import sys
import argparse
from pathlib import Path

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames, get_preset
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, resolve_optimal_frames, OPTIMAL_DURATIONS

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["studio", "interactive", "ui", "tui", "-i", "--interactive"]:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    parser = argparse.ArgumentParser(
        description="👑 H3MLX Universal CLI - 1:1 Antirez h3.c Compatible with Metal 4 NAX Acceleration & All 5 Frontiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Antirez Model & IO Options
    parser.add_argument("-d", "--model-dir", type=str, default="", help="Path to MiniMax H3 checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, default="", help="Text generation prompt (defaults to preset prompt if --preset is provided)")
    parser.add_argument("-o", "--output", type=str, default="outputs/h3mlx_output.mp4", help="Output MP4 file path")
    parser.add_argument("--preset", type=str, default="", choices=list(PRESETS.keys()), help="Load a pre-configured video preset")
    parser.add_argument("--frontier", "--level", dest="frontier_level", type=str, default="",
                        choices=["1", "2", "3", "4", "5", "champion"],
                        help="Select Frontier Level: 1 (Isolated NAX + GPU Sampler), 2 (Token Reduction 4:34), 3 (Monolithic 3D VAE), 4 (14-Step PDD), 5/champion (Master 4K + Audio)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive studio director")
    
    # 2. Dimensions & Temporal Grid
    parser.add_argument("--width", type=int, default=768, help="Internal render width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Internal render height in pixels")
    parser.add_argument("--render-width", type=int, default=0, help="Optional lower internal model canvas width")
    parser.add_argument("--render-height", type=int, default=0, help="Optional lower internal model canvas height")
    parser.add_argument("--frames", type=int, default=0, help="Total video frames (e.g. 48 for 2s, 73 for 3s, 90 for 4s)")
    parser.add_argument("--seconds", type=float, default=0.0, help="Duration in seconds (calculated at 24 fps)")
    parser.add_argument("--duration", type=str, default="",
                        help="Standard optimal duration: '3s', '4s', '6s', '8s', '10s', '12s', '15s', '20s', '30s'")
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
    parser.add_argument("--smart-filter", type=str, default="auto",
                        choices=["auto", "portrait", "cinema", "anime", "action", "macro", "clean"],
                        help="Smart Mastering Filter: 'auto' (content-aware), 'portrait' (AMD CAS+Bilateral), 'cinema', 'anime', 'action', 'macro', or 'clean'")
    parser.add_argument("--profile", action="store_true", default=True, help="Print per-phase Metal profiling metrics")
    parser.add_argument("--frames-dir", type=str, default="", help="Directory to dump individual decoded RGB frames (.ppm)")
    parser.add_argument("--nax-st", action="store_true", help="Enable NAX-Spatiotemporal Multimodal Attention for long video")
    parser.add_argument("--nax-chunk", type=int, default=4, help="Frames per local temporal chunk (default: 4)")
    parser.add_argument("--nax-stride", type=int, default=4, help="Keyframe anchor stride in frames (default: 4)")
    
    args = parser.parse_args()
    
    if args.interactive:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    # Handle Frontier Levels
    if args.frontier_level == "1":
        print("🏛️ Attivazione Frontiera Livello 1: Test Isolato Livello 1 (NAX + GPU Sampler)")
        args.engine_mode = "boosted"
        args.layers = 50
        args.steps = 14
        args.reuse = 1
        args.token_reduction = False
        args.int8 = True
        args.solver = "euler"
    elif args.frontier_level == "2":
        print("⚡ Attivazione Frontiera Livello 2: Spatial Token Reduction Adattiva Multi-Scala (4:34)")
        args.engine_mode = "boosted"
        args.layers = 50
        args.token_reduction = True
        args.int8 = True
    elif args.frontier_level == "3":
        print("💎 Attivazione Frontiera Livello 3: Monolithic 3D VAE Zero-Stitch")
        args.engine_mode = "boosted"
    elif args.frontier_level == "4":
        print("🚀 Attivazione Frontiera Livello 4: 14-Step PDD Optimal Trajectory")
        args.engine_mode = "boosted"
        args.steps = 14
        args.reuse = 2
        args.int8 = True
        args.token_reduction = True
    elif args.frontier_level in ["5", "champion"]:
        print("👑 Attivazione Frontiera Livello 5: Champion Master (Cooke Anamorphic S4/i MTF + 4K Broadcast)")
        args.engine_mode = "boosted"
        args.steps = 14
        args.reuse = 2
        args.int8 = True
        args.token_reduction = True
        args.upscale_4k = True

    # Apply Preset if specified
    if args.preset:
        preset_cfg = get_preset(args.preset)
        if preset_cfg:
            print(f"🎬 Caricamento Preset: {preset_cfg['name']} ({preset_cfg['description']})\n")
            args.width = preset_cfg.get("width", args.width)
            args.height = preset_cfg.get("height", args.height)
            args.seconds = preset_cfg.get("seconds", args.seconds)
            args.steps = preset_cfg.get("steps", args.steps)
            args.engine_mode = preset_cfg.get("mode", preset_cfg.get("engine_mode", args.engine_mode))
            args.solver = preset_cfg.get("solver", args.solver)
            args.int8 = preset_cfg.get("int8", args.int8)
            args.token_reduction = preset_cfg.get("token_reduction", args.token_reduction)
            args.upscale_4k = preset_cfg.get("upscale_4k", args.upscale_4k)
            args.layers = preset_cfg.get("layers", args.layers)
            args.reuse = preset_cfg.get("reuse", args.reuse)
            if not args.prompt and "prompt" in preset_cfg:
                args.prompt = preset_cfg["prompt"]

    if not args.prompt:
        args.prompt = "Cinematic close-up portrait of Brad Pitt smiling, natural soft lighting, highly detailed"

    # Resolution override if canonical
    if args.canonical:
        args.engine_mode = "canonical"
    elif args.boosted:
        args.engine_mode = "boosted"

    # Frame calculations using MiniMax-H3 optimal temporal lattice (17k + 5)
    if args.duration:
        total_frames = resolve_optimal_frames(duration=args.duration)
    elif args.frames > 0:
        total_frames = resolve_optimal_frames(frames=args.frames)
    elif args.seconds > 0:
        total_frames = resolve_optimal_frames(seconds=args.seconds)
    else:
        total_frames = 90  # Default ~4s standard

    # Frame calculations using MiniMax-H3 optimal temporal lattice (17k + 5)

    print("=" * 70)
    print(f"🚀 H3MLX Master Pipeline | Engine: {args.engine_mode.upper()} | Steps: {args.steps}")
    print(f"📐 Canvas: {args.width}x{args.height} | Frames: {total_frames} ({total_frames/24:.1f}s @ 24fps) | Seed: {args.seed}")
    print(f"⚡ Accelerations: NAX={'ON' if args.engine_mode != 'canonical' else 'OFF'}, INT8={'ON' if args.int8 else 'OFF'}, TokenReduction={'ON' if args.token_reduction else 'OFF'}")
    print(f"📝 Prompt: \"{args.prompt[:80]}...\"")
    print(f"💾 Output: {args.output}")
    print("=" * 70 + "\n")

    res = execute_h3_generation(
        prompt=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        frames=total_frames,
        steps=args.steps,
        seed=args.seed,
        engine_mode=args.engine_mode,
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
        smart_filter=args.smart_filter,
        model_dir=args.model_dir if args.model_dir else None,
        profile=args.profile,
        nax_st=args.nax_st,
        nax_chunk=args.nax_chunk,
        nax_stride=args.nax_stride
    )

    if res.success:
        wall_time = res.wall_time_s
        fps = total_frames / wall_time if wall_time > 0 else 0
        raw_mb = Path(res.raw_output_path).stat().st_size / (1024 * 1024) if Path(res.raw_output_path).exists() else 0.0
        print("\n" + "=" * 70)
        print("🎉 GENERAZIONE ALTA FEDELTÀ COMPLETATA CON SUCCESSO!")
        print(f"⏱️  Tempo Totale Reale:       {wall_time:.2f}s  (Throughput: {fps:.2f} FPS)")
        print(f"🎬  Video RAW (Nativo {args.width}x{args.height}): {res.raw_output_path} ({raw_mb:.2f} MB)")
        if res.master_output_path:
            master_mb = Path(res.master_output_path).stat().st_size / (1024 * 1024) if Path(res.master_output_path).exists() else 0.0
            print(f"💎  Video MASTER (Smart 4K):   {res.master_output_path} ({master_mb:.2f} MB)")
        print(f"📐  Risoluzione & Frame:      {args.width}x{args.height} {'-> 4K UHD' if args.upscale_4k else ''} | {total_frames} frames ({total_frames/24:.2f}s @ 24fps)")
        if res.profile_data:
            print("\n📊 Profiling GPU Metal & Smart Mastering:")
            for k, v in res.profile_data.items():
                print(f"   • {k:<25}: {v:.2f}s")
        print("=" * 70 + "\n")
    else:
        print(f"\n❌ Errore durante l'esecuzione di H3:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
