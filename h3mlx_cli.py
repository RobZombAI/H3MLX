#!/usr/bin/env python3
"""
👑 H3MLX Unified Universal CLI (v2.5 Master Edition)
1:1 Complete & Faithful Drop-in Replacement for Salvatore Sanfilippo (antirez) h3.c CLI
with RobZomb H3MLX Metal 4 NAX Acceleration, All 5 Frontier Levels, INT8 FC2, 3D VAE Zero-Stitch, & 4K Cinema Upscaler.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames, get_preset
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, resolve_optimal_frames, OPTIMAL_DURATIONS
import re

def derive_static_anchor_prompt(prompt: str) -> str:
    """
    Transforms a dynamic action prompt into a stable, high-fidelity starting pose
    for Stage 1 anchor synthesis. Replaces rotational or kinetic motion phrases
    with poised posture cues and reinforces anatomical hand and facial definitions.
    """
    anchor = prompt
    replacements = [
        (r"executing a sharp acrobatic spin and landing smoothly", "standing in a poised athletic ready stance"),
        (r"executing a sharp acrobatic spin", "standing in a poised athletic stance"),
        (r"performing an explosive spinning breakdance flare on the floor", "standing in a poised athletic ready pose"),
        (r"performing an explosive breakdance", "in a stylish athletic stance"),
        (r"acrobatic spin", "athletic pose"),
        (r"drifting through a rain-soaked", "parked gleaming in a rain-soaked"),
        (r"banking aggressively into a sharp curve", "positioned dynamically on"),
        (r"running at full sprint", "standing heroically"),
    ]
    for pattern, repl in replacements:
        anchor = re.sub(pattern, repl, anchor, flags=re.IGNORECASE)

    # Ensure anatomical constraints on human subjects
    human_keywords = ["dancer", "person", "man", "woman", "portrait", "tyler", "brad", "girl", "boy"]
    if any(k in prompt.lower() for k in human_keywords):
        if "hands" not in anchor.lower():
            anchor += ", crisp anatomically correct hands with distinct articulated fingers"
        if "facial" not in anchor.lower() and "face" not in anchor.lower():
            anchor += ", clear symmetrical facial features, looking towards camera"
        if "photorealistic" not in anchor.lower():
            anchor += ", 8k photorealistic RAW portrait"
    return anchor

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["studio", "interactive", "ui", "tui", "-i", "--interactive"]:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    parser = argparse.ArgumentParser(
        description="👑 H3MLX Universal CLI - 1:1 Antirez h3.c Compatible with Metal 4 NAX Acceleration & All 11 Frontiers (v3.4 2026 SOTA Edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Antirez Model & IO Options
    parser.add_argument("-d", "--model-dir", type=str, default="", help="Path to MiniMax H3 checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, default="", help="Text generation prompt (defaults to preset prompt if --preset is provided)")
    parser.add_argument("-o", "--output", type=str, default="outputs/h3mlx_output.mp4", help="Output MP4 file path")
    parser.add_argument("--preset", type=str, default="", choices=list(PRESETS.keys()), help="Load a pre-configured video preset")
    parser.add_argument("--list-presets", action="store_true", help="List all available studio video presets and exit")
    parser.add_argument("--frontier", "--level", dest="frontier_level", type=str, default="",
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "champion", "ultra", "sfmc", "fast-master"],
                        help="Select Frontier Level: 1-5, 6 (Temporal-FreqFlow), 7 (Cinema Optics), 8 (TFM Momentum), 9 (C1 Hann Tile Rectification), 10 (Chebyshev Curvature Warp), 11 (Spectral Multi-Physics), 12 (S-FMC 5-Step Symplectic Flow)")
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
    parser.add_argument("--no-activity-mask", action="store_true", help="Disable spatial-temporal activity matrix calculation (ensures full token compute across all regions)")
    parser.add_argument("--ssd-streaming", action="store_true", help="Keep only 2 DiT blocks in memory and stream from SSD")
    
    # 4. Conditioning & Multimodal References
    parser.add_argument("--first-frame", "--i2v", dest="first_frame", type=str, default="", help="First conditioning frame (Image-to-Video)")
    parser.add_argument("--auto-anchor", action="store_true", help="Two-Stage Pipeline: automatically synthesize and anchor the critical initial frame with dedicated spatial attention before temporal video rollout")
    parser.add_argument("--anchor-steps", type=int, default=8, help="Dedicated steps for the initial anchor frame synthesis (default: 8)")
    parser.add_argument("--anchor-prompt", type=str, default="", help="Optional dedicated prompt refinement for the anchor frame synthesis")
    parser.add_argument("--last-frame", type=str, default="", help="Last conditioning frame (Interpolation)")
    parser.add_argument("--ref-image", type=str, default="", help="Reference image conditioning")
    parser.add_argument("--ref-video", type=str, default="", help="Reference video conditioning (including embedded audio)")
    parser.add_argument("--ref-silent-video", type=str, default="", help="Reference video conditioning without its audio track")
    parser.add_argument("--ref-video-audio", nargs=2, metavar=("VIDEO", "AUDIO"), help="Reference video plus independent audio soundtrack")
    parser.add_argument("--ref-audio", type=str, default="", help="Standalone ordered reference audio clip conditioning")
    parser.add_argument("--export-audio", action="store_true", help="Extract and export lossless standalone audio stream (.aac) alongside video")
    
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
                        choices=["auto", "portrait", "cinema", "anime", "action", "macro", "clean", "master-optics", "optics", "frontier-c1"],
                        help="Smart Mastering Filter: 'auto', 'portrait', 'cinema', 'anime', 'action', 'macro', 'clean', 'master-optics', or 'frontier-c1'")
    parser.add_argument("--fps", type=int, default=24, help="Target output video framerate cadence: 24 (cinematic 35mm) or 48 (HFR motion smooth)")
    parser.add_argument("--profile", action="store_true", default=True, help="Print per-phase Metal profiling metrics")
    parser.add_argument("--frames-dir", type=str, default="", help="Directory to dump individual decoded RGB frames (.ppm)")
    parser.add_argument("--nax-st", action="store_true", help="Enable NAX-Spatiotemporal Multimodal Attention for long video")
    parser.add_argument("--nax-chunk", type=int, default=4, help="Frames per local temporal chunk (default: 4)")
    parser.add_argument("--nax-stride", type=int, default=4, help="Keyframe anchor stride in frames (default: 4)")
    parser.add_argument("--bandpass-limiter", "--spark-preserve", dest="bandpass_limiter", action="store_true",
                        help="Bandpass Spectral Limiter: preserves isolated impulsive high-frequency particles (sparks, droplets, glints)")
    
    # 6. Physical ControlNet, Timeline & Forensic Pipeline
    parser.add_argument("--control-pose", type=str, default="",
                        help="Physical OpenPose / biomechanical skeletal guide video for precise limb & trajectory control")
    parser.add_argument("--timeline", type=str, default="",
                        help="Path to JSON storyboard timeline configuration for multi-beat narrative sequences")
    parser.add_argument("--refine-subjects", "--forensic-refine", dest="refine_subjects", action="store_true",
                        help="Run post-generation forensic micro-subject detailer on miniature characters, hands, and typography")
    
    args = parser.parse_args()
    
    if args.interactive:
        import h3mlx_studio
        h3mlx_studio.main()
        return

    if args.list_presets:
        print("\n" + "=" * 78)
        print("👑 H3MLX CANONICAL STUDIO PRESETS (Frontier 12 & Full Hardware Suite)")
        print("=" * 78)
        for pid, cfg in PRESETS.items():
            if pid in ["h3mlx_champion_4s", "h3mlx_livello1", "h3mlx_cinema_4k_master", "h3mlx_ghibli_watercolor_4s"]:
                continue
            name = cfg.get("name", pid)
            w, h = cfg.get("width", 768), cfg.get("height", 512)
            steps = cfg.get("steps", 8)
            frames = cfg.get("frames", 90)
            frontier = cfg.get("frontier", "12")
            desc = cfg.get("description", "")
            print(f" • {pid:<22} | {name:<26} | {w}x{h} ({frames}f) | {steps}st (F{frontier})")
            print(f"   Desc: {desc}")
            print(f"   Prompt: \"{cfg.get('prompt', '')[:70]}...\"\n")
        print("Usage: python3 h3mlx_cli.py --preset <preset_name> [-o output.mp4]")
        print("=" * 78 + "\n")
        return

    # Timeline Storyboard Mode
    if args.timeline:
        from h3_timeline_director import execute_timeline_storyboard
        import json
        timeline_path = Path(args.timeline).resolve()
        if not timeline_path.exists():
            print(f"❌ Error: Timeline configuration not found: {timeline_path}", file=sys.stderr)
            sys.exit(1)
        with open(timeline_path, "r") as f:
            timeline_cfg = json.load(f)
        
        timeline_res = execute_timeline_storyboard(
            timeline_config=timeline_cfg,
            output_path=args.output,
            width=args.width,
            height=args.height,
            base_seed=args.seed,
            smart_filter=args.smart_filter if args.smart_filter != "auto" else "macro",
            no_activity_mask=args.no_activity_mask
        )
        if not timeline_res.get("success", False):
            sys.exit(1)
            
        if args.refine_subjects:
            from h3_subject_detailer import refine_video_subjects
            refined_path = Path(args.output).parent / f"{Path(args.output).stem}_refined.mp4"
            print("🔬 Running Micro-Subject Forensic Detailer on timeline output...")
            refine_video_subjects(args.output, str(refined_path))
            print(f"✨ Forensic Refined Output Ready: {refined_path}")
        return

    # Apply Preset first if specified
    if args.preset:
        preset_cfg = get_preset(args.preset)
        if preset_cfg:
            print(f"🎬 Loading Preset: {preset_cfg['name']} ({preset_cfg['description']})\n")
            if "--width" not in sys.argv:
                args.width = preset_cfg.get("width", args.width)
            if "--height" not in sys.argv:
                args.height = preset_cfg.get("height", args.height)
            if "--seconds" not in sys.argv and "--duration" not in sys.argv and "--frames" not in sys.argv:
                args.seconds = preset_cfg.get("seconds", args.seconds)
            if "--steps" not in sys.argv and "-s" not in sys.argv:
                args.steps = preset_cfg.get("steps", args.steps)
            args.engine_mode = preset_cfg.get("mode", preset_cfg.get("engine_mode", args.engine_mode))
            args.solver = preset_cfg.get("solver", args.solver)
            args.int8 = preset_cfg.get("int8", args.int8)
            args.token_reduction = preset_cfg.get("token_reduction", args.token_reduction)
            args.upscale_4k = preset_cfg.get("upscale_4k", args.upscale_4k)
            args.layers = preset_cfg.get("layers", args.layers)
            args.reuse = preset_cfg.get("reuse", args.reuse)
            if "seed" in preset_cfg and args.seed == 42:
                args.seed = preset_cfg["seed"]
            if "frames" in preset_cfg and args.frames == 0 and not args.duration and args.seconds == 0.0:
                args.frames = preset_cfg["frames"]
            if "auto_anchor" in preset_cfg:
                args.auto_anchor = preset_cfg["auto_anchor"]
            if "anchor_steps" in preset_cfg:
                args.anchor_steps = preset_cfg["anchor_steps"]
            if "smart_filter" in preset_cfg and args.smart_filter == "auto":
                args.smart_filter = preset_cfg["smart_filter"]
            if not args.frontier_level and "frontier" in preset_cfg:
                args.frontier_level = str(preset_cfg["frontier"])
            if not args.prompt and "prompt" in preset_cfg:
                args.prompt = preset_cfg["prompt"]

    # Handle Frontier Levels
    if args.frontier_level == "1":
        print("🏛️ Activating Frontier Level 1: Isolated Metal 4 NAX + GPU Sampler")
        args.engine_mode = "boosted"
        args.layers = 50
        args.steps = 14
        args.reuse = 1
        args.token_reduction = False
        args.int8 = True
        args.solver = "euler"
    elif args.frontier_level == "2":
        print("⚡ Activating Frontier Level 2: Adaptive Multi-Scale Spatial Token Reduction (4:34)")
        args.engine_mode = "boosted"
        args.layers = 50
        args.token_reduction = True
        args.int8 = True
    elif args.frontier_level == "3":
        print("💎 Activating Frontier Level 3: Monolithic 3D VAE Zero-Stitch")
        args.engine_mode = "boosted"
    elif args.frontier_level == "4":
        print("🚀 Activating Frontier Level 4: 14-Step PDD Optimal Trajectory")
        args.engine_mode = "boosted"
        args.steps = 14
        args.reuse = 2
        args.int8 = True
        args.token_reduction = True
    elif args.frontier_level in ["5", "champion"]:
        print("👑 Activating Frontier Level 5: Champion Master (Cooke Anamorphic S4/i MTF + 4K Broadcast)")
        args.engine_mode = "boosted"
        args.steps = 14
        args.reuse = 2
        args.int8 = True
        args.token_reduction = True
        args.upscale_4k = True
    elif args.frontier_level == "6":
        print("🌊 Activating Frontier Level 6: FreqFlow Dynamic High-Frequency Spectral Velocity Boost")
        args.engine_mode = "boosted"
    elif args.frontier_level == "7":
        print("🔭 Activating Frontier Level 7: Super-Nyquist Pre-VAE Phase Alignment & Kodak Vision3 Optics")
        args.engine_mode = "boosted"
        args.upscale_4k = True
    elif args.frontier_level == "8":
        print("🏃 Activating Frontier Level 8: Temporal Block-Tridiagonal Momentum Regularization (TFM)")
        args.engine_mode = "boosted"
    elif args.frontier_level == "9":
        print("📐 Activating Frontier Level 9: Raised-Cosine C1 Latent Manifold Rectification")
        args.engine_mode = "boosted"
        args.smart_filter = "master-optics"
    elif args.frontier_level == "10":
        print("📈 Activating Frontier Level 10: Curvature-Adaptive Chebyshev Time-Warping (CACFM)")
        args.engine_mode = "boosted"
        args.reuse = 1
    elif args.frontier_level in ["11", "ultra"]:
        print("✨ Activating Frontier Level 11: Complete 2026 SOTA Multi-Physics Stack (Chebyshev + TFM + Hann C1 + 4K)")
        args.engine_mode = "boosted"
        args.upscale_4k = True
        args.smart_filter = "master-optics"
    elif args.frontier_level in ["12", "sfmc", "fast-master"]:
        print("⚡ Activating Frontier Level 12: S-FMC (Symplectic Flow Curvature + Radau-Chebyshev Anchoring)")
        args.engine_mode = "boosted"
        args.upscale_4k = True
        args.smart_filter = "master-optics"
        if "--steps" not in sys.argv and "-s" not in sys.argv and not args.preset:
            args.steps = 5

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
    # Two-Stage Auto-Anchor Workflow
    if args.auto_anchor and not args.first_frame:
        print("=" * 70)
        print("🎯 TWO-STAGE AUTO-ANCHOR ACTIVATED")
        print("Stage 1: Synthesizing critical anchor frame with dedicated spatial fidelity...")
        print("=" * 70)

        anchor_dir = Path(args.output).parent / "anchors"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.output).stem
        temp_anchor_video = str(anchor_dir / f"{stem}_anchor_raw.mp4")
        anchor_frame_path = str(anchor_dir / f"{stem}_anchor_frame.jpg")

        prompt_stage1 = args.anchor_prompt if args.anchor_prompt else derive_static_anchor_prompt(args.prompt)
        print(f"📌 Stage 1 Anchor Prompt: \"{prompt_stage1[:90]}...\"\n")

        anchor_res = execute_h3_generation(
            prompt=prompt_stage1,
            output_path=temp_anchor_video,
            width=args.width,
            height=args.height,
            frames=22,
            steps=args.anchor_steps,
            seed=args.seed,
            engine_mode=args.engine_mode,
            solver=args.solver,
            reuse=1,
            layers=args.layers,
            token_reduction=False,
            int8=args.int8,
            upscale_4k=False,
            model_dir=args.model_dir if args.model_dir else None,
            profile=False,
            frontier=args.frontier_level
        )

        if anchor_res.success and Path(temp_anchor_video).exists():
            cmd_extract = [
                "ffmpeg", "-y", "-i", temp_anchor_video,
                "-vf", "select=eq(n\\,0)", "-vsync", "0",
                anchor_frame_path
            ]
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"✅ Stage 1 Complete: Anchor frame synthesized & locked:\n   {anchor_frame_path}")
            print("\nStage 2: Rolling out temporal motion conditioned on pristine anchor...")
            print("=" * 70 + "\n")
            args.first_frame = anchor_frame_path
        else:
            print("⚠️ Stage 1 Anchor generation encountered an issue, falling back to direct rollout.")

    activity_bin_path = None
    if args.no_activity_mask or os.environ.get("H3_DISABLE_ACTIVITY_GATE") == "1":
        if "H3_ACTIVITY_MASK" in os.environ:
            del os.environ["H3_ACTIVITY_MASK"]
        os.environ["H3_DISABLE_ACTIVITY_GATE"] = "1"
        print("⚡ Matrice di Attività Disattivata: calcolo 100% full-token abilitato su tutto il canvas.")
    elif args.first_frame and Path(args.first_frame).exists():
        try:
            from h3_spatial_matrix import compute_activity_matrix
            matrix_dir = Path(args.output).parent / "matrix"
            matrix_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.output).stem
            act_bin = str(matrix_dir / f"{stem}_activity.bin")
            mat_res = compute_activity_matrix(
                args.first_frame,
                target_width=args.width,
                target_height=args.height,
                output_bin_path=act_bin
            )
            activity_bin_path = act_bin
            os.environ["H3_ACTIVITY_MASK"] = act_bin
            print("🧠 Matrice di Attività Spazio-Temporale Calcolata:")
            print(f"   • Zone Statiche / Già Fatte: {mat_res['static_coverage_pct']:.1f}% (Calcolo VAE & DiT congelato)")
            print(f"   • Soggetto Dinamico Primario: {mat_res['active_coverage_pct']:.1f}% (Priorità di calcolo 100%)")
            print(f"   • Preview Mappa:             {mat_res['preview_png']}\n")
        except Exception as e:
            print(f"⚠️ Nota: Impossibile generare la matrice di attività ({e}), procedo con rollout standard.")

    if args.control_pose:
        if not args.ref_video:
            args.ref_video = args.control_pose
        print(f"🦴 Physical ControlNet Pose Guidance Active: {args.control_pose}")

    extra_env = {"H3_ACTIVITY_MASK": activity_bin_path} if activity_bin_path else {"H3_DISABLE_ACTIVITY_GATE": "1"}
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
        core_reuse=args.core_reuse,
        token_reduction=args.token_reduction,
        int8=args.int8,
        first_frame=args.first_frame if args.first_frame else None,
        last_frame=args.last_frame if args.last_frame else None,
        ref_image=args.ref_image if args.ref_image else None,
        ref_video=args.ref_video if args.ref_video else None,
        ref_silent_video=args.ref_silent_video if args.ref_silent_video else None,
        ref_video_audio=tuple(args.ref_video_audio) if args.ref_video_audio else None,
        ref_audio=args.ref_audio if args.ref_audio else None,
        export_audio=args.export_audio,
        ssd_streaming=args.ssd_streaming,
        upscale_4k=args.upscale_4k,
        smart_filter=args.smart_filter,
        model_dir=args.model_dir if args.model_dir else None,
        profile=args.profile,
        nax_st=args.nax_st,
        nax_chunk=args.nax_chunk,
        nax_stride=args.nax_stride,
        frontier=args.frontier_level,
        fps=args.fps,
        bandpass_limiter=args.bandpass_limiter,
        extra_env=extra_env
    )

    if res.success:
        wall_time = res.wall_time_s
        fps = total_frames / wall_time if wall_time > 0 else 0
        raw_mb = Path(res.raw_output_path).stat().st_size / (1024 * 1024) if Path(res.raw_output_path).exists() else 0.0
        print("\n" + "=" * 70)
        print("GENERATION COMPLETED SUCCESSFULLY")
        print(f"  • Total Wall Time:  {wall_time:.2f}s  (Throughput: {fps:.2f} FPS)")
        print(f"  • Video RAW:        {res.raw_output_path} ({raw_mb:.2f} MB)")
        if res.audio_output_path:
            audio_mb = Path(res.audio_output_path).stat().st_size / (1024 * 1024) if Path(res.audio_output_path).exists() else 0.0
            print(f"  • Audio NATIVE:     {res.audio_output_path} ({audio_mb:.2f} MB)")
        if res.master_output_path:
            master_mb = Path(res.master_output_path).stat().st_size / (1024 * 1024) if Path(res.master_output_path).exists() else 0.0
            print(f"  • Video MASTER 4K:  {res.master_output_path} ({master_mb:.2f} MB)")
        print(f"  • Resolution:       {args.width}x{args.height} {'-> 4K UHD' if args.upscale_4k else ''} | {total_frames} frames (~{total_frames/24:.2f}s @ 24fps)")
        if res.profile_data:
            print("\nMetal GPU Profiling:")
            for k, v in res.profile_data.items():
                if isinstance(v, (int, float)):
                    print(f"   • {k:<25}: {v:.2f}s")
                else:
                    print(f"   • {k:<25}: {v}")
        print("=" * 70 + "\n")

        if args.refine_subjects and res.raw_output_path and Path(res.raw_output_path).exists():
            from h3_subject_detailer import refine_video_subjects
            refined_output = str(Path(res.raw_output_path).parent / f"{Path(res.raw_output_path).stem}_refined.mp4")
            print("🔬 Running Micro-Subject Forensic Detailer on output...")
            refine_video_subjects(res.raw_output_path, refined_output)
            print(f"✨ Forensic Refined Output Ready: {refined_output}\n")
    else:
        print(f"\nError during H3 execution:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
