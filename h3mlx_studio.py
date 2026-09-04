#!/usr/bin/env python3
"""
H3MLX Studio & Interactive Generation Interface
Interactive CLI for MiniMax H3 inference on Apple Silicon.
Supports Text-to-Video (T2V) and Image-to-Video (I2V) workflows.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, BASE_DIR

# Terminal Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_WHITE = "\033[97m"

STUDIO_PRESETS = [
    {
        "id": "h3mlx_champion_gold",
        "title": "Champion Master (3:2)",
        "resolution": "768x512 -> 4K UHD Master (3072x2048)",
        "width": 768,
        "height": 512,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "~40s (M5 Max)",
        "description": "Standard 3:2 canvas (768x512). 50 dense DiT layers with DPM++ 2M flow matching.",
        "default_prompt": "Cinematic portrait of a person in warm evening light, sharp focus, natural skin texture"
    },
    {
        "id": "h3mlx_cinema_16x9",
        "title": "Cinema Widescreen (16:9)",
        "resolution": "960x544 -> 4K Widescreen Master (3840x2176)",
        "width": 960,
        "height": 544,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "~50s (M5 Max)",
        "description": "Cinematic 16:9 widescreen canvas (960x544). Optimal for landscapes, action, and cityscapes.",
        "default_prompt": "Cinematic wide shot of a futuristic neon city street at dusk with wet asphalt reflections"
    },
    {
        "id": "h3mlx_macro_square",
        "title": "Square Canvas (1:1)",
        "resolution": "640x640 -> 2.5K Master (2560x2560)",
        "width": 640,
        "height": 640,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "~45s (M5 Max)",
        "description": "Balanced 1:1 square canvas (640x640, 1600 latent tokens). High spatial density for macro shots.",
        "default_prompt": "Close-up macro shot of dew drops on a vibrant green leaf, soft blurred background"
    },
    {
        "id": "h3mlx_vertical_reel",
        "title": "Vertical Cinema (9:16)",
        "resolution": "576x1024 -> Vertical Master (2304x4096)",
        "width": 576,
        "height": 1024,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "~52s (M5 Max)",
        "description": "9:16 vertical ratio (576x1024). Optimized for mobile full-screen portraits and fashion sequences.",
        "default_prompt": "Cinematic vertical full-body shot of a fashion model walking down a sunlit city avenue"
    },
    {
        "id": "h3mlx_ghibli_master",
        "title": "Stylized / Anime (3:2)",
        "resolution": "768x512 -> 4K Master (3072x2048)",
        "width": 768,
        "height": 512,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "~40s (M5 Max)",
        "description": "Calibrated for hand-painted, watercolor, and animation aesthetics.",
        "default_prompt": "Lush green rolling hills with giant windmill, fluffy white clouds, vibrant anime landscape"
    }
]

def print_header():
    width = min(shutil.get_terminal_size().columns, 85)
    print("\n" + C_CYAN + "═" * width + C_RESET)
    print(f"{C_BOLD}{C_WHITE}H3MLX STUDIO - Interactive Video Generator{C_RESET}")
    print(f"{C_DIM}Pure C / Metal 4 Engine for MiniMax H3 on Apple Silicon (M1-M5 Max/Ultra){C_RESET}")
    print(C_CYAN + "═" * width + C_RESET)
    
    # Audio status disclaimer
    print(f"\n{C_YELLOW}[!] Notice on Audio Generation:{C_RESET}")
    print(f"{C_DIM}    The Audio VAE pipeline is currently experimental/unstable in local inference.")
    print(f"    Audio is muted by default. Community PRs to stabilize the audio decoder are welcome.{C_RESET}\n")

def interactive_prompt(default_val: str, prompt_text: str) -> str:
    print(f"{C_CYAN}?{C_RESET} {C_BOLD}{prompt_text}{C_RESET} [{C_GREEN}{default_val}{C_RESET}]: ", end="", flush=True)
    try:
        val = input().strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        sys.exit(0)
    return val if val else default_val

def prompt_file_path(prompt_text: str, optional: bool = True) -> Optional[str]:
    while True:
        print(f"{C_CYAN}?{C_RESET} {C_BOLD}{prompt_text}{C_RESET}" + (" (press Enter to skip): " if optional else ": "), end="", flush=True)
        try:
            val = input().strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)
            
        if not val:
            if optional:
                return None
            print(f"{C_RED}File path cannot be empty.{C_RESET}")
            continue
            
        # Clean path quotes if user dragged and dropped file into terminal
        val = val.strip("'\"")
        path = Path(val).expanduser().resolve()
        if path.exists() and path.is_file():
            return str(path)
        else:
            print(f"{C_RED}File not found at: {path}. Please provide a valid file path.{C_RESET}")

def main():
    print_header()
    
    # Verify model weights
    try:
        model_path = resolve_model_path(steps=8)
        print(f"{C_GREEN}✓ Model Weights Detected:{C_RESET} {model_path.name}\n")
    except FileNotFoundError as e:
        print(f"\n{C_YELLOW}⚠️  Model Weights Missing:{C_RESET}\n{e}\n")
        answer = input(f"{C_CYAN}?{C_RESET} Download official MiniMax H3 weights now via download_models.sh? [Y/n]: ").strip().lower()
        if answer in ["", "y", "yes"]:
            import subprocess
            subprocess.run([sys.executable, str(BASE_DIR / "download_models.py")])
        else:
            print(f"{C_RED}Execution aborted: model weights required.{C_RESET}")
            sys.exit(1)

    # 1. Preset Selection
    print(f"{C_BOLD}{C_WHITE}SELECT VIDEO PRESET:{C_RESET}")
    for i, p in enumerate(STUDIO_PRESETS, 1):
        print(f"\n  {C_BOLD}{C_CYAN}[{i}]{C_RESET} {C_BOLD}{p['title']}{C_RESET}")
        print(f"      Resolution : {C_WHITE}{p['resolution']}{C_RESET} | Est. Time: {C_GREEN}{p['est_time_m5']}{C_RESET}")
        print(f"      Description: {C_DIM}{p['description']}{C_RESET}")
        
    print(f"\n  {C_BOLD}{C_CYAN}[0]{C_RESET} {C_DIM}Exit{C_RESET}\n")
    
    while True:
        choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Select preset [1-{len(STUDIO_PRESETS)}, 0 to exit]:{C_RESET} ").strip()
        if choice == "0":
            print("\nExiting H3MLX Studio. Goodbye!\n")
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(STUDIO_PRESETS):
                selected = STUDIO_PRESETS[idx]
                break
        except ValueError:
            pass
        print(f"{C_RED}Invalid selection. Please enter a number between 1 and {len(STUDIO_PRESETS)}.{C_RESET}")

    print(f"\n{C_GREEN}✓ Selected: {selected['title']}{C_RESET}\n")
    
    # 2. Pipeline Mode: Text-to-Video vs Image-to-Video
    print(f"{C_CYAN}?{C_RESET} {C_BOLD}Select Generation Mode:{C_RESET}")
    print(f"  {C_CYAN}[1]{C_RESET} Text-to-Video (T2V) - Generate motion purely from text description")
    print(f"  {C_CYAN}[2]{C_RESET} Image-to-Video (I2V) - Animate a starting conditioning image")
    print(f"  {C_CYAN}[3]{C_RESET} First & Last Frame Interpolation - Morph between two images")
    gen_mode_choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Choose mode [1-3, default: 1]:{C_RESET} ").strip()
    
    first_frame_path = None
    last_frame_path = None
    
    if gen_mode_choice == "2":
        print(f"\n{C_MAGENTA}{C_BOLD}🖼️  IMAGE-TO-VIDEO SETUP:{C_RESET}")
        first_frame_path = prompt_file_path("Enter path to initial image (.jpg / .png / .webp)", optional=False)
        print(f"{C_GREEN}✓ Starting image loaded:{C_RESET} {first_frame_path}\n")
    elif gen_mode_choice == "3":
        print(f"\n{C_MAGENTA}{C_BOLD}🖼️  INTERPOLATION SETUP:{C_RESET}")
        first_frame_path = prompt_file_path("Enter path to first image (.jpg / .png)", optional=False)
        last_frame_path = prompt_file_path("Enter path to target last image (.jpg / .png)", optional=False)
        print(f"{C_GREEN}✓ First image:{C_RESET} {first_frame_path}")
        print(f"{C_GREEN}✓ Last image :{C_RESET} {last_frame_path}\n")

    # 3. Prompt input
    prompt = interactive_prompt(selected["default_prompt"], "Enter text prompt")
    
    # 4. Duration seconds
    sec_str = interactive_prompt(str(selected["default_seconds"]), "Duration in seconds")
    try:
        seconds = float(sec_str)
    except ValueError:
        seconds = selected["default_seconds"]
        
    # 5. Output Path
    timestamp = int(time.time())
    prefix = "i2v" if first_frame_path else "t2v"
    default_out = f"outputs/{selected['id']}_{prefix}_{timestamp}.mp4"
    out_path = interactive_prompt(default_out, "Output video path")
    
    # 6. Quality & Frontier Features
    print(f"\n{C_CYAN}?{C_RESET} {C_BOLD}Optical & Frontier Pipeline Quality Level:{C_RESET}")
    print(f"  {C_CYAN}[1]{C_RESET} Frontier Level 7 (FreqFlow + Pre-VAE Phase Align + Kodak Vision3 5219 Optics) [Recommended]")
    print(f"  {C_CYAN}[2]{C_RESET} Frontier Level 6 (FreqFlow Late-Step Velocity Boost)")
    print(f"  {C_CYAN}[3]{C_RESET} Standard Baseline (Exact Unmodified Flow Matching)")
    q_choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Choose quality level [1-3, default: 1]:{C_RESET} ").strip()
    
    if q_choice == "2":
        frontier_level = "6"
    elif q_choice == "3":
        frontier_level = None
    else:
        frontier_level = "7"
        
    # 7. Mastering & Upscale Profile
    print(f"\n{C_CYAN}?{C_RESET} {C_BOLD}Mastering & Resolution Scaling:{C_RESET}")
    print(f"  {C_CYAN}[1]{C_RESET} 4K UHD Master (VideoToolbox Main 10-bit HEVC + AMD CAS adaptive contrast)")
    print(f"  {C_CYAN}[2]{C_RESET} Native RAW Only (Direct GPU output, no post-processing)")
    m_choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Choose profile [1-2, default: 1]:{C_RESET} ").strip()
    upscale_4k = (m_choice != "2")
    
    # Resolve Canvas Dimensions & Frames
    width = selected.get("width", 768)
    height = selected.get("height", 512)
    frames = calculate_canonical_frames(seconds, width, height)
    steps = selected["default_steps"]
    
    # Summary Card before launch
    print("\n" + C_CYAN + "─" * 70 + C_RESET)
    print(f"{C_BOLD}{C_WHITE}GENERATION CONFIGURATION SUMMARY:{C_RESET}")
    print(f"  • Preset      : {selected['title']}")
    print(f"  • Pipeline    : {'Image-to-Video (I2V)' if first_frame_path else 'Text-to-Video (T2V)'}")
    if first_frame_path:
        print(f"  • Input Image : {first_frame_path}")
    if last_frame_path:
        print(f"  • Target Image: {last_frame_path}")
    print(f"  • Resolution  : {width}x{height} {'-> 4K UHD Master' if upscale_4k else '(Native RAW)'}")
    print(f"  • Frames      : {frames} frames (~{frames/24:.2f}s @ 24fps) | DiT Steps: {steps}")
    print(f"  • Frontier    : Level {frontier_level if frontier_level else 'None (Baseline)'}")
    print(f"  • Audio       : Muted (Audio VAE pipeline currently experimental)")
    print(f"  • Output Path : {out_path}")
    print(f"  • Prompt      : \"{prompt[:80]}...\"")
    print(C_CYAN + "─" * 70 + C_RESET)
    
    confirm = input(f"\n{C_CYAN}?{C_RESET} {C_BOLD}Start generation now? [Y/n]:{C_RESET} ").strip().lower()
    if confirm in ["n", "no"]:
        print(f"{C_YELLOW}Generation cancelled.{C_RESET}\n")
        sys.exit(0)
        
    print(f"\n{C_GREEN}{C_BOLD}Starting H3MLX inference engine...{C_RESET}\n")
    
    t0 = time.perf_counter()
    res = execute_h3_generation(
        prompt=prompt,
        output_path=out_path,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        seed=42,
        engine_mode=selected.get("mode", "boosted"),
        solver=selected.get("solver", "dpm3m"),
        reuse=selected.get("reuse", 1),
        layers=selected.get("layers", 50),
        token_reduction=False,
        int8=selected.get("int8", True),
        upscale_4k=upscale_4k,
        first_frame=first_frame_path,
        last_frame=last_frame_path,
        frontier=frontier_level,
        profile=True
    )
    t1 = time.perf_counter()
    
    if res.success:
        wall_time = res.wall_time_s
        fps = frames / wall_time if wall_time > 0 else 0
        print("\n" + C_GREEN + "═" * 70 + C_RESET)
        print(f"{C_BOLD}{C_WHITE}GENERATION FINISHED SUCCESSFULLY{C_RESET}")
        print(f"  • Wall Time   : {C_GREEN}{C_BOLD}{wall_time:.2f}s{C_RESET} (Throughput: {C_BOLD}{fps:.2f} FPS{C_RESET})")
        print(f"  • Output File : {C_CYAN}{C_BOLD}{res.output_path}{C_RESET}")
        if res.master_output_path and os.path.exists(res.master_output_path):
            print(f"  • Master 4K   : {C_CYAN}{C_BOLD}{res.master_output_path}{C_RESET}")
            
        if res.profile_data:
            print(f"\n  Profiling Metrics:")
            for phase, dur in res.profile_data.items():
                print(f"     • {phase:26s}: {C_CYAN}{dur:.2f}s{C_RESET}")
                
        print(C_GREEN + "═" * 70 + C_RESET + "\n")
    else:
        print(f"\n{C_RED}Error during generation:{C_RESET}\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
