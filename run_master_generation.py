#!/usr/bin/env python3
"""
⚡ MiniMax H3 Master Generator with Universal Step Set & Resolution Compiler & Pre-Cooling
Supports all step sets: Turbo (4), Fast (8), Balanced (20), Master (40), Extreme (60), Museum (100) and any custom step count.
"""

import os
import sys
import time
import json
import re
import math
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Dict

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
DEFAULT_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
FULL_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3")

class H3UniversalResolutionCompiler:
    """
    Continuous analytical compiler that maps ANY (width, height) AND ANY step set (4 to 100 steps)
    to optimal optical framing, step-reuse ratio, warp gamma, and sharpness boost.
    """

    STANDARD_PRESETS = {
        # 16:9 Widescreen Cinema
        "16:9_UHD": (1280, 704),
        "16:9_FHD": (1024, 576),
        "16:9_CINEMA_MASTER": (960, 544),
        "16:9_LARGE": (896, 512),
        "16:9_MEDIUM": (832, 480),
        "16:9_COMPACT": (768, 448),
        "16:9_SMALL": (640, 352),
        "16:9_TINY": (512, 288),

        # 9:16 Mobile / TikTok / Reels / Shorts
        "9:16_UHD": (704, 1280),
        "9:16_FHD": (576, 1024),
        "9:16_VERTICAL_MASTER": (544, 960),
        "9:16_LARGE": (512, 896),
        "9:16_MEDIUM": (480, 832),
        "9:16_COMPACT": (448, 768),
        "9:16_SMALL": (352, 640),

        # 1:1 Square
        "1:1_ULTRA_MASTER": (768, 768),
        "1:1_MASTER": (640, 640),
        "1:1_LARGE": (576, 576),
        "1:1_BASE": (512, 512),
        "1:1_COMPACT": (448, 448),
        "1:1_TINY": (384, 384),

        # 21:9 Ultra-Widescreen Cinemascope
        "21:9_CINEMASCOPE": (1024, 448),
        "21:9_LARGE": (896, 384),
        "21:9_MEDIUM": (768, 320),

        # 9:21 Ultra-Tall Banner / Stories
        "9:21_STORIES": (448, 1024),
        "9:21_LARGE": (384, 896),
        "9:21_MEDIUM": (320, 768),

        # 4:3 Classic Television / IMAX
        "4:3_LARGE": (960, 704),
        "4:3_MASTER": (768, 576),
        "4:3_MEDIUM": (640, 480),
        "4:3_BASE": (512, 384),

        # 3:4 Classic Portrait
        "3:4_LARGE": (704, 960),
        "3:4_MASTER": (576, 768),
        "3:4_MEDIUM": (480, 640),
        "3:4_BASE": (384, 512),

        # 3:2 35mm Photography
        "3:2_MASTER": (960, 640),
        "3:2_MEDIUM": (768, 512),
        "3:2_BASE": (576, 384),

        # 2:3 35mm Vertical Photography
        "2:3_MASTER": (640, 960),
        "2:3_MEDIUM": (512, 768),
        "2:3_BASE": (384, 576),
    }

    STEP_PRESETS = {
        "turbo": (4, 1),       # 4 steps, exact/reuse 1 (Ultra fast preview)
        "fast": (8, 1),        # 8 steps, exact/reuse 1 (Fast high-res)
        "balanced": (20, 4),   # 20 steps, reuse 4 (5 forward GPU evals)
        "master": (40, 6),     # 40 steps, reuse 6 (7-8 forward GPU evals - STANDARD)
        "extreme": (60, 8),    # 60 steps, reuse 8 (8 forward GPU evals)
        "museum": (100, 12),   # 100 steps, reuse 12 (9 forward GPU evals)
    }

    @classmethod
    def get_optimal_reuse(cls, steps: int, is_pdd: bool = False) -> int:
        """Calculates optimal mathematical step-reuse ratio for any step count and model."""
        if is_pdd:
            return 1 if steps <= 8 else 2
        else:
            if steps <= 8:
                return 1
            elif steps <= 16:
                return 2
            elif steps <= 25:
                return 4
            elif steps <= 45:
                return 6
            elif steps <= 65:
                return 8
            elif steps <= 85:
                return 10
            else:
                return 12

    @classmethod
    def compile(cls, raw_prompt: str, width: int, height: int, steps: int = 40, reuse: int = 0, is_pdd: bool = False) -> Tuple[str, Dict[str, str], int, int]:
        w_clamped = max(32, (width // 32) * 32)
        h_clamped = max(32, (height // 32) * 32)
        tokens = (w_clamped // 16) * (h_clamped // 16)
        aspect_ratio = w_clamped / h_clamped
        is_small_res = tokens <= 1100

        actual_reuse = reuse if reuse > 0 else cls.get_optimal_reuse(steps, is_pdd=is_pdd)

        # 1. Continuous Analytical Hardware Parameter Derivation based on (Steps, Tokens)
        t_factor = max(0.0, min(1.5, (tokens - 1024.0) / (2304.0 - 1024.0)))
        step_factor = max(0.0, math.log2(max(4, steps) / 8.0))
        
        # Warp Gamma & Sharpness Boost tuning
        if is_small_res:
            warp_val = 1.12 + (0.02 * step_factor)
            sharp_val = 1.58 + (0.04 * step_factor)
        else:
            warp_val = 1.06 + (0.04 * step_factor) + (0.08 * t_factor)
            sharp_val = 1.35 + (0.06 * step_factor) + (0.22 * t_factor)

        warp_gamma = f"{min(1.25, max(1.02, warp_val)):.2f}"
        sharpness_boost = f"{min(1.70, max(1.20, sharp_val)):.2f}"

        # 2. Aspect Ratio & Token Concentration Optical Framing Synthesizer
        if is_small_res:
            if aspect_ratio >= 1.3:
                framing = "Cinematic dynamic medium close-up shot, 35mm lens, concentrated foreground subject with exquisite facial micro-details and soft atmospheric background"
            elif aspect_ratio <= 0.7:
                framing = "Cinematic 9:16 vertical dynamic medium close-up to bust shot, intense concentrated perspective on character face, expressive eyes, and sculpted torso with towering vertical background lighting"
            else:
                framing = "Cinematic centered dynamic close-up portrait, maximum token concentration on expressive face, sharp irises, and kinetic posture"
        else:
            if aspect_ratio >= 2.2:
                framing = "Cinematic 21:9 ultra-widescreen Cinemascope framing, sweeping anamorphic lens, grand panoramic scale with dynamic foreground character tracking and massive atmospheric horizon"
            elif aspect_ratio >= 1.55:
                framing = "Cinematic dynamic medium shot, anamorphic 35mm lens, 16:9 widescreen composition with subject positioned gracefully in the foreground and deep atmospheric background staging"
            elif aspect_ratio >= 1.35:
                framing = "Cinematic 3:2 classic 35mm photography framing, natural golden ratio composition with exquisite spatial separation and organic background bokeh"
            elif aspect_ratio >= 1.15:
                framing = "Cinematic 4:3 IMAX format framing, vertical and horizontal balance with powerful central subject presence and towering geometric scale"
            elif aspect_ratio >= 0.85:
                framing = "Cinematic dynamic medium close-up, 1:1 centered portrait composition with maximum token density focused on subject anatomy, expressive face, and kinetic posture"
            elif aspect_ratio >= 0.70:
                framing = "Cinematic 3:4 editorial portrait framing, elegant vertical staging from torso to head with dramatic overhead volumetric lighting"
            elif aspect_ratio >= 0.60:
                framing = "Cinematic 2:3 high-fashion vertical photography framing, dynamic three-quarter length perspective with deep vertical perspective"
            elif aspect_ratio >= 0.45:
                framing = "Cinematic 9:16 vertical dynamic shot, full-figure to medium tracking shot optimized for mobile perspective with towering vertical depth"
            else:
                framing = "Cinematic 9:21 ultra-tall vertical composition, dramatic low-angle vertigo perspective from floor to sky"

        # 3. Dynamic Domain & Semantic Isolation
        p_lower = raw_prompt.lower()
        is_cinema_narrative = any(k in p_lower for k in ["tarantino", "pulp", "fiction", "cinema", "film still", "film scene", "movie", "diner", "vincent", "vega", "wallace", "actor", "two-shot", "35mm", "retro 90s", "heist", "hitman"])
        is_human_dance = not is_cinema_narrative and any(k in p_lower for k in ["dance", "dancer", "ballerina", "popping", "tutting", "choreography", "suit", "girl", "woman", "maya", "samurai", "human", "face", "portrait", "dj", "singer", "martial", "kick", "fighter"])
        is_wildlife = any(k in p_lower for k in ["cheetah", "wolf", "eagle", "animal", "lion", "tiger", "bear", "bird", "hawk", "falcon"])
        is_macro = any(k in p_lower for k in ["watch", "tourbillon", "macro", "clockwork", "lens", "close-up", "micro", "mechanism", "gear", "jewelry"])
        is_scifi_city = any(k in p_lower for k in ["cyber", "neon", "city", "street", "car", "blade runner", "matrix", "tokyo", "spaceship", "hoverbike", "wingsuit"])

        clean_text = raw_prompt.strip()
        clean_text = re.sub(r'\b160\s*BPM\b', 'smooth rhythmic pace', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\bsharp,\s*staccato popping and tutting\b', 'graceful, controlled, fluid liquid arm movements and hypnotic wave choreography', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\bblistering\b', 'atmospheric', clean_text, flags=re.IGNORECASE)

        if is_cinema_narrative:
            motion_lighting = "authentic 35mm film grain, Kodak Vision3 5219 film stock, atmospheric warm chiaroscuro diner lighting, smooth eye-level medium tracking shot, stable character 3D spatial consistency, natural human facial micro-expressions, articulated hands with distinct joint separation, geometric ambient neon strip illumination, soft natural depth of field, 24 fps cinematic motion blur"
        elif is_human_dance:
            motion_lighting = "graceful classical contemporary dancer posture, open extended arm phrasing with distinctly articulated wrists and separated elegant fingers, clean anatomical silhouette with clear spatial clearance from torso, natural 180-degree motion blur at 24 fps, volumetric golden amber rim lighting, crisp wet-look hair strands, photorealistic skin texture, dramatic rim-lit contours"
        elif is_wildlife:
            motion_lighting = "majestic natural continuous locomotion, realistic muscular anatomy, sub-pixel fur and feather details, golden hour rim lighting, organic 24 fps motion blur"
        elif is_macro:
            motion_lighting = "ultra-crisp 8k optical definition, sub-pixel specular highlights, microscopic bevel reflections, smooth slow-motion camera drift, shallow depth of field with soft circular bokeh"
        elif is_scifi_city:
            motion_lighting = "ultra-crisp 8k optical definition, volumetric atmospheric haze, glowing neon reflections on wet asphalt and reflective glass, cinematic camera tracking at 24 fps"
        else:
            motion_lighting = "ultra-crisp 8k optical definition, volumetric atmospheric lighting, physically-based surface reflections, natural 24 fps motion dynamics"

        audio_spec = "48kHz synchronized high-fidelity spatial audio."

        if "integrated_multimodal_description:" in raw_prompt:
            compiled_prompt = clean_text
        elif any(clean_text.lower().startswith(prefix) for prefix in ["cinematic", "extreme", "dynamic", "macro", "portrait", "intense", "breathtaking", "quentin"]):
            compiled_prompt = f"{clean_text}, {motion_lighting}, {audio_spec}"
        else:
            compiled_prompt = f"{framing}, {clean_text}, {motion_lighting}, {audio_spec}"

        env_vars = {
            "H3_PROFILE": "1",
            "H3_NAX": "qkv-attn",
            "H3_DIT_COMMAND_BLOCKS": "0",
            "H3_ZERO_COPY_WEIGHTS": "1",
            "H3_REUSE_MPS_COMMAND": "1",
            "H3_FACIAL_WARP": "1",
            "H3_WARP_GAMMA": warp_gamma,
            "H3_SHARPNESS_BOOST": sharpness_boost,
            "H3_HOLISTIC_NGRAM": "1",
            "H3_NGRAM_SPECULATIVE": "1",
            "H3_TRIGRAM_TREE": "1",
            "H3_OCTREE_NGRAM": "1",
            "H3_OPTICAL_FLOW_WARP": "1",
            "H3_VAE_NGRAM_SPECULATIVE": "1",
            "H3_AUDIO_NGRAM_SPECULATIVE": "1",
            "H3_VAE_INT8": "1",
            "H3_CRF": "14",
            "H3_TSSAA": "1",
            "OMP_NUM_THREADS": "18",
            "METAL_DEVICE_WRAPPER_TYPE": "0",
            "MTL_DEBUG_LAYER": "0",
            "METAL_CAPTURE_ENABLED": "0"
        }

        return compiled_prompt, env_vars, actual_reuse, steps

def main():
    parser = argparse.ArgumentParser(description="Master High-Fidelity Video Generator with Universal Step Set & Resolution Compiler")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Video prompt or request")
    parser.add_argument("--preset", type=str, default="", help="Standard resolution preset (e.g. 16:9_CINEMA_MASTER, 1:1_MASTER, 9:16_VERTICAL_MASTER)")
    parser.add_argument("--step-preset", type=str, default="", choices=["turbo", "fast", "balanced", "master", "extreme", "museum"], help="Step preset: turbo (4), fast (8), balanced (20), master (40), extreme (60), museum (100)")
    parser.add_argument("--model", type=str, default="", choices=["pdd", "full"], help="Model type: full (MiniMax-H3) or pdd (8-step)")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration in seconds (e.g. 1.0, 2.0, 4.0, 15.0)")
    parser.add_argument("--width", type=int, default=0, help="Width (default 960)")
    parser.add_argument("--height", type=int, default=0, help="Height (default 544)")
    parser.add_argument("--steps", type=int, default=0, help="Sampling steps (e.g. 4, 8, 20, 40, 60, 100)")
    parser.add_argument("--reuse", type=int, default=-1, help="Step reuse (-1 = auto-optimal derived, 0 = exact evaluation)")
    parser.add_argument("--output", "-o", type=str, default="outputs/master_gen.mp4", help="Output MP4 path")
    parser.add_argument("--seed", type=int, default=333, help="RNG seed")
    parser.add_argument("--solver", type=str, default="auto", choices=["auto", "dpm3m", "euler"], help="ODE Flow Matching Solver: auto (routes by model), dpm3m (3rd-order), or euler")
    parser.add_argument("--raw", action="store_true", help="Disable adaptive prompt compiler and use raw prompt directly")
    
    args = parser.parse_args()

    # 1. Resolve Width & Height
    width = 960
    height = 544
    if args.preset and args.preset.upper() in H3UniversalResolutionCompiler.STANDARD_PRESETS:
        width, height = H3UniversalResolutionCompiler.STANDARD_PRESETS[args.preset.upper()]
    if args.width > 0:
        width = args.width
    if args.height > 0:
        height = args.height

    # 2. Resolve Step Preset and Steps
    steps = 40
    reuse = args.reuse
    if args.step_preset:
        steps, default_reuse = H3UniversalResolutionCompiler.STEP_PRESETS[args.step_preset]
        if reuse == -1:
            reuse = default_reuse
    elif args.steps > 0:
        steps = args.steps

    # 3. Model selection & Multi-Level Compatibility Routing
    if args.model:
        is_pdd = (args.model == "pdd")
        model_path = DEFAULT_MODEL if is_pdd else FULL_MODEL
    else:
        # PDD Distilled (Fast 4-16 step ladder) vs Full Continuous Flow (20-100 step master)
        is_pdd = (steps <= 16)
        model_path = DEFAULT_MODEL if is_pdd else FULL_MODEL

    if reuse == -1:
        reuse = H3UniversalResolutionCompiler.get_optimal_reuse(steps, is_pdd=is_pdd)

    # 4. Optical Prompt & Continuous Hardware Compilation
    if not args.raw:
        compiled_prompt, env_tuning, actual_reuse, actual_steps = H3UniversalResolutionCompiler.compile(args.prompt, width, height, steps, reuse, is_pdd=is_pdd)
    else:
        compiled_prompt = args.prompt
        _, env_tuning, actual_reuse, actual_steps = H3UniversalResolutionCompiler.compile("", width, height, steps, reuse, is_pdd=is_pdd)

    # 5. Solver Dynamic Routing
    if args.solver == "auto":
        resolved_solver = "euler" if is_pdd else "dpm3m"
    else:
        resolved_solver = args.solver

    if resolved_solver == "dpm3m":
        env_tuning["H3_CPU_SAMPLER"] = "1"
        env_tuning["H3_SOLVER"] = "dpm3m"
        env_tuning.pop("H3_GPU_SAMPLER", None)
    else:
        env_tuning["H3_CPU_SAMPLER"] = "1"
        env_tuning["H3_SOLVER"] = "euler"
        env_tuning.pop("H3_GPU_SAMPLER", None)

    # 6. Calculate exact frames according to Causal Lattice T = 17n + 5
    fps = 24
    raw_frames = int(round(args.seconds * fps))
    if raw_frames <= 22:
        frames = 22  # n=1 (1.0s)
    elif raw_frames <= 48:
        frames = 39 if raw_frames <= 39 else 48  # n=2 (2.0s)
    elif raw_frames <= 96:
        frames = 90 if raw_frames <= 90 else 96  # n=5 (4.0s)
    else:
        frames = raw_frames

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Set Hardware Environment
    env = os.environ.copy()
    env.update(env_tuning)

    h3_bin = H3_DIR / "h3"
    resident_sock = Path("/tmp/h3_resident.sock")
    is_resident = resident_sock.exists()

    if is_resident:
        cmd = [
            str(h3_bin),
            "--client", str(resident_sock),
            "-p", compiled_prompt,
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--steps", str(actual_steps),
            "--layers", "50",
            "--reuse", str(actual_reuse),
            "--seed", str(args.seed),
            "-o", str(output_path)
        ]
    else:
        cmd = [
            str(h3_bin), "--profile",
            "-d", str(model_path),
            "-p", compiled_prompt,
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--steps", str(actual_steps),
            "--layers", "50",
            "--reuse", str(actual_reuse),
            "--use-int8-row-fc2",
            "--ngram",
            "--seed", str(args.seed),
            "-o", str(output_path)
        ]

    gpu_passes = actual_steps // actual_reuse + (1 if actual_steps % actual_reuse else 0) if actual_reuse > 1 else actual_steps
    print(f"🎬 Avvio Generazione Master [Universal Dynamic Compatibility Matrix]:")
    print(f"   Modalità Esecuzione : {'⚡ Residente UMA (/tmp/h3_resident.sock - 0s Load)' if is_resident else 'Standard Diretto'}")
    print(f"   Modello Auto-Routed : {model_path.name} ({'PDD Distilled Ladder' if is_pdd else 'Full Continuous Flow'})")
    print(f"   Risoluzione & Ratio : {width}x{height} ({(width//16)*(height//16)} token | Aspect Ratio {width/height:.2f}:1)")
    print(f"   Frame / Durata      : {frames} frame ({args.seconds:.1f}s @ 24 fps | Causal Lattice T={frames})")
    print(f"   Step & Riuso        : {actual_steps} step (Reuse {actual_reuse} -> {gpu_passes} forward GPU reali)")
    print(f"   Solutore ODE        : {resolved_solver.upper()} ({'Piecewise Linear Flow' if resolved_solver == 'euler' else 'Adams-Bashforth 3M + Symplectic Flow'})")
    print(f"   Tuning Hardware     : Warp Gamma={env_tuning.get('H3_WARP_GAMMA')} | Sharpness Boost={env_tuning.get('H3_SHARPNESS_BOOST')}")
    print(f"   Prompt Compilato    : \"{compiled_prompt[:130]}...\"")
    print(f"   Output MP4          : {output_path}")
    print("-" * 95)

    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(H3_DIR), env=env)
    total_time = time.time() - start_time
    print("-" * 95)
    if proc.returncode == 0:
        print(f"✅ GENERAZIONE COMPLETATA CON SUCCESSO in {total_time:.2f}s!")
    else:
        print(f"⚠️ Processo terminato con codice di errore: {proc.returncode}")

if __name__ == "__main__":
    main()
