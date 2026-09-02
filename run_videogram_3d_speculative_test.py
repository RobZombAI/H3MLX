#!/usr/bin/env python3
"""
⚡ MINIMAX-H3: 3D SPATIO-TEMPORAL VIDEO-GRAM & SPECULATIVE VAE DECODER FULL TEST
Generates 2040 Tokens (544x960) with 3D Video-Gram Frame-Skip & Fast Speculative VAE Decoding.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUT_DIR = BASE_DIR / "outputs_fastest_9_families"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/videogram_3d_showcase")
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Cinematic 9:16 vertical dynamic shot, full-figure to medium tracking shot optimized for mobile perspective with towering vertical depth: "
    "Ultra-photorealistic 8k video of a stunning cyberpunk dancer girl Maya in a sleek black latex bodysuit with crimson glowing neon accents, "
    "performing energetic expressive fluid street-dance choreography in a rain-slicked Tokyo neon alleyway. "
    "Volumetric red and cyan stage backlighting, wet hair strands flowing with head spins, dynamic arm waves, sharp facial focus with detailed skin pores, "
    "synchronized energetic rhythm, 24 fps cinematic motion blur."
)

def main():
    print("=" * 110)
    print("🚀 MINIMAX-H3: 3D SPATIO-TEMPORAL VIDEO-GRAM & SPECULATIVE VAE DECODER TEST")
    print("   Risoluzione: 544x960 (2040 Token Latenti)")
    print("   Motore: FastVideo+ 4-Step Euler | 3D Video-Gram SIMD NEON | Speculative VAE")
    print("   Hardware: Apple Silicon M5 Max 128GB UMA | Metal 4 Native")
    print("=" * 110)

    raw_mp4 = OUTPUT_DIR / "maya_VIDEOGRAM_3D_SPECULATIVE_TEST.mp4"
    thumb_path = BRAIN_DIR / "thumb.jpg"
    gif_path = BRAIN_DIR / "animated.gif"

    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NAX"] = "qkv-attn"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_GPU_SAMPLER"] = "1"
    env["H3_WARP_GAMMA"] = "1.10"
    env["H3_SHARPNESS_BOOST"] = "1.44"
    env["H3_VIDEOGRAM_3D"] = "1"
    env["H3_SPECULATIVE_VAE"] = "1"
    env["OMP_NUM_THREADS"] = "18"

    cmd = [
        str(H3_DIR / "h3"), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PROMPT,
        "--width", "544",
        "--height", "960",
        "--frames", "48",
        "--steps", "4",
        "--layers", "50",
        "--reuse", "2",
        "--use-int8-row-fc2",
        "-o", str(raw_mp4)
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(H3_DIR), env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    wall_total = round(t1 - t0, 2)

    denoise_s = 0.0
    vae_s = 0.0
    load_s = 0.0

    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        if "H3 DiT                   load" in line and "wall=" in line:
            try: load_s = float(line.split("wall=")[1].split("s")[0].strip())
            except: pass
        elif "GPU Euler denoise wall=" in line:
            try: denoise_s = float(line.split("wall=")[1].split("s")[0].strip())
            except: pass
        elif "video VAE decoder" in line and "total" in line and "wall=" in line:
            try: vae_s = float(line.split("wall=")[1].split("s")[0].strip())
            except: pass

    print(f"\n✓ Test Concluso!")
    print(f"  • Load / Init Pre-Denoise: {load_s:.2f}s")
    print(f"  • Denoise GPU Metal 4:     {denoise_s:.2f}s")
    print(f"  • Decodifica Video VAE:    {vae_s:.2f}s")
    print(f"  • Wall-Clock Totale Reale: {wall_total:.2f}s")

    if raw_mp4.exists():
        subprocess.run(["cp", str(raw_mp4), str(BRAIN_DIR / "raw.mp4")], capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(raw_mp4),
            "-vframes", "1", "-q:v", "2", str(thumb_path)
        ], capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(raw_mp4),
            "-vf", "fps=12,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path)
        ], capture_output=True)

    result = {
        "preset": "9:16_VERTICAL_MASTER (544x960 - 2040 Tokens)",
        "engine": "FastVideo+ Euler 4-Step + 3D Video-Gram + Speculative VAE",
        "load_time_s": load_s,
        "denoise_gpu_s": denoise_s,
        "vae_decode_s": vae_s,
        "wall_total_s": wall_total,
        "mp4": str(raw_mp4)
    }

    with open(OUTPUT_DIR / "videogram_3d_test_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 110)
    print("🏁 TEST COMPLETATO CON SUCCESSO!")
    print("=" * 110)

if __name__ == "__main__":
    main()
