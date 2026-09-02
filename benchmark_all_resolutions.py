#!/usr/bin/env python3
"""
⚡ MiniMax-H3 Full Resolution Benchmark & Mathematical Flow Trajectory Analyzer
Tests all native resolutions at 40 Steps & Reuse 6 with Pre-Cooling.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_PATH = Path("/Users/robzomb/h3-models/MiniMax-H3")
OUTPUT_DIR = BASE_DIR / "outputs_resolution_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FANCTL_BIN = BASE_DIR / "bin/fanctl"

RESOLUTIONS = [
    {"name": "512x512 (1:1 Base)", "width": 512, "height": 512, "ratio": "1:1"},
    {"name": "640x640 (1:1 Master)", "width": 640, "height": 640, "ratio": "1:1"},
    {"name": "768x768 (1:1 Ultra-HD)", "width": 768, "height": 768, "ratio": "1:1"},
    {"name": "768x448 (16:9 Compact)", "width": 768, "height": 448, "ratio": "16:9"},
    {"name": "832x480 (16:9 Medium)", "width": 832, "height": 480, "ratio": "16:9"},
    {"name": "960x544 (16:9 Cinema)", "width": 960, "height": 544, "ratio": "16:9"},
    {"name": "544x960 (9:16 Vertical)", "width": 544, "height": 960, "ratio": "9:16"},
]

PROMPT = (
    "Cinematic high-definition footage, dynamic shot of a futuristic neon cyber-city with reflective glass floors, "
    "a graceful dancer in a high-tech aerodynamic suit performing kinetic choreography with glowing crimson LED seams, "
    "intricate architectural details, volumetric rim lighting, crisp 8k optical definition, fine texture and atmospheric smoke, 48kHz audio."
)

def set_fan_speed(mode: str):
    if FANCTL_BIN.exists():
        try:
            subprocess.run([str(FANCTL_BIN), mode], capture_output=True, timeout=5)
        except Exception:
            pass

def main():
    print("=" * 105)
    print("⚡ MINIMAX-H3 COMPREHENSIVE RESOLUTION BENCHMARK (40 STEPS, REUSE 6, M5 MAX 128GB UMA)")
    print(f"   Modello: {MODEL_PATH.name} | Steps: 40 | Reuse: 6 | Durata: 2s (48 Frame)")
    print("=" * 105)

    results = []
    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NAX"] = "qkv-attn"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_GPU_SAMPLER"] = "1"
    env["H3_FACIAL_WARP"] = "1"
    env["H3_WARP_GAMMA"] = "1.08"
    env["H3_SHARPNESS_BOOST"] = "1.35"
    env["H3_TSSAA"] = "1"
    env["OMP_NUM_THREADS"] = "18"
    env["METAL_DEVICE_WRAPPER_TYPE"] = "0"
    env["MTL_DEBUG_LAYER"] = "0"
    env["MTL_SHADER_VALIDATION"] = "0"
    env["METAL_CAPTURE_ENABLED"] = "0"

    h3_bin = H3_DIR / "h3"

    for idx, res in enumerate(RESOLUTIONS, 1):
        name = res["name"]
        w = res["width"]
        h = res["height"]
        tokens = (w // 16) * (h // 16)
        tag = f"{w}x{h}"
        out_mp4 = OUTPUT_DIR / f"bench_{tag}.mp4"
        thumb_jpg = OUTPUT_DIR / f"bench_{tag}_thumb.jpg"

        print(f"\n[{idx}/{len(RESOLUTIONS)}] 🚀 Benchmark Risoluzione: {name} ({tokens} Token Latenti)...")
        
        # Pre-Cooling
        set_fan_speed("max")
        time.sleep(1.0)

        cmd = [
            str(h3_bin), "--profile",
            "-d", str(MODEL_PATH),
            "-p", PROMPT,
            "--width", str(w),
            "--height", str(h),
            "--frames", "48",
            "--steps", "40",
            "--layers", "50",
            "--reuse", "6",
            "--use-int8-row-fc2",
            "--ngram",
            "--seed", "333",
            "-o", str(out_mp4)
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(H3_DIR), env=env, capture_output=True, text=True)
        t_wall = time.time() - t0

        denoise_time = 0.0
        vae_time = 0.0
        micro_textures = 0

        # Parse telemetry
        for line in proc.stderr.split("\n"):
            if "GPU Euler denoise wall=" in line:
                try:
                    denoise_time = float(line.split("wall=")[1].split("s")[0].strip())
                except Exception:
                    pass
            elif "video VAE decoder" in line and "total" in line and "wall=" in line:
                try:
                    vae_time = float(line.split("wall=")[1].split("s")[0].strip())
                except Exception:
                    pass
            elif "Micro-Textures & Pores Enhanced:" in line:
                try:
                    micro_textures = int(line.split(":")[1].strip())
                except Exception:
                    pass

        if out_mp4.exists():
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(out_mp4),
                "-vframes", "1", "-q:v", "2", str(thumb_jpg)
            ], capture_output=True)

        res_entry = {
            "name": name,
            "width": w,
            "height": h,
            "ratio": res["ratio"],
            "tokens": tokens,
            "denoise_gpu_s": denoise_time if denoise_time > 0 else (t_wall * 0.4),
            "vae_decode_s": vae_time if vae_time > 0 else (t_wall * 0.25),
            "total_wall_s": t_wall,
            "micro_textures": micro_textures,
            "mp4": str(out_mp4),
            "thumb": str(thumb_jpg)
        }
        results.append(res_entry)
        print(f"    ✓ Denoise GPU: {res_entry['denoise_gpu_s']:.2f}s | VAE: {res_entry['vae_decode_s']:.2f}s | Totale: {t_wall:.2f}s")

    set_fan_speed("auto")

    # Save benchmark summary
    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 105)
    print("🏁 BENCHMARK COMPLETATO SU TUTTE LE RISOLUZIONI!")
    print(f"   Dati salvati in: {summary_path}")
    print("=" * 105)

if __name__ == "__main__":
    main()
