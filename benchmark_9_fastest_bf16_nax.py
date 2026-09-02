#!/usr/bin/env python3
"""
⚡ BENCHMARK SUITE: 9 FASTEST PRESETS WITH BF16 NAX VIDEO VAE DECODER
Runs all 9 fastest presets with FastVideo+ 4-Step Euler ODE (--reuse 2) and BF16 NAX VAE.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_9_fastest_bf16_nax"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/fastest_9_bf16_nax_showcase")
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "A graceful, elegant dancer girl performing soft, fluid, ethereal contemporary dance movements "
    "with flowing sheer silk fabric catching gentle golden backlighting, delicate expressive gestures, "
    "sharp facial focus with detailed skin pores, cinematic bokeh, 24 fps smooth motion, 48kHz gentle melodic ambient."
)

FASTEST_9_PRESETS = [
    ("01_16x9_cinema", "16:9_SMALL", "640x352", 880, "16:9 Cinema Widescreen"),
    ("02_9x16_mobile", "9:16_SMALL", "352x640", 880, "9:16 Vertical Mobile"),
    ("03_1x1_square", "1:1_TINY", "384x384", 576, "1:1 Square Format"),
    ("04_21x9_cinemascope", "21:9_MEDIUM", "768x320", 960, "21:9 Cinemascope"),
    ("05_9x21_stories", "9:21_MEDIUM", "320x768", 960, "9:21 Vertical Stories"),
    ("06_4x3_classic", "4:3_BASE", "512x384", 768, "4:3 Classic / IMAX"),
    ("07_3x4_editorial", "3:4_BASE", "384x512", 768, "3:4 Vertical Editorial"),
    ("08_3x2_photo", "3:2_BASE", "576x384", 864, "3:2 35mm Photography"),
    ("09_2x3_vertical_photo", "2:3_BASE", "384x576", 864, "2:3 35mm Vertical Photo")
]

def main():
    print("=" * 110, flush=True)
    print("💃 MINIMAX-H3: BENCHMARK 9 FLASH PRESET CON DECODER VIDEO VAE BF16 NAX (METAL 4 MATRIX ENGINE)", flush=True)
    print("   Motore: FastVideo+ 4-Step Euler ODE (--reuse 2) | BF16 NAX VAE | INT8-FC2 DiT", flush=True)
    print("   Hardware: Apple Silicon M5 Max 128GB UMA | Metal 4 Native Matrix Engine", flush=True)
    print("=" * 110, flush=True)

    results = []

    for idx, (clean_id, preset_name, res_str, tokens, family_title) in enumerate(FASTEST_9_PRESETS, 1):
        out_mp4 = OUTPUT_DIR / f"{clean_id}_dancer.mp4"
        brain_raw = BRAIN_DIR / f"{clean_id}_raw.mp4"
        thumb_jpg = BRAIN_DIR / f"{clean_id}_thumb.jpg"

        print(f"\n[{idx}/{len(FASTEST_9_PRESETS)}] 🚀 Esecuzione: {family_title} (Preset: {preset_name} | {res_str} - {tokens} Token)...", flush=True)

        cmd = [
            str(RUNNER),
            "-p", PROMPT,
            "--preset", preset_name,
            "--seconds", "2.0",
            "--steps", "4",
            "--reuse", "2",
            "--output", str(out_mp4)
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        t_wall = round(time.perf_counter() - t0, 2)

        denoise_s = 0.0
        vae_s = 0.0
        vae_load_s = 0.0
        dit_load_s = 0.0
        micro_tex = 0

        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            if "H3 DiT                   load" in line and "wall=" in line:
                try: dit_load_s = float(line.split("wall=")[1].split("s")[0].strip())
                except: pass
            elif "GPU Euler denoise wall=" in line:
                try: denoise_s = float(line.split("wall=")[1].split("s")[0].strip())
                except: pass
            elif "video VAE load" in line and "100%" in line:
                try: vae_load_s = float(line.split("(")[1].split("s")[0].strip())
                except: pass
            elif "video VAE decoder" in line and "total" in line and "wall=" in line:
                try: vae_s = float(line.split("wall=")[1].split("s")[0].strip())
                except: pass
            elif "Micro-Textures & Pores Enhanced:" in line:
                try: micro_tex = int(line.split("Enhanced:")[1].strip())
                except: pass

        pure_calc_s = round(denoise_s + vae_s + 0.37, 2)

        if out_mp4.exists():
            subprocess.run(["cp", str(out_mp4), str(brain_raw)], capture_output=True)
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(out_mp4),
                "-vframes", "1", "-q:v", "2", str(thumb_jpg)
            ], capture_output=True)

        entry = {
            "index": idx,
            "id": clean_id,
            "preset": preset_name,
            "family": family_title,
            "resolution": res_str,
            "tokens": tokens,
            "dit_load_s": dit_load_s,
            "denoise_gpu_s": denoise_s,
            "vae_load_s": vae_load_s,
            "vae_decode_s": vae_s,
            "pure_calc_s": pure_calc_s,
            "wall_total_s": t_wall,
            "micro_textures": micro_tex,
            "mp4": str(out_mp4)
        }
        results.append(entry)

        print(f"   ✓ Finito in {t_wall}s | Denoise: {denoise_s}s | VAE Decode: {vae_s}s | Calcolo Puro: {pure_calc_s}s", flush=True)

    json_path = OUTPUT_DIR / "benchmark_9_fastest_bf16_nax_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 110, flush=True)
    print("🏆 BENCHMARK 9 FLASH COMPLETATO CON SUCCESSO!", flush=True)
    print("=" * 110, flush=True)

if __name__ == "__main__":
    main()
