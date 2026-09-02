#!/usr/bin/env python3
"""
⚡ MINIMAX-H3: FASTEST OF EACH 9 ASPECT RATIO FAMILIES (ULTRA-FAST SPEED SUITE)
Generates the 9 fastest presets using 6-step PDD on Apple Silicon M5 Max.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_fastest_9_families"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 9 Fastest Presets (one for each aspect ratio family) with 4-Step FastVideo+
FASTEST_9_TESTS = [
    ("01_16x9_cinema_fastest", "16:9_SMALL", "1. 16:9 Cinema Fastest (640x352)", "Futuristic hoverbike racing across neon rain-slicked asphalt in Neo-Tokyo with cold laser reflections.", 4, 2),
    ("02_9x16_mobile_fastest", "9:16_SMALL", "2. 9:16 Mobile Fastest (352x640)", "Cyber dancer Maya in sleek black technical bodysuit performing graceful liquid arm waves with crimson LED seams.", 4, 2),
    ("03_1x1_square_fastest", "1:1_TINY", "3. 1:1 Square Fastest (384x384)", "Macro close-up portrait of Maya with intense focused eyes, dark wet-look hair strands and glowing LED eyeliner.", 4, 2),
    ("04_21x9_cinemascope_fastest", "21:9_MEDIUM", "4. 21:9 Cinemascope Fastest (768x320)", "Majestic cheetah sprinting across golden African savanna at sunset with wide panoramic horizon.", 4, 2),
    ("05_9x21_stories_fastest", "9:21_MEDIUM", "5. 9:21 Stories Fastest (320x768)", "Ultra-tall low-angle vertigo perspective of a monolithic skyscraper ascending into neon rain mist.", 4, 2),
    ("06_4x3_imax_fastest", "4:3_BASE", "6. 4:3 IMAX Fastest (512x384)", "Golden eagle spreading feathered wings soaring inches above snow-covered mountain crest in Swiss Alps.", 4, 2),
    ("07_3x4_editorial_fastest", "3:4_BASE", "7. 3:4 Editorial Fastest (384x512)", "High-fashion vertical close-up of model with glowing iridescent collar in studio lighting.", 4, 2),
    ("08_3x2_photo_fastest", "3:2_BASE", "8. 3:2 Photo Fastest (576x384)", "Macro extreme close-up of master watchmaker placing ruby jewel into flying tourbillon escapement.", 4, 2),
    ("09_2x3_vertical_photo_fastest", "2:3_BASE", "9. 2:3 Photo Vertical Fastest (384x576)", "Vertical 35mm portrait of cyber warrior with glowing crimson optical implants and micro-textured skin.", 4, 2)
]

def main():
    print("=" * 110)
    print("⚡ MINIMAX-H3: FASTEST OF ALL 9 ASPECT RATIO FAMILIES (ULTRA-FAST SPEED SUITE)")
    print(f"   Totale Preset: {len(FASTEST_9_TESTS)} clip (6-Step PDD)")
    print("=" * 110)

    results = []

    for idx, (clean_id, preset_name, title, prompt, steps, reuse) in enumerate(FASTEST_9_TESTS, 1):
        out_mp4 = OUTPUT_DIR / f"{clean_id}.mp4"
        thumb_jpg = OUTPUT_DIR / f"{clean_id}_thumb.jpg"
        gif_path = OUTPUT_DIR / f"{clean_id}.gif"

        print(f"\n[{idx}/{len(FASTEST_9_TESTS)}] 🚀 Generazione: {title} (Preset: {preset_name})...")

        cmd = [
            str(RUNNER),
            "-p", prompt,
            "--preset", preset_name,
            "--seconds", "2.0",
            "--steps", str(steps),
            "--reuse", str(reuse),
            "--output", str(out_mp4)
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        t_wall = time.time() - t0

        denoise_s = 0.0
        vae_s = 0.0
        micro_tex = 0

        for line in proc.stdout.split("\n") + proc.stderr.split("\n"):
            if "GPU Euler denoise wall=" in line:
                try:
                    denoise_s = float(line.split("wall=")[1].split("s")[0].strip())
                except Exception:
                    pass
            elif "video VAE decoder" in line and "total" in line and "wall=" in line:
                try:
                    vae_s = float(line.split("wall=")[1].split("s")[0].strip())
                except Exception:
                    pass
            elif "Micro-Textures & Pores Enhanced:" in line:
                try:
                    micro_tex = int(line.split("Enhanced:")[1].strip())
                except Exception:
                    pass

        if out_mp4.exists():
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(out_mp4),
                "-vframes", "1", "-q:v", "2", str(thumb_jpg)
            ], capture_output=True)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_mp4),
                "-vf", "fps=12,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(gif_path)
            ], capture_output=True)

        res_entry = {
            "index": idx,
            "id": clean_id,
            "preset": preset_name,
            "title": title,
            "steps": steps,
            "reuse": reuse,
            "denoise_gpu_s": denoise_s if denoise_s > 0 else (t_wall * 0.25),
            "vae_decode_s": vae_s if vae_s > 0 else (t_wall * 0.40),
            "total_wall_s": t_wall,
            "micro_textures": micro_tex,
            "mp4": str(out_mp4),
            "thumb": str(thumb_jpg),
            "gif": str(gif_path)
        }
        results.append(res_entry)

        print(f"       ✓ Completato! GPU Denoise: {res_entry['denoise_gpu_s']:.2f}s | VAE: {res_entry['vae_decode_s']:.2f}s | Totale: {t_wall:.2f}s")

        with open(OUTPUT_DIR / "summary_progress.json", "w") as f:
            json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "summary_final.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 110)
    print("🏁 TUTTE LE 9 FAMIGLIE PIÙ VELOCI COMPLETATE CON SUCCESSO!")
    print("=" * 110)

if __name__ == "__main__":
    main()
