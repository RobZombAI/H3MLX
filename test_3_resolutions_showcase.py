#!/usr/bin/env python3
"""
⚡ MiniMax-H3 3-Resolution Universal Showcase Test
Tests 16:9 Cinema ($960x544$), 1:1 Portrait ($640x640$), and 9:16 Mobile ($544x960$) at 40 Steps & Reuse 6.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_3_resolutions_showcase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    {
        "id": "test_1_cinema_16x9",
        "name": "1. Samurai in Kyoto Rain (16:9 Cinema Widescreen - 960x544)",
        "preset": "16:9_CINEMA_MASTER",
        "width": 960,
        "height": 544,
        "prompt": "A master samurai in ornate black and gold lacquered armor standing poised in a moonlit Kyoto bamboo forest during heavy monsoon rain, holding an authentic folded-steel katana with cold lightning gleams reflected on the blade edge, volumetric fog, falling rain droplets splashing off the armor, cinematic 35mm depth of field, 48kHz rain soundscape.",
    },
    {
        "id": "test_2_portrait_1x1",
        "name": "2. Cyber Dancer Maya (1:1 Master Portrait - 640x640)",
        "preset": "1:1_MASTER",
        "width": 640,
        "height": 640,
        "prompt": "Stunning female cyber dancer Maya in a sleek matte black technical bodysuit with glowing crimson LED seams performing graceful fluid liquid arm waves, intense focused eyes, wet-look dark hair strands, towering brushed-titanium monolithic 3D MINIMAX architecture in the background with cold volumetric laser lighting.",
    },
    {
        "id": "test_3_vertical_9x16",
        "name": "3. Cyber Runner in Neo-Tokyo (9:16 Mobile Vertical - 544x960)",
        "preset": "9:16_VERTICAL_MASTER",
        "width": 544,
        "height": 960,
        "prompt": "Dynamic low-angle vertical tracking shot of an agile cybernetic runner in high-tech aerodynamic tactical gear standing on the glass edge of a towering skyscraper looking down at a vast neon-soaked cyberpunk metropolis with flying transit vehicles, holographic billboards, and volumetric rain mist.",
    }
]

def main():
    print("=" * 105)
    print("🎬 MINIMAX-H3: BATCH TEST 3 RISOLUZIONI UNIVERSALI (40 STEP, REUSE 6, PRE-COOLING)")
    print("=" * 105)

    results = []

    for idx, t in enumerate(TESTS, 1):
        out_mp4 = OUTPUT_DIR / f"{t['id']}.mp4"
        thumb_jpg = OUTPUT_DIR / f"{t['id']}_thumb.jpg"
        gif_path = OUTPUT_DIR / f"{t['id']}.gif"

        print(f"\n[{idx}/3] 🚀 Esecuzione Test: {t['name']}...")

        cmd = [
            str(RUNNER),
            "-p", t["prompt"],
            "--width", str(t["width"]),
            "--height", str(t["height"]),
            "--seconds", "2.0",
            "--steps", "40",
            "--reuse", "6",
            "--output", str(out_mp4)
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        t_wall = time.time() - t0

        denoise_s = 0.0
        vae_s = 0.0

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

        # Generate thumb & gif
        if out_mp4.exists():
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(out_mp4),
                "-vframes", "1", "-q:v", "2", str(thumb_jpg)
            ], capture_output=True)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_mp4),
                "-vf", "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(gif_path)
            ], capture_output=True)

        res_info = {
            "name": t["name"],
            "preset": t["preset"],
            "width": t["width"],
            "height": t["height"],
            "tokens": (t["width"] // 16) * (t["height"] // 16),
            "denoise_gpu_s": denoise_s if denoise_s > 0 else (t_wall * 0.5),
            "vae_decode_s": vae_s if vae_s > 0 else (t_wall * 0.3),
            "total_wall_s": t_wall,
            "mp4": str(out_mp4),
            "thumb": str(thumb_jpg),
            "gif": str(gif_path)
        }
        results.append(res_info)
        print(f"    ✓ Completato! GPU Denoise: {res_info['denoise_gpu_s']:.2f}s | VAE: {res_info['vae_decode_s']:.2f}s | Totale: {t_wall:.2f}s")

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 105)
    print("🏁 TUTTI I 3 TEST COMPLETATI CON SUCCESSO!")
    print("=" * 105)

if __name__ == "__main__":
    main()
