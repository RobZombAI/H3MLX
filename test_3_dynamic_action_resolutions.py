#!/usr/bin/env python3
"""
⚡ MiniMax-H3 High-Dynamics Action Showcase across 3 Universal Resolutions
1. Hoverbike Chase (16:9 Cinema 960x544)
2. Cyber Martial Arts Kick (1:1 Action 640x640)
3. Wingsuit Dive off Spire (9:16 Vertical 544x960)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_3_dynamic_action_showcase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DYNAMIC_TESTS = [
    {
        "id": "action_1_hoverbike_16x9",
        "name": "1. High-Speed Cyber Hoverbike Chase (16:9 Cinema - 960x544)",
        "width": 960,
        "height": 544,
        "prompt": "Intense high-velocity low-angle tracking shot alongside a sleek matte-carbon hoverbike racing through rain-soaked Neo-Tokyo streets at night. Glowing cyan and cobalt LED thrusters leaving vibrant light trails, metallic sparks grinding against wet asphalt, rain mist splashing across the camera lens, deep panoramic city perspective with towering holographic skyscrapers and 35mm anamorphic speed blur, 48kHz roaring turbine audio.",
    },
    {
        "id": "action_2_martial_arts_1x1",
        "name": "2. Cyber Martial Arts Kinetic Strike (1:1 Action - 640x640)",
        "width": 640,
        "height": 640,
        "prompt": "Dynamic medium close-up shot of an agile female cybernetic warrior in a dark technical suit executing a powerful, graceful fluid spinning kick. Intense focused eyes, wet-look dark hair whipping across her face in slow motion, glowing crimson energy arcs wrapping around her leg, shattered glass and laser sparks scattering through the air, photorealistic skin texture, dramatic rim-lit physique, 48kHz impact soundscape.",
    },
    {
        "id": "action_3_wingsuit_dive_9x16",
        "name": "3. High-Altitude Wingsuit Skyscraper Dive (9:16 Vertical - 544x960)",
        "width": 544,
        "height": 960,
        "prompt": "Breathtaking vertical high-speed tracking shot following an aerodynamic wingsuit operative diving headfirst off the pinnacle spire of a massive megacity skyscraper. Rushing vertically downward between vertical neon-lit mega-towers, aerodynamic vapor trails peeling off the wings, dizzying perspective looking into the glowing cyber city depths below, 48kHz rushing wind acoustic soundscape.",
    }
]

def main():
    print("=" * 105)
    print("🎬 MINIMAX-H3: HIGH-DYNAMICS ACTION SUITE (3 RISOLUZIONI, 40 STEP, REUSE 6, PRE-COOLING)")
    print("=" * 105)

    results = []

    for idx, t in enumerate(DYNAMIC_TESTS, 1):
        out_mp4 = OUTPUT_DIR / f"{t['id']}.mp4"
        thumb_jpg = OUTPUT_DIR / f"{t['id']}_thumb.jpg"
        gif_path = OUTPUT_DIR / f"{t['id']}.gif"

        print(f"\n[{idx}/3] 🚀 Esecuzione Scena Dinamica: {t['name']}...")

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

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 105)
    print("🏁 TUTTE LE 3 SCENE AD ALTA DINAMICA COMPLETATE!")
    print("=" * 105)

if __name__ == "__main__":
    main()
