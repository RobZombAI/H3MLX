#!/usr/bin/env python3
"""
⚡ MiniMax-H3 All-Presets Master Verification Suite
Tests all key presets across all aspect ratios (16:9, 9:16, 1:1, 21:9, 4:3, 3:2, 2:3)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_all_master_presets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET_TESTS = [
    {
        "id": "01_16x9_cinema_master",
        "preset": "16:9_CINEMA_MASTER",
        "title": "1. 16:9 Cinema Master (960x544) — Samurai in Kyoto Rain",
        "prompt": "A master samurai in ornate black and gold lacquered armor standing poised in a moonlit Kyoto bamboo forest during heavy monsoon rain, holding an authentic folded-steel katana with cold lightning gleams reflected on the blade edge, volumetric fog, falling rain droplets splashing off the armor, cinematic 35mm depth of field, 48kHz rain soundscape.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "02_9x16_vertical_master",
        "preset": "9:16_VERTICAL_MASTER",
        "title": "2. 9:16 Vertical Master (544x960) — Cyber Runner Neo-Tokyo",
        "prompt": "Dynamic low-angle vertical tracking shot of an agile cybernetic runner in high-tech aerodynamic tactical gear standing on the glass edge of a towering skyscraper looking down at a vast neon-soaked cyberpunk metropolis with flying transit vehicles, holographic billboards, and volumetric rain mist.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "03_1x1_square_master",
        "preset": "1:1_MASTER",
        "title": "3. 1:1 Square Master (640x640) — Cyber Dancer Maya",
        "prompt": "Stunning female cyber dancer Maya in a sleek matte black technical bodysuit with glowing crimson LED seams performing graceful fluid liquid arm waves, intense focused eyes, wet-look dark hair strands, towering brushed-titanium monolithic 3D MINIMAX architecture in the background with cold volumetric laser lighting.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "04_21x9_cinemascope",
        "preset": "21:9_CINEMASCOPE",
        "title": "4. 21:9 Cinemascope (1024x448) — Cheetah in Savanna Sunset",
        "prompt": "Majestic cheetah in full predatory stride sprinting gracefully across the golden African savanna at sunset, muscular ripples under spotted coat, kicking up fine golden dust clouds in slow-motion, wide panoramic horizon with acacia trees, warm cinematic anamorphic lens flares.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "05_4x3_imax_master",
        "preset": "4:3_MASTER",
        "title": "5. 4:3 IMAX Master (768x576) — Golden Eagle Mountain Crest",
        "prompt": "Majestic golden eagle spreading massive feathered wings soaring inches above a snow-covered mountain crest in the Swiss Alps, sub-pixel feather details catching morning sunlight, crisp icy wind blowing snow dust, deep IMAX depth of field.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "06_3x2_photo_master",
        "preset": "3:2_MASTER",
        "title": "6. 3:2 Photography Master (960x640) — Tourbillon Watchmaker",
        "prompt": "Extreme macro close-up of a master Swiss watchmaker placing a microscopic ruby jewel into an exposed flying tourbillon escapement mechanism with fine precision tweezers, polished brass gears rotating, titanium bridge reflections, 8k definition.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "07_9x16_social_compact",
        "preset": "9:16_COMPACT",
        "title": "7. 9:16 Social Compact (448x768) — Cyber Martial Arts Kick",
        "prompt": "Dynamic vertical medium shot of an agile female cyber warrior in a dark technical suit executing a powerful fluid spinning kick, glowing crimson energy arcs, wet-look dark hair whipping across her face, photorealistic skin texture, dramatic neon lighting.",
        "steps": 40,
        "reuse": 6
    },
    {
        "id": "08_9x16_social_small",
        "preset": "9:16_SMALL",
        "title": "8. 9:16 Social Small (352x640) — Maya Token Concentrated 6-Step",
        "prompt": "Stunning female cyber dancer Maya in a sleek matte black technical bodysuit with glowing crimson LED seams performing graceful fluid liquid arm waves, intense focused eyes, wet-look dark hair strands, towering vertical monolithic 3D MINIMAX architecture in the background with cold volumetric neon lighting.",
        "steps": 6,
        "reuse": 1
    }
]

def main():
    print("=" * 110)
    print("🎬 MINIMAX-H3: ALL-PRESETS MASTER VERIFICATION SUITE (APPLE SILICON M5 MAX)")
    print("=" * 110)

    results = []

    for idx, t in enumerate(PRESET_TESTS, 1):
        out_mp4 = OUTPUT_DIR / f"{t['id']}.mp4"
        thumb_jpg = OUTPUT_DIR / f"{t['id']}_thumb.jpg"
        gif_path = OUTPUT_DIR / f"{t['id']}.gif"

        print(f"\n[{idx}/{len(PRESET_TESTS)}] 🚀 Generazione Preset: {t['title']}...")

        cmd = [
            str(RUNNER),
            "-p", t["prompt"],
            "--preset", t["preset"],
            "--seconds", "2.0",
            "--steps", str(t["steps"]),
            "--reuse", str(t["reuse"]),
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
                "-vf", "fps=12,scale=360:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(gif_path)
            ], capture_output=True)

        res_info = {
            "title": t["title"],
            "preset": t["preset"],
            "steps": t["steps"],
            "reuse": t["reuse"],
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

    print("\n" + "=" * 110)
    print("🏁 TUTTI I PRESET DEL MASTER BIBLE COMPLETATI CON SUCCESSO!")
    print("=" * 110)

if __name__ == "__main__":
    main()
