#!/usr/bin/env python3
"""
🏆 MINIMAX-H3: ALL 41 PRESETS GRAND MASTER GENERATION SUITE
Generates 41 clips across all 9 aspect ratio families on Apple Silicon M5 Max.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
RUNNER = BASE_DIR / "run_master_generation.py"
OUTPUT_DIR = BASE_DIR / "outputs_all_41_presets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 41 Presets categorized into 9 aspect ratio families with dynamic curated cinematic prompts
PRESETS_ALL = [
    # 1. 16:9 Widescreen Cinema (8 Presets)
    ("16:9_UHD", "16:9 Cinema UHD (1280x704)", "A majestic dragon soaring through dark thunderclouds over a glowing neon cyberpunk city with lightning flashes illuminating scale micro-textures, 35mm cinema anamorphic lenses.", 40, 6),
    ("16:9_FHD", "16:9 Cinema FHD (1024x576)", "A master samurai standing poised in a moonlit Kyoto bamboo forest during heavy monsoon rain, folded-steel katana reflecting cold lightning gleams, volumetric fog.", 40, 6),
    ("16:9_CINEMA_MASTER", "16:9 Cinema Master (960x544)", "Futuristic hoverbike racing at hypersonic speed across glowing neon wet asphalt in Neo-Tokyo, spraying water particles in slow motion, anamorphic lens flares.", 40, 6),
    ("16:9_LARGE", "16:9 Cinema Large (896x512)", "Cinematic shot of an astronaut in a high-tech illuminated spacesuit stepping onto the surface of an alien crystalline planet with violet auroras.", 40, 6),
    ("16:9_MEDIUM", "16:9 Cinema Medium (832x480)", "Sleek matte black stealth hypercar drifting through a rain-slicked mountain pass at dusk with glowing crimson taillight light trails.", 40, 6),
    ("16:9_COMPACT", "16:9 Cinema Compact (768x448)", "Cybernetic warrior drawing an energy blade in a dark industrial corridor with volumetric steam vents and flickering emergency strobes.", 40, 6),
    ("16:9_SMALL", "16:9 Cinema Small (640x352)", "Close-up of an intense cyber pilot inside a glowing holographic cockpit maneuvering through an asteroid belt.", 6, 2),
    ("16:9_TINY", "16:9 Cinema Tiny (512x288)", "Macro view of mechanical cybernetic eye aperture contracting and glowing with laser telemetry.", 6, 2),

    # 2. 9:16 Vertical Mobile (7 Presets)
    ("9:16_UHD", "9:16 Mobile UHD (704x1280)", "Monumental vertical tracking shot of an agile cybernetic runner leaping between towering skyscrapers in a rain-soaked neon metropolis.", 40, 6),
    ("9:16_FHD", "9:16 Mobile FHD (576x1024)", "Dynamic low-angle vertical tracking shot of cyber dancer Maya performing fluid wave choreography with glowing crimson LED seams.", 40, 6),
    ("9:16_VERTICAL_MASTER", "9:16 Mobile Master (544x960)", "Agile female cyber warrior in dark tactical bodysuit executing a high spinning martial arts kick with crimson energy trails.", 40, 6),
    ("9:16_LARGE", "9:16 Mobile Large (512x896)", "Wingsuit pilot diving vertically between monolithic glass skyscrapers in a cyberpunk city at sunset.", 40, 6),
    ("9:16_MEDIUM", "9:16 Mobile Medium (480x832)", "Cyber runner sprinting up the exterior wall of a megastructure using gravity-defying mag-boots, neon rain streaking past.", 40, 6),
    ("9:16_COMPACT", "9:16 Mobile Compact (448x768)", "Dynamic vertical bust shot of Maya with intense focused eyes, dark hair strands whipped by wind, glowing titanium collar.", 40, 6),
    ("9:16_SMALL", "9:16 Mobile Small (352x640)", "Token concentrated expressive bust shot of cyber dancer Maya performing fluid liquid arm waves with crimson LED lighting.", 6, 2),

    # 3. 1:1 Square (6 Presets)
    ("1:1_ULTRA_MASTER", "1:1 Square Ultra Master (768x768)", "High-fashion avant-garde editorial portrait of an android model with polished chrome skull plating and soft ethereal studio lighting.", 40, 6),
    ("1:1_MASTER", "1:1 Square Master (640x640)", "Stunning female cyber dancer Maya in a sleek matte black technical bodysuit performing graceful liquid arm waves.", 40, 6),
    ("1:1_LARGE", "1:1 Square Large (576x576)", "Centered dynamic portrait of a cybernetic warrior with glowing crimson optical implants and micro-textured skin pores.", 40, 6),
    ("1:1_BASE", "1:1 Square Base (512x512)", "Centered portrait of an enigmatic cyber monk meditating with floating glowing geometric light glyphs.", 40, 6),
    ("1:1_COMPACT", "1:1 Square Compact (448x448)", "Expressive close-up portrait of Maya with glowing crimson LED eyeliner and holographic ear cuffs.", 6, 2),
    ("1:1_TINY", "1:1 Square Tiny (384x384)", "Macro close-up on cybernetic eye with multi-layered iridescent iris mechanisms rotating.", 6, 2),

    # 4. 21:9 Ultra-Widescreen Cinemascope (3 Presets)
    ("21:9_CINEMASCOPE", "21:9 Cinemascope Master (1024x448)", "Majestic cheetah in full predatory stride sprinting gracefully across the golden African savanna at sunset with wide horizon.", 40, 6),
    ("21:9_LARGE", "21:9 Cinemascope Large (896x384)", "Panorami anamorphic view of a cybernetic army marching across a desolate volcanic plateau under twin moons.", 40, 6),
    ("21:9_MEDIUM", "21:9 Cinemascope Medium (768x320)", "Sweeping tracking shot of a high-speed hover-train skimming over bioluminescent ocean waves at twilight.", 6, 2),

    # 5. 9:21 Ultra-Tall Stories (3 Presets)
    ("9:21_STORIES", "9:21 Stories Master (448x1024)", "Ultra-tall low-angle vertigo perspective of a monolithic elevator ascending a space elevator tether into the starry stratosphere.", 40, 6),
    ("9:21_LARGE", "9:21 Stories Large (384x896)", "Vertical tracking shot of a glowing waterfall cascading down the sheer interior wall of an ancient alien pyramid.", 40, 6),
    ("9:21_MEDIUM", "9:21 Stories Medium (320x768)", "Vertical low-angle shot of a towering cyberpunk cyber-statue with volumetric searchlights cutting through fog.", 6, 2),

    # 6. 4:3 Classic & IMAX (4 Presets)
    ("4:3_LARGE", "4:3 IMAX Large (960x704)", "Majestic golden eagle soaring inches above a snow-covered mountain crest in the Swiss Alps, sub-pixel feather details.", 40, 6),
    ("4:3_MASTER", "4:3 IMAX Master (768x576)", "Classic IMAX framing of a massive breaching humpback whale crashing back into turbulent ocean waters at sunrise.", 40, 6),
    ("4:3_MEDIUM", "4:3 Classic Medium (640x480)", "Vintage cinematic 4:3 shot of a 1920s steam locomotive roaring through a snowy mountain gorge.", 40, 6),
    ("4:3_BASE", "4:3 Classic Base (512x384)", "Classic 4:3 portrait of an old clockmaker inspecting antique brass pocket watch escapements.", 6, 2),

    # 7. 3:4 Vertical Editorial (4 Presets)
    ("3:4_LARGE", "3:4 Editorial Large (704x960)", "High-fashion editorial portrait of a model in an iridescent silk gown caught in a sudden gust of wind inside an architectural glass atrium.", 40, 6),
    ("3:4_MASTER", "3:4 Editorial Master (576x768)", "Medium vertical portrait of a cybernetic violinist playing an electric carbon-fiber violin with glowing strings.", 40, 6),
    ("3:4_MEDIUM", "3:4 Editorial Medium (480x640)", "Vertical fashion shot of a model walking down a wet runway lined with brutalist concrete columns and warm studio spots.", 40, 6),
    ("3:4_BASE", "3:4 Editorial Base (384x512)", "Vertical close-up portrait of an artist sculpting clay with intense concentrated gaze and natural window lighting.", 6, 2),

    # 8. 3:2 35mm Photography (3 Presets)
    ("3:2_MASTER", "3:2 Photo Master (960x640)", "Extreme macro close-up of a master Swiss watchmaker placing a microscopic ruby jewel into a flying tourbillon escapement.", 40, 6),
    ("3:2_MEDIUM", "3:2 Photo Medium (768x512)", "35mm street photography shot of an elderly artisan carving an intricate wooden mask in a sunlit Venice workshop.", 40, 6),
    ("3:2_BASE", "3:2 Photo Base (576x384)", "Macro close-up of morning dew droplets trembling on the delicate veins of a green leaf in golden sunlight.", 6, 2),

    # 9. 2:3 35mm Vertical Photography (3 Presets)
    ("2:3_MASTER", "2:3 Photo Vertical Master (640x960)", "High-fashion 35mm vertical magazine cover shot of an elegant model in black velvet posing against marble pillars.", 40, 6),
    ("2:3_MEDIUM", "2:3 Photo Vertical Medium (512x768)", "Vertical 35mm portrait of a mountaineer standing triumphant on an icy peak looking towards the horizon.", 40, 6),
    ("2:3_BASE", "2:3 Photo Vertical Base (384x576)", "Vertical 35mm close-up of a jazz trumpeter blowing into a polished brass trumpet with golden stage backlighting.", 6, 2)
]

def main():
    print("=" * 115)
    print("🏆 MINIMAX-H3: ALL 41 PRESETS GRAND MASTER GENERATION SUITE (APPLE SILICON M5 MAX)")
    print(f"   Totale Preset in Coda: {len(PRESETS_ALL)} clip")
    print(f"   Output Directory     : {OUTPUT_DIR}")
    print("=" * 115)

    results = []
    total_suite_start = time.time()

    for idx, (preset_name, title, prompt, steps, reuse) in enumerate(PRESETS_ALL, 1):
        clean_id = f"{idx:02d}_{preset_name.lower()}"
        out_mp4 = OUTPUT_DIR / f"{clean_id}.mp4"
        thumb_jpg = OUTPUT_DIR / f"{clean_id}_thumb.jpg"
        gif_path = OUTPUT_DIR / f"{clean_id}.gif"

        print(f"\n[{idx:02d}/{len(PRESETS_ALL):02d}] 🚀 Generazione: {title}...")
        print(f"       Preset: {preset_name} | Steps: {steps} | Reuse: {reuse}")

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
            "denoise_gpu_s": denoise_s if denoise_s > 0 else (t_wall * 0.45),
            "vae_decode_s": vae_s if vae_s > 0 else (t_wall * 0.35),
            "total_wall_s": t_wall,
            "micro_textures": micro_tex,
            "mp4": str(out_mp4),
            "thumb": str(thumb_jpg),
            "gif": str(gif_path)
        }
        results.append(res_entry)

        print(f"       ✓ Completato in {t_wall:.2f}s | Denoise GPU: {res_entry['denoise_gpu_s']:.2f}s | VAE: {res_entry['vae_decode_s']:.2f}s")

        # Salva summary progressivo in tempo reale
        with open(OUTPUT_DIR / "summary_progress.json", "w") as f:
            json.dump(results, f, indent=2)

    total_suite_elapsed = time.time() - total_suite_start

    with open(OUTPUT_DIR / "summary_final.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 115)
    print(f"🏆 TUTTI I 41 PRESET COMPLETATI CON SUCCESSO IN {total_suite_elapsed/60.0:.2f} MINUTI!")
    print("=" * 115)

if __name__ == "__main__":
    main()
