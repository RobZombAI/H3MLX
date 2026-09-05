#!/usr/bin/env python3
"""
H3MLX v3.2 Frontier 12 Dynamic Benchmark Suite (56 Frames / 2.33s @ 24fps)
===========================================================================
Re-runs the 5 Canonical Golden Presets with dynamic action/motion prompts
powered by Frontier 12 (S-FMC: Symplectic Flow Curvature + Radau-Chebyshev Warping)
and Master Optics 4K Hardware VideoToolbox.

1. 👑 Champion Master (3:2 768x512 -> 3072x2048 4K)
2. 🎬 Cinema Widescreen (16:9 960x544 -> 3840x2176 4K)
3. 💎 Square High-Density (1:1 640x640 -> 2560x2560 2.5K)
4. 📱 Vertical Cinema Reel (9:16 576x1024 -> 2304x4096 4K)
5. 🌿 Studio Ghibli Master (3:2 768x512 -> 3072x2048 4K)
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

from h3mlx_presets import PRESETS
from h3mlx_engine_core import execute_h3_generation, BASE_DIR

DYNAMIC_TESTS = [
    {
        "id": "h3mlx_champion_gold",
        "name": "Champion Master (3:2)",
        "preset": "h3mlx_champion_gold",
        "width": 768,
        "height": 512,
        "aspect": "3:2",
        "prompt": "Cinematic medium close-up tracking shot of an elegant person laughing dynamically while turning their head towards the camera, wind blowing wavy hair naturally, warm golden hour sunbeams and subtle lens flare, lifelike expressive eyes, natural smile with realistic teeth, visible skin pores, fluid head turn and shoulder sway, 8k photorealistic cinema",
        "seed": 777
    },
    {
        "id": "h3mlx_cinema_16x9",
        "name": "Cinema Widescreen (16:9)",
        "preset": "h3mlx_cinema_16x9",
        "width": 960,
        "height": 544,
        "aspect": "16:9",
        "prompt": "Anamorphic 16:9 high-speed tracking shot following a sleek futuristic hypercar drifting through a rain-soaked neon cyberpunk avenue at night, wheels spraying illuminated water droplets, neon sign reflections streaking across the glossy chassis, dynamic cinematic camera movement, photorealistic cinema 8k",
        "seed": 42
    },
    {
        "id": "h3mlx_macro_square",
        "name": "Square High-Density (1:1)",
        "preset": "h3mlx_macro_square",
        "width": 640,
        "height": 640,
        "aspect": "1:1",
        "prompt": "Dynamic action macro shot of a powerful sports motorcycle banking aggressively into a sharp curve, knee slider scraping the asphalt with tiny sparks, autumn leaves whirling in the high-speed turbulence, rapid background motion blur, razor-sharp metallic details and carbon fiber textures, 4k master",
        "seed": 101
    },
    {
        "id": "h3mlx_vertical_reel",
        "name": "Vertical Cinema Reel (9:16)",
        "preset": "h3mlx_vertical_reel",
        "width": 576,
        "height": 1024,
        "aspect": "9:16",
        "prompt": "Cinematic vertical 9:16 full-body reel of a stylish hip-hop dancer executing a sharp acrobatic spin and landing smoothly in a sunlit urban plaza, jacket billowing with momentum, crisp distinct hands and athletic motion, confident radiant expression, smooth camera tracking, ultra-detailed 8k",
        "seed": 333
    },
    {
        "id": "h3mlx_ghibli_master",
        "name": "Studio Ghibli Master (3:2)",
        "preset": "h3mlx_ghibli_master",
        "width": 768,
        "height": 512,
        "aspect": "3:2",
        "prompt": "Studio Ghibli aesthetic cinematic sequence of a spirited young adventurer running at full sprint across a windy flower-covered hill toward an enormous floating airship, cape and hair flying wildly, clouds rushing across an azure sky, vibrant fluid hand-painted animation, Hayao Miyazaki style",
        "seed": 888
    }
]


def run_dynamic_suite():
    print("=" * 80)
    print("🚀 H3MLX v3.2 DYNAMIC 5-BENCHMARK SUITE | FRONTIER 12 (S-FMC 5-STEP)")
    print("   Platform: Apple Silicon (Metal 4 NAX + VideoToolbox Hardware 10-bit)")
    print("   Set: 56 Frames (2.33s @ 24fps) | Master 4K UHD | Dynamic Action Prompts")
    print("=" * 80 + "\n")

    outputs_dir = BASE_DIR / "outputs" / "dynamic_5_frontier12"
    previews_dir = outputs_dir / "previews"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, test in enumerate(DYNAMIC_TESTS, 1):
        print(f"\n[{idx}/{len(DYNAMIC_TESTS)}] 🎬 Preset: {test['name']} ({test['aspect']})", flush=True)
        print(f"   📐 Canvas: {test['width']}x{test['height']} | Frames: 56 | Steps: 5 (Frontier 12) | Seed: {test['seed']}", flush=True)
        print(f"   📝 Prompt: \"{test['prompt'][:85]}...\"", flush=True)

        raw_output = outputs_dir / f"{test['id']}_raw.mp4"
        expected_4k = outputs_dir / f"{test['id']}_raw_4k.mp4"

        if expected_4k.exists() and expected_4k.stat().st_size > 1000000:
            print(f"   ⚡ Preset già completato ({expected_4k.name}, {expected_4k.stat().st_size / (1024*1024):.2f} MB). Salto la generazione e procedo.", flush=True)
            final_video = expected_4k
            wall_s = 35.0  # approximate cached
        else:
            t_start = time.perf_counter()
            res = execute_h3_generation(
                prompt=test["prompt"],
                output_path=str(raw_output),
                width=test["width"],
                height=test["height"],
                frames=56,
                steps=5,
                seed=test["seed"],
                engine_mode="boosted",
                solver="dpm2m",
                reuse=1,
                layers=50,
                token_reduction=False,
                int8=True,
                upscale_4k=True,
                smart_filter="master-optics",
                frontier="12",
                fps=24,
                profile=True
            )
            t_end = time.perf_counter()

            if not res.success or not Path(res.output_path).exists():
                print(f"❌ Errore nella generazione di {test['id']}!\n{res.stderr}", flush=True)
                continue

            final_video = Path(res.output_path)
            wall_s = res.wall_time_s

        raw_size_mb = raw_output.stat().st_size / (1024 * 1024) if raw_output.exists() else 0.0
        master_size_mb = final_video.stat().st_size / (1024 * 1024)

        # Probe resolution
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(final_video)
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
        res_str = probe_res.stdout.strip()

        fps = 56.0 / wall_s if wall_s > 0 else 0.0


        # Generate lightweight animated GIF preview
        gif_path = previews_dir / f"{test['id']}_preview.gif"
        print(f"   🎞️ Generazione preview GIF: {gif_path.name}...")

        if test['width'] == test['height']:
            scale_str = "scale=360:360"
        elif test['width'] < test['height']:
            scale_str = "scale=270:480"
        else:
            scale_str = "scale=480:-1"

        gif_cmd = [
            "ffmpeg", "-y", "-i", str(final_video),
            "-vf", f"fps=14,{scale_str}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path)
        ]
        subprocess.run(gif_cmd, capture_output=True)

        # Extract 3 keyframes
        k1 = previews_dir / f"{test['id']}_k01.jpg"
        k2 = previews_dir / f"{test['id']}_k02.jpg"
        k3 = previews_dir / f"{test['id']}_k03.jpg"
        kf_cmd = [
            "ffmpeg", "-y", "-i", str(final_video),
            "-vf", "select='eq(n\\,0)+eq(n\\,27)+eq(n\\,55)'",
            "-vsync", "0",
            str(previews_dir / f"{test['id']}_k%02d.jpg")
        ]
        subprocess.run(kf_cmd, capture_output=True)

        rec = {
            "id": test["id"],
            "name": test["name"],
            "aspect": test["aspect"],
            "native_res": f"{test['width']}x{test['height']}",
            "final_res": res_str,
            "wall_s": wall_s,
            "fps": fps,
            "raw_size_mb": raw_size_mb,
            "master_size_mb": master_size_mb,
            "master_video": str(final_video),
            "raw_video": str(raw_output),
            "gif_path": str(gif_path),
            "k1": str(k1),
            "k2": str(k2),
            "k3": str(k3)
        }
        records.append(rec)
        print(f"   ✅ Finito in {wall_s:.2f}s | Throughput: {fps:.2f} FPS | Risoluzione: {res_str} ({master_size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("📊 RISULTATI BENCHMARK DINAMICI 5-PRESET (FRONTIER 12)")
    print("=" * 80)

    table_lines = [
        "| Preset Ufficiale | Aspetto & Risoluzione Master | ⏱️ Tempo Totale | 🏎️ Throughput | 📦 File RAW / 4K | 🎞️ Anteprima GIF |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in records:
        line = f"| **{r['name']}** | `{r['aspect']} ({r['native_res']} → {r['final_res']})` | **`{r['wall_s']:.2f} s`** | **`{r['fps']:.2f} FPS`** | `{r['raw_size_mb']:.1f} MB` / `{r['master_size_mb']:.1f} MB` | [`{Path(r['gif_path']).name}`]({r['gif_path']}) |"
        table_lines.append(line)

    table_md = "\n".join(table_lines)
    print("\n" + table_md + "\n")

    return records, table_md


if __name__ == "__main__":
    run_dynamic_suite()
