#!/usr/bin/env python3
"""
H3MLX v3.0 Official Benchmark Suite (4 Seconds / 90 Frames @ 24fps)
====================================================================
Systematically benchmarks the 5 Golden Presets on Apple Silicon M5 Max:
1. 👑 Champion Master Gold (3:2 768x512 -> 3072x2048 4K)
2. 🎬 Cinema Anamorphic (16:9 960x544 -> 3840x2176 4K)
3. 💎 Square High-Density (1:1 640x640 -> 2560x2560 2.5K)
4. 📱 Vertical Cinema Reel (9:16 576x1024 -> 2304x4096 4K)
5. 🌿 Studio Ghibli Master (3:2 768x512 -> 3072x2048 4K)

Saves 4-second videos, generates animated GIF previews, and writes BENCHMARKS.md.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

from h3mlx_presets import PRESETS
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, BASE_DIR

GOLDEN_PRESET_IDS = [
    "h3mlx_champion_gold",
    "h3mlx_cinema_16x9",
    "h3mlx_macro_square",
    "h3mlx_vertical_reel",
    "h3mlx_ghibli_master"
]

def run_benchmarks():
    print("=" * 80)
    print("🚀 H3MLX v3.0 OFFICIAL BENCHMARK RUNNER (4 SECONDS / 90 FRAMES)")
    print("   Platform: Apple Silicon M5 Max (128GB Unified Memory, >400 GB/s)")
    print("=" * 80 + "\n")
    
    model_path = resolve_model_path(steps=8)
    print(f"📦 Checkpoint: {model_path.name}")
    print(f"🎯 Presets da testare: {len(GOLDEN_PRESET_IDS)}\n")
    
    assets_dir = BASE_DIR / "assets"
    outputs_dir = BASE_DIR / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    for idx, pid in enumerate(GOLDEN_PRESET_IDS, 1):
        cfg = PRESETS[pid]
        print(f"\n[{idx}/{len(GOLDEN_PRESET_IDS)}] 🎬 Esecuzione Benchmark: {cfg['name']}")
        print(f"   📐 Dimensioni: {cfg['width']}x{cfg['height']} | Layer: 50 | Step: 8 | Frames: 90 (4.0s @ 24fps)")
        print(f"   📝 Prompt: \"{cfg['prompt']}\"")
        
        out_mp4 = outputs_dir / f"benchmark_v3_{pid}.mp4"
        
        t_start = time.perf_counter()
        res = execute_h3_generation(
            prompt=cfg["prompt"],
            output_path=str(out_mp4),
            width=cfg["width"],
            height=cfg["height"],
            frames=90,  # 4 seconds exact causal temporal lattice (5*17 + 5)
            steps=cfg.get("steps", 8),
            seed=42,
            engine_mode=cfg.get("mode", "boosted"),
            solver=cfg.get("solver", "dpm3m"),
            reuse=cfg.get("reuse", 1),
            layers=cfg.get("layers", 50),
            token_reduction=False,
            int8=cfg.get("int8", True),
            upscale_4k=True,
            smart_filter="auto",
            profile=True
        )
        t_end = time.perf_counter()
        
        if not res.success or not Path(res.output_path).exists():
            print(f"❌ Errore nella generazione di {pid}!\n{res.stderr}")
            continue
            
        final_video = Path(res.output_path)
        file_size_mb = final_video.stat().st_size / (1024 * 1024)
        
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
        
        # Extract timings from profiling
        denoise_s = res.profile_data.get("denoise_s", 0.0)
        vae_s = res.profile_data.get("vae_decode_s", 0.0)
        wall_s = res.wall_time_s
        fps = 90.0 / wall_s if wall_s > 0 else 0.0
        
        # Generate lightweight animated GIF preview for GitHub table
        gif_path = assets_dir / f"preview_v3_{pid}.gif"
        print(f"   🎞️  Generazione preview animata GIF: {gif_path.name}...")
        gif_cmd = [
            "ffmpeg", "-y", "-i", str(final_video),
            "-vf", "fps=12,scale=360:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path)
        ]
        subprocess.run(gif_cmd, capture_output=True)
        
        rec = {
            "id": pid,
            "name": cfg["name"],
            "native_res": f"{cfg['width']}x{cfg['height']}",
            "final_res": res_str,
            "prompt": cfg["prompt"],
            "denoise_s": denoise_s,
            "vae_s": vae_s,
            "total_s": wall_s,
            "fps": fps,
            "size_mb": file_size_mb,
            "video_path": str(final_video),
            "gif_name": gif_path.name
        }
        records.append(rec)
        print(f"   ✅ Completato in {wall_s:.2f}s | Throughput: {fps:.2f} FPS | Risoluzione: {res_str} ({file_size_mb:.2f} MB)")
        
    print("\n" + "=" * 80)
    print("📊 RISULTATI BENCHMARK UFFICIALI H3MLX v3.0 (4 SECONDI / 90 FRAMES)")
    print("=" * 80)
    
    # Generate Markdown Table
    table_lines = [
        "| Preset Ufficiale | Risoluzione & 4K | ⚡ Denoise GPU | 💎 3D VAE | ⏱️ Tempo Totale (4s / 90fr) | 🏎️ Throughput | 🎥 Anteprima Video |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for r in records:
        line = f"| **{r['name']}** | `{r['native_res']} → {r['final_res']}` | `{r['denoise_s']:.2f}s` | `{r['vae_s']:.2f}s` | **`{r['total_s']:.2f}s`** | **`{r['fps']:.2f} FPS`** | [▶ Guarda Video](outputs/{Path(r['video_path']).name}) |"
        table_lines.append(line)
        
    table_md = "\n".join(table_lines)
    print("\n" + table_md + "\n")
    
    return records, table_md

if __name__ == "__main__":
    run_benchmarks()
