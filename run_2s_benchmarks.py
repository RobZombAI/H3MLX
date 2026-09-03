#!/usr/bin/env python3
"""
H3MLX v3.1 Official 2-Second Benchmark Suite (56 Frames @ 24fps)
================================================================
Systematically benchmarks the 5 Golden Presets on Apple Silicon M5 Max:
1. 👑 Champion Master Gold: 768x512 (3:2) -> 3072x2048 True 35mm Master
2. 🎬 Cinema Anamorphic: 960x544 (16:9) -> 3840x2176 True Anamorphic Master
3. 💎 Square High-Density: 640x640 (1:1) -> 2560x2560 True Square Master
4. 📱 Vertical Cinema Reel: 576x1024 (9:16) -> 2304x4096 True Vertical Master
5. 🌿 Studio Ghibli Master: 768x512 (3:2) -> 3072x2048 True 35mm Master
"""

import sys
import os
import time
import subprocess
from pathlib import Path

from h3mlx_presets import PRESETS
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, BASE_DIR

GOLDEN_PRESET_IDS = [
    "h3mlx_champion_gold",
    "h3mlx_cinema_16x9",
    "h3mlx_macro_square",
    "h3mlx_vertical_reel",
    "h3mlx_ghibli_master"
]

def run_2s_benchmarks():
    print("=" * 80)
    print("🚀 H3MLX v3.1 OFFICIAL 2-SECOND BENCHMARK SUITE (56 FRAMES @ 24FPS)")
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
        print(f"\n[{idx}/{len(GOLDEN_PRESET_IDS)}] 🎬 Esecuzione Benchmark 2s: {cfg['name']}")
        print(f"   📐 Dimensioni: {cfg['width']}x{cfg['height']} | Layer: 50 | Step: 8 | Frames: 56 (2.33s @ 24fps)")
        print(f"   📝 Prompt: \"{cfg['prompt']}\"")
        
        out_mp4 = outputs_dir / f"benchmark_2s_{pid}.mp4"
        
        t_start = time.perf_counter()
        res = execute_h3_generation(
            prompt=cfg["prompt"],
            output_path=str(out_mp4),
            width=cfg["width"],
            height=cfg["height"],
            frames=56,  # 2.33 seconds exact causal temporal lattice (3*17 + 5)
            steps=cfg.get("steps", 8),
            seed=42,
            engine_mode=cfg.get("mode", "boosted"),
            solver="dpm2m",
            reuse=1,
            layers=50,
            token_reduction=False,
            int8=True,
            upscale_4k=True,
            smart_filter="auto",
            profile=True
        )
        t_end = time.perf_counter()
        
        if not res.success or not Path(res.output_path).exists():
            print(f"❌ Errore nella generazione di {pid}!\n{res.stderr}")
            continue
            
        final_video = Path(res.output_path)
        raw_video = outputs_dir / f"benchmark_2s_{pid}.mp4"
        raw_size_mb = raw_video.stat().st_size / (1024 * 1024) if raw_video.exists() else 0.0
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
        
        denoise_s = res.profile_data.get("denoise_s", 0.0)
        vae_s = res.profile_data.get("vae_decode_s", 0.0)
        wall_s = res.wall_time_s
        fps = 56.0 / wall_s if wall_s > 0 else 0.0
        
        # Generate lightweight animated GIF preview for GitHub table
        gif_path = assets_dir / f"preview_v3_{pid}.gif"
        print(f"   🎞️  Generazione preview animata GIF: {gif_path.name}...")
        
        # Scale GIF preserving aspect ratio
        if cfg['width'] == cfg['height']:
            scale_str = "scale=360:360"
        elif cfg['width'] < cfg['height']:
            scale_str = "scale=270:480"
        else:
            scale_str = "scale=480:-1"
            
        gif_cmd = [
            "ffmpeg", "-y", "-i", str(final_video),
            "-vf", f"fps=12,{scale_str}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path)
        ]
        subprocess.run(gif_cmd, capture_output=True)
        
        rec = {
            "id": pid,
            "name": cfg["name"],
            "aspect": "3:2" if cfg['width']==768 else ("16:9" if cfg['width']==960 else ("1:1" if cfg['width']==640 else "9:16")),
            "native_res": f"{cfg['width']}x{cfg['height']}",
            "final_res": res_str,
            "prompt": cfg["prompt"],
            "denoise_s": denoise_s,
            "vae_s": vae_s,
            "total_s": wall_s,
            "fps": fps,
            "raw_size_mb": raw_size_mb,
            "master_size_mb": file_size_mb,
            "video_path": str(final_video),
            "gif_name": gif_path.name
        }
        records.append(rec)
        print(f"   ✅ Completato in {wall_s:.2f}s (Denoise GPU: {denoise_s:.2f}s) | Throughput: {fps:.2f} FPS | Risoluzione Reale: {res_str} ({file_size_mb:.2f} MB)")
        
    print("\n" + "=" * 80)
    print("📊 RISULTATI BENCHMARK UFFICIALI H3MLX v3.1 (2 SECONDI / 56 FRAMES)")
    print("=" * 80)
    
    table_lines = [
        "| Preset Ufficiale | Aspect & Risoluzione Reale | ⚡ Denoise GPU | ⏱️ Tempo Totale (56 fr / 2.3s) | 🏎️ Throughput | 📦 Dimensioni (RAW / Master) | 🎞️ Anteprima Animata |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for r in records:
        line = f"| **{r['name']}** | `{r['aspect']} ({r['native_res']} → {r['final_res']})` | **`{r['denoise_s']:.2f} s`** | **`{r['total_s']:.2f} s`** | **`{r['fps']:.2f} FPS`** | `{r['raw_size_mb']:.1f} MB` / `{r['master_size_mb']:.1f} MB` | ![{r['name']}](assets/{r['gif_name']}) |"
        table_lines.append(line)
        
    table_md = "\n".join(table_lines)
    print("\n" + table_md + "\n")
    
    return records, table_md

if __name__ == "__main__":
    run_2s_benchmarks()
