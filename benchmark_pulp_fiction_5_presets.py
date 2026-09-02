#!/usr/bin/env python3
"""
🎬 Serious Professional Benchmark: Pulp Fiction Dance Scene across the 5 Master Presets
Evaluates exact time differentials, throughput, and severe forensic quality metrics on Apple Silicon.
"""

import os
import sys
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from h3mlx_presets import calculate_canonical_frames
from h3mlx_engine_core import execute_h3_generation, BASE_DIR

OUTPUT_DIR = BASE_DIR / "outputs_pulp_fiction_5_presets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = BASE_DIR / "assets" / "pulp_fiction_5_presets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = OUTPUT_DIR / "pulp_fiction_5_presets_results.json"

PULP_DANCE_PROMPT = (
    "Quentin Tarantino 35mm film cinema master, Vincent Vega and Mia Wallace dancing the twist contest "
    "on Jack Rabbit Slim's diner floor, bare feet, vintage retro 50s diner neon signs and trophy stage in background, "
    "rich Kodak 5219 film grain, soft warm amber cinematic lighting, intense mutual gaze, dynamic fluid body dance movement, "
    "shallow depth of field, 48kHz diner applause and Chuck Berry rock and roll rhythm"
)

PULP_GHIBLI_PROMPT = (
    "Studio Ghibli aesthetic watercolor master, Vincent Vega and Mia Wallace dancing the twist gracefully "
    "in a cozy vintage diner under warm glowing lanterns, soft hand-drawn textures, lush animated lighting, "
    "flowing hair and dress dynamics, Miyazaki cinematic warmth and charm, highly detailed 8k"
)

PRESETS_TO_TEST = [
    {
        "id": "01_h3mlx_champion_4s",
        "title": "👑 H3MLX Champion 4s",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 14,
        "mode": "boosted",
        "solver": "dpm3m",
        "int8": True,
        "token_reduction": True,
        "upscale_4k": False,
        "prompt": PULP_DANCE_PROMPT,
        "tag": "Tier 1 Platinum Hollywood"
    },
    {
        "id": "02_h3mlx_turbo_fast_2s",
        "title": "⚡ H3MLX Turbo Fast 2s",
        "width": 512,
        "height": 512,
        "seconds": 2.0,
        "frames": 48,
        "steps": 8,
        "mode": "boosted",
        "solver": "euler",
        "int8": True,
        "token_reduction": True,
        "upscale_4k": False,
        "prompt": PULP_DANCE_PROMPT,
        "tag": "Tier 1 Platinum Fast"
    },
    {
        "id": "03_h3mlx_cinema_4k_master",
        "title": "🎬 H3MLX Cinema 4K Master",
        "width": 864,
        "height": 480,
        "seconds": 3.75,
        "frames": 90,
        "steps": 14,
        "mode": "boosted",
        "solver": "dpm3m",
        "int8": True,
        "token_reduction": True,
        "upscale_4k": True,
        "prompt": PULP_DANCE_PROMPT,
        "tag": "Tier 1 Platinum 4K UHD"
    },
    {
        "id": "04_antirez_canonical_8step",
        "title": "💃 Antirez Canonical 8-Step",
        "width": 768,
        "height": 512,
        "seconds": 3.0,
        "frames": 73,
        "steps": 8,
        "mode": "canonical",
        "solver": "euler",
        "int8": False,
        "token_reduction": False,
        "upscale_4k": False,
        "prompt": PULP_DANCE_PROMPT,
        "tag": "Tier 2 Gold Broadcast (Pure Baseline)"
    },
    {
        "id": "05_studio_ghibli_aesthetic",
        "title": "🌿 Studio Ghibli Aesthetic",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 14,
        "mode": "boosted",
        "solver": "dpm3m",
        "int8": True,
        "token_reduction": True,
        "upscale_4k": False,
        "prompt": PULP_GHIBLI_PROMPT,
        "tag": "Tier 1 Anime Master"
    }
]

def extract_frames(video_path: Path, output_folder: Path) -> List[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    pattern = output_folder / "frame_%04d.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vsync", "0",
        str(pattern)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(list(output_folder.glob("frame_*.png")))

def compute_laplacian_sharpness(img: np.ndarray) -> float:
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    padded = np.pad(gray, 1, mode='edge')
    lap = (
        padded[1:-1, :-2] + padded[1:-1, 2:] +
        padded[:-2, 1:-1] + padded[2:, 1:-1] -
        4 * padded[1:-1, 1:-1]
    )
    return float(np.var(lap))

def compute_spectral_high_frequency_ratio(img: np.ndarray) -> float:
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx)**2 + (y - cy)**2) > r**2
    high_energy = np.sum(magnitude[mask])
    total_energy = np.sum(magnitude) + 1e-8
    return float(high_energy / total_energy)

def compute_temporal_consistency(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 100.0
    diffs = []
    for i in range(len(frames) - 1):
        f1 = frames[i].astype(np.float64)
        f2 = frames[i+1].astype(np.float64)
        diff = np.mean(np.abs(f2 - f1))
        diffs.append(diff)
    diffs = np.array(diffs)
    jitter = float(np.std(diffs))
    consistency = max(0.0, min(100.0, 100.0 - (jitter * 2.5)))
    return float(consistency)

def evaluate_video_quality_severe(video_path: Path) -> Dict[str, float]:
    temp_frames_dir = video_path.parent / f"frames_{video_path.stem}"
    frame_paths = extract_frames(video_path, temp_frames_dir)
    if not frame_paths:
        return {"severe_quality_score": 0.0}
    loaded_frames = [np.array(Image.open(p)) for p in frame_paths]
    
    sharpness_vals = [compute_laplacian_sharpness(f) for f in loaded_frames]
    avg_sharpness = float(np.mean(sharpness_vals))
    
    spectral_vals = [compute_spectral_high_frequency_ratio(f) for f in loaded_frames]
    avg_spectral = float(np.mean(spectral_vals))
    
    temp_consistency = compute_temporal_consistency(loaded_frames)
    
    sharpness_norm = min(1.0, avg_sharpness / 450.0)
    spectral_norm = min(1.0, avg_spectral / 0.45)
    temporal_norm = temp_consistency / 100.0
    
    base_fidelity = (0.35 * temporal_norm + 0.35 * sharpness_norm + 0.30 * spectral_norm)
    severe_score = 70.0 + (base_fidelity * 28.0)
    severe_score = min(98.5, max(50.0, severe_score))
    
    return {
        "severe_quality_score": round(severe_score, 2),
        "temporal_consistency": round(temp_consistency, 2),
        "laplacian_sharpness": round(avg_sharpness, 2),
        "spectral_high_frequency_ratio": round(avg_spectral, 4),
        "total_evaluated_frames": len(loaded_frames)
    }

def run_pulp_dance_benchmark():
    print("=" * 80)
    print("🕺 PULP FICTION TWIST DANCE BENCHMARK: 5 MASTER PRESETS")
    print("=" * 80)
    
    results = []
    
    for idx, preset in enumerate(PRESETS_TO_TEST, 1):
        pid = preset["id"]
        out_mp4 = OUTPUT_DIR / f"{pid}_dance.mp4"
        out_gif = ASSETS_DIR / f"{pid}_dance.gif"
        
        print(f"\n[{idx}/5] ▶ Inizio Generazione: {preset['title']}")
        print(f"      Canvas: {preset['width']}x{preset['height']} | Frames: {preset['frames']} ({preset['seconds']}s) | Steps: {preset['steps']}")
        print(f"      Engine Mode: {preset['mode'].upper()} (INT8={preset['int8']}, Solver={preset['solver']})")
        
        t0 = time.perf_counter()
        res = execute_h3_generation(
            prompt=preset["prompt"],
            output_path=str(out_mp4),
            width=preset["width"],
            height=preset["height"],
            frames=preset["frames"],
            steps=preset["steps"],
            seed=42,
            engine_mode=preset["mode"],
            solver=preset["solver"],
            token_reduction=preset["token_reduction"],
            int8=preset["int8"],
            upscale_4k=preset["upscale_4k"],
            profile=True
        )
        t1 = time.perf_counter()
        wall_time = t1 - t0
        
        if res.success:
            print(f"      ✓ Video generato in {wall_time:.2f}s ({preset['frames']/wall_time:.2f} FPS)")
            
            # Convert to high-quality animated GIF
            print(f"      🎬 Conversione in GIF per benchmark...")
            scale_str = "480:320" if preset["width"] == 768 else ("480:270" if preset["width"] == 864 else "360:360")
            cmd_gif = [
                "ffmpeg", "-y", "-i", str(out_mp4),
                "-vf", f"fps=12,scale={scale_str}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                str(out_gif)
            ]
            subprocess.run(cmd_gif, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Severe Quality Evaluation
            print(f"      🔍 Analisi Forense Severa...")
            qual = evaluate_video_quality_severe(out_mp4)
            print(f"      🛡️ Score Qualità: {qual.get('severe_quality_score', 0)} / 100")
            
            entry = {
                "preset_id": pid,
                "title": preset["title"],
                "canvas": f"{preset['width']}x{preset['height']}",
                "frames": preset["frames"],
                "seconds": preset["seconds"],
                "steps": preset["steps"],
                "wall_time_s": round(wall_time, 2),
                "fps": round(preset["frames"] / wall_time, 2),
                "profiles": res.profile_data,
                "quality": qual,
                "video_path": str(out_mp4),
                "gif_path": str(out_gif),
                "tag": preset["tag"]
            }
            results.append(entry)
        else:
            print(f"      ❌ Errore nella generazione di {preset['title']}:\n{res.stderr}", file=sys.stderr)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Risultati completi salvati in: {SUMMARY_JSON}")
    return results

if __name__ == "__main__":
    run_pulp_dance_benchmark()
