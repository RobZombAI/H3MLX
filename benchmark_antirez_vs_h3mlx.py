#!/usr/bin/env python3
"""
🔬 Severe Forensic Benchmark Suite: Antirez Canonical h3.c vs H3MLX Boosted Engine
Evaluates exact time differentials, throughput, and world-class video quality metrics.
"""

import os
import sys
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames
from h3mlx_engine_core import execute_h3_generation, BASE_DIR

OUTPUT_DIR = BASE_DIR / "outputs_antirez_vs_h3mlx_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUTPUT_DIR / "benchmark_summary.json"

def extract_frames(video_path: Path, output_folder: Path) -> List[Path]:
    """Extracts video frames as PNGs for forensic quality analysis."""
    output_folder.mkdir(parents=True, exist_ok=True)
    pattern = output_folder / "frame_%04d.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vsync", "0",
        str(pattern)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(list(output_folder.glob("frame_*.png")))

def compute_frame_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio between two RGB images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(20 * math.log10(max_pixel / math.sqrt(mse)))

def compute_frame_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) on luminance channel."""
    # Convert RGB to Luminance (Y)
    y1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
    y2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(y1)
    mu2 = np.mean(y2)
    sigma1_sq = np.var(y1)
    sigma2_sq = np.var(y2)
    sigma12 = np.cov(y1.flat, y2.flat)[0, 1]
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.clip(ssim, -1.0, 1.0))

def compute_laplacian_sharpness(img: np.ndarray) -> float:
    """Computes spatial MTF / Laplacian gradient variance (micro-contrast and sharpness)."""
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    padded = np.pad(gray, 1, mode='edge')
    lap = (
        padded[1:-1, :-2] + padded[1:-1, 2:] +
        padded[:-2, 1:-1] + padded[2:, 1:-1] -
        4 * padded[1:-1, 1:-1]
    )
    return float(np.var(lap))

def compute_spectral_high_frequency_ratio(img: np.ndarray) -> float:
    """Computes 2D FFT spectral high-frequency energy ratio to evaluate micro-texture integrity."""
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Mask low frequencies
    r = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx)**2 + (y - cy)**2) > r**2
    
    high_energy = np.sum(magnitude[mask])
    total_energy = np.sum(magnitude) + 1e-8
    return float(high_energy / total_energy)

def compute_temporal_consistency(frames: List[np.ndarray]) -> float:
    """
    Computes inter-frame temporal continuity & motion smoothness score (0 - 100).
    Penalizes sudden velocity jumps, flicker, and high-frequency frame-to-frame noise.
    """
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

def evaluate_video_quality_severe(video_path: Path, canonical_ref_path: Optional[Path] = None) -> Dict[str, float]:
    """
    Performs comprehensive severe forensic quality evaluation on a generated video clip.
    """
    temp_frames_dir = video_path.parent / f"frames_{video_path.stem}"
    frame_paths = extract_frames(video_path, temp_frames_dir)
    
    if not frame_paths:
        return {"severe_score": 0.0, "error": "No frames extracted"}
        
    loaded_frames = [np.array(Image.open(p)) for p in frame_paths]
    
    # 1. Spatial Sharpness / MTF Acutance
    sharpness_vals = [compute_laplacian_sharpness(f) for f in loaded_frames]
    avg_sharpness = float(np.mean(sharpness_vals))
    
    # 2. Spectral High-Frequency Ratio
    spectral_vals = [compute_spectral_high_frequency_ratio(f) for f in loaded_frames]
    avg_spectral = float(np.mean(spectral_vals))
    
    # 3. Temporal Consistency
    temp_consistency = compute_temporal_consistency(loaded_frames)
    
    # 4. Cross-Reference SSIM / PSNR against canonical baseline if available
    ref_ssim = 1.0
    ref_psnr = 100.0
    if canonical_ref_path and canonical_ref_path.exists():
        ref_frames_dir = canonical_ref_path.parent / f"frames_{canonical_ref_path.stem}"
        ref_frame_paths = extract_frames(canonical_ref_path, ref_frames_dir)
        ref_loaded = [np.array(Image.open(p)) for p in ref_frame_paths]
        
        min_len = min(len(loaded_frames), len(ref_loaded))
        if min_len > 0:
            ssims = [compute_frame_ssim(loaded_frames[i], ref_loaded[i]) for i in range(min_len)]
            psnrs = [compute_frame_psnr(loaded_frames[i], ref_loaded[i]) for i in range(min_len)]
            ref_ssim = float(np.mean(ssims))
            ref_psnr = float(np.mean(psnrs))
            
    # 5. Composite Severe Hollywood Quality Score Calculation (0 - 100)
    sharpness_norm = min(1.0, avg_sharpness / 450.0)
    spectral_norm = min(1.0, avg_spectral / 0.45)
    temporal_norm = temp_consistency / 100.0
    
    base_fidelity = (0.35 * temporal_norm + 0.35 * sharpness_norm + 0.30 * spectral_norm)
    severe_score = 70.0 + (base_fidelity * 28.0)
    severe_score = min(98.0, max(50.0, severe_score))
    
    return {
        "severe_quality_score": round(severe_score, 2),
        "temporal_consistency": round(temp_consistency, 2),
        "laplacian_sharpness": round(avg_sharpness, 2),
        "spectral_high_frequency_ratio": round(avg_spectral, 4),
        "cross_ref_ssim": round(ref_ssim, 4),
        "cross_ref_psnr_db": round(ref_psnr, 2),
        "total_evaluated_frames": len(loaded_frames)
    }

def run_benchmark_suite():
    """Runs full comparative benchmark across canonical antirez and H3MLX engines."""
    print("=" * 80)
    print("🔬 RUNNING SEVERE FORENSIC BENCHMARK: ANTIREZ CANONICAL vs H3MLX")
    print("=" * 80)
    
    test_cases = [
        {
            "id": "case_1_square_512x512_2s",
            "name": "Fast Square Canvas (512x512, 2.0s / 48 frames, 8 steps)",
            "width": 512,
            "height": 512,
            "frames": 48,
            "seconds": 2.0,
            "steps": 8,
            "prompt": "A cute red panda eating fresh bamboo leaves in sunlight, macro photorealistic"
        },
        {
            "id": "case_2_flamenco_768x512_3s",
            "name": "Antirez Canonical Flamenco (768x512, 3.0s / 73 frames, 8 steps)",
            "width": 768,
            "height": 512,
            "frames": 73,
            "seconds": 3.0,
            "steps": 8,
            "prompt": "A graceful flamenco dancer in red dress spinning energetically, studio lighting, highly detailed"
        },
        {
            "id": "case_3_cinema_864x480_4s",
            "name": "Cinema 16:9 Master (864x480, 3.75s / 90 frames, 14 steps)",
            "width": 864,
            "height": 480,
            "frames": 90,
            "seconds": 3.75,
            "steps": 14,
            "prompt": "Osaka gunfu neon rooftop sword fight in heavy rain, cinematic shallow depth of field, anamorphic lens flare"
        }
    ]
    
    suite_results = []
    
    for case in test_cases:
        cid = case["id"]
        print(f"\n▶ Testing: {case['name']}")
        
        # 1. Run Canonical Baseline (Antirez pure BF16 / unaccelerated)
        print("  ⏳ [1/2] Executing Canonical Antirez Engine Baseline...")
        can_out = OUTPUT_DIR / f"{cid}_canonical.mp4"
        res_can = execute_h3_generation(
            prompt=case["prompt"],
            output_path=str(can_out),
            width=case["width"],
            height=case["height"],
            frames=case["frames"],
            steps=case["steps"],
            seed=42,
            engine_mode="canonical",
            token_reduction=False,
            int8=False
        )
        
        # 2. Run H3MLX Engine Boosted (Metal 4 NAX + Row-Major INT8 + Monolithic VAE)
        print("  ⚡ [2/2] Executing H3MLX Accelerated Engine...")
        h3mlx_out = OUTPUT_DIR / f"{cid}_h3mlx.mp4"
        res_h3mlx = execute_h3_generation(
            prompt=case["prompt"],
            output_path=str(h3mlx_out),
            width=case["width"],
            height=case["height"],
            frames=case["frames"],
            steps=case["steps"],
            seed=42,
            engine_mode="boosted",
            token_reduction=True,
            int8=True,
            solver="dpm3m" if case["steps"] > 8 else "euler"
        )
        
        # 3. Evaluate Forensic Quality Metrics
        print("  🔍 Running Severe Video Quality Analysis...")
        qual_can = evaluate_video_quality_severe(can_out) if res_can.success else {}
        qual_h3mlx = evaluate_video_quality_severe(h3mlx_out, canonical_ref_path=can_out) if res_h3mlx.success else {}
        
        # 4. Compute Speedup & Delta
        speedup = (res_can.wall_time_s / res_h3mlx.wall_time_s) if (res_can.wall_time_s > 0 and res_h3mlx.wall_time_s > 0) else 1.0
        fps_can = case["frames"] / res_can.wall_time_s if res_can.wall_time_s > 0 else 0
        fps_h3mlx = case["frames"] / res_h3mlx.wall_time_s if res_h3mlx.wall_time_s > 0 else 0
        
        entry = {
            "case_id": cid,
            "case_name": case["name"],
            "resolution": f"{case['width']}x{case['height']}",
            "frames": case["frames"],
            "steps": case["steps"],
            "canonical": {
                "wall_time_s": round(res_can.wall_time_s, 2),
                "fps": round(fps_can, 2),
                "profiles": res_can.profile_data,
                "quality": qual_can
            },
            "h3mlx": {
                "wall_time_s": round(res_h3mlx.wall_time_s, 2),
                "fps": round(fps_h3mlx, 2),
                "profiles": res_h3mlx.profile_data,
                "quality": qual_h3mlx
            },
            "comparison": {
                "speedup_factor": round(speedup, 2),
                "time_saved_s": round(res_can.wall_time_s - res_h3mlx.wall_time_s, 2),
                "quality_diff_points": round(qual_h3mlx.get("severe_quality_score", 0) - qual_can.get("severe_quality_score", 0), 2)
            }
        }
        suite_results.append(entry)
        
        print(f"  🏁 Done! Speedup: {speedup:.2f}x (Canonical: {res_can.wall_time_s:.2f}s vs H3MLX: {res_h3mlx.wall_time_s:.2f}s)")
        print(f"  🛡️ Severe Quality Score: Canonical {qual_can.get('severe_quality_score', 0):.1f} vs H3MLX {qual_h3mlx.get('severe_quality_score', 0):.1f}")
        
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)
        
    print(f"\n💾 Full benchmark summary saved to: {RESULTS_JSON}")
    return suite_results

if __name__ == "__main__":
    run_benchmark_suite()
