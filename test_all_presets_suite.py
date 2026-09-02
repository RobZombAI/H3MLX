#!/usr/bin/env python3
"""
🧪 H3MLX Systematic Suite Validator
Executes and validates all 6 high-quality presets across the causal temporal lattice.
"""

import sys
import time
import subprocess
from pathlib import Path

from h3mlx_presets import PRESETS
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, BASE_DIR

PRESET_LIST = [
    "h3mlx_champion_gold",
    "h3mlx_cinema_16x9",
    "h3mlx_macro_square",
    "h3mlx_vertical_reel",
    "h3mlx_ghibli_master",
    "antirez_canonical_bf16"
]

def main():
    print("=" * 80)
    print("🧪 H3MLX SYSTEMATIC PRESETS VALIDATION SUITE (APPLE SILICON M5 MAX)")
    print("=" * 80)
    
    model_path = resolve_model_path(steps=8)
    print(f"📦 Model: {model_path.name}")
    print(f"🎯 Target Presets to validate: {len(PRESET_LIST)}\n")
    
    results = []
    
    for i, pid in enumerate(PRESET_LIST, 1):
        cfg = PRESETS[pid]
        print(f"[{i}/{len(PRESET_LIST)}] ▶ Validating: {cfg['name']} ({cfg['width']}x{cfg['height']})...")
        out_file = BASE_DIR / "outputs" / f"test_suite_{pid}.mp4"
        
        # Test on 22 frames (causal lattice n=1: 0.92s) for rapid systematic coverage
        test_frames = 22
        
        t0 = time.perf_counter()
        res = execute_h3_generation(
            prompt=cfg["prompt"],
            output_path=str(out_file),
            width=cfg["width"],
            height=cfg["height"],
            frames=test_frames,
            steps=cfg.get("steps", 8),
            seed=42,
            engine_mode=cfg.get("mode", "boosted"),
            solver=cfg.get("solver", "dpm3m"),
            reuse=cfg.get("reuse", 1),
            layers=cfg.get("layers", 50),
            token_reduction=False,
            int8=cfg.get("int8", True),
            upscale_4k=cfg.get("upscale_4k", False),
            profile=True
        )
        t1 = time.perf_counter()
        dur = t1 - t0
        
        # Verify video file exists and probe with ffprobe
        valid = False
        meta = {}
        check_file = Path(res.output_path)
        if res.success and check_file.exists():
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames,codec_name", "-of", "default=noprint_wrappers=1", str(check_file)],
                capture_output=True, text=True
            )
            if probe.returncode == 0 and "width=" in probe.stdout:
                valid = True
                for line in probe.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        meta[k] = v
                        
                # Extract sample frame for visual inspection
                frame_jpg = Path(f"/tmp/suite_{pid}.jpg")
                subprocess.run(["ffmpeg", "-y", "-i", str(check_file), "-vframes", "1", "-ss", "00:00:00.5", str(frame_jpg)], capture_output=True)
                
        actual_res = f"{meta.get('width', cfg['width'])}x{meta.get('height', cfg['height'])}"
        results.append({
            "id": pid,
            "name": cfg["name"],
            "resolution": actual_res,
            "frames": test_frames,
            "wall_sec": dur,
            "fps": test_frames / dur if dur > 0 else 0,
            "success": valid,
            "output": str(check_file)
        })
        
        status_sym = "✅ PASS" if valid else "❌ FAIL"
        print(f"    {status_sym} in {dur:.2f}s | Output Res: {actual_res} | Output: {check_file.name}\n")
        
    print("=" * 80)
    print("📊 SUITE VALIDATION SUMMARY REPORT")
    print("=" * 80)
    print(f"{'PRESET':<38} | {'RES':<9} | {'SEC':<7} | {'FPS':<8} | {'STATUS'}")
    print("-" * 80)
    all_pass = True
    for r in results:
        status_str = "✅ PASS" if r["success"] else "❌ FAIL"
        if not r["success"]:
            all_pass = False
        print(f"{r['name']:<38} | {r['resolution']:<9} | {r['wall_sec']:<6.2f}s | {r['fps']:<7.2f} | {status_str}")
    print("=" * 80)
    
    if all_pass:
        print("\n🎉 ALL 6 SYSTEMATIC PRESETS PASSED WITH 100% SUCCESS!\n")
    else:
        print("\n⚠️ SOME PRESETS ENCOUNTERED ISSUES.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
