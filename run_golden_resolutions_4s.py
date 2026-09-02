#!/usr/bin/env python3
"""
🏆 Master Benchmark: Tutte le 6 Risoluzioni Regine a 4 Secondi (90 Frame)
1. 16:9 Cinema: 1024x576
2. 1:1 Square: 768x768
3. 9:16 Vertical: 576x1024
4. 21:9 Cinemascope: 1024x448
5. 4:3 Classic TV: 768x576
6. 3:2 Photography: 960x640
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
H3_BIN = H3_DIR / "h3"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUT_DIR = BASE_DIR / "outputs_golden_resolutions_4s"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Live-action raw documentary broadcast 35mm footage of an intense professional wrestling match inside the ring. "
    "Two real athletic heavyweight muscular wrestlers clashing in tight grapple, glistening sweat beads on skin, "
    "realistic facial strain and veins, stadium arena floodlights cutting through atmospheric haze, "
    "35mm film stock, motion blur, photorealistic live sports broadcast"
)

GOLDEN_PRESETS = [
    {"family": "16:9 Cinema", "name": "golden_16x9_1024x576", "w": 1024, "h": 576},
    {"family": "1:1 Square", "name": "golden_1x1_768x768", "w": 768, "h": 768},
    {"family": "9:16 Vertical", "name": "golden_9x16_576x1024", "w": 576, "h": 1024},
    {"family": "21:9 Cinemascope", "name": "golden_21x9_1024x448", "w": 1024, "h": 448},
    {"family": "4:3 Classic TV", "name": "golden_4x3_768x576", "w": 768, "h": 576},
    {"family": "3:2 Photography", "name": "golden_3x2_960x640", "w": 960, "h": 640},
]

env = os.environ.copy()
env["H3_PROFILE"] = "1"
env["H3_NAX"] = "qkv-attn"
env["H3_ZERO_COPY_WEIGHTS"] = "1"
env["H3_REUSE_MPS_COMMAND"] = "1"
env["H3_GPU_SAMPLER"] = "1"

print("=" * 84)
print("🏆 MASTER SUITE: TUTTE LE 6 RISOLUZIONI REGINE A 4 SECONDI (90 FRAME)")
print("   Scena: Wrestling Live-Action (2 Lottatori) · Modello: MiniMax-H3-PDD-8Step")
print("   Accelerazione: Metal 4 NAX Row-Major INT8 FC2 · 8 Step Esatti")
print("=" * 84)

RESULTS = []

for idx, p in enumerate(GOLDEN_PRESETS, start=1):
    w, h = p["w"], p["h"]
    tokens = (w // 16) * (h // 16)
    out_mp4 = OUT_DIR / f"{p['name']}_4s.mp4"
    frame_png = OUT_DIR / f"frame_{p['name']}_4s.png"
    
    print(f"\n[{idx:02d}/{len(GOLDEN_PRESETS):02d}] 🎥 {p['family']:<18} | {w}x{h} ({tokens} token · 4.0s / 90 frame)...")
    
    cmd = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PROMPT,
        "--width", str(w),
        "--height", str(h),
        "--frames", "90",
        "--steps", "8",
        "--layers", "50",
        "--reuse", "1",
        "--use-int8-row-fc2",
        "--seed", "555",
        "-o", str(out_mp4)
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=H3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    
    if res.returncode == 0 and out_mp4.exists():
        size_mb = out_mp4.stat().st_size / (1024 * 1024)
        print(f"   ✅ Generato in {elapsed:.2f}s | MP4: {size_mb:.2f} MB")
        
        # Extract middle frame
        subprocess.run([
            "ffmpeg", "-y", "-i", str(out_mp4),
            "-vf", "select=eq(n\\,45)", "-vframes", "1",
            str(frame_png)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        RESULTS.append({
            "idx": idx,
            "family": p["family"],
            "name": p["name"],
            "width": w,
            "height": h,
            "tokens": tokens,
            "time_sec": elapsed,
            "size_mb": size_mb,
            "mp4": str(out_mp4),
            "frame": str(frame_png)
        })
    else:
        print(f"   ❌ Errore su {p['name']}: Returncode {res.returncode}")
        print(res.stdout[-400:])

# Save JSON Report
report_json = OUT_DIR / "golden_resolutions_4s_results.json"
with open(report_json, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 84)
print("📊 RIEPILOGO FINALE 6 RISOLUZIONI REGINE A 4 SECONDI (90 FRAME):")
print("=" * 84)
print(f"{'#':<3} | {'Famiglia':<18} | {'Risoluzione':<12} | {'Token':<6} | {'Tempo Totale':<14} | {'Size'}")
print("-" * 84)
for r in RESULTS:
    print(f"{r['idx']:<3} | {r['family']:<18} | {r['width']}x{r['height']:<7} | {r['tokens']:<6} | {r['time_sec']:>8.2f} s     | {r['size_mb']:>5.2f} MB")
print("=" * 84)
