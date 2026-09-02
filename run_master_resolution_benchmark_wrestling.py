#!/usr/bin/env python3
"""
🏆 Master Multi-Resolution & Aspect-Ratio Benchmark: Wrestling Royal Rumble (2 Lottatori)
Tests ALL aspect ratios on Apple Silicon M5 Max 128GB UMA with Metal 4 NAX Row-Major INT8 FC2:
1. 16:9 Widescreen (768x512, 864x480, 960x544, 1024x576, 1280x704)
2. 1:1 Quadrato (512x512, 640x640, 768x768)
3. 9:16 Verticale (512x896, 576x1024, 704x1280)
4. 21:9 Cinemascope (896x384, 1024x448)
5. 4:3 Classic TV (640x480, 768x576)
6. 3:2 Photography (768x512, 960x640)
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
OUT_DIR = BASE_DIR / "outputs_wrestling_all_resolutions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Live-action raw documentary broadcast 35mm footage of an intense professional wrestling match inside the ring. "
    "Two real athletic heavyweight muscular wrestlers clashing in tight grapple, glistening sweat beads on skin, "
    "realistic facial strain and veins, stadium arena floodlights cutting through atmospheric haze, "
    "35mm film stock, motion blur, photorealistic live sports broadcast"
)

PRESETS = [
    # 16:9 Widescreen Cinema / Broadcast
    {"group": "16:9 Widescreen", "name": "16x9_768x512", "w": 768, "h": 512},
    {"group": "16:9 Widescreen", "name": "16x9_864x480", "w": 864, "h": 480},
    {"group": "16:9 Widescreen", "name": "16x9_960x544", "w": 960, "h": 544},
    {"group": "16:9 Widescreen", "name": "16x9_1024x576", "w": 1024, "h": 576},
    {"group": "16:9 Widescreen", "name": "16x9_1280x704", "w": 1280, "h": 704},

    # 1:1 Quadrato
    {"group": "1:1 Square", "name": "1x1_512x512", "w": 512, "h": 512},
    {"group": "1:1 Square", "name": "1x1_640x640", "w": 640, "h": 640},
    {"group": "1:1 Square", "name": "1x1_768x768", "w": 768, "h": 768},

    # 9:16 Verticale (Reels / TikTok / Shorts)
    {"group": "9:16 Vertical", "name": "9x16_512x896", "w": 512, "h": 896},
    {"group": "9:16 Vertical", "name": "9x16_576x1024", "w": 576, "h": 1024},
    {"group": "9:16 Vertical", "name": "9x16_704x1280", "w": 704, "h": 1280},

    # 21:9 Ultra-Widescreen Cinemascope
    {"group": "21:9 Cinemascope", "name": "21x9_896x384", "w": 896, "h": 384},
    {"group": "21:9 Cinemascope", "name": "21x9_1024x448", "w": 1024, "h": 448},

    # 4:3 Classic TV
    {"group": "4:3 Classic TV", "name": "4x3_640x480", "w": 640, "h": 480},
    {"group": "4:3 Classic TV", "name": "4x3_768x576", "w": 768, "h": 576},

    # 3:2 Photography
    {"group": "3:2 Photography", "name": "3x2_960x640", "w": 960, "h": 640},
]

env = os.environ.copy()
env["H3_PROFILE"] = "1"
env["H3_NAX"] = "qkv-attn"
env["H3_ZERO_COPY_WEIGHTS"] = "1"
env["H3_REUSE_MPS_COMMAND"] = "1"
env["H3_GPU_SAMPLER"] = "1"

print("=" * 84)
print("🏆 MASTER MULTI-RESOLUTION & ASPECT-RATIO BENCHMARK")
print("   Scena: Wrestling Live-Action (2 Lottatori) · Modello: MiniMax-H3-PDD-8Step")
print("   Accelerazione: Metal 4 NAX Row-Major INT8 FC2 · 8 Step Esatti (Reuse 1)")
print("=" * 84)

RESULTS = []

for idx, p in enumerate(PRESETS, start=1):
    w, h = p["w"], p["h"]
    tokens = (w // 16) * (h // 16)
    out_mp4 = OUT_DIR / f"wrestling_{p['name']}.mp4"
    frame_png = OUT_DIR / f"frame_{p['name']}.png"
    
    print(f"\n[{idx:02d}/{len(PRESETS):02d}] 🎥 {p['group']:<18} | {p['name']:<16} ({w}x{h} · {tokens} token)...")
    
    cmd = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PROMPT,
        "--width", str(w),
        "--height", str(h),
        "--frames", "22",
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
            "-vf", "select=eq(n\\,10)", "-vframes", "1",
            str(frame_png)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        RESULTS.append({
            "idx": idx,
            "group": p["group"],
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
report_json = OUT_DIR / "wrestling_benchmark_results.json"
with open(report_json, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 84)
print("📊 RIEPILOGO COMPLETO BENCHMARK TUTTE LE RISOLUZIONI & ASPECT-RATIO:")
print("=" * 84)
print(f"{'#':<3} | {'Famiglia':<18} | {'Preset':<16} | {'Risoluzione':<12} | {'Token':<6} | {'Tempo':<9} | {'Size'}")
print("-" * 84)
for r in RESULTS:
    print(f"{r['idx']:<3} | {r['group']:<18} | {r['name']:<16} | {r['width']}x{r['height']:<7} | {r['tokens']:<6} | {r['time_sec']:>6.2f} s | {r['size_mb']:>5.2f} MB")
print("=" * 84)
