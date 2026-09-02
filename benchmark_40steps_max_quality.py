#!/usr/bin/env python3
"""
🏆 40-Step Ultimate Quality Benchmark across Key Resolutions
Evaluates visual fidelity, skin micro-pores, lighting, and rendering speed at 40 Steps:
1. 512x512 (1.024 tokens)
2. 640x640 (1.600 tokens)
3. 768x512 (1.536 tokens)
4. 1024x576 (2.304 tokens - Cinema 16:9)
5. 576x1024 (2.304 tokens - Vertical 9:16)
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
OUT_DIR = BASE_DIR / "outputs_40step_quality_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path.home() / "Downloads"

PROMPT = (
    "Shot on Arri Alexa LF with Cooke Anamorphic S4i Prime 50mm T2.3 lens, MTF optical sub-pixel phase coherence, "
    "pristine Hollywood master medium-close action tracking shot in heavy torrential night rain, John Wick in crisp "
    "tailored black wool suit with white shirt and black tie facing 3/4 frontally with razor-sharp Keanu Reeves likeness "
    "and intense fierce eyes, executing a rapid tactical Gun-Fu double-tap with custom handgun, instantaneous brilliant "
    "golden-amber muzzle flash illuminating wet facial skin pores and airborne rain droplets, brass shell casing ejecting "
    "in mid-air with crisp specular highlights, vibrant neon cyan background bokeh, wet Tokyo street reflections, 4k 24fps master"
)

PRESETS = [
    {"name": "40step_512x512", "label": "⏹️ 512x512 (1.024 token)", "w": 512, "h": 512},
    {"name": "40step_640x640", "label": "⏹️ 640x640 (1.600 token)", "w": 640, "h": 640},
    {"name": "40step_768x512", "label": "🎥 768x512 (1.536 token)", "w": 768, "h": 512},
    {"name": "40step_1024x576", "label": "👑 1024x576 Cinema 16:9 (2.304 token)", "w": 1024, "h": 576},
    {"name": "40step_576x1024", "label": "📱 576x1024 Vertical 9:16 (2.304 token)", "w": 576, "h": 1024},
]

env = os.environ.copy()
env["H3_PROFILE"] = "1"
env["H3_NAX"] = "qkv-attn"
env["H3_ZERO_COPY_WEIGHTS"] = "1"
env["H3_REUSE_MPS_COMMAND"] = "1"
env["H3_GPU_SAMPLER"] = "1"
env["OMP_NUM_THREADS"] = "18"

print("=" * 88)
print("🏆 BENCHMARK QUALITÀ MASSIMA ASSOLUTA A 40 STEP (JOHN WICK ARRI ALEXA LF)")
print("   Hardware: Apple Silicon M5 Max (128GB UMA · Metal 4 NAX Row-Major INT8 FC2)")
print("   Schedulazione: 40 Step Esatti (Reuse 1) · 50 Layer DiT Completi")
print("=" * 88)

RESULTS = []

for idx, p in enumerate(PRESETS, start=1):
    w, h = p["w"], p["h"]
    tokens = (w // 16) * (h // 16)
    name = p["name"]
    label = p["label"]
    
    raw_mp4 = OUT_DIR / f"{name}_raw.mp4"
    master_4k_mp4 = DOWNLOADS_DIR / f"{name}_4k_master.mp4"
    frame_png = OUT_DIR / f"frame_{name}.png"
    
    print(f"\n▶️ [{idx}/{len(PRESETS)}] {label} (Risoluzione: {w}x{h} · 48 frame / 2.0s)...")
    
    cmd_dit = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PROMPT,
        "--width", str(w),
        "--height", str(h),
        "--frames", "48",
        "--steps", "40",
        "--layers", "50",
        "--reuse", "1",
        "--use-int8-row-fc2",
        "--seed", "5555",
        "-o", str(raw_mp4)
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd_dit, env=env, cwd=H3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_dit = time.time() - t0
    
    if res.returncode == 0 and raw_mp4.exists():
        raw_size = raw_mp4.stat().st_size / (1024 * 1024)
        print(f"   ⚡ DIT Generation completata in {t_dit:.2f}s | Raw MP4: {raw_size:.2f} MB")
        
        # Pipeline 4K UHD Master (10-bit YUV420P10LE, Lanczos-4, Audio 48kHz)
        t_4k_0 = time.time()
        cmd_4k = [
            "ffmpeg", "-y", "-i", str(raw_mp4),
            "-af", "stereowiden=crossfeed=0.4:feedback=0.3:drymix=0.8,bass=g=8.0:f=75:w=0.6,treble=g=6.0:f=11000:w=0.7,dynaudnorm=p=0.95:m=10.0:r=0.9:b=1",
            "-vf", "scale=3840:2160:flags=lanczos+accurate_rnd+full_chroma_int,unsharp=5:5:0.85:5:5:0.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-preset", "fast", "-crf", "14",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            str(master_4k_mp4)
        ]
        subprocess.run(cmd_4k, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_4k = time.time() - t_4k_0
        total_time = t_dit + t_4k
        master_size = master_4k_mp4.stat().st_size / (1024 * 1024) if master_4k_mp4.exists() else 0
        
        print(f"   💎 4K UHD Master (10-bit) in {t_4k:.2f}s | Totale: {total_time:.2f}s | Dimensione 4K: {master_size:.2f} MB")
        
        # Extract middle frame
        subprocess.run([
            "ffmpeg", "-y", "-i", str(master_4k_mp4),
            "-vf", "select=eq(n\\,24)", "-vframes", "1",
            str(frame_png)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        RESULTS.append({
            "idx": idx,
            "label": label,
            "name": name,
            "width": w,
            "height": h,
            "tokens": tokens,
            "time_dit": t_dit,
            "time_4k": t_4k,
            "time_total": total_time,
            "master_4k": str(master_4k_mp4),
            "frame": str(frame_png),
            "size_4k_mb": master_size
        })
    else:
        print(f"   ❌ Errore su {name}: Returncode {res.returncode}")
        print(res.stdout[-400:])

report_json = OUT_DIR / "results_40step_benchmark.json"
with open(report_json, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 88)
print("📊 RIEPILOGO FINALE BENCHMARK MASSIMA QUALITÀ A 40 STEP:")
print("=" * 88)
print(f"{'#':<3} | {'Preset Risoluzione':<38} | {'Token':<6} | {'Denoise':<10} | {'Totale (4K)':<12} | {'Output 4K'}")
print("-" * 88)
for r in RESULTS:
    print(f"{r['idx']:<3} | {r['label']:<38} | {r['tokens']:<6} | {r['time_dit']:>7.2f} s | {r['time_total']:>9.2f} s | {Path(r['master_4k']).name}")
print("=" * 88)
