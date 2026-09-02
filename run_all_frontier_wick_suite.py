#!/usr/bin/env python3
"""
👑 Master Suite: Esecuzione di TUTTE le Frontiere Campione di John Wick & Action Cinema
1. Gun-Fu Osaka 14-Step Optimal Master (Record 74s 4K)
2. Fast Katana Sword Combat Master
3. Capriola Acrobatica 360° + Water Splash
4. Wing Chun Rapid Centerline Trapping Combat
5. Lateral Dolly Cinema 16:9 Master (1024x576)
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
OUT_DIR = BASE_DIR / "outputs_frontier_suite"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path.home() / "Downloads"

SUITE = [
    {
        "id": 1,
        "name": "gunfu_osaka_14step",
        "title": "👑 Gun-Fu Osaka 14-Step Master (Record 74s)",
        "w": 768, "h": 512, "frames": 90, "steps": 14, "reuse": 2, "seed": 5555,
        "prompt": (
            "Shot on Arri Alexa LF with Cooke Anamorphic S4i Prime 50mm T2.3 lens, MTF optical sub-pixel phase coherence, "
            "pristine Hollywood master medium-close action tracking shot in heavy torrential night rain, John Wick in crisp "
            "tailored black wool suit with white shirt and black tie facing 3/4 frontally with razor-sharp Keanu Reeves likeness "
            "and intense fierce eyes, executing a rapid tactical Gun-Fu double-tap with custom handgun, instantaneous brilliant "
            "golden-amber muzzle flash illuminating wet facial skin pores and airborne rain droplets, brass shell casing ejecting "
            "in mid-air with crisp specular highlights, vibrant neon cyan background bokeh, wet Tokyo street reflections, 4k 24fps master"
        )
    },
    {
        "id": 2,
        "name": "katana_sword_combat",
        "title": "⚔️ Fast Katana Sword Combat Master",
        "w": 768, "h": 512, "frames": 90, "steps": 14, "reuse": 2, "seed": 4444,
        "prompt": (
            "Shot on Arri Alexa LF 50mm lens, cinematic 8k footage of John Wick in black tactical suit wielding a razor-sharp "
            "Japanese katana sword in pouring neon rain, executing a high-speed lethal sword parry and strike, bright specular "
            "blade reflection cutting through water droplets, intense facial expression of Keanu Reeves with wet hair strands, "
            "neon sign bokeh reflections on wet asphalt, ultra photorealistic, dynamic camera motion"
        )
    },
    {
        "id": 3,
        "name": "acrobatic_flip_360_splash",
        "title": "🤸 Capriola Acrobatica 360° + Water Splash",
        "w": 768, "h": 512, "frames": 90, "steps": 14, "reuse": 2, "seed": 3333,
        "prompt": (
            "Pristine cinematic action slow-motion shot of John Wick in black suit performing an athletic 360-degree aerial combat flip "
            "over a shallow water puddle in Tokyo night rain, dynamic camera tracking his mid-air rotation, landing with heavy splash "
            "of water droplets illuminated by neon streetlights, razor-sharp Keanu Reeves likeness, realistic cloth physics and rain, 4k"
        )
    },
    {
        "id": 4,
        "name": "wing_chun_trapping",
        "title": "🥋 Wing Chun Rapid Centerline Trapping Combat",
        "w": 768, "h": 512, "frames": 73, "steps": 14, "reuse": 2, "seed": 2222,
        "prompt": (
            "Cinematic master close-up action shot of John Wick engaged in rapid Wing Chun martial arts hand trapping and parrying, "
            "deflecting incoming strikes with lightning speed, intense focused eyes, sweat and rain on brow, Cooke anamorphic lens flare, "
            "shallow depth of field, neon city background bokeh, hyper-realistic skin texture, 24fps master"
        )
    },
    {
        "id": 5,
        "name": "lateral_dolly_16x9_cinema",
        "title": "🎥 Lateral Dolly Cinema Widescreen Master (1024x576)",
        "w": 1024, "h": 576, "frames": 90, "steps": 14, "reuse": 2, "seed": 1111,
        "prompt": (
            "Masterpiece 16:9 anamorphic cinematic tracking dolly shot of John Wick in tailored suit walking determinedly through a "
            "crowded Osaka neon alleyway in heavy rain, drawing his tactical weapon, razor-sharp Keanu Reeves facial detail with water "
            "dripping from hair, stunning atmospheric fog, vibrant neon signs reflecting on wet ground, ultra high definition 8k"
        )
    }
]

env = os.environ.copy()
env["H3_PROFILE"] = "1"
env["H3_NAX"] = "qkv-attn"
env["H3_ZERO_COPY_WEIGHTS"] = "1"
env["H3_REUSE_MPS_COMMAND"] = "1"
env["H3_GPU_SAMPLER"] = "1"
env["OMP_NUM_THREADS"] = "18"

print("=" * 88)
print("👑 MASTER SUITE: ESECUZIONE COMPLETA DI TUTTE LE FRONTIERE CHAMPION (JOHN WICK)")
print("   Hardware: Apple Silicon M5 Max (128GB UMA · Metal 4 NAX INT8 FC2)")
print("   Pipeline: Denoise DiT + Video VAE 3D + Audio 48kHz + Pipeline 4K UHD Master")
print("=" * 88)

RESULTS = []

for item in SUITE:
    num = item["id"]
    name = item["name"]
    title = item["title"]
    w, h = item["w"], item["h"]
    frames = item["frames"]
    steps = item["steps"]
    reuse = item["reuse"]
    seed = item["seed"]
    prompt = item["prompt"]
    
    raw_mp4 = OUT_DIR / f"{name}_raw.mp4"
    master_4k_mp4 = DOWNLOADS_DIR / f"{name}_4k_master.mp4"
    frame_png = OUT_DIR / f"frame_{name}.png"
    
    print(f"\n▶️ [{num}/{len(SUITE)}] {title}")
    print(f"   📐 Risoluzione: {w}x{h} | Frame: {frames} ({frames/24:.1f}s) | Step: {steps} (Reuse {reuse}) | Seed: {seed}")
    
    cmd_gen = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", prompt,
        "--width", str(w),
        "--height", str(h),
        "--frames", str(frames),
        "--steps", str(steps),
        "--layers", "50",
        "--reuse", str(reuse),
        "--use-int8-row-fc2",
        "--seed", str(seed),
        "-o", str(raw_mp4)
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd_gen, env=env, cwd=H3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time_dit = time.time() - t0
    
    if res.returncode == 0 and raw_mp4.exists():
        raw_size = raw_mp4.stat().st_size / (1024 * 1024)
        print(f"   ⚡ DIT Generation completata in {time_dit:.2f}s | Raw MP4: {raw_size:.2f} MB")
        
        # Pipeline 4K UHD + Audio Foley 48kHz
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
        time_4k = time.time() - t_4k_0
        total_time = time_dit + time_4k
        master_size = master_4k_mp4.stat().st_size / (1024 * 1024) if master_4k_mp4.exists() else 0
        
        print(f"   💎 4K UHD Master (10-bit) renderizzato in {time_4k:.2f}s | Totale: {total_time:.2f}s | Dimensione 4K: {master_size:.2f} MB")
        
        # Extract middle frame
        mid_frame = frames // 2
        subprocess.run([
            "ffmpeg", "-y", "-i", str(master_4k_mp4),
            "-vf", f"select=eq(n\\,{mid_frame})", "-vframes", "1",
            str(frame_png)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        RESULTS.append({
            "id": num,
            "name": name,
            "title": title,
            "resolution": f"{w}x{h}",
            "frames": frames,
            "time_dit": time_dit,
            "time_4k": time_4k,
            "time_total": total_time,
            "master_4k": str(master_4k_mp4),
            "frame": str(frame_png),
            "size_4k_mb": master_size
        })
    else:
        print(f"   ❌ Errore nella generazione di {name}: Returncode {res.returncode}")
        print(res.stdout[-400:])

report_json = OUT_DIR / "frontier_suite_results.json"
with open(report_json, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n" + "=" * 88)
print("📊 RIEPILOGO FINALE SUITE FRONTIERE JOHN WICK (TUTTI I VIDEO MASTER 4K):")
print("=" * 88)
print(f"{'#':<3} | {'Scena / Frontiera':<40} | {'Res':<10} | {'Denoise':<10} | {'Totale (4K)':<12} | {'Output 4K'}")
print("-" * 88)
for r in RESULTS:
    print(f"{r['id']:<3} | {r['title']:<40} | {r['resolution']:<10} | {r['time_dit']:>7.2f} s | {r['time_total']:>9.2f} s | {Path(r['master_4k']).name}")
print("=" * 88)
