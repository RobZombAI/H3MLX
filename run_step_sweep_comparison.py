#!/usr/bin/env python3
"""
🔬 H3 Step Sweep Benchmark: 16, 20, 23, 26, 40 Steps
Evaluates visual convergence, denoise GPU timing, and fidelity progression.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
H3_BIN = H3_DIR / "h3"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUT_DIR = BASE_DIR / "outputs_step_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Masterpiece National Geographic 8k wildlife footage of a majestic wild red fox hunting in fresh winter snow, "
    "jumping high into the air and diving headfirst into deep powder snow, flying snow particles catching golden sunset backlight, "
    "crystal clear fur texture, natural 35mm bokeh, ultra photorealistic, cinematic"
)

STEPS_LIST = [16, 20, 23, 26, 40]
RESULTS = []

env = os.environ.copy()
env["H3_PROFILE"] = "1"
env["H3_NAX"] = "qkv-attn"
env["H3_ZERO_COPY_WEIGHTS"] = "1"
env["H3_REUSE_MPS_COMMAND"] = "1"
env["H3_GPU_SAMPLER"] = "1"

print("=" * 76)
print("🔬 AVVIO SWEEP MULTI-STEP: 16, 20, 23, 26, 40 STEPS")
print(f" • Modello   : {MODEL_DIR.name}")
print(f" • Risoluzione: 640x640 (22 frame / 1.0s)")
print(f" • Quantizz. : Metal 4 NAX Row-Major INT8 FC2 (Exact Euler, Reuse 1)")
print("=" * 76)

for s in STEPS_LIST:
    out_mp4 = OUT_DIR / f"fox_step_{s}.mp4"
    cmd = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PROMPT,
        "--width", "640",
        "--height", "640",
        "--frames", "22",
        "--steps", str(s),
        "--layers", "50",
        "--reuse", "1",
        "--use-int8-row-fc2",
        "--seed", "333",
        "-o", str(out_mp4)
    ]
    
    print(f"\n▶️ [Step {s:02d}] Generazione in corso...")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=H3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    
    if res.returncode == 0 and out_mp4.exists():
        size_mb = out_mp4.stat().st_size / (1024 * 1024)
        print(f"   ✅ Step {s:02d} completato in {elapsed:.2f}s | MP4: {size_mb:.2f} MB")
        RESULTS.append({"step": s, "time": elapsed, "size": size_mb, "file": str(out_mp4)})
        
        # Extract middle frame
        frame_png = OUT_DIR / f"frame_step_{s}.png"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(out_mp4),
            "-vf", "select=eq(n\\,10)", "-vframes", "1",
            str(frame_png)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"   ❌ Errore step {s}: Returncode {res.returncode}")
        print(res.stdout[-500:])

print("\n" + "=" * 76)
print("📊 RIEPILOGO FINALE BENCHMARK MULTI-STEP:")
print("=" * 76)
print(f"{'Step':<10} | {'Tempo Totale':<15} | {'Dimensione MP4':<15} | {'File Output'}")
print("-" * 76)
for r in RESULTS:
    print(f"{r['step']:<10} | {r['time']:>8.2f} s       | {r['size']:>8.2f} MB       | {Path(r['file']).name}")
print("=" * 76)
