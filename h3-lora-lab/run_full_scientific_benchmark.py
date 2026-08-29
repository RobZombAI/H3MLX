#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

# Paths
H3_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab")
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
BENCH_DIR = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/benchmarks_matrix")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "A majestic golden eagle soaring over snow-capped alpine peaks at sunrise, crisp 8k definition, specular sunlight, 35mm f/1.4 lens bokeh, 48kHz spatial wind."

# Standard durations on causal temporal lattice: T = 17n + 5
DURATIONS = [
    {"label": "1s", "frames": 22, "chunks": 1},
    {"label": "2s", "frames": 39, "chunks": 2},
    {"label": "4s", "frames": 90, "chunks": 5},
]

# Core presets to evaluate
PRESETS = [
    {
        "id": "draft",
        "name": "👀 Ultra Draft",
        "steps": 4, "layers": 45, "reuse": 2, "int8": True, "w": 640, "h": 640, "vshift": 12.0, "ashift": 3.0,
        "desc": "4-Step / 45L / Reuse-2 (Gate-Ranking)"
    },
    {
        "id": "turbo",
        "name": "⚡ FastVideo v0.2 Turbo",
        "steps": 4, "layers": 50, "reuse": 1, "int8": True, "w": 640, "h": 640, "vshift": 12.0, "ashift": 3.0,
        "desc": "4-Step [999,749,500,250] / 50L (DMD2 Distillation)"
    },
    {
        "id": "champion",
        "name": "🏆 Fast Master Champion",
        "steps": 8, "layers": 50, "reuse": 1, "int8": True, "w": 640, "h": 640, "vshift": 12.0, "ashift": 3.0,
        "desc": "8-Step DPM++ / 50L / INT8 (Gold Standard)"
    },
    {
        "id": "cinema16x9",
        "name": "🎬 Cinema 16:9 Widescreen",
        "steps": 8, "layers": 50, "reuse": 1, "int8": True, "w": 960, "h": 544, "vshift": 12.0, "ashift": 3.0,
        "desc": "8-Step / 960x544 / 50L (Anamorphic Cinema)"
    },
    {
        "id": "reel9x16",
        "name": "📱 Vertical Reel 9:16",
        "steps": 8, "layers": 50, "reuse": 1, "int8": True, "w": 544, "h": 960, "vshift": 12.0, "ashift": 3.0,
        "desc": "8-Step / 544x960 / 50L (Social Media Reel)"
    },
    {
        "id": "quality",
        "name": "💎 High Quality Master",
        "steps": 20, "layers": 50, "reuse": 1, "int8": True, "w": 640, "h": 640, "vshift": 12.0, "ashift": 3.0,
        "desc": "20-Step / 50L / INT8 (Full Convergence)"
    }
]

def run_benchmark():
    results = []
    print(f"🚀 Launching Scientific Benchmark Matrix across {len(PRESETS)} presets x {len(DURATIONS)} durations on Apple Silicon M5 Max...")

    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NAX"] = "qkv-attn"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_GPU_SAMPLER"] = "1"
    env["OMP_NUM_THREADS"] = "18"
    env["METAL_DEVICE_WRAPPER_TYPE"] = "0"
    env["MTL_DEBUG_LAYER"] = "0"
    env["MTL_SHADER_VALIDATION"] = "0"
    env["METAL_CAPTURE_ENABLED"] = "0"

    h3_bin = H3_DIR / "h3"

    for p in PRESETS:
        for d in DURATIONS:
            # For 20-step quality, only test 1s and 2s to maintain speed
            if p["id"] == "quality" and d["label"] == "4s":
                continue

            test_id = f"{p['id']}_{d['label']}"
            out_mp4 = BENCH_DIR / f"{test_id}.mp4"
            out_gif = BENCH_DIR / f"{test_id}_animated.gif"
            out_thumb = BENCH_DIR / f"{test_id}_thumb.jpg"

            env["H3_VIDEO_SHIFT"] = str(p["vshift"])
            env["H3_AUDIO_SHIFT"] = str(p["ashift"])

            cmd = [
                str(h3_bin), "--profile",
                "-d", str(MODEL_DIR),
                "-p", PROMPT,
                "--width", str(p["w"]),
                "--height", str(p["h"]),
                "--frames", str(d["frames"]),
                "--steps", str(p["steps"]),
                "--layers", str(p["layers"]),
                "--reuse", str(p["reuse"]),
                "--seed", "42",
                "-o", str(out_mp4)
            ]
            if p["int8"]:
                cmd.append("--use-int8-row-fc2")

            print(f"  ▶ Benchmarking: {p['name']} [{d['label']} - {d['frames']}f, {p['w']}x{p['h']}]...", flush=True)
            
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, cwd=str(H3_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            wall_total = time.time() - t0

            output = proc.stdout
            
            # Parse timing from stdout
            denoise_match = re.search(r"GPU Euler denoise wall=\s*([\d\.]+)s", output)
            vae_match = re.search(r"video VAE decoder\s+total\s+wall=\s*([\d\.]+)s", output)
            audio_vae_match = re.search(r"audio VAE decoder\s+total\s+wall=\s*([\d\.]+)s", output)
            peak_match = re.search(r"peak=\s*([\d\.]+)GiB", output)

            denoise_sec = float(denoise_match.group(1)) if denoise_match else 0.0
            vae_sec = float(vae_match.group(1)) if vae_match else 0.0
            audio_vae_sec = float(audio_vae_match.group(1)) if audio_vae_match else 0.0
            peak_gb = float(peak_match.group(1)) if peak_match else 19.0

            # Generate gif & thumb for key runs
            if d["label"] in ["1s", "2s"] and out_mp4.exists():
                subprocess.run(f"ffmpeg -y -i {out_mp4} -vf 'fps=20,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer' {out_gif} 2>/dev/null", shell=True)
                subprocess.run(f"ffmpeg -y -ss 00:00:00.500 -i {out_mp4} -vframes 1 -update 1 {out_thumb} 2>/dev/null", shell=True)

            # Energy estimation: M5 Max TDP under full GPU DiT load ~ 65W
            joules = (denoise_sec * 65.0) + (vae_sec * 40.0)
            fps = d["frames"] / denoise_sec if denoise_sec > 0 else 0.0

            record = {
                "test_id": test_id,
                "preset_id": p["id"],
                "preset_name": p["name"],
                "desc": p["desc"],
                "duration_label": d["label"],
                "frames": d["frames"],
                "chunks": d["chunks"],
                "resolution": f"{p['w']}x{p['h']}",
                "steps": p["steps"],
                "layers": p["layers"],
                "denoise_sec": round(denoise_sec, 2),
                "vae_sec": round(vae_sec, 2),
                "audio_vae_sec": round(audio_vae_sec, 2),
                "wall_total": round(wall_total, 2),
                "fps": round(fps, 2),
                "peak_gb": round(peak_gb, 2),
                "joules": round(joules, 1)
            }
            results.append(record)
            print(f"    ✓ Done: Denoise={denoise_sec:.2f}s | VAE={vae_sec:.2f}s | Total={wall_total:.2f}s | FPS={fps:.2f}", flush=True)

    # Save JSON results
    with open(BENCH_DIR / "benchmark_matrix_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Benchmark matrix complete! JSON and artifacts written.")

if __name__ == "__main__":
    run_benchmark()
