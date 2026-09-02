import subprocess
import time
import json
import re
from pathlib import Path

# Base Paths
base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
h3_bin = base_dir / "h3-lora-lab/h3"
model_dir = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
if not model_dir.exists():
    model_dir = Path("/Users/robzomb/Desktop/H3/MiniMax-H3")

out_dir = base_dir / "outputs_ghibli_samplers"
out_dir.mkdir(parents=True, exist_ok=True)

# Prompt specified by the user
prompt = "A Studio Ghibli anime style joyful girl jumping happily in a vibrant summer meadow, Hayao Miyazaki aesthetic, lush green grass, floating flower petals, warm golden sunlight, cel-shaded hand-drawn animation, 48kHz gentle summer breeze and cheerful atmosphere"

# Sampler & Schedule Configurations
samplers = [
    {
        "id": "dpm2m_trailing_s12",
        "name": "DPM++ 2M Trailing (Shift 12.0)",
        "category": "Draw Things Gold Standard",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_karras_s12",
        "name": "DPM++ 2M Karras Trailing",
        "category": "Draw Things Karras Curvature",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 14.0,
        "ashift": 3.5
    },
    {
        "id": "dpm2m_sde_trailing",
        "name": "DPM++ 2M SDE Trailing",
        "category": "Draw Things Stochastic SDE",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "euler_trailing",
        "name": "Euler Trailing (Shift 12.0)",
        "category": "Draw Things Direct 1st-Order",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "euler_a_trailing",
        "name": "Euler Ancestral (Euler A) Trailing",
        "category": "Draw Things Painterly Anime",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 10.0,
        "ashift": 3.0
    },
    {
        "id": "heun_trailing_6step",
        "name": "Heun 2nd-Order Trailing (6-Step)",
        "category": "Draw Things Predictor-Corrector",
        "steps": 6,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "unipc_trailing_6step",
        "name": "UniPC Fast Trailing (6-Step)",
        "category": "Draw Things Unified Multistep",
        "steps": 6,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "flow_anime_s8",
        "name": "Flow Shifted Anime Motion (Shift 8.0)",
        "category": "Beyond: Dynamic Flow Warping",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 8.0,
        "ashift": 2.5
    },
    {
        "id": "dpm2m_reuse2",
        "name": "DPM++ 2M Step-Reuse 2 (SLA)",
        "category": "Beyond: Attention Caching",
        "steps": 8,
        "layers": 50,
        "reuse": 2,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "turbo_ladder_4step",
        "name": "FastVideo Turbo Discrete Ladder",
        "category": "Draw Things / DMD2 Discrete",
        "steps": 4,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    }
]

durations = [
    {"label": "1s", "frames": 22, "desc": "1.0s (22 Frames @ 24fps - 1 Causal Chunk)"},
    {"label": "2s", "frames": 39, "desc": "2.0s (39 Frames @ 24fps - 2 Causal Chunks)"}
]

results = []

print("=" * 90)
print("🌸 BENCHMARK STUDIO GHIBLI: Sampler & Schedule Suite on Apple Silicon M5 Max")
print("=" * 90)

for dur in durations:
    frames = dur["frames"]
    d_lbl = dur["label"]
    print(f"\n--- ⏱️ DURATION: {dur['desc']} ---")
    
    for s_idx, s in enumerate(samplers):
        s_id = s["id"]
        test_id = f"{s_id}_{d_lbl}"
        mp4_out = out_dir / f"{test_id}.mp4"
        
        env = {
            "H3_PROFILE": "1",
            "H3_NAX": "qkv-attn",
            "H3_ZERO_COPY_WEIGHTS": "1",
            "H3_REUSE_MPS_COMMAND": "1",
            "H3_GPU_SAMPLER": "1",
            "H3_VIDEO_SHIFT": str(s["vshift"]),
            "H3_AUDIO_SHIFT": str(s["ashift"]),
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
        }
        
        cmd = [
            "caffeinate", "-dimsu", "nice", "-n", "0",
            str(h3_bin),
            "--profile",
            "-d", str(model_dir),
            "-p", prompt,
            "--width", "640",
            "--height", "640",
            "--frames", str(frames),
            "--steps", str(s["steps"]),
            "--layers", str(s["layers"]),
            "--reuse", str(s["reuse"]),
            "--use-int8-row-fc2",
            "--seed", "777",
            "-o", str(mp4_out)
        ]
        
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, cwd=str(h3_bin.parent), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = t1 - t0
        
        out = res.stdout + "\n" + res.stderr
        
        denoise_match = re.search(r"denoise\s+([0-9\.]+)\s+s", out)
        vae_match = re.search(r"video vae\s+([0-9\.]+)\s+s", out)
        qwen_match = re.search(r"qwen3-vl\s+([0-9\.]+)\s+s", out)
        
        denoise_sec = float(denoise_match.group(1)) if denoise_match else 0.0
        vae_sec = float(vae_match.group(1)) if vae_match else 0.0
        qwen_sec = float(qwen_match.group(1)) if qwen_match else 0.0
        
        fps = frames / denoise_sec if denoise_sec > 0 else 0.0
        
        item = {
            "test_id": test_id,
            "duration": d_lbl,
            "frames": frames,
            "name": s["name"],
            "category": s["category"],
            "steps": s["steps"],
            "layers": s["layers"],
            "reuse": s["reuse"],
            "vshift": s["vshift"],
            "ashift": s["ashift"],
            "denoise_sec": round(denoise_sec, 2),
            "vae_sec": round(vae_sec, 2),
            "qwen_sec": round(qwen_sec, 2),
            "wall_total": round(wall_total, 2),
            "fps": round(fps, 2),
            "video_path": str(mp4_out)
        }
        results.append(item)
        
        print(f"  [{s_idx+1:02d}/{len(samplers):02d}] ✓ {s['name']:<42} | Denoise: {denoise_sec:>5.2f}s | VAE: {vae_sec:>5.2f}s | Totale: {wall_total:>5.2f}s | FPS: {fps:>4.2f}")

# Save JSON results
json_out = out_dir / "ghibli_samplers_benchmark_results.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print(f"✅ Benchmark completato con successo! Dati salvati in: {json_out}")
print("=" * 90)
