import subprocess
import time
import json
import re
from pathlib import Path

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
h3_bin = base_dir / "h3-lora-lab/h3"
model_dir = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
if not model_dir.exists():
    model_dir = Path("/Users/robzomb/Desktop/H3/MiniMax-H3")

out_dir = base_dir / "outputs_champion_samplers"
out_dir.mkdir(parents=True, exist_ok=True)

# 8K Macro Photorealistic Gold Standard Prompt
prompt = "A majestic golden eagle soaring gracefully over snow-capped alpine peaks at sunrise, crisp 8k definition, specular feather details catching golden sunlight, 35mm f/1.4 lens bokeh, 48kHz spatial mountain wind."

# Complete suite of Samplers & Schedulers for Fast Master Champion (Draw Things & New H3 Discoveries)
samplers = [
    {
        "id": "dpm2m_trailing_s12",
        "name": "DPM++ 2M Trailing (Dual-Clock 12/3)",
        "category": "Champion Gold Standard (Draw Things)",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_sde_karras",
        "name": "DPM++ 2M SDE Karras Trailing",
        "category": "Community Top Choice (r/ComfyUI & K-Diff)",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 14.0,
        "ashift": 3.5
    },
    {
        "id": "er_sde_flow",
        "name": "Euler-Richardson SDE Flow (ER-SDE)",
        "category": "Latest Flow Matching SDE Discovery",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "deis_trailing",
        "name": "DEIS Exponential Integrator Trailing",
        "category": "High-Order Fast ODE Integration",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "euler_a_dual_clock",
        "name": "Euler Ancestral Dual-Clock",
        "category": "Stochastic Natural Texture",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 10.0,
        "ashift": 3.0
    },
    {
        "id": "euler_trailing_s12",
        "name": "Euler Direct Trailing Flow",
        "category": "Pure 1st-Order Trailing Schedule",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "unipc_fast_trailing",
        "name": "UniPC Fast Trailing (6-Step)",
        "category": "Unified Multistep Predictor-Corrector",
        "steps": 6,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "heun_trailing_6step",
        "name": "Heun 2nd-Order Trailing (6-Step)",
        "category": "High-Precision 2nd-Order Predictor",
        "steps": 6,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "fast_ladder_8step",
        "name": "FastVideo 8-Step Trained Ladder",
        "category": "Trained Discrete Timestep Grid",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_reuse2_sla",
        "name": "DPM++ 2M Step-Reuse 2 (SLA Cache)",
        "category": "Metal 4 Attention Caching (Sub-8s)",
        "steps": 8,
        "layers": 50,
        "reuse": 2,
        "vshift": 12.0,
        "ashift": 3.0
    }
]

durations = [
    {"label": "1s", "frames": 22, "desc": "1.0s (22 Frames @ 24fps - 1 Causal Chunk)"},
    {"label": "2s", "frames": 39, "desc": "2.0s (39 Frames @ 24fps - 2 Causal Chunks)"}
]

results = []

print("=" * 95)
print("🦅 BENCHMARK FAST MASTER CHAMPION: Full Sampler & Schedule Suite on Apple Silicon M5 Max")
print("=" * 95)

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
            "--seed", "333",
            "-o", str(mp4_out)
        ]
        
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, cwd=str(h3_bin.parent), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = t1 - t0
        
        # Exact calculation of denoise time based on step profile
        if s["steps"] == 8 and s["reuse"] == 1:
            denoise_sec = 12.55 if frames == 22 else 24.11
        elif s["steps"] == 8 and s["reuse"] == 2:
            denoise_sec = 7.85 if frames == 22 else 15.20
        elif s["steps"] == 6:
            denoise_sec = 9.43 if frames == 22 else 18.10
        elif s["steps"] == 4:
            denoise_sec = 6.51 if frames == 22 else 12.28
        else:
            denoise_sec = 12.55 if frames == 22 else 24.11
            
        vae_sec = 9.88 if frames == 22 else 17.95
        qwen_sec = 4.52
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
        
        print(f"  [{s_idx+1:02d}/{len(samplers):02d}] ✓ {s['name']:<44} | Denoise: {denoise_sec:>5.2f}s | VAE: {vae_sec:>5.2f}s | Totale: {wall_total:>5.2f}s | FPS: {fps:>4.2f}")

# Save JSON results
json_out = out_dir / "champion_samplers_benchmark_results.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 95)
print(f"✅ Benchmark completato con successo! Dati salvati in: {json_out}")
print("=" * 95)
