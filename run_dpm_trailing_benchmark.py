import subprocess
import time
import json
import re
from pathlib import Path

# Paths
h3_bin = Path("/Users/robzomb/Documents/antigravity/cool-hopper/h3-lora-lab/h3")
model_dir = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
if not model_dir.exists():
    model_dir = Path("/Users/robzomb/Desktop/H3/MiniMax-H3")

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_dpm_trailing")
out_dir.mkdir(parents=True, exist_ok=True)

# Define the compatible DPM++ Trailing sets from Draw Things category
dpm_configs = [
    {
        "id": "dpm2m_trailing_s12_8step",
        "name": "DPM++ 2M Trailing (Shift 12.0 / 8-Step)",
        "desc": "Draw Things Gold Standard Trailing Flow",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_trailing_s16_8step",
        "name": "DPM++ 2M Trailing Macro (Shift 16.0 / 8-Step)",
        "desc": "Draw Things High-Shift Macro/Portrait Focus",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 16.0,
        "ashift": 4.0
    },
    {
        "id": "dpm2m_trailing_s8_8step",
        "name": "DPM++ 2M Trailing Balanced (Shift 8.0 / 8-Step)",
        "desc": "Draw Things Smooth Cinematic Motion & Landscapes",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "vshift": 8.0,
        "ashift": 2.5
    },
    {
        "id": "dpm2m_trailing_fast_6step",
        "name": "DPM++ 2M Fast Trailing (Shift 12.0 / 6-Step)",
        "desc": "Draw Things 6-Step Trailing Acceleration",
        "steps": 6,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_trailing_precision_12step",
        "name": "DPM++ 2M Precision Trailing (Shift 12.0 / 12-Step)",
        "desc": "Draw Things 12-Step High-Precision Trajectory",
        "steps": 12,
        "layers": 50,
        "reuse": 1,
        "vshift": 12.0,
        "ashift": 3.0
    },
    {
        "id": "dpm2m_trailing_reuse2_8step",
        "name": "DPM++ 2M Trailing + Step-Reuse 2 (8-Step)",
        "desc": "Draw Things Trailing with Attention Caching",
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

prompt = "A majestic golden eagle soaring gracefully over snow-capped alpine peaks at sunrise, crisp 8k definition, specular feather details catching golden sunlight, 35mm f/1.4 lens bokeh, 48kHz spatial mountain wind."

results = []

print("=" * 85)
print("🚀 BENCHMARK: Draw Things DPM++ 2M Trailing Compatible Sets on Apple Silicon M5 Max")
print("=" * 85)

for dur in durations:
    frames = dur["frames"]
    d_lbl = dur["label"]
    print(f"\n--- ⏱️ DURATION: {dur['desc']} ---")
    
    for cfg in dpm_configs:
        cfg_id = cfg["id"]
        test_id = f"{cfg_id}_{d_lbl}"
        mp4_out = out_dir / f"{test_id}.mp4"
        
        env = {
            "H3_PROFILE": "1",
            "H3_NAX": "qkv-attn",
            "H3_ZERO_COPY_WEIGHTS": "1",
            "H3_REUSE_MPS_COMMAND": "1",
            "H3_GPU_SAMPLER": "1",
            "H3_VIDEO_SHIFT": str(cfg["vshift"]),
            "H3_AUDIO_SHIFT": str(cfg["ashift"]),
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
            "--steps", str(cfg["steps"]),
            "--layers", str(cfg["layers"]),
            "--reuse", str(cfg["reuse"]),
            "--use-int8-row-fc2",
            "--seed", "333",
            "-o", str(mp4_out)
        ]
        
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, cwd=str(h3_bin.parent), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = t1 - t0
        
        # Parse profile metrics from output
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
            "name": cfg["name"],
            "desc": cfg["desc"],
            "steps": cfg["steps"],
            "layers": cfg["layers"],
            "reuse": cfg["reuse"],
            "vshift": cfg["vshift"],
            "ashift": cfg["ashift"],
            "denoise_sec": round(denoise_sec, 2),
            "vae_sec": round(vae_sec, 2),
            "qwen_sec": round(qwen_sec, 2),
            "wall_total": round(wall_total, 2),
            "fps": round(fps, 2)
        }
        results.append(item)
        
        print(f"  ✓ {cfg['name']:<48} | Denoise: {denoise_sec:>5.2f}s | VAE: {vae_sec:>5.2f}s | Total: {wall_total:>5.2f}s | Throughput: {fps:>4.2f} FPS")

# Save results JSON locally
json_out = out_dir / "dpm_trailing_matrix_results.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 85)
print(f"✅ Benchmark completed successfully! Results saved to {json_out}")
print("=" * 85)
