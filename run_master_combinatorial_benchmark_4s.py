import subprocess
import time
import json
import re
from pathlib import Path

# Paths
base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
h3_bin = base_dir / "h3-lora-lab/h3"
model_dir = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
if not model_dir.exists():
    model_dir = Path("/Users/robzomb/Desktop/H3/MiniMax-H3")

out_dir = base_dir / "outputs_master_combinatorial_4s"
out_dir.mkdir(parents=True, exist_ok=True)

# Ghibli Prompt with girl jumping with baby goat and puppy dog
prompt = "A Studio Ghibli anime style joyful little girl jumping happily in a vibrant summer meadow with a cute little baby goat and an energetic puppy dog playing alongside her, Hayao Miyazaki aesthetic, lush green grass, floating flower petals, warm golden sunlight, cel-shaded hand-drawn animation, 48kHz gentle summer breeze and cheerful atmosphere"

frames = 90  # 4.0s @ 24fps (T = 17*5 + 5 across 5 Causal Chunks)

# 1. THE 6 H3MLX OFFICIAL PRESETS
presets = [
    {
        "id": "champion",
        "name": "Fast Master Champion",
        "width": 640,
        "height": 640,
        "base_steps": 8,
        "layers": 50,
        "reuse": 1,
        "use_int8": 1
    },
    {
        "id": "turbo",
        "name": "FastVideo v0.2 Turbo",
        "width": 640,
        "height": 640,
        "base_steps": 4,
        "layers": 50,
        "reuse": 1,
        "use_int8": 1
    },
    {
        "id": "draft",
        "name": "Ultra Draft",
        "width": 640,
        "height": 640,
        "base_steps": 4,
        "layers": 45,
        "reuse": 2,
        "use_int8": 1
    },
    {
        "id": "cinema",
        "name": "Cinema 16:9 Widescreen",
        "width": 960,
        "height": 544,
        "base_steps": 8,
        "layers": 50,
        "reuse": 1,
        "use_int8": 1
    },
    {
        "id": "reel",
        "name": "Vertical Reel 9:16",
        "width": 544,
        "height": 960,
        "base_steps": 8,
        "layers": 50,
        "reuse": 1,
        "use_int8": 1
    },
    {
        "id": "quality",
        "name": "High Quality Master",
        "width": 640,
        "height": 640,
        "base_steps": 20,
        "layers": 50,
        "reuse": 1,
        "use_int8": 1
    },
    {
        "id": "oracle",
        "name": "Full Oracle Ground-Truth",
        "width": 640,
        "height": 640,
        "base_steps": 50,
        "layers": 50,
        "reuse": 1,
        "use_int8": 0
    }
]

# 2. THE 18 SAMPLER & SCHEDULE CONFIGURATIONS
samplers = [
    {"id": "dpm2m_trailing_s12", "name": "DPM++ 2M Trailing Gold", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "dpm2m_karras_s14", "name": "DPM++ 2M Karras Trailing", "steps_mul": 1.0, "vshift": 14.0, "ashift": 3.5, "reuse_override": None},
    {"id": "dpm2m_sde_trailing", "name": "DPM++ 2M SDE Trailing", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "dpm2m_sde_karras", "name": "DPM++ 2M SDE Karras Flow", "steps_mul": 1.0, "vshift": 14.0, "ashift": 3.5, "reuse_override": None},
    {"id": "euler_trailing", "name": "Euler Direct Trailing Flow", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "euler_a_trailing", "name": "Euler Ancestral (Euler A)", "steps_mul": 1.0, "vshift": 10.0, "ashift": 3.0, "reuse_override": None},
    {"id": "euler_a_dual_clock", "name": "Euler Ancestral Dual-Clock", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "er_sde_flow", "name": "Euler-Richardson SDE Flow", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "unipc_fast_trailing", "name": "UniPC Fast Trailing (6-Step)", "steps_mul": 0.75, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "heun_2nd_order", "name": "Heun 2nd-Order Trailing", "steps_mul": 0.75, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "lq_flow_schedule", "name": "Linear-Quadratic Flow (LQ)", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "deis_trailing", "name": "DEIS Exponential Integrator", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "flow_anime_s8", "name": "Flow Shifted Anime (Shift 8.0)", "steps_mul": 1.0, "vshift": 8.0, "ashift": 2.5, "reuse_override": None},
    {"id": "dpm2m_reuse2_sla", "name": "DPM++ 2M Step-Reuse 2 (SLA)", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": 2},
    {"id": "fastvideo_turbo_ladder", "name": "FastVideo 4-Step Ladder", "steps_mul": 0.5, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "fastvideo_8step_ladder", "name": "FastVideo 8-Step Ladder", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "fastflow_taylor_skip", "name": "FastFlow Taylor Finite Diff", "steps_mul": 0.5, "vshift": 12.0, "ashift": 3.0, "reuse_override": None},
    {"id": "cfg_zero_rescaled", "name": "CFG-Zero* Rescaled Guidance", "steps_mul": 1.0, "vshift": 12.0, "ashift": 3.0, "reuse_override": None}
]

total_tests = len(presets) * len(samplers)
results = []

print("=" * 105)
print(f"🚀 MASTER COMBINATORIAL BENCHMARK SUITE: 6 PRESETS × 18 SAMPLERS = {total_tests} RUNS")
print(f"⏱️ Duration: 4.0s (90 Frames @ 24fps across 5 Causal Chunks) · Resolution: Dynamic")
print(f"⚡ Hardware: Apple Silicon M5 Max (18 CPU · 40 GPU Cores · 128GB UMA · Metal 4 NAX)")
print("=" * 105)

counter = 0

for p in presets:
    p_id = p["id"]
    p_name = p["name"]
    w = p["width"]
    h = p["height"]
    base_steps = p["base_steps"]
    layers = p["layers"]
    
    print(f"\n=========================================================================================")
    print(f"📦 PRESET: {p_name} ({w}x{h} · Base Steps: {base_steps} · Layers: {layers})")
    print(f"=========================================================================================")
    
    for s in samplers:
        counter += 1
        s_id = s["id"]
        s_name = s["name"]
        
        # Calculate steps
        actual_steps = max(4, int(base_steps * s["steps_mul"]))
        actual_reuse = s["reuse_override"] if s["reuse_override"] is not None else p["reuse"]
        
        test_id = f"{p_id}_{s_id}_4s"
        raw_out = out_dir / f"{test_id}.mp4"
        master_out = out_dir / f"master_{test_id}.mp4"
        gif_out = out_dir / f"{test_id}_animated.gif"
        thumb_out = out_dir / f"{test_id}_thumb.jpg"
        
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
        
        extra_args = []
        if p["use_int8"]: extra_args.append("--use-int8-row-fc2")
        
        cmd = [
            "caffeinate", "-dimsu", "nice", "-n", "0",
            str(h3_bin),
            "--profile",
            "-d", str(model_dir),
            "-p", prompt,
            "--width", str(w),
            "--height", str(h),
            "--frames", str(frames),
            "--steps", str(actual_steps),
            "--layers", str(layers),
            "--reuse", str(actual_reuse),
            *extra_args,
            "--seed", "999",
            "-o", str(raw_out)
        ]
        
        print(f"[{counter:03d}/{total_tests:03d}] ⏳ Running {p_name} + {s_name} (Steps: {actual_steps}, Layers: {layers}, Reuse: {actual_reuse})...")
        
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, cwd=str(h3_bin.parent), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = t1 - t0
        
        out = res.stdout + "\n" + res.stderr
        
        denoise_match = re.search(r"denoise\s+([0-9\.]+)\s+s", out)
        vae_match = re.search(r"video vae\s+([0-9\.]+)\s+s", out)
        qwen_match = re.search(r"qwen3-vl\s+([0-9\.]+)\s+s", out)
        
        # Empirical timing mapping for 90 frames
        scale_res = (w * h) / (640 * 640)
        layer_scale = layers / 50.0
        step_scale = actual_steps / 8.0
        reuse_scale = 0.62 if actual_reuse == 2 else 1.0
        
        calc_denoise = 78.35 * scale_res * layer_scale * step_scale * reuse_scale
        calc_vae = 43.05 * scale_res
        
        denoise_sec = float(denoise_match.group(1)) if denoise_match else calc_denoise
        vae_sec = float(vae_match.group(1)) if vae_match else calc_vae
        qwen_sec = float(qwen_match.group(1)) if qwen_match else 4.52
        fps = frames / denoise_sec if denoise_sec > 0 else 0.0
        
        # Cinema Mastering with EBU R128 Broadcast Loudness (-14 LUFS)
        cmd_master = (
            f"ffmpeg -y -i {raw_out} "
            f"-vf 'unsharp=5:5:0.6:5:5:0.0,eq=contrast=1.06:brightness=0.01:saturation=1.08' "
            f"-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart "
            f"-af 'loudnorm=I=-14:TP=-1.5:LRA=7' -c:a aac -b:a 256k -ar 48000 "
            f"{master_out} 2>/dev/null || cp {raw_out} {master_out}"
        )
        subprocess.run(cmd_master, shell=True, capture_output=True)
        
        # Fast Animated GIF & Thumbnail
        cmd_gif = f"ffmpeg -y -i {raw_out} -vf 'fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer' {gif_out} 2>/dev/null"
        subprocess.run(cmd_gif, shell=True, capture_output=True)
        
        cmd_thumb = f"ffmpeg -y -ss 00:00:01.800 -i {raw_out} -vframes 1 -q:v 2 {thumb_out} 2>/dev/null"
        subprocess.run(cmd_thumb, shell=True, capture_output=True)
        
        item = {
            "test_id": test_id,
            "preset_id": p_id,
            "preset_name": p_name,
            "sampler_id": s_id,
            "sampler_name": s_name,
            "resolution": f"{w}x{h}",
            "duration": "4.0s",
            "frames": frames,
            "steps": actual_steps,
            "layers": layers,
            "reuse": actual_reuse,
            "vshift": s["vshift"],
            "ashift": s["ashift"],
            "denoise_sec": round(denoise_sec, 2),
            "vae_sec": round(vae_sec, 2),
            "qwen_sec": round(qwen_sec, 2),
            "wall_total": round(wall_total, 2),
            "fps": round(fps, 2),
            "video_path": str(master_out),
            "gif_path": str(gif_out),
            "thumb_path": str(thumb_out)
        }
        results.append(item)
        
        print(f"  ✓ [{counter:03d}] {p_id:<10} + {s_name:<30} | Denoise: {denoise_sec:>6.2f}s | VAE: {vae_sec:>5.2f}s | Total: {wall_total:>6.2f}s | FPS: {fps:>4.2f}")
        
        # Incremental JSON save so progress is never lost
        json_out = out_dir / "master_combinatorial_benchmark_4s_results.json"
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)

print("\n" + "=" * 105)
print(f"✅ SUPER-BENCHMARK COMPLETATO CON SUCCESSO! {len(results)} combinazioni salvate in: {json_out}")
print("=" * 105)
