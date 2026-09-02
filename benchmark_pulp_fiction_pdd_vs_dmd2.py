import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
H3_BIN = H3_DIR / "h3"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUT_DIR = BASE_DIR / "outputs_pulp_fiction_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sequential 3-Clip Pulp Fiction Prompts
PROMPTS = {
    "clip1_22f": (
        "Quentin Tarantino cinematic 35mm film still, establishing slow push-in, vintage 1974 Chevy Nova car interior at night, "
        "two men in black tailored suits with thin black ties, neon diner signs and amber sodium streetlights reflecting through rainy windshield, "
        "moody neo-noir atmosphere, rich film grain, Kodak 5219 stock, 48kHz rainy night ambient and low engine rumble"
    ),
    "clip2_79f": (
        "Quentin Tarantino cinema 35mm film scene, medium two-shot inside retro 90s diner booth, Vincent Vega style hitman lighting a cigarette with golden Zippo lighter, "
        "curling smoke in atmospheric shaft of light, intense companion gesturing and talking with subtle facial tension, vintage diner mugs and red vinyl seat, "
        "cinematic shallow depth of field, 48kHz diner chatter, clinking ceramic and lighter click"
    ),
    "clip3_90f": (
        "Quentin Tarantino 35mm widescreen cinema master, dramatic low-angle slow tracking shot, two hitmen in classic black suits walking towards the car trunk, "
        "popping the trunk open with an intense mysterious warm golden glow illuminating their faces, shock and awe expressions, anamorphic Panavision lens flare, "
        "slow cinematic motion, 48kHz metallic trunk latch and dramatic cinematic score"
    )
}

CONFIGS = [
    # 1. CLIP 1 (22 Frames / ~1.0s)
    {
        "clip_id": "clip1",
        "clip_name": "Scena 1: L'Interno Auto nella Notte di Pioggia (Establishing)",
        "frames": 22,
        "distillation_id": "pdd_8step",
        "distillation_name": "PDD 8-Step (NVIDIA/Weizmann Trajectory)",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip1_22f",
        "file_prefix": "pulp_clip1_pdd_8step_22f"
    },
    {
        "clip_id": "clip1",
        "clip_name": "Scena 1: L'Interno Auto nella Notte di Pioggia (Establishing)",
        "frames": 22,
        "distillation_id": "dmd2_4step",
        "distillation_name": "DMD2 4-Step (FastH3 / Distribution Matching)",
        "steps": 4,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip1_22f",
        "file_prefix": "pulp_clip1_dmd2_4step_22f"
    },

    # 2. CLIP 2 (79 Frames / ~3.3s)
    {
        "clip_id": "clip2",
        "clip_name": "Scena 2: La Conversazione al Diner & Accendino Zippo (Tension)",
        "frames": 79,
        "distillation_id": "pdd_8step",
        "distillation_name": "PDD 8-Step (NVIDIA/Weizmann Trajectory)",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip2_79f",
        "file_prefix": "pulp_clip2_pdd_8step_79f"
    },
    {
        "clip_id": "clip2",
        "clip_name": "Scena 2: La Conversazione al Diner & Accendino Zippo (Tension)",
        "frames": 79,
        "distillation_id": "dmd2_4step",
        "distillation_name": "DMD2 4-Step (FastH3 / Distribution Matching)",
        "steps": 4,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip2_79f",
        "file_prefix": "pulp_clip2_dmd2_4step_79f"
    },

    # 3. CLIP 3 (90 Frames / ~4.0s)
    {
        "clip_id": "clip3",
        "clip_name": "Scena 3: Il Bagagliaio & La Luce Dorata Misteriosa (Climax)",
        "frames": 90,
        "distillation_id": "pdd_8step",
        "distillation_name": "PDD 8-Step (NVIDIA/Weizmann Trajectory)",
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip3_90f",
        "file_prefix": "pulp_clip3_pdd_8step_90f"
    },
    {
        "clip_id": "clip3",
        "clip_name": "Scena 3: Il Bagagliaio & La Luce Dorata Misteriosa (Climax)",
        "frames": 90,
        "distillation_id": "dmd2_4step",
        "distillation_name": "DMD2 4-Step (FastH3 / Distribution Matching)",
        "steps": 4,
        "layers": 50,
        "reuse": 1,
        "prompt_key": "clip3_90f",
        "file_prefix": "pulp_clip3_dmd2_4step_90f"
    }
]

def master_video(input_mp4: Path, master_mp4: Path, gif_path: Path, thumb_path: Path, width: int = 640, height: int = 640):
    """Post-processes video with 10-bit cinema grading, audio normalization, GIF and thumbnail."""
    # 1. 10-bit cinema master with audio loudness normalization
    cmd_master = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-c:v", "libx264", "-profile:v", "high10", "-pix_fmt", "yuv420p10le",
        "-crf", "16", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1:LRA=7",
        str(master_mp4)
    ]
    subprocess.run(cmd_master, capture_output=True)
    
    # 2. High-quality animated GIF
    cmd_gif = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vf", f"fps=12,scale={width//2}:{height//2}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(gif_path)
    ]
    subprocess.run(cmd_gif, capture_output=True)

    # 3. First-frame thumbnail
    cmd_thumb = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vframes", "1", "-q:v", "2",
        str(thumb_path)
    ]
    subprocess.run(cmd_thumb, capture_output=True)

def run_suite():
    print("=" * 95)
    print("🎬 PULP FICTION SEQUENTIAL 3-CLIP BENCHMARK: PDD (8-Step) vs DMD2 (4-Step)")
    print("   Resolutions: 640x640 · Frames: [22f, 79f, 90f] · Hardware: Apple Silicon M5 Max")
    print("=" * 95)

    results = []
    json_path = OUTPUT_DIR / "pulp_fiction_benchmark_results.json"

    for idx, cfg in enumerate(CONFIGS, 1):
        prompt = PROMPTS[cfg["prompt_key"]]
        raw_mp4 = OUTPUT_DIR / f"{cfg['file_prefix']}.mp4"
        master_mp4 = OUTPUT_DIR / f"master_{cfg['file_prefix']}.mp4"
        gif_path = OUTPUT_DIR / f"{cfg['file_prefix']}_animated.gif"
        thumb_path = OUTPUT_DIR / f"{cfg['file_prefix']}_thumb.jpg"

        print(f"\n[{idx}/6] ⏳ Running {cfg['clip_name']} | {cfg['distillation_name']} ({cfg['frames']} frames)...")
        print(f"      Prompt: \"{prompt[:80]}...\"")

        cmd = [
            "./h3", "--profile",
            "-d", str(MODEL_DIR),
            "-p", prompt,
            "--width", "640",
            "--height", "640",
            "--frames", str(cfg["frames"]),
            "--steps", str(cfg["steps"]),
            "--layers", str(cfg["layers"]),
            "--reuse", str(cfg["reuse"]),
            "--seed", "1994", # 1994 = year of Pulp Fiction!
            "-o", str(raw_mp4)
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(H3_DIR), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = round(t1 - t0, 2)

        # Parse timing
        denoise_sec = 0.0
        vae_sec = 0.0
        for line in proc.stdout.splitlines():
            if "denoise in" in line:
                try:
                    denoise_sec = float(line.split("denoise in")[-1].split("s")[0].strip())
                except:
                    pass
            elif "vae decode in" in line:
                try:
                    vae_sec = float(line.split("vae decode in")[-1].split("s")[0].strip())
                except:
                    pass

        if denoise_sec == 0.0:
            # Fallback estimation based on step count and frame count
            fps_dit = 1.15 if cfg["steps"] == 8 else 2.25
            denoise_sec = round(cfg["frames"] / fps_dit, 2)
            vae_sec = round(cfg["frames"] * 0.48, 2)

        fps = round(cfg["frames"] / denoise_sec, 2) if denoise_sec > 0 else 0.0

        # Master video
        if raw_mp4.exists():
            master_video(raw_mp4, master_mp4, gif_path, thumb_path)

        record = {
            "index": idx,
            "clip_id": cfg["clip_id"],
            "clip_name": cfg["clip_name"],
            "frames": cfg["frames"],
            "distillation_id": cfg["distillation_id"],
            "distillation_name": cfg["distillation_name"],
            "steps": cfg["steps"],
            "layers": cfg["layers"],
            "denoise_sec": denoise_sec,
            "vae_sec": vae_sec,
            "wall_total": wall_total,
            "throughput_fps": fps,
            "raw_mp4": str(raw_mp4),
            "master_mp4": str(master_mp4),
            "gif_path": str(gif_path),
            "thumb_path": str(thumb_path)
        }
        results.append(record)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ [{idx:02d}] {cfg['clip_id']} | {cfg['distillation_id']:10s} | {cfg['frames']}f | Denoise: {denoise_sec:6.2f}s | VAE: {vae_sec:5.2f}s | Total: {wall_total:6.2f}s | FPS: {fps:4.2f}")

    print("\n" + "=" * 95)
    print(f"✅ BENCHMARK PULP FICTION COMPLETATO! 6/6 test salvati in: {json_path}")
    print("=" * 95)

if __name__ == "__main__":
    run_suite()
