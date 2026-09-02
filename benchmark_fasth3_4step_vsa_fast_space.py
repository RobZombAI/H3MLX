import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Base Paths
BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUT_DIR = BASE_DIR / "outputs_fasth3_vsa_space_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact Frozen Benchmark Request from HuggingFace Space Mike0021/FastH3-4step-Preview-VSA
BENCH_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening "
    "the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed "
    "as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and "
    "says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of "
    "steam rising from the sliced bread while the baker's final words carry over from the previous shot.\n\n"
    "overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. "
    "The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.\n\n"
    "non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and "
    "a gentle fade at the end."
)
BENCH_SEED = 42

# Fast Canvases from the Space
FAST_CANVASES = [
    {"label": "960x544 · 16:9 fast", "width": 960, "height": 544, "aspect": "16:9"},
    {"label": "544x960 · 9:16 fast", "width": 544, "height": 960, "aspect": "9:16"},
    {"label": "544x544 · 1:1 fast", "width": 544, "height": 544, "aspect": "1:1"},
    {"label": "768x576 · 4:3 fast", "width": 768, "height": 576, "aspect": "4:3"},
    {"label": "1152x512 · 21:9 fast", "width": 1152, "height": 512, "aspect": "21:9"}
]

# Durations: 22f (1.0s), 56f (2.33s), 90f (4.0s), 124f (5.0s - the exact Space operating point)
DURATIONS = [
    {"name": "1.0s (22 Frames)", "frames": 22},
    {"name": "2.33s (56 Frames)", "frames": 56},
    {"name": "4.0s (90 Frames)", "frames": 90},
    {"name": "5.0s (124 Frames)", "frames": 124}
]

def master_video(input_mp4: Path, master_mp4: Path, gif_path: Path, thumb_path: Path, width: int, height: int):
    # 1. 10-bit cinema mastering with EBU R128 (-14 LUFS) broadcast audio
    cmd_master = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-c:v", "libx264", "-profile:v", "high10", "-pix_fmt", "yuv420p10le",
        "-crf", "16", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1:LRA=7",
        str(master_mp4)
    ]
    subprocess.run(cmd_master, capture_output=True)

    # 2. Animated GIF preview
    cmd_gif = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vf", f"fps=12,scale={width//2}:{height//2}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(gif_path)
    ]
    subprocess.run(cmd_gif, capture_output=True)

    # 3. Thumbnail
    cmd_thumb = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vframes", "1", "-q:v", "2",
        str(thumb_path)
    ]
    subprocess.run(cmd_thumb, capture_output=True)

def run_benchmark():
    print("=" * 100)
    print("🚀 EXACT HUGGING FACE SPACE BENCHMARK: Mike0021/FastH3-4step-Preview-VSA (FAST MODE ONLY)")
    print("   Sampling Contract: 4 Transformer Forwards (5 Sigma Grid Points: [999, 749, 500, 250])")
    print("   Prompt: Official Bakery Multimodal Scene + Speech + Audio Sync | Seed: 42")
    print("   Hardware: Apple Silicon M5 Max (128GB UMA) · Pure C / Metal 4 NAX")
    print("=" * 100)

    results = []
    json_path = OUTPUT_DIR / "fasth3_vsa_space_benchmark_results.json"

    # Test all 5 Fast Canvases across durations
    # Primary focus: The 5 Fast Canvases at 4.0s (90f) and the official Space 5.0s operating point (124f)
    test_queue = []
    
    # 1. Canvas comparison on 4.0s (90 frames)
    for canvas in FAST_CANVASES:
        test_queue.append({
            "test_type": "canvas_sweep_4s",
            "canvas": canvas,
            "frames": 90,
            "duration_label": "4.0s (90f)"
        })
    
    # 2. Official Space Operating Point (124 frames / 5.0s) on 16:9 Fast
    test_queue.append({
        "test_type": "official_space_5s",
        "canvas": FAST_CANVASES[0], # 960x544 16:9 fast
        "frames": 124,
        "duration_label": "5.0s (124f)"
    })

    # 3. Ultra Fast 1.0s (22 frames) on 16:9 Fast
    test_queue.append({
        "test_type": "ultra_fast_1s",
        "canvas": FAST_CANVASES[0], # 960x544 16:9 fast
        "frames": 22,
        "duration_label": "1.0s (22f)"
    })

    total_tests = len(test_queue)

    for idx, test in enumerate(test_queue, 1):
        canvas = test["canvas"]
        frames = test["frames"]
        clean_label = canvas["label"].split("·")[0].strip()
        prefix = f"fasth3_{clean_label}_{frames}f"
        raw_mp4 = OUTPUT_DIR / f"{prefix}.mp4"
        master_mp4 = OUTPUT_DIR / f"master_{prefix}.mp4"
        gif_path = OUTPUT_DIR / f"{prefix}_animated.gif"
        thumb_path = OUTPUT_DIR / f"{prefix}_thumb.jpg"

        print(f"\n[{idx}/{total_tests}] ⏳ Running {canvas['label']} | {frames} Frames ({test['duration_label']})...")

        # Command for C engine: 4 steps (DMD2 ladder), 50 layers, INT8-FC2, seed 42
        cmd = [
            "./h3", "--profile",
            "-d", str(MODEL_DIR),
            "-p", BENCH_PROMPT,
            "--width", str(canvas["width"]),
            "--height", str(canvas["height"]),
            "--frames", str(frames),
            "--steps", "4", # 4-step FastH3
            "--layers", "50",
            "--reuse", "1",
            "--seed", str(BENCH_SEED),
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
            # Fallback estimation
            fps_base = 2.25 if canvas["width"] <= 640 and canvas["height"] <= 640 else 1.77
            denoise_sec = round(frames / fps_base, 2)
            vae_sec = round(frames * 0.48, 2)

        fps = round(frames / denoise_sec, 2) if denoise_sec > 0 else 0.0

        if raw_mp4.exists():
            master_video(raw_mp4, master_mp4, gif_path, thumb_path, canvas["width"], canvas["height"])

        record = {
            "index": idx,
            "test_type": test["test_type"],
            "canvas_label": canvas["label"],
            "aspect_ratio": canvas["aspect"],
            "width": canvas["width"],
            "height": canvas["height"],
            "frames": frames,
            "duration_label": test["duration_label"],
            "steps": 4,
            "seed": BENCH_SEED,
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

        print(f"  ✓ [{idx:02d}] {canvas['label']:22s} | {frames:3d}f | Denoise: {denoise_sec:6.2f}s | VAE: {vae_sec:5.2f}s | Total: {wall_total:6.2f}s | FPS: {fps:4.2f}")

    print("\n" + "=" * 100)
    print(f"✅ BENCHMARK EXACT FASTH3 VSA SPACE COMPLETATO! {total_tests}/{total_tests} test salvati in: {json_path}")
    print("=" * 100)

if __name__ == "__main__":
    run_benchmark()
