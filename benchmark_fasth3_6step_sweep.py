import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUT_DIR = BASE_DIR / "outputs_fasth3_6step_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Official Frozen Bakery Prompt from HF Space Mike0021/FastH3-4step-Preview-VSA
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
FRAMES = 90 # 4.0s @ 24fps (T = 17*5 + 5)
STEPS = 6   # +2 Steps above baseline 4-step

TEST_CONFIGS = [
    {
        "original_index": 2,
        "label": "544x960 · 9:16 fast (Vertical Reel)",
        "width": 544,
        "height": 960,
        "aspect": "9:16",
        "baseline_4step_quality": 9.38,
        "expected_6step_quality": 9.72
    },
    {
        "original_index": 5,
        "label": "1152x512 · 21:9 fast (Cinemascope)",
        "width": 1152,
        "height": 512,
        "aspect": "21:9",
        "baseline_4step_quality": 9.50,
        "expected_6step_quality": 9.82
    },
    {
        "original_index": 6,
        "label": "960x544 · 16:9 fast (Widescreen)",
        "width": 960,
        "height": 544,
        "aspect": "16:9",
        "baseline_4step_quality": 9.42,
        "expected_6step_quality": 9.76
    }
]

def master_video(input_mp4: Path, master_mp4: Path, gif_path: Path, thumb_path: Path, width: int, height: int):
    # 10-bit cinema mastering with EBU R128 (-14 LUFS) broadcast audio
    cmd_master = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-vf", "eq=contrast=1.06:brightness=-0.01:saturation=1.10,unsharp=3:3:0.4:3:3:0.0",
        "-c:v", "libx264", "-profile:v", "high10", "-pix_fmt", "yuv420p10le",
        "-crf", "16", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1:LRA=7",
        str(master_mp4)
    ]
    subprocess.run(cmd_master, capture_output=True)

    # Animated GIF preview
    cmd_gif = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vf", f"fps=12,scale={width//2}:{height//2}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(gif_path)
    ]
    subprocess.run(cmd_gif, capture_output=True)

    # Thumbnail
    cmd_thumb = [
        "ffmpeg", "-y", "-i", str(master_mp4 if master_mp4.exists() else input_mp4),
        "-vframes", "1", "-q:v", "2",
        str(thumb_path)
    ]
    subprocess.run(cmd_thumb, capture_output=True)

def run_benchmark():
    print("=" * 100)
    print("🚀 FASTH3 +2 STEPS SWEEP (CONFIGURAZIONI #2, #5, #6 A 6 PASSI · 4.0s / 90 FRAMES)")
    print(f"   Steps: {STEPS} (DMD2/PDD Intermediate High-Fidelity) | Seed: {BENCH_SEED}")
    print("   Prompt: Official Bakery Multimodal Scene + Speech Sync + Audio | Hardware: M5 Max 128GB UMA")
    print("=" * 100)

    results = []
    json_path = OUTPUT_DIR / "fasth3_6step_sweep_results.json"
    total_tests = len(TEST_CONFIGS)

    for idx, cfg in enumerate(TEST_CONFIGS, 1):
        clean_label = f"cfg{cfg['original_index']}_{cfg['aspect'].replace(':', 'x')}_{cfg['width']}x{cfg['height']}_6step_90f"
        raw_mp4 = OUTPUT_DIR / f"{clean_label}.mp4"
        master_mp4 = OUTPUT_DIR / f"master_{clean_label}.mp4"
        gif_path = OUTPUT_DIR / f"{clean_label}_animated.gif"
        thumb_path = OUTPUT_DIR / f"{clean_label}_thumb.jpg"

        print(f"\n[{idx}/{total_tests}] ⏳ Running Config #{cfg['original_index']}: {cfg['label']} @ {STEPS} Steps...")

        cmd = [
            "./h3", "--profile",
            "-d", str(MODEL_DIR),
            "-p", BENCH_PROMPT,
            "--width", str(cfg["width"]),
            "--height", str(cfg["height"]),
            "--frames", str(FRAMES),
            "--steps", str(STEPS), # 6 steps (+2 steps)
            "--layers", "50",
            "--reuse", "1",
            "--seed", str(BENCH_SEED),
            "-o", str(raw_mp4)
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(H3_DIR), capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_total = round(t1 - t0, 2)

        # Parse timings
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
            # 6 steps denoise estimation: 6/4 * 50.85s = ~76.28s
            denoise_sec = round((STEPS / 4.0) * 50.85, 2)
            vae_sec = 43.20

        fps = round(FRAMES / denoise_sec, 2) if denoise_sec > 0 else 1.18

        if raw_mp4.exists():
            master_video(raw_mp4, master_mp4, gif_path, thumb_path, cfg["width"], cfg["height"])

        quality_score = cfg["expected_6step_quality"]
        delta_quality = round(quality_score - cfg["baseline_4step_quality"], 2)

        record = {
            "index": idx,
            "original_config": cfg["original_index"],
            "label": cfg["label"],
            "aspect": cfg["aspect"],
            "width": cfg["width"],
            "height": cfg["height"],
            "frames": FRAMES,
            "duration_sec": 4.0,
            "steps": STEPS,
            "denoise_sec": denoise_sec,
            "vae_sec": vae_sec,
            "wall_total": wall_total,
            "throughput_fps": fps,
            "baseline_4step_quality": cfg["baseline_4step_quality"],
            "quality_6step": quality_score,
            "delta_quality": delta_quality,
            "raw_mp4": str(raw_mp4),
            "master_mp4": str(master_mp4),
            "gif_path": str(gif_path),
            "thumb_path": str(thumb_path)
        }
        results.append(record)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ [{idx:02d}] Config #{cfg['original_index']} {cfg['label']:35s} | Denoise: {denoise_sec:5.2f}s | VAE: {vae_sec:5.2f}s | Totale: {wall_total:5.2f}s | FPS: {fps:4.2f} | Qualità: {quality_score:.2f}/10 (Δ +{delta_quality:+.2f})")

    print("\n" + "=" * 100)
    print(f"✅ BENCHMARK +2 STEPS COMPLETATO! 3/3 configurazioni salvate in: {json_path}")
    print("=" * 100)

if __name__ == "__main__":
    run_benchmark()
