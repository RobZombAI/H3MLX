#!/usr/bin/env python3
"""
👑 H3 MASTER CINEMA 4K RUNNER
Encapsulates the undisputed Champion Gold Configuration (John Wick / Hollywood Master Standard):
- Camera: Arri Alexa LF + Cooke Anamorphic S4/i Prime 50mm T2.3
- MTF Sub-Pixel Phase Coherence + Specular Rain/Pores
- Apple Silicon Metal 4 NAX Row-Major INT8 FC2 (50 Full Layers)
- 14-Step Optimal / 23-Step Euler Sampler
- 4K UHD Master (3840x2160, 10-bit yuv420p10le, Lanczos-4)
- 48 kHz Spatial Audio Foley Mastering
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
H3_BIN = H3_DIR / "h3"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
DOWNLOADS_DIR = Path.home() / "Downloads"

def build_master_prompt(user_subject_action: str) -> str:
    """Inietta la formula ottica Cooke Anamorphic / Arri Alexa LF sul soggetto e azione richiesti."""
    return (
        f"Shot on Arri Alexa LF with Cooke Anamorphic S4i Prime 50mm T2.3 lens, MTF optical sub-pixel phase coherence, "
        f"pristine Hollywood master medium-close action tracking shot in heavy torrential night rain, {user_subject_action}, "
        f"crisp tailored costume texture with realistic cloth physics, wet facial skin pores and airborne rain droplets, "
        f"vibrant neon cyan and amber background bokeh, wet street reflections, anamorphic lens flare, 4k 24fps master"
    )

def run_master_generation(
    subject_action: str,
    output_name: str = "cinema_master",
    width: int = 768,
    height: int = 512,
    seconds: float = 4.0,
    steps: int = 14,
    reuse: int = 2,
    seed: int = 5555
):
    frames = int(seconds * 24)
    # Temporal lattice constraint T = 17n + 5 approx
    if frames == 96:
        frames = 90
    elif frames == 72:
        frames = 73
        
    prompt = build_master_prompt(subject_action)
    raw_mp4 = BASE_DIR / "outputs" / f"{output_name}_raw.mp4"
    raw_mp4.parent.mkdir(parents=True, exist_ok=True)
    master_4k_mp4 = DOWNLOADS_DIR / f"{output_name}_4k_master.mp4"
    
    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NAX"] = "qkv-attn"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_GPU_SAMPLER"] = "1"
    env["OMP_NUM_THREADS"] = "18"
    
    print("=" * 84)
    print("👑 H3 MASTER CINEMA 4K GENERATOR (HOLLYWOOD STANDARD)")
    print(f" • Soggetto & Azione : {subject_action}")
    print(f" • Risoluzione Nativa: {width}x{height} (-> 4K UHD 3840x2160 10-bit)")
    print(f" • Durata            : {seconds}s ({frames} frame @ 24fps)")
    print(f" • Step & Schedulazione: {steps} Step (Reuse {reuse}) · 50 Layer INT8 FC2")
    print(f" • Seed              : {seed}")
    print("=" * 84)
    
    cmd_dit = [
        str(H3_BIN), "--profile",
        "-d", str(MODEL_DIR),
        "-p", prompt,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--steps", str(steps),
        "--layers", "50",
        "--reuse", str(reuse),
        "--use-int8-row-fc2",
        "--seed", str(seed),
        "-o", str(raw_mp4)
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd_dit, env=env, cwd=H3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_dit = time.time() - t0
    
    if res.returncode != 0 or not raw_mp4.exists():
        print(f"❌ Errore durante l'inferenza DiT: {res.returncode}")
        print(res.stdout[-500:])
        return False
        
    print(f"✅ DiT Generation & VAE completati in {t_dit:.2f}s")
    
    # 4K Super-Resolution + Audio Foley Mastering
    print("💎 Esecuzione Mastering 4K UHD & Sound Design Foley Dinamico (48 kHz Stereo)...")
    from h3_cinema_sound_designer import inject_foley_to_video
    t_4k_0 = time.time()
    
    temp_4k = BASE_DIR / "outputs" / f"{output_name}_temp_4k.mp4"
    cmd_4k = [
        "ffmpeg", "-y", "-i", str(raw_mp4),
        "-vf", "scale=3840:2160:flags=lanczos+accurate_rnd+full_chroma_int,unsharp=5:5:0.85:5:5:0.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-preset", "fast", "-crf", "14",
        "-an",
        str(temp_4k)
    ]
    subprocess.run(cmd_4k, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Inietta il Sound Design Foley sintetizzato a 48 kHz
    inject_foley_to_video(temp_4k, prompt, master_4k_mp4)
    if temp_4k.exists():
        temp_4k.unlink()
        
    t_4k = time.time() - t_4k_0
    
    total_time = t_dit + t_4k
    size_mb = master_4k_mp4.stat().st_size / (1024 * 1024) if master_4k_mp4.exists() else 0
    
    print("=" * 84)
    print("🏆 MASTER 4K COMPLETATO CON SUCCESSO!")
    print(f" • Tempo Totale Generazione : {total_time:.2f}s (DIT: {t_dit:.2f}s + 4K: {t_4k:.2f}s)")
    print(f" • Dimensione Master 4K     : {size_mb:.2f} MB")
    print(f" • File Salvato in          : {master_4k_mp4}")
    print("=" * 84)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3 Master Cinema 4K Runner")
    parser.add_argument("subject", type=str, help="Soggetto e azione da filmare")
    parser.add_argument("-o", "--output", type=str, default="cinema_master", help="Nome file output")
    parser.add_argument("-w", "--width", type=int, default=768, help="Larghezza nativa")
    parser.add_argument("-H", "--height", type=int, default=512, help="Altezza nativa")
    parser.add_argument("-s", "--seconds", type=float, default=4.0, help="Secondi di video")
    parser.add_argument("--steps", type=int, default=14, help="Numero di step")
    parser.add_argument("--seed", type=int, default=5555, help="Seed di generazione")
    args = parser.parse_args()
    
    run_master_generation(
        subject_action=args.subject,
        output_name=args.output,
        width=args.width,
        height=args.height,
        seconds=args.seconds,
        steps=args.steps,
        seed=args.seed
    )
