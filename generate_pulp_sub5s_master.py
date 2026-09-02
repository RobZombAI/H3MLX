#!/usr/bin/env python3
"""
🎬 Unified Production Engine for MiniMax H3 High-Fidelity Video Generation
========================================================================
Supports:
  1. Monolithic 5s 1080p Cinema Master (Pyramidal Layer Scheduling + N-Gram Detail)
  2. Sub-5s Fast Chained Master (2x 56-frame chunks, Causal Lattice Validated)
  3. Continuous 15s/30s Cinematic Masterpiece (Jack Rabbit Slims Dance Floor)
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")

PULP_PROMPTS = {
    "dance_opening": (
        "Quentin Tarantino 35mm cinema master, 1994 Jack Rabbit Slims diner dance floor, "
        "cinematic medium two-shot at eye level, Mia Wallace and Vincent Vega in the iconic twist contest. "
        "Mia Wallace with jet-black blunt bob haircut, straight bangs, intense dark brown eyes with specular reflections, "
        "crimson red lipstick, crisp oversized white button-up collared shirt. Vincent Vega in black tailored suit, "
        "white collared shirt, silver bolo tie, slicked-back dark hair. Both smiling with authentic eye contact, "
        "dancing with natural 1950s twist arm rhythm at waist level, warm ambient chiaroscuro diner lighting, "
        "glowing horizontal red and turquoise neon background, vintage diner booths with soft circular bokeh, "
        "photorealistic 8k, authentic Kodak Vision3 5219 film stock, 48kHz vintage rock acoustics"
    ),
    "dance_middle": (
        "Quentin Tarantino 35mm cinema master, Jack Rabbit Slims diner dance floor, "
        "Mia Wallace and Vincent Vega in continuous twist rhythm. Mia Wallace smiling playfully, "
        "crisp white collared shirt folds, black cigarette pants, fluid hip sway. "
        "Vincent Vega laughing with eye wrinkles, tailored black suit jacket, silver bolo tie, "
        "rhythmic shoulder bounce, warm neon diner reflections, Kodak Vision3 5219 film stock, 48kHz synchronized acoustics"
    ),
    "dance_finale": (
        "Quentin Tarantino 35mm cinema master, Jack Rabbit Slims diner dance floor, "
        "Mia Wallace and Vincent Vega smoothly completing their iconic twist dance routine. "
        "Mia Wallace in white collared shirt and black bob, Vincent Vega in black tailored suit, "
        "warm mutual smile, relaxed posture, atmospheric diner neon backdrop with soft bokeh, "
        "35mm film master, 48kHz rock'n'roll diner acoustics"
    )
}

def get_optimized_env(mode="pyramid"):
    env = os.environ.copy()
    env.update({
        "H3_PROFILE": "1",
        "H3_NAX": "qkv-attn",
        "H3_CPU_SAMPLER": "1",
        "H3_ZERO_COPY_WEIGHTS": "1",
        "H3_REUSE_MPS_COMMAND": "1",
        "H3_DIT_COMMAND_BLOCKS": "0",
        "H3_WARP_GAMMA": "1.15",
        "H3_SHARPNESS_BOOST": "1.65",
        "H3_SOLVER": "euler",
        "OMP_NUM_THREADS": "18"
    })
    if mode == "pyramid":
        env["H3_LAYER_SCHEDULE"] = "20,24,28,34,40,46,50,50"
        env["H3_NGRAM"] = "1"
        env["H3_NGRAM_DETAIL"] = "1"
    return env

def generate_monolithic_5s(output_mp4: Path, width=512, height=320, seed=42, upscale_1080p=True):
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    raw_mp4 = output_mp4.parent / f"_raw_{output_mp4.stem}.mp4"
    env = get_optimized_env(mode="pyramid")
    
    cmd = [
        str(H3_DIR / "h3"), "--profile",
        "-d", str(MODEL_DIR),
        "-p", PULP_PROMPTS["dance_opening"],
        "--width", str(width), "--height", str(height),
        "--frames", "107",
        "--steps", "8",
        "--use-int8-row-fc2",
        "--seed", str(seed),
        "-o", str(raw_mp4)
    ]
    
    print("=" * 80)
    print("🚀 GENERAZIONE MONOLITICA 5S (107 FRAME @ 24 FPS) CON MOTORE PIRAMIDALE")
    print(f"   Risoluzione Base: {width}x{height}")
    print(f"   Destinazione   : {output_mp4}")
    print("=" * 80)
    
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(H3_DIR), env=env, capture_output=True, text=True)
    t_gen = time.time() - t0
    
    if res.returncode != 0 or not raw_mp4.exists():
        print(f"❌ Errore durante la generazione:\n{res.stderr}", file=sys.stderr)
        return False
        
    print(f"✓ Generazione RAW completata in: {t_gen:.2f}s")
    
    if upscale_1080p:
        print("\n✨ Applicazione Super-Risoluzione Cinema 1080p (Lanczos-8 + CAS + Kodak 5219)...")
        upscale_cmd = [
            "python3", str(BASE_DIR / "upscale_cinema_master.py"),
            "-i", str(raw_mp4),
            "-o", str(output_mp4),
            "-r", "1080p",
            "--cas", "0.45",
            "--grain", "0.04"
        ]
        res_up = subprocess.run(upscale_cmd, capture_output=True, text=True)
        if res_up.returncode == 0:
            print(f"✓ Master 1080p salvato con successo: {output_mp4}")
            if raw_mp4.exists():
                raw_mp4.unlink()
        else:
            print(f"⚠️ Errore upscaling, mantenuto RAW: {raw_mp4}")
            raw_mp4.rename(output_mp4)
    else:
        raw_mp4.rename(output_mp4)
        
    return True

def generate_fast_sub5s_chained(output_mp4: Path, duration_sec=5.0, seed=42):
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_mp4.parent / f"_temp_sub5s_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    env = get_optimized_env(mode="chained")
    
    # Use 56 frames per chunk (2.33s each, fully valid causal lattice T=17*3+5=56)
    chunk_prompts = [
        PULP_PROMPTS["dance_opening"],
        PULP_PROMPTS["dance_finale"]
    ] if duration_sec <= 5.0 else [
        PULP_PROMPTS["dance_opening"],
        PULP_PROMPTS["dance_middle"],
        PULP_PROMPTS["dance_middle"],
        PULP_PROMPTS["dance_finale"]
    ]
    
    chunk_files = []
    last_frame = None
    
    print("=" * 80)
    print(f"⚡ GENERAZIONE CAUSAL CHAINED ({duration_sec}s in {len(chunk_prompts)} Chunk da 56 frame)")
    print(f"   Risoluzione     : 704x384 (Cinema 16:9)")
    print(f"   Denoise Target  : ~12.2s per chunk da 2.33s (Denoise rate: ~5.2s/s)")
    print("=" * 80)
    
    start_total = time.time()
    for idx, prompt in enumerate(chunk_prompts):
        chunk_mp4 = temp_dir / f"chunk_{idx:02d}.mp4"
        
        cmd = [
            str(H3_DIR / "h3"), "--profile",
            "-d", str(MODEL_DIR),
            "-p", prompt,
            "--width", "704", "--height", "384",
            "--frames", "56",
            "--steps", "8",
            "--layers", "44",
            "--reuse", "2",
            "--use-int8-row-fc2",
            "--seed", str(seed + idx * 17),
            "-o", str(chunk_mp4)
        ]
        if last_frame and last_frame.exists():
            cmd.extend(["--first", str(last_frame)])
            
        print(f"\n⚡ Generazione Chunk {idx + 1}/{len(chunk_prompts)}...")
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(H3_DIR), env=env, capture_output=True, text=True)
        t_chk = time.time() - t0
        
        if res.returncode != 0 or not chunk_mp4.exists():
            print(f"❌ Errore generazione chunk {idx}:\n{res.stderr}", file=sys.stderr)
            return False
            
        print(f"✓ Chunk {idx + 1} completato in {t_chk:.2f}s -> {chunk_mp4.name}")
        chunk_files.append(chunk_mp4)
        
        # Extract boundary anchor
        last_frame = temp_dir / f"anchor_{idx:02d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-0.08", "-i", str(chunk_mp4),
            "-vsync", "vfr", "-q:v", "2", "-update", "1", str(last_frame)
        ], capture_output=True)
        
    concat_list = temp_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for cf in chunk_files:
            f.write(f"file '{cf.resolve()}'\n")
            
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(output_mp4)
    ], capture_output=True)
    
    total_time = time.time() - start_total
    print("=" * 80)
    print(f"✅ GENERAZIONE CAUSAL CHAINED COMPLETATA IN {total_time:.2f}s!")
    print(f"   File Salvato: {output_mp4}")
    print("=" * 80)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax H3 Video Master Engine")
    parser.add_argument("--mode", choices=["monolithic", "sub5s", "15s"], default="monolithic")
    parser.add_argument("-o", "--output", type=str, default="pulp_fiction_master_output.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    out_path = Path(args.output).resolve()
    if args.mode == "monolithic":
        generate_monolithic_5s(out_path, seed=args.seed)
    elif args.mode == "sub5s":
        generate_fast_sub5s_chained(out_path, duration_sec=5.0, seed=args.seed)
    elif args.mode == "15s":
        generate_fast_sub5s_chained(out_path, duration_sec=15.0, seed=args.seed)
