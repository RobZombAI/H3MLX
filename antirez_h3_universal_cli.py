#!/usr/bin/env python3
"""
👑 Universal Antirez H3 Bridge & INT8 Frontier Engine CLI (v2.0 Hyper-Master)
Co-designed for Apple Silicon (M1-M5) 128GB UMA
Seamless 1:1 drop-in compatibility with Salvatore Sanfilippo (antirez) h3.c
with all RobZomb H3XML / Metal 4 NAX INT8 Frontiers & 4K Cinema Upscaler unlocked.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
H3_BIN = BASE_DIR / "h3-lora-lab" / "h3"
DEFAULT_PDD_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
DEFAULT_FULL_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3")
RESIDENT_SOCKET = Path("/tmp/h3_resident.sock")

try:
    from h3_cinema_upscaler import upscale_video_to_4k
except ImportError:
    upscale_video_to_4k = None

def main():
    parser = argparse.ArgumentParser(
        description="👑 Universal Antirez H3 Bridge with Full INT8 Frontiers & 4K Master Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 1:1 antirez arguments
    parser.add_argument("-d", "--model-dir", type=str, default="", help="Path to MiniMax H3 checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, default="A red fox in fresh winter snow, photorealistic 8k, cinematic lighting", help="Generation prompt")
    parser.add_argument("-o", "--output", type=str, default="outputs/antirez_int8_output.mp4", help="Output MP4 file path")
    parser.add_argument("--width", type=int, default=512, help="Internal render width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Internal render height in pixels")
    parser.add_argument("--frames", type=int, default=0, help="Total video frames (e.g. 48 for 2.0s, 90 for 4.0s)")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration in seconds (at 24 fps)")
    parser.add_argument("--steps", type=int, default=8, help="Denoising steps (8 for PDD, 25-40 for Full Flow)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--first-frame", "--i2v", dest="first_frame", type=str, default="", help="First-frame conditioning image for Image-to-Video")
    
    # Frontier INT8, Solvers & 4K Cinema Options
    parser.add_argument("--mode", type=str, default="boosted", choices=["boosted", "canonical", "antirez", "h3xml"],
                        help="Engine mode: 'boosted' (Metal 4 NAX + INT8 + AB3) or 'canonical' (antirez pure baseline)")
    parser.add_argument("--int8", action="store_true", default=True, help="Enable Metal 4 NAX Row-Major INT8 dynamic FC2 quantization")
    parser.add_argument("--no-int8", dest="int8", action="store_false", help="Disable INT8 quantization and use pure BF16")
    parser.add_argument("--solver", type=str, default="auto", choices=["auto", "dpm3m", "ab3", "euler"],
                        help="ODE flow solver: 'dpm3m' (3rd order), 'ab3', or 'euler'")
    parser.add_argument("--reuse", type=int, default=-1, help="Predictive step reuse (-1 = auto-optimal derived)")
    parser.add_argument("--4k", "--upscale", dest="upscale_4k", action="store_true", help="Automatically upscale output video to 4K UHD Master (3840x2160)")
    parser.add_argument("--profile", action="store_true", help="Print per-phase Metal profiling statistics")
    
    args = parser.parse_args()
    
    # 1. Resolve Model Directory
    if args.model_dir:
        model_path = Path(args.model_dir)
        is_pdd = "pdd" in model_path.name.lower() or args.steps <= 16
    else:
        is_pdd = (args.steps <= 16)
        model_path = DEFAULT_PDD_MODEL if is_pdd else DEFAULT_FULL_MODEL
        
    if not model_path.exists():
        print(f"❌ Errore: Directory del modello non trovata: {model_path}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Resolve Frames & Lattice
    if args.frames > 0:
        frames = args.frames
    else:
        raw_frames = int(round(args.seconds * 24))
        if raw_frames <= 22:
            frames = 22
        elif raw_frames <= 48:
            frames = 39 if raw_frames <= 39 else 48
        elif raw_frames <= 96:
            frames = 90 if raw_frames <= 90 else 96
        else:
            frames = raw_frames
            
    # 3. Resolve Step Reuse
    if args.reuse == -1:
        if args.mode in ["canonical", "antirez"]:
            reuse = 1
        else:
            if args.steps <= 8:
                reuse = 1
            elif args.steps <= 16:
                reuse = 2
            elif args.steps <= 30:
                reuse = 4
            else:
                reuse = 6
    else:
        reuse = args.reuse
        
    # 4. Resolve Output Directory
    out_file = Path(args.output)
    if not out_file.is_absolute():
        out_file = BASE_DIR / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. Environment configuration for INT8 & NAX
    env = os.environ.copy()
    is_canonical = args.mode in ["canonical", "antirez"]
    
    if not is_canonical and args.int8:
        env["H3_INT8_FC2"] = "1"
        env["H3_NAX"] = "qkv-attn"
        env["H3_GPU_SAMPLER"] = "1"
        
    if args.solver == "auto":
        resolved_solver = "euler" if (args.steps <= 8 or is_canonical) else "dpm3m"
    else:
        resolved_solver = args.solver
        
    env["H3_SOLVER"] = resolved_solver
    if resolved_solver != "euler":
        env["H3_CPU_SAMPLER"] = "1"
        env.pop("H3_GPU_SAMPLER", None)
        
    # 6. Build Command
    is_resident = RESIDENT_SOCKET.exists() and not is_canonical
    
    if is_resident:
        cmd = [
            str(H3_BIN),
            "--client", str(RESIDENT_SOCKET),
            "-p", args.prompt,
            "--width", str(args.width),
            "--height", str(args.height),
            "--frames", str(frames),
            "--steps", str(args.steps),
            "--layers", "50",
            "--reuse", str(reuse),
            "--seed", str(args.seed),
            "-o", str(out_file)
        ]
        if args.first_frame:
            cmd.extend(["--first-frame", str(Path(args.first_frame).resolve())])
    else:
        cmd = [
            str(H3_BIN),
            "-d", str(model_path),
            "-p", args.prompt,
            "--width", str(args.width),
            "--height", str(args.height),
            "--frames", str(frames),
            "--steps", str(args.steps),
            "--layers", "50",
            "--reuse", str(reuse),
            "--seed", str(args.seed),
            "-o", str(out_file)
        ]
        if args.first_frame:
            cmd.extend(["--first-frame", str(Path(args.first_frame).resolve())])
            
        if not is_canonical:
            if args.int8:
                cmd.append("--use-int8-row-fc2")
            cmd.append("--ngram")
            
        if args.profile:
            cmd.append("--profile")
            
    gpu_passes = args.steps // reuse + (1 if args.steps % reuse else 0) if reuse > 1 else args.steps
    
    print("=" * 72)
    print("👑 UNIVERSAL ANTIREZ H3 BRIDGE v2.0 — HYPER-MASTER ACCELERATOR")
    print("=" * 72)
    print(f" • Modello           : {model_path.name} ({'PDD Distilled' if is_pdd else 'Full Continuous Flow'})")
    print(f" • Modalità Motore   : {'🏛️ Canonical Pure (antirez baseline)' if is_canonical else '🚀 H3XML Boosted (RobZomb Frontier)'}")
    print(f" • Quantizzazione    : {'⚡ Metal 4 NAX INT8 Row-Major (FC2 W8A8)' if (args.int8 and not is_canonical) else 'BF16 Standard'}")
    print(f" • Risoluzione Base  : {args.width}x{args.height} ({frames} frame | {args.seconds:.1f}s @ 24fps)")
    print(f" • Denoise Steps     : {args.steps} step (Reuse: {reuse} -> {gpu_passes} forward GPU reali)")
    print(f" • Solutore ODE      : {resolved_solver.upper()} ({'Piecewise Linear' if resolved_solver == 'euler' else 'Adams-Bashforth 3M + Symplectic Flow'})")
    print(f" • Runtime UMA       : {'⚡ Residente (/tmp/h3_resident.sock - 0.00s Load)' if is_resident else 'Standard Diretto'}")
    if args.first_frame:
        print(f" • Image-to-Video    : 🖼️ First-Frame Conditioning ({args.first_frame})")
    print(f" • 4K Super-Res      : {'💎 Attivo (Lanczos-4 Sub-pixel 3840x2160)' if args.upscale_4k else 'Disattivo (Output nativo)'}")
    print(f" • Output File       : {out_file}")
    print("=" * 72)
    
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=H3_BIN.parent)
    elapsed = time.time() - t0
    
    if res.returncode == 0 and out_file.exists():
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print("=" * 72)
        print(f"✅ GENERAZIONE DIRETTA COMPLETATA IN {elapsed:.2f}s!")
        print(f"🎬 Video nativo salvato: {out_file} ({size_mb:.2f} MB)")
        
        # 4K Super-Resolution Pipeline
        if args.upscale_4k and upscale_video_to_4k is not None:
            t_up = time.time()
            out_4k = upscale_video_to_4k(str(out_file))
            elapsed_up = time.time() - t_up
            print(f"👑 PIPELINE 4K MASTER COMPLETATA IN {elapsed + elapsed_up:.2f}s TOTALI!")
            print(f"🌟 Video 4K Master: {out_4k}")
        print("=" * 72)
    else:
        print(f"❌ Errore durante l'esecuzione del motore H3 (Exit code: {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
