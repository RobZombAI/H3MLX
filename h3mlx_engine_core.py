#!/usr/bin/env python3
"""
⚡ H3MLX Engine Core Bridge & Dual-Execution Driver
Connects Salvatore Sanfilippo (antirez) canonical h3 pipeline with H3MLX Metal 4 NAX acceleration.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
H3_BIN = BASE_DIR / "h3-lora-lab" / "h3"
DEFAULT_PDD_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
DEFAULT_FULL_MODEL = Path("/Users/robzomb/h3-models/MiniMax-H3")

class H3EngineResult:
    def __init__(self,
                 success: bool,
                 output_path: str,
                 wall_time_s: float,
                 stdout: str,
                 stderr: str,
                 profile_data: Optional[Dict[str, float]] = None):
        self.success = success
        self.output_path = output_path
        self.wall_time_s = wall_time_s
        self.stdout = stdout
        self.stderr = stderr
        self.profile_data = profile_data or {}

def resolve_model_path(model_dir: Optional[str] = None, steps: int = 8) -> Path:
    """Resolve model path with fallback to PDD or Full model."""
    if model_dir:
        p = Path(model_dir)
        if p.exists():
            return p
    if steps <= 16 and DEFAULT_PDD_MODEL.exists():
        return DEFAULT_PDD_MODEL
    if DEFAULT_FULL_MODEL.exists():
        return DEFAULT_FULL_MODEL
    if DEFAULT_PDD_MODEL.exists():
        return DEFAULT_PDD_MODEL
    raise FileNotFoundError("MiniMax H3 model directory not found in default paths or user argument.")

def execute_h3_generation(
    prompt: str,
    output_path: str,
    width: int = 768,
    height: int = 512,
    frames: int = 73,
    steps: int = 8,
    seed: int = 42,
    engine_mode: str = "boosted",  # "canonical" or "boosted" (h3mlx)
    solver: str = "auto",          # "euler", "dpm3m", "ab3", "auto"
    reuse: int = 1,
    layers: int = 50,
    token_reduction: bool = False,
    int8: bool = True,
    first_frame: Optional[str] = None,
    last_frame: Optional[str] = None,
    ref_image: Optional[str] = None,
    ref_video: Optional[str] = None,
    ref_audio: Optional[str] = None,
    speech_audio: Optional[str] = None,
    ssd_streaming: bool = False,
    upscale_4k: bool = False,
    model_dir: Optional[str] = None,
    profile: bool = True,
    extra_env: Optional[Dict[str, str]] = None
) -> H3EngineResult:
    """
    Executes an H3 inference job using either canonical antirez engine or boosted H3MLX.
    """
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = BASE_DIR / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    model_path = resolve_model_path(model_dir, steps)
    
    # Base command arguments (faithful to antirez h3.c CLI)
    cmd = [
        str(H3_BIN),
        "-d", str(model_path),
        "-p", prompt,
        "-o", str(out_file),
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--steps", str(steps),
        "--seed", str(seed),
        "--layers", str(layers),
        "--reuse", str(reuse)
    ]
    
    if profile:
        cmd.append("--profile")
        
    if first_frame:
        cmd.extend(["--first-frame", str(first_frame)])
    if last_frame:
        cmd.extend(["--last-frame", str(last_frame)])
    if ref_image:
        cmd.extend(["--ref-image", str(ref_image)])
    if ref_video:
        cmd.extend(["--ref-video", str(ref_video)])
    if ref_audio:
        cmd.extend(["--ref-audio", str(ref_audio)])
    if speech_audio:
        cmd.extend(["--speech-audio", str(speech_audio)])
    if ssd_streaming:
        cmd.append("--ssd-streaming")
        
    env = os.environ.copy()
    
    # Enable Apple Silicon UMA Zero-Copy & Command Reuse by default
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_PROFILE"] = "1"
    env["OMP_NUM_THREADS"] = "18"
    
    if engine_mode in ["canonical", "pure", "antirez"]:
        cmd.append("--canonical")
        cmd.append("--sol-cache")
        cmd.append("--use-int8-row-fc2")
        env["H3_NAX"] = "0"
        env["H3_INT8_FC2"] = "1"
        env["H3_GPU_SAMPLER"] = "0"
        env["H3_SOLVER"] = "euler"
    else:  # H3MLX boosted engine
        cmd.append("--boosted")
        cmd.append("--sol-cache")
        if int8:
            cmd.append("--int8")
            cmd.append("--use-int8-row-fc2")
            env["H3_INT8_FC2"] = "1"
        else:
            env["H3_INT8_FC2"] = "0"
            
        if token_reduction:
            cmd.append("--token-reduction")
            
        env["H3_NAX"] = "qkv-attn"
        env["H3_GPU_SAMPLER"] = "1"
        
        if solver == "auto":
            env["H3_SOLVER"] = "dpm3m" if steps > 8 else "euler"
        else:
            env["H3_SOLVER"] = solver
            
    if extra_env:
        env.update(extra_env)
        
    t_start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(BASE_DIR / "h3-lora-lab"), capture_output=True, text=True)
    t_end = time.perf_counter()
    wall_time = t_end - t_start
    
    # Parse profiling logs from stderr or stdout if available
    profile_data = {}
    combined_logs = proc.stderr + "\n" + proc.stdout
    for line in combined_logs.splitlines():
        if "✓" in line and "(" in line and "s)" in line:
            parts = line.split("✓")[-1].split("[")[0].strip()
            time_part = line.split("(")[-1].split("s)")[0].strip()
            try:
                profile_data[parts] = float(time_part)
            except ValueError:
                pass
        elif "Forward Time:" in line:
            try:
                profile_data["Forward Time (Denoise)"] = float(line.split("Forward Time:")[-1].split("s")[0].strip())
            except ValueError:
                pass
        elif "video VAE decoder" in line and "wall=" in line:
            try:
                profile_data["VAE Decoder Wall"] = float(line.split("wall=")[-1].split("s")[0].strip())
            except ValueError:
                pass
                
    # If 4K upscaling requested, run upscaler
    if upscale_4k and proc.returncode == 0 and out_file.exists():
        try:
            from h3_cinema_upscaler import upscale_video_to_4k
            out_4k = out_file.parent / f"{out_file.stem}_4k.mp4"
            upscale_video_to_4k(str(out_file), str(out_4k))
            profile_data["4k_upscale_applied"] = 1.0
        except Exception as e:
            sys.stderr.write(f"Warning: 4K upscaling failed: {e}\n")
            
    return H3EngineResult(
        success=(proc.returncode == 0 and out_file.exists()),
        output_path=str(out_file),
        wall_time_s=wall_time,
        stdout=proc.stdout,
        stderr=proc.stderr,
        profile_data=profile_data
    )
