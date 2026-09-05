#!/usr/bin/env python3
"""
⚡ H3MLX Master Engine Core: All 5 Frontiers on Apple Silicon Metal 4 NAX
Incorporates Level 1-5 Frontiers:
 - Level 1: Metal 4 NAX Fused Attention + Native GPU Trajectory Sampler + UMA Zero-Copy
 - Level 2: Spatial Multi-Scale Token Reduction (Blocks 4:34)
 - Level 3: Monolithic 3D VAE Zero-Stitch on 128GB Unified Memory
 - Level 4: 14-Step PDD / 8-Step Fast Master Optimal Trajectory (Reuse=2, INT8-FC2)
 - Level 5: Cooke Anamorphic S4/i MTF Optical Phase Coherence & Broadcast 4K Mastering
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
                 profile_data: Optional[Dict[str, float]] = None,
                 raw_output_path: str = "",
                 master_output_path: Optional[str] = None):
        self.success = success
        self.output_path = output_path
        self.wall_time_s = wall_time_s
        self.stdout = stdout
        self.stderr = stderr
        self.profile_data = profile_data or {}
        self.raw_output_path = raw_output_path or output_path
        self.master_output_path = master_output_path

# Standard Optimal Duration Table for MiniMax-H3 Multimodal Architecture
# Aligned strictly with the 3D Video VAE Temporal Lattice: frames = 17k + 5, latents = 5k + 2
OPTIMAL_DURATIONS = {
    "2s": 56,     # 2.33s @ 24fps (k=3, lat=17)
    "3s": 73,     # 3.04s @ 24fps (k=4, lat=22)
    "4s": 90,     # 3.75s (Default MiniMax-H3 Short, k=5, lat=27)
    "6s": 141,    # 5.88s (~6s, k=8, lat=42)
    "8s": 192,    # 8.00s esatti @ 24fps (Cinema Extended, k=11, lat=57)
    "10s": 243,   # 10.12s (~10s, k=14, lat=72)
    "12s": 294,   # 12.25s (~12s, k=17, lat=87)
    "15s": 362,   # 15.08s (~15s Master, k=21, lat=107)
    "20s": 481,   # 20.04s (~20s Ultra, k=28, lat=142)
    "30s": 719,   # 29.96s (~30s Cinema Epic, k=42, lat=212)
}

def resolve_optimal_frames(duration: Optional[str] = None,
                           seconds: Optional[float] = None,
                           frames: Optional[int] = None) -> int:
    """
    Resolves the exact mathematically optimal frame count aligned with
    MiniMax-H3 3D VAE lattice (17k + 5).
    """
    if duration:
        d = duration.strip().lower()
        if d in OPTIMAL_DURATIONS:
            return OPTIMAL_DURATIONS[d]
        try:
            seconds = float(d.rstrip("s"))
        except ValueError:
            pass
    if seconds is not None and seconds > 0:
        target_frames = seconds * 24.0
        k = max(0, int(round((target_frames - 5.0) / 17.0)))
        return int(17 * k + 5)
    if frames is not None and frames > 0:
        k = max(0, int(round((frames - 5.0) / 17.0)))
        return int(17 * k + 5)
    return 90

def resolve_model_path(model_dir: Optional[str] = None, steps: int = 14) -> Path:
    """Resolve model path with fallback to PDD or Full model across standard Mac directories."""
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
        if p.exists():
            return p

    candidates = [
        Path.home() / "h3-models" / "MiniMax-H3-PDD-8Step",
        Path.home() / "h3-models" / "MiniMax-H3",
        Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step"),
        Path("/Users/robzomb/h3-models/MiniMax-H3"),
        Path.home() / "Desktop" / "H3" / "MiniMax-H3",
        BASE_DIR / "models" / "MiniMax-H3-PDD-8Step",
        BASE_DIR / "models" / "MiniMax-H3"
    ]

    for c in candidates:
        if c.exists() and (c / "FL2VA").exists() or (c / "model.safetensors").exists() or any(c.glob("*.safetensors")):
            return c
        if c.exists():
            return c

    raise FileNotFoundError(
        "❌ No MiniMax H3 model directory found in standard search paths.\n"
        "💡 To automatically download weights on this Mac, run:\n"
        "   ./download_models.sh\n"
        "Or pass the checkpoint path explicitly via -d / --model-dir <path>."
    )

from h3mlx_smart_filters import build_smart_video_filter

def execute_h3_generation(
    prompt: str,
    output_path: str,
    width: int = 768,
    height: int = 512,
    frames: int = 90,
    steps: int = 14,
    seed: int = 42,
    engine_mode: str = "boosted",  # "canonical" or "boosted" (h3mlx)
    solver: str = "auto",          # "euler", "dpm3m", "ab3", "auto"
    reuse: int = 2,
    layers: int = 50,
    token_reduction: bool = True,
    token_reduction_blocks: str = "4:34",
    int8: bool = True,
    first_frame: Optional[str] = None,
    last_frame: Optional[str] = None,
    ref_image: Optional[str] = None,
    ref_video: Optional[str] = None,
    ref_audio: Optional[str] = None,
    speech_audio: Optional[str] = None,
    ssd_streaming: bool = False,
    upscale_4k: bool = False,
    smart_filter: str = "auto",
    model_dir: Optional[str] = None,
    profile: bool = True,
    nax_st: bool = False,
    nax_chunk: int = 4,
    nax_stride: int = 4,
    frontier: Optional[Union[str, int]] = None,
    fps: int = 24,
    bandpass_limiter: bool = False,
    extra_env: Optional[Dict[str, str]] = None
) -> H3EngineResult:
    """
    Executes an H3 inference job using the Frontier-optimized engine pipeline.
    """
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = BASE_DIR / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    model_path = resolve_model_path(model_dir, steps)
    
    # Base command arguments
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
        
    def _abs_path(p):
        if not p: return None
        path_obj = Path(p)
        return str((BASE_DIR / path_obj).resolve() if not path_obj.is_absolute() else path_obj)

    if first_frame:
        cmd.extend(["--first-frame", _abs_path(first_frame)])
    if last_frame:
        cmd.extend(["--last-frame", _abs_path(last_frame)])
    if ref_image:
        cmd.extend(["--ref-image", _abs_path(ref_image)])
    if ref_video:
        cmd.extend(["--ref-video", _abs_path(ref_video)])
    if ref_audio:
        cmd.extend(["--ref-audio", _abs_path(ref_audio)])
    if speech_audio:
        cmd.extend(["--speech-audio", _abs_path(speech_audio)])
    if ssd_streaming:
        cmd.append("--ssd-streaming")
    if nax_st:
        cmd.append("--nax-st")
        if nax_chunk > 0:
            cmd.extend(["--nax-chunk", str(nax_chunk)])
        if nax_stride > 0:
            cmd.extend(["--nax-stride", str(nax_stride)])
        
    env = os.environ.copy()
    
    # Frontier Optimization Environment (Metal 4 NAX + UMA Zero-Copy + Fast Driver)
    env["H3_PROFILE"] = "1"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_DIT_COMMAND_BLOCKS"] = "0"
    env["H3_GPU_SAMPLER_WINDOW"] = "0"
    env["OMP_NUM_THREADS"] = "18"
    env["METAL_DEVICE_WRAPPER_TYPE"] = "0"
    env["MTL_DEBUG_LAYER"] = "0"
    env["MTL_SHADER_VALIDATION"] = "0"
    env["METAL_CAPTURE_ENABLED"] = "0"
    
    if engine_mode in ["canonical", "pure", "antirez"]:
        cmd.append("--canonical")
        cmd.append("--use-int8-row-fc2")
        env["H3_NAX"] = "0"
        env["H3_INT8_FC2"] = "1"
        env["H3_GPU_SAMPLER"] = "0"
        env["H3_SOLVER"] = "euler"
    else:  # H3MLX boosted engine (Level 1-5 Frontiers)
        if token_reduction:
            cmd.append("--token-reduction")
            env["H3_TOKEN_REDUCTION"] = "1"
            env["H3_TOKEN_REDUCTION_BLOCKS"] = token_reduction_blocks
        else:
            env["H3_TOKEN_REDUCTION"] = "0"

        if int8:
            cmd.append("--int8")
            cmd.append("--use-int8-row-fc2")
            env["H3_INT8_FC2"] = "1"
        else:
            env["H3_INT8_FC2"] = "0"
            
        env["H3_NAX"] = "qkv-attn"
        env["H3_GPU_SAMPLER"] = "1"
        
        if solver == "auto":
            env["H3_SOLVER"] = "dpm2m"
        else:
            env["H3_SOLVER"] = solver
            
        env["H3_WARP_GAMMA"] = "1.0"
        env["H3_TEMPORAL_CRISP"] = "0.04"

        f_val = 0
        try:
            f_val = int(frontier) if frontier is not None else 0
        except (ValueError, TypeError):
            if frontier in ["champion", "optics", "cinema-optics"]:
                f_val = 7
            elif frontier in ["ultra", "frontier-2026"]:
                f_val = 11
            elif frontier in ["12", 12, "sfmc", "fast-master"]:
                f_val = 12

        if f_val >= 6 or frontier in ["6", 6]:
            env["H3_FREQFLOW"] = "0.08"
        if f_val >= 7 or frontier in ["7", 7, "champion", "optics", "cinema-optics"]:
            env["H3_FREQFLOW"] = "0.08"
            env["H3_SPATIAL_CRISP"] = "0.035"
            if smart_filter == "auto":
                smart_filter = "master-optics"
        if f_val >= 8 or frontier in ["8", 8]:
            env["H3_TFM_MOMENTUM"] = "0.06"
            env["H3_FRONTIER"] = str(max(8, f_val))
        if f_val >= 9 or frontier in ["9", 9]:
            if smart_filter in ["auto", "master-optics"]:
                smart_filter = "master-optics"
            env["H3_FRONTIER"] = str(max(9, f_val))
        if f_val >= 10 or frontier in ["10", 10]:
            env["H3_CHEBYSHEV_WARP"] = "1"
            env["H3_FRONTIER"] = str(max(10, f_val))
        if f_val >= 11 or frontier in ["11", 11, "ultra", "frontier-2026"]:
            # Empirical ablation proof: Spectral eigen-clamping without spatial attention masks
            # suppresses ocular and dental high-frequency phase. Keep opt-in via extra_env only.
            env["H3_SPECTRAL_CLAMP"] = "0"
            env["H3_FRONTIER"] = str(max(11, f_val))
            if smart_filter in ["auto", "master-optics"]:
                smart_filter = "master-optics"
        if f_val >= 12 or frontier in ["12", 12, "sfmc", "fast-master"]:
            # Frontier Level 12: S-FMC (Symplectic Flow-Matching Curvature + Radau Anchoring)
            # Damped Hermite-Taylor operator + Radau boundary collocation enables 5-step photorealism.
            env["H3_FRONTIER"] = str(max(12, f_val))
            env["H3_RADAU_WARP"] = "1"
            env["H3_CHEBYSHEV_WARP"] = "1"
            env["H3_FLUX_LIMITER"] = "1"
            env["H3_BANDPASS_LIMITER"] = "1"
            env["H3_SPECTRAL_CLAMP"] = "0"
            env["H3_TFM_MOMENTUM"] = "0.04"
            env["H3_SPATIAL_CRISP"] = "0.025"
            env["H3_FREQFLOW"] = "0.05"
            if smart_filter in ["auto", "master-optics"]:
                smart_filter = "master-optics"

    if bandpass_limiter:
        env["H3_BANDPASS_LIMITER"] = "1"
            
    if extra_env:
        env.update(extra_env)
        
    t_start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(BASE_DIR / "h3-lora-lab"), capture_output=True, text=True)
    t_end = time.perf_counter()
    wall_time = t_end - t_start
    
    profile_data = {}
    if proc.stdout:
        for line in proc.stdout.splitlines():
            line_str = line.strip()
            if "Total wall-clock:" in line_str:
                try:
                    profile_data["total_wall_s"] = float(line_str.split(":")[-1].replace("s", "").strip())
                except Exception:
                    pass
            elif "DiT sampling total:" in line_str or "Forward time:" in line_str:
                try:
                    profile_data["denoise_s"] = float(line_str.split(":")[-1].replace("s", "").strip())
                except Exception:
                    pass
            elif "Video VAE decode:" in line_str or "VAE decode total:" in line_str:
                try:
                    profile_data["vae_decode_s"] = float(line_str.split(":")[-1].replace("s", "").strip())
                except Exception:
                    pass

    # Level 5 Smart Cinema Mastering & 48kHz Audio Foley (Wavelet Bayes + CAS + VideoToolbox)
    final_output_path = str(out_file)
    master_path_str = None
    if proc.returncode == 0 and out_file.exists() and upscale_4k:
        four_k_path = out_file.parent / f"{out_file.stem}_4k.mp4"
        try:
            from h3_cinema_upscaler import upscale_video
            # True Aspect Ratio 4x Scaling (Preserves true pixel geometry)
            if width == 640 and height == 640:
                t_w, t_h = 2560, 2560  # 1:1 True Square Master
            elif width == 576 and height == 1024:
                t_w, t_h = 2304, 4096  # 9:16 True Vertical Cinema Reel
            elif width == 576 and height == 768:
                t_w, t_h = 2304, 3072  # 3:4 True Vertical Cinema Master
            elif width == 960 and height == 544:
                t_w, t_h = 3840, 2176  # 16:9 True Cinema Anamorphic
            elif width == 768 and height == 512:
                t_w, t_h = 3072, 2048  # 3:2 True 35mm Photographic Master
            else:
                scale_factor = max(1, 3840 // width)
                t_w = ((width * scale_factor) // 2) * 2
                t_h = ((height * scale_factor) // 2) * 2

            result_path = upscale_video(
                input_path=str(out_file),
                output_path=str(four_k_path),
                target_width=t_w,
                target_height=t_h,
                enable_denoise=True,
                cas_strength=0.25,
                use_videotoolbox=True,
                smart_filter=smart_filter,
                fps=fps
            )
            final_output_path = result_path
            master_path_str = result_path
        except Exception as e:
            print(f"Mastering step failed ({e}), falling back to raw output.")
            
    success = (proc.returncode == 0 and out_file.exists())
    return H3EngineResult(
        success=success,
        output_path=final_output_path,
        wall_time_s=wall_time,
        stdout=proc.stdout,
        stderr=proc.stderr,
        profile_data=profile_data,
        raw_output_path=str(out_file),
        master_output_path=master_path_str
    )
