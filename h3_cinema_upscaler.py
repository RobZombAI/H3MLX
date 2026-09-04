#!/usr/bin/env python3
"""
🎬 H3 Cinema 4K Super-Resolution & Intraframe Detail Restoration Engine
Combines Wavelet Bayesian Denoising (vaguedenoiser), Contrast Adaptive Sharpening (CAS),
Lanczos-4 scaling, Apple Silicon VideoToolbox 10-bit HEVC encoding, and EBU R128 audio normalization.
Adapted with high-fidelity principles from X-MinimaxH3.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DETAIL_RESTORE_FILTER = (
    "vaguedenoiser=threshold=1.2:method=garrote:nsteps=4:percent=45:planes=7:type=bayes,"
    "cas=strength=0.25"
)


def build_filtergraph(
    target_width: int,
    target_height: int,
    aspect_ratio: str = "keep",
    enable_denoise: bool = True,
    cas_strength: float = 0.25,
    pix_fmt: str = "yuv420p10le",
    smart_filter: str = "auto",
    sensitometric_grain: bool = False,
) -> str:
    """Build high-fidelity video filter chain."""
    filters = []

    if aspect_ratio == "9:16":
        filters.append("crop=ih*9/16:ih:(iw-ih*9/16)/2:0")

    filters.append(
        f"scale={target_width}:{target_height}:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"
    )

    if enable_denoise:
        filters.append("vaguedenoiser=threshold=1.2:method=garrote:nsteps=4:percent=45:planes=7:type=bayes")

    if cas_strength > 0.0:
        filters.append(f"cas=strength={cas_strength:.2f}")

    if sensitometric_grain or smart_filter in ["master-optics", "optics", "cinema-35mm"]:
        # Kodak Vision3 5219 Sensitometric Optical Grain Emulation
        filters.append("noise=alls=1.8:allf=t+u")

    if pix_fmt:
        filters.append(f"format={pix_fmt}")

    return ",".join(filters)


def has_audio_stream(video_path: Path) -> bool:
    """Check if input video file contains an audio stream."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        return "audio" in res.stdout
    except Exception:
        return False


def upscale_video(
    input_path: str,
    output_path: str | None = None,
    target_width: int = 3840,
    target_height: int = 2160,
    aspect_ratio: str = "keep",
    enable_denoise: bool = True,
    cas_strength: float = 0.25,
    bitrate: str = "60M",
    use_videotoolbox: bool = True,
    smart_filter: str = "auto",
    sensitometric_grain: bool = False,
) -> str:
    """Upscale and restore video using Apple Silicon native acceleration."""
    in_file = Path(input_path).resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"Video not found: {in_file}")

    if output_path is None:
        suffix = "4k_master" if target_width >= 3840 else "1080p_master"
        out_file = in_file.parent / f"{in_file.stem}_{suffix}.mp4"
    else:
        out_file = Path(output_path).resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg non trovato sul sistema.")

    has_audio = has_audio_stream(in_file)
    print(f"Starting Intraframe Detail Restoration & Upscale: {in_file.name} -> {out_file.name}")
    print(f"   Resolution: {target_width}x{target_height} | Denoise: {enable_denoise} | CAS: {cas_strength} | Audio: {'Present' if has_audio else 'Muted'} | Filter: {smart_filter}")

    if use_videotoolbox:
        vf_filter = build_filtergraph(
            target_width, target_height, aspect_ratio, enable_denoise, cas_strength, "yuv420p10le",
            smart_filter=smart_filter, sensitometric_grain=sensitometric_grain
        )

        if has_audio:
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                "-filter_complex",
                f"[0:v]{vf_filter}[v];[0:a]aresample=48000,loudnorm=I=-14:TP=-1.0:LRA=11[a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "hevc_videotoolbox",
                "-profile:v", "main10",
                "-pix_fmt", "p010le",
                "-b:v", bitrate,
                "-tag:v", "hvc1",
                "-r", "24",
                "-c:a", "aac",
                "-b:a", "320k",
                "-ar", "48000",
                "-movflags", "+faststart",
                str(out_file),
            ]
        else:
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                "-vf", vf_filter,
                "-an",
                "-c:v", "hevc_videotoolbox",
                "-profile:v", "main10",
                "-pix_fmt", "p010le",
                "-b:v", bitrate,
                "-tag:v", "hvc1",
                "-r", "24",
                "-movflags", "+faststart",
                str(out_file),
            ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode == 0:
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"VideoToolbox Main10 mastering complete: {out_file.name} ({size_mb:.2f} MB)")
            return str(out_file)
        else:
            print("VideoToolbox encoding failed, falling back to libx264...")

    vf_fallback = build_filtergraph(
        target_width, target_height, aspect_ratio, enable_denoise, cas_strength, "yuv420p"
    )
    cmd_fallback = [
        ffmpeg, "-y",
        "-i", str(in_file),
        "-vf", vf_fallback,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "16",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    res_fb = subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res_fb.returncode != 0:
        vf_simple = f"scale={target_width}:{target_height}:flags=lanczos+accurate_rnd,unsharp=5:5:0.5,format=yuv420p"
        cmd_simple = [
            ffmpeg, "-y",
            "-i", str(in_file),
            "-vf", vf_simple,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "16",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_file),
        ]
        res_simple = subprocess.run(cmd_simple, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res_simple.returncode != 0:
            err = res_simple.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"FFmpeg mastering failed: {err}")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Upscaling complete: {out_file.name} ({size_mb:.2f} MB)")
    return str(out_file)


def upscale_video_to_4k(input_path: str, output_path: str | None = None, target_width: int = 3840, target_height: int = 2160, crf: int = 16) -> str:
    return upscale_video(input_path, output_path, target_width=target_width, target_height=target_height)


def main():
    parser = argparse.ArgumentParser(description="H3 Cinema 4K & Intraframe Detail Restoration")
    parser.add_argument("input", help="Input MP4 video file path")
    parser.add_argument("output", nargs="?", default=None, help="Output MP4 video file path (optional)")
    parser.add_argument("--res", choices=["1080p", "4k", "reel_9x16"], default="4k", help="Target resolution profile")
    parser.add_argument("--cas", type=float, default=0.25, help="Contrast Adaptive Sharpening intensity (default: 0.25)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable Wavelet Bayesian Denoising")
    args = parser.parse_args()

    if args.res == "1080p":
        tw, th, ar = 1920, 1080, "keep"
    elif args.res == "reel_9x16":
        tw, th, ar = 1080, 1920, "9:16"
    else:
        tw, th, ar = 3840, 2160, "keep"

    upscale_video(
        args.input,
        args.output,
        target_width=tw,
        target_height=th,
        aspect_ratio=ar,
        enable_denoise=not args.no_denoise,
        cas_strength=args.cas,
    )


if __name__ == "__main__":
    main()

