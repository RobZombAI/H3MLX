#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h3_timeline_director.py - Temporal Multi-Beat Storyboard Scheduler for H3MLX
=============================================================================
Orchestrates multi-beat cinematic commercials and complex narrative scenes.
Bridges disparate storytelling beats with frame-accurate causal anchoring
and C2 Quintic Smoothstep latent continuity.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from h3mlx_engine_core import execute_h3_generation, resolve_optimal_frames

BASE_DIR = Path(__file__).resolve().parent

def execute_timeline_storyboard(
    timeline_config: List[Dict[str, Any]],
    output_path: str,
    width: int = 768,
    height: int = 512,
    base_seed: int = 777,
    smart_filter: str = "macro",
    no_activity_mask: bool = True
) -> Dict[str, Any]:
    """
    Executes a multi-beat narrative storyboard with sequential causal chaining.
    """
    out_master = Path(output_path).resolve()
    out_dir = out_master.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / f"_timeline_{out_master.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    total_beats = len(timeline_config)
    print("=" * 75)
    print(f"🎬 H3MLX TIMELINE DIRECTOR: {total_beats} NARRATIVE BEATS")
    print(f"Canvas: {width}x{height} | Smart Filter: {smart_filter} | Base Seed: {base_seed}")
    print("=" * 75)
    
    beat_outputs = []
    current_anchor_frame = None
    t_start_total = time.time()
    
    for idx, beat in enumerate(timeline_config):
        beat_idx = idx + 1
        name = beat.get("name", f"beat_{beat_idx}")
        duration_s = beat.get("duration", 2.5)
        frames = resolve_optimal_frames(duration=f"{int(round(duration_s))}s" if duration_s >= 3.0 else "3s")
        steps = beat.get("steps", 8)
        prompt = beat.get("prompt", "")
        ref_video = beat.get("ref_video", None)
        ref_image = beat.get("ref_image", None)
        seed = beat.get("seed", base_seed + idx * 37)
        
        out_beat = temp_dir / f"beat_{beat_idx:02d}_{name}.mp4"
        beat_outputs.append(str(out_beat))
        
        print(f"\n▶️ [Beat {beat_idx}/{total_beats}] \"{name}\" ({duration_s}s, {frames} frames, {steps} steps):")
        print(f"   Prompt: \"{prompt[:80]}...\"")
        if current_anchor_frame:
            print(f"   Anchor: {current_anchor_frame}")
        if ref_video:
            print(f"   Pose/Control: {ref_video}")
            
        env_extra = {}
        if no_activity_mask:
            env_extra["H3_DISABLE_ACTIVITY_GATE"] = "1"
            
        t0 = time.time()
        res = execute_h3_generation(
            prompt=prompt,
            output_path=str(out_beat),
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            reuse=1,
            layers=50,
            token_reduction=False,
            int8=True,
            first_frame=current_anchor_frame,
            ref_video=ref_video,
            ref_image=ref_image,
            smart_filter=smart_filter,
            extra_env=env_extra
        )
        t_elapsed = time.time() - t0
        
        if not res.success or not out_beat.exists():
            print(f"❌ Error in Beat {beat_idx}: {res.error}")
            return {"success": False, "error": res.error}
            
        print(f"   ✓ Beat {beat_idx} completed in {t_elapsed:.2f}s ({out_beat.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Extract the terminal frame to serve as pristine C1 anchor for the next beat
        if beat_idx < total_beats:
            next_anchor_path = temp_dir / f"anchor_to_beat_{beat_idx+1:02d}.jpg"
            cmd_extract = [
                "ffmpeg", "-y",
                "-i", str(out_beat),
                "-vf", f"select='eq(n\\,{frames-1})',scale={width}:{height}",
                "-vframes", "1",
                str(next_anchor_path)
            ]
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            current_anchor_frame = str(next_anchor_path)
            
    # Assemble master commercial with ffmpeg
    print(f"\n🎞️ Assembling Final Commercial: {out_master}...")
    concat_list_file = temp_dir / "concat_list.txt"
    with open(concat_list_file, "w") as f:
        for b_path in beat_outputs:
            f.write(f"file '{b_path}'\n")
            
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(out_master)
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    total_time = time.time() - t_start_total
    print("=" * 75)
    print(f"🏆 TIMELINE MASTER COMPLETED: {out_master}")
    print(f"   • Total Time: {total_time:.2f}s")
    print(f"   • File Size:  {out_master.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 75)
    
    return {
        "success": True,
        "master_video": str(out_master),
        "total_time": total_time,
        "beats": beat_outputs
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3MLX Timeline Storyboard Director")
    parser.add_argument("--config", type=str, required=True, help="Path to timeline JSON config")
    parser.add_argument("-o", "--output", type=str, default="outputs/timeline_commercial_master.mp4")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--smart-filter", type=str, default="macro")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    execute_timeline_storyboard(
        timeline_config=cfg,
        output_path=args.output,
        width=args.width,
        height=args.height,
        base_seed=args.seed,
        smart_filter=args.smart_filter
    )
