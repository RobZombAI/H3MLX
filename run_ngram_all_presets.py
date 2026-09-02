#!/usr/bin/env python3
"""
N-Gram Unified Runner — 5 Presets × 4 Samplers
================================================
Runs the full 20-combination matrix with --ngram enabled and collects
performance telemetry. All outputs go to the flat outputs/ directory.

Usage:
    python3 run_ngram_all_presets.py [--dry-run]
"""

import subprocess
import json
import time
import os
import sys
import re

# Paths
H3_BIN = os.path.join(os.path.dirname(__file__), "h3-lora-lab", "h3")
MODEL_DIR = "/Users/robzomb/h3-models/MiniMax-H3"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Default prompt for benchmarking
BENCHMARK_PROMPT = (
    "A majestic red fox walks through fresh winter snow in a sunlit "
    "forest clearing, ultra photorealistic, cinematic depth of field, "
    "golden hour light, 8K quality"
)

# === 5 PRESETS ===
PRESETS = {
    "ultra_draft": {
        "name": "Ultra Draft",
        "width": 512, "height": 512,
        "steps": 4, "layers": 45, "reuse": 2,
        "seconds": 2,
    },
    "fastvideo_turbo": {
        "name": "FastVideo Turbo",
        "width": 512, "height": 512,
        "steps": 4, "layers": 50, "reuse": 1,
        "seconds": 2,
    },
    "fast_master_champion": {
        "name": "Fast Master Champion",
        "width": 512, "height": 512,
        "steps": 8, "layers": 50, "reuse": 1,
        "seconds": 2,
    },
    "cinema_16x9": {
        "name": "Cinema Widescreen 16:9",
        "width": 960, "height": 544,
        "steps": 8, "layers": 50, "reuse": 1,
        "seconds": 2,
    },
    "high_quality": {
        "name": "High Quality Master",
        "width": 512, "height": 512,
        "steps": 16, "layers": 50, "reuse": 1,
        "seconds": 2,
    },
}

# === 4 TOP SAMPLERS ===
# Note: h3 uses --reuse for velocity reuse (DPM++ 2M style by default)
# The sampler type is controlled via environment variables for shift values
SAMPLERS = {
    "dpm2m_trailing": {
        "name": "DPM++ 2M Trailing (Gold)",
        "env": {"H3_VIDEO_SHIFT": "12.0", "H3_AUDIO_SHIFT": "3.0"},
        "extra_flags": [],
    },
    "euler_ancestral": {
        "name": "Euler Ancestral",
        "env": {"H3_VIDEO_SHIFT": "10.0", "H3_AUDIO_SHIFT": "3.0"},
        "extra_flags": [],
    },
    "flow_anime": {
        "name": "Flow Shifted Anime",
        "env": {"H3_VIDEO_SHIFT": "8.0", "H3_AUDIO_SHIFT": "2.5"},
        "extra_flags": [],
    },
    "dpm2m_sla_reuse2": {
        "name": "DPM++ 2M Reuse 2 + SLA Cache",
        "env": {"H3_VIDEO_SHIFT": "12.0", "H3_AUDIO_SHIFT": "3.0"},
        "extra_flags": ["--reuse", "2", "--sol-cache"],
        "override_reuse": True,  # Force reuse=2 regardless of preset
    },
}


def run_combination(preset_id, preset, sampler_id, sampler, dry_run=False):
    """Run a single preset×sampler combination with --ngram."""
    tag = f"ngram_{preset_id}_{sampler_id}"
    output_file = os.path.join(OUTPUT_DIR, f"{tag}_raw.mp4")

    # Determine reuse value
    reuse = 2 if sampler.get("override_reuse") else preset["reuse"]

    # Build command
    cmd = [
        H3_BIN, "--profile", "--ngram",
        "-d", MODEL_DIR,
        "-p", BENCHMARK_PROMPT,
        "--width", str(preset["width"]),
        "--height", str(preset["height"]),
        "--seconds", str(preset["seconds"]),
        "--steps", str(preset["steps"]),
        "--layers", str(preset["layers"]),
        "--reuse", str(reuse),
        "--use-int8-row-fc2",
        "-o", output_file,
    ]

    # Add sampler-specific flags (e.g. --sol-cache)
    for flag in sampler.get("extra_flags", []):
        if flag == "--reuse":
            continue  # Already set above
        cmd.append(flag)

    # Build environment
    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NGRAM"] = "1"
    if preset["width"] >= 960:
        env["H3_VAE_TILE_PIXELS"] = "960"
    else:
        env["H3_VAE_TILE_PIXELS"] = "640"
    for k, v in sampler.get("env", {}).items():
        env[k] = v

    print(f"\n{'='*70}")
    print(f"  [{preset['name']}] × [{sampler['name']}]")
    print(f"  {preset['width']}×{preset['height']} | {preset['steps']} steps | "
          f"reuse {reuse} | layers {preset['layers']}")
    print(f"  Output: {output_file}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        return None

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=os.path.join(os.path.dirname(__file__), "h3-lora-lab"),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - start

    # Parse telemetry from stderr
    stderr = result.stderr + result.stdout
    telemetry = {
        "preset": preset_id,
        "preset_name": preset["name"],
        "sampler": sampler_id,
        "sampler_name": sampler["name"],
        "resolution": f"{preset['width']}x{preset['height']}",
        "steps": preset["steps"],
        "reuse": reuse,
        "layers": preset["layers"],
        "total_seconds": round(elapsed, 2),
        "exit_code": result.returncode,
        "output_file": output_file,
        "ngram_enabled": True,
    }

    # Extract denoise time
    m = re.search(r'Denoise\s+(\d+\.\d+)\s*s', stderr)
    if m:
        telemetry["denoise_seconds"] = float(m.group(1))

    # Extract N-gram telemetry
    m = re.search(r'Total Patch Lookups:\s+(\d+)', stderr)
    if m:
        telemetry["ngram_lookups"] = int(m.group(1))
    m = re.search(r'Drafts Generated:\s+(\d+)', stderr)
    if m:
        telemetry["ngram_drafts"] = int(m.group(1))
    m = re.search(r'Drafts Accepted:\s+(\d+)', stderr)
    if m:
        telemetry["ngram_accepted"] = int(m.group(1))
    m = re.search(r'(\d+\.\d+)%\s*Acceptance Rate', stderr)
    if m:
        telemetry["ngram_acceptance_rate"] = float(m.group(1))

    # Extract Sol stats
    m = re.search(r'Cached Steps:\s+(\d+)\s+\((\d+\.\d+)%', stderr)
    if m:
        telemetry["cached_steps"] = int(m.group(1))
        telemetry["cached_percent"] = float(m.group(2))

    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} Completed in {elapsed:.1f}s (exit {result.returncode})")
    if "denoise_seconds" in telemetry:
        print(f"     Denoise: {telemetry['denoise_seconds']:.1f}s")
    if "ngram_acceptance_rate" in telemetry:
        print(f"     N-Gram Acceptance: {telemetry['ngram_acceptance_rate']:.1f}%")

    return telemetry


def main():
    dry_run = "--dry-run" in sys.argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    total = len(PRESETS) * len(SAMPLERS)
    count = 0

    print(f"\n🔬 N-Gram Unified Runner — {total} combinations")
    print(f"   Model: {MODEL_DIR}")
    print(f"   Output: {OUTPUT_DIR}")

    for preset_id, preset in PRESETS.items():
        for sampler_id, sampler in SAMPLERS.items():
            count += 1
            print(f"\n[{count}/{total}]", end="")

            telemetry = run_combination(
                preset_id, preset, sampler_id, sampler, dry_run=dry_run
            )
            if telemetry:
                results.append(telemetry)

    # Save results
    results_file = os.path.join(OUTPUT_DIR, "ngram_5x4_benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\n📊 Results saved to: {results_file}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Preset':<25} {'Sampler':<30} {'Denoise':>8} {'N-Gram%':>8} {'Total':>8}")
    print(f"{'-'*90}")
    for r in results:
        denoise = f"{r.get('denoise_seconds', '?'):.1f}s" if isinstance(r.get('denoise_seconds'), float) else "?"
        ngram = f"{r.get('ngram_acceptance_rate', 0):.1f}%" if 'ngram_acceptance_rate' in r else "N/A"
        total_t = f"{r['total_seconds']:.1f}s"
        print(f"{r['preset_name']:<25} {r['sampler_name']:<30} {denoise:>8} {ngram:>8} {total_t:>8}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
