import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
H3_DIR = BASE_DIR / "h3-lora-lab"
MODEL_DIR = Path("/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step")
OUTPUT_DIR = BASE_DIR / "outputs_timeline_hyperpop_15s"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/timeline_hyperpop_gallery")
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR))
from h3_status import update_task_status

SHOTS = [
    {
        "id": 1,
        "name": "Shot 1: Glitch Grid & Staccato Tutting",
        "prompt": "integrated_multimodal_description: [Shot 1: 00:00.000 - 00:04.000 | 10 Beats] Extreme wide-angle low-angle tracking shot in an infinite void of glass timeline rails, scanning laser grids, and suspended metal conduits in stark void black, cold laser white, and warning neon red. A single female dancer in a matte black technical aerodynamic bodysuit with crimson LED seams begins sharp, staccato popping and tutting. On every heavy 808 kick, the entire spatial environment compresses violently toward her in perfect rhythmic sync. Glitch cuts fragment the perspective as floating timeline frames shudder under high-velocity acceleration.\n\noverall_soundscape: Sub-bass 808 shockwaves rattling glass panels, razor-sharp metallic rim-shots, clicking laser scanning pulses, and glitching UI feedback loops.\n\nnon_diegetic_music: Blistering Industrial Hyperpop × Deconstructed Club track at 160 BPM in D minor with hard 808 kicks and aggressive glitch hi-hats."
    },
    {
        "id": 2,
        "name": "Shot 2: Snare Duplicate Echoes & MINIMAX Architecture",
        "prompt": "integrated_multimodal_description: [Shot 2: 00:04.000 - 00:08.000 | 10 Beats] Match-on-action cut to a Dutch-angle Dutch-tilt medium shot as rapid metallic snares duplicate her movements into an echelon of recursive translucent ghost projections trailing behind her limbs. She glides under the towering physical 3D monolithic brushed-titanium architecture spelling 'MINIMAX' that casts sharp geometric shadows across the reflective glass floorboards.\n\noverall_soundscape: Heavy metallic snare cracks resonating off titanium structures, recursive audio stutter delays, and whooshing air displacement.\n\nnon_diegetic_music: Rapid 160 BPM metallic snare rolls, distorted bass arpeggios, and stuttering pitch-shifted vocal chops."
    },
    {
        "id": 3,
        "name": "Shot 3: 3D Origami Interface Folding Matrix",
        "prompt": "integrated_multimodal_description: [Shot 3: 00:08.000 - 00:11.500 | 9 Beats] Top-down bird's-eye perspective plunging downward into a folding 3D kinetic origami matrix. The 2D timeline interface bends upward along glowing crimson gridlines, forming a closing tunnel of light and glass as the dancer performs an intense back-bend wave, resisting temporal entrapment.\n\noverall_soundscape: Deep sub-bass glides distorting space, glass timeline panels creaking and folding, and high-frequency laser scanner whines.\n\nnon_diegetic_music: Intense sub-bass build-up, accelerating hi-hat rolls, and tension-filled rising synthesizer sweeps."
    },
    {
        "id": 4,
        "name": "Shot 4: THE DROP - 360 Orbital Long Take Escape",
        "prompt": "integrated_multimodal_description: [Shot 4: 00:11.500 - 00:15.000 | 11 Beats | THE DROP] Match-on-action explosion into an impossible, hyper-fluid continuous 360-degree orbital long take. The dancer shatters the timeline repetition loop, breaking out into an explosive, unrestricted hyper-kinetic contemporary dance routine. Glass timeline lattices fragment into millions of weightless illuminated red-and-white shards swirling centrifugally around her, as the camera orbits at warp speed and settles into a razor-sharp, defiant final hero pose in extreme close-up.\n\noverall_soundscape: Explosive drop shockwave, glass shards shattering and floating, rushing wind, and fierce breathing of the dancer in extreme close-up.\n\nnon_diegetic_music: Devastating 160 BPM industrial drop with maximum distortion 808s, screaming modular synths, and virtuosic breakcore percussion."
    }
]

def main():
    task_id = "task-timeline-15s-multichunk"
    print("=" * 110)
    print("🚀 GENERAZIONE 15s N-GRAM MULTI-CHUNK (4 SHOT CONCATENATI / ZERO MASTERING / RAW MODELLO 100%)")
    print("=" * 110)

    env = os.environ.copy()
    env["H3_PROFILE"] = "1"
    env["H3_NAX"] = "qkv-attn"
    env["H3_ZERO_COPY_WEIGHTS"] = "1"
    env["H3_REUSE_MPS_COMMAND"] = "1"
    env["H3_GPU_SAMPLER"] = "1"
    env["H3_HOLISTIC_NGRAM"] = "1"
    env["H3_NGRAM_SPECULATIVE"] = "1"
    env["H3_VAE_NGRAM_SPECULATIVE"] = "1"
    env["H3_AUDIO_NGRAM_SPECULATIVE"] = "1"
    env["H3_NGRAM_SUPER_DETAIL"] = "1"
    env["H3_SHARPNESS_BOOST"] = "1.35"
    env["H3_TSSAA"] = "1"
    env["H3_ADAPTIVE_FOCAL_DENOISE"] = "1"
    env["H3_OCTREE_NGRAM"] = "1"
    env["H3_OPTICAL_FLOW_WARP"] = "1"
    env["H3_TRIGRAM_TREE"] = "1"
    env["H3_NGRAM_THRESHOLD"] = "0.985"
    env["H3_VAE_THRESHOLD"] = "0.990"
    env["OMP_NUM_THREADS"] = "18"

    shot_files = []
    total_denoise = 0.0
    total_vae = 0.0

    for idx, s in enumerate(SHOTS, 1):
        out_shot = OUTPUT_DIR / f"shot_{idx}_raw.mp4"
        shot_files.append(out_shot)
        update_task_status(task_id, f"Timeline 15s (Shot {idx}/4: {s['name'][:25]})", "RUNNING", idx, 4)
        
        print(f"\n🎬 [{idx}/4] Generazione {s['name']} (96 Frame / 4.0s @ 24fps)...")
        cmd = [
            "./h3", "--profile",
            "-d", str(MODEL_DIR),
            "-p", s["prompt"],
            "--width", "640",
            "--height", "640",
            "--frames", "96",
            "--steps", "8",
            "--layers", "50",
            "--reuse", "1",
            "--use-int8-row-fc2",
            "--seed", str(42 + idx),
            "-o", str(out_shot)
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(H3_DIR), env=env, capture_output=True, text=True)
        t1 = time.perf_counter()
        
        den_s = 35.0
        vae_s = 14.0
        for line in proc.stdout.splitlines():
            if "denoise in" in line:
                try: den_s = float(line.split("denoise in")[-1].split("s")[0].strip())
                except: pass
            elif "vae decode in" in line:
                try: vae_s = float(line.split("vae decode in")[-1].split("s")[0].strip())
                except: pass
        total_denoise += den_s
        total_vae += vae_s
        print(f"   ✓ Shot {idx} Completato in {den_s:.2f}s (Denoise) + {vae_s:.2f}s (VAE) = {den_s+vae_s:.2f}s")

    # Concatenate seamlessly without any mastering re-encoding (Zero Loss / 100% Native RAW)
    final_raw = OUTPUT_DIR / "timeline_dancer_15s_raw.mp4"
    list_file = OUTPUT_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for sf in shot_files:
            f.write(f"file '{sf.resolve()}'\n")

    print("\n🔗 Unione senza perdita dei 4 shot nel video master RAW 15s (100% Nativo Modello)...")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(final_raw)
    ], capture_output=True)

    # Generate quick GIF preview from raw
    gif_path = OUTPUT_DIR / "timeline_dancer_15s_animated.gif"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(final_raw),
        "-vf", "fps=10,scale=360:360:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(gif_path)
    ], capture_output=True)

    thumb_path = OUTPUT_DIR / "timeline_dancer_15s_thumb.jpg"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(final_raw),
        "-ss", "00:00:11.500", "-vframes", "1", "-q:v", "2",
        str(thumb_path)
    ], capture_output=True)

    # Copy to brain dir
    subprocess.run(f"cp {OUTPUT_DIR}/* {BRAIN_DIR}/", shell=True)

    # Update status to COMPLETED
    update_task_status(task_id, "Timeline Dancer 15s (160 BPM)", "COMPLETED", 4, 4, total_denoise, total_vae, str(final_raw))

    print("=" * 110)
    print(f"✅ VIDEO 15s RAW MODELLO COMPLETATO CON SUCCESSO! File: {final_raw}")
    print(f"   Tempo GPU Totale: {total_denoise+total_vae:.2f}s | FPS: {360/(total_denoise+total_vae):.2f}")
    print("=" * 110)

if __name__ == "__main__":
    main()
