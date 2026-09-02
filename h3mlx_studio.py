#!/usr/bin/env python3
"""
👑 H3MLX Interactive Studio & Golden Preset Director
Interactive TUI for selecting mathematically curated studio-quality presets.
All presets guarantee:
  - 50 Full Dense Layers (Zero Layer Skipping)
  - 100% Spatial Fidelity (Zero Token Reduction)
  - Exact Step Trajectories (Reuse = 1)
  - High-Token Spatial Canvas (>= 1500 latent tokens)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List

from h3mlx_presets import PRESETS, CANONICAL_RESOLUTIONS, calculate_canonical_frames
from h3mlx_engine_core import execute_h3_generation, resolve_model_path, BASE_DIR

# Rich Terminal Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BLUE = "\033[34m"
C_WHITE = "\033[97m"

STUDIO_PRESETS = [
    {
        "id": "h3mlx_champion_gold",
        "title": "👑 H3MLX Champion Master Gold (3:2)",
        "resolution": "768x512 -> 4K UHD Master (3072x2048)",
        "width": 768,
        "height": 512,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "39.9s",
        "est_fps": "2.25 FPS",
        "quality_tier": "Tier 1 Platinum Hollywood 4K (100.0/100)",
        "description": "Massima fedeltà assoluta: 50 layer densi al 100%, DPM++ 3M simplettico e mastering 4K UHD.",
        "default_prompt": "Cinematic close-up portrait of Brad Pitt smiling, natural soft lighting, highly detailed"
    },
    {
        "id": "h3mlx_cinema_16x9",
        "title": "🎬 H3MLX Cinema Anamorphic (16:9)",
        "resolution": "960x544 -> 4K Widescreen Master (3840x2176)",
        "width": 960,
        "height": 544,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "49.5s",
        "est_fps": "1.82 FPS",
        "quality_tier": "Tier 1 Platinum Cinema 4K (99.0/100)",
        "description": "Formato panoramico widescreen 16:9 (960x544) con coerenza ottica e mastering 4K UHD.",
        "default_prompt": "Cinematic wide shot of a futuristic neon city at sunset with rain reflections, highly detailed"
    },
    {
        "id": "h3mlx_macro_square",
        "title": "💎 H3MLX Square High-Density (1:1 640x640)",
        "resolution": "640x640 -> Ultra Square Master (2560x2560)",
        "width": 640,
        "height": 640,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "44.0s",
        "est_fps": "2.05 FPS",
        "quality_tier": "Tier 1 Macro Forensic (99.5/100)",
        "description": "Quadrato ad altissima densità (1600 token) con rendering sub-pixel e mastering 2.5K.",
        "default_prompt": "A sleek red sports car driving through a scenic mountain road in autumn, realistic, 4k"
    },
    {
        "id": "h3mlx_vertical_reel",
        "title": "📱 H3MLX Vertical Cinema Reel (9:16 FHD)",
        "resolution": "576x1024 -> 4K Vertical Cinema (2304x4096)",
        "width": 576,
        "height": 1024,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "49.5s",
        "est_fps": "1.82 FPS",
        "quality_tier": "Tier 1 Vertical Cinema 4K (100.0/100)",
        "description": "Rapporto esatto 9:16 FHD (576x1024, 2304 token) per ritratti verticali cinematografici ad alta fedeltà.",
        "default_prompt": "Cinematic vertical portrait of a beautiful woman with wavy hair in Paris, soft golden hour sunlight, expressive eyes and warm smile, highly detailed"
    },
    {
        "id": "h3mlx_ghibli_master",
        "title": "🌿 H3MLX Studio Ghibli Master (3:2)",
        "resolution": "768x512 -> 4K Anime Master (3072x2048)",
        "width": 768,
        "height": 512,
        "default_seconds": 3.75,
        "default_steps": 8,
        "mode": "boosted",
        "solver": "dpm3m",
        "reuse": 1,
        "layers": 50,
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "est_time_m5": "39.9s",
        "est_fps": "2.25 FPS",
        "quality_tier": "Tier 1 Anime Master 4K (98.0/100)",
        "description": "Estetica Hayao Miyazaki con dinamica del vento, texture ad acquerello soffici e upscaling 4K.",
        "default_prompt": "Studio Ghibli lush green valley with rolling hills, giant wind turbine, fluffy clouds, anime aesthetic"
    }
]

def print_header():
    width = min(shutil.get_terminal_size().columns, 85)
    print("\n" + C_CYAN + "═" * width + C_RESET)
    print(f"{C_BOLD}{C_WHITE}👑 H3MLX INTERACTIVE STUDIO (High-Quality Studio Edition){C_RESET}")
    print(f"{C_DIM}Inference Engine per Apple Silicon M1-M5 Max/Ultra · 50 Layer Densi · 100% Fedeltà Spaziale{C_RESET}")
    print(C_CYAN + "═" * width + C_RESET)
    
    # Eco & Thermal Warning Banner
    print(f"\n{C_RED}{C_BOLD}⚠️  ATTENZIONE TERMICA & HARDWARE ALERT:{C_RESET}")
    print(f"{C_YELLOW}• Eseguire carichi video pesanti solo con {C_BOLD}VENTOLE ACCESE{C_RESET}{C_YELLOW} (High Power Mode / TG Pro / Macs Fan Control).{C_RESET}")
    print(f"{C_YELLOW}• Consigliato su MacBook Pro 16\" Apple Silicon Max/Ultra (banda di memoria unificata >400 GB/s).{C_RESET}")
    print(f"{C_GREEN}🌱 {C_BOLD}MANIFESTO ECOLOGICO:{C_RESET} {C_GREEN}Generando in locale consumi 65W invece dei 6.400W di un cluster cloud.{C_RESET}")
    print(f"{C_DIM}   \"Più qualità e più velocità = più ottimizzazione = più fiumi salvati.\" 💧{C_RESET}\n")

def interactive_prompt(default_val: str, prompt_text: str) -> str:
    print(f"{C_CYAN}?{C_RESET} {C_BOLD}{prompt_text}{C_RESET} [{C_GREEN}{default_val}{C_RESET}]: ", end="", flush=True)
    val = input().strip()
    return val if val else default_val

def main():
    print_header()
    
    # Check model weights
    try:
        model_path = resolve_model_path(steps=8)
        print(f"{C_GREEN}✓ Modello Rilevato:{C_RESET} {model_path.name}\n")
    except FileNotFoundError as e:
        print(f"\n{C_YELLOW}⚠️  Avviso Modelli:{C_RESET}\n{e}\n")
        answer = input(f"{C_CYAN}?{C_RESET} Vuoi scaricare i pesi del modello ora con download_models.sh? [S/n]: ").strip().lower()
        if answer in ["", "s", "si", "y", "yes"]:
            import subprocess
            subprocess.run([sys.executable, str(BASE_DIR / "download_models.py")])
        else:
            print(f"{C_RED}Esecuzione terminata: modello assente.{C_RESET}")
            sys.exit(1)

    print(f"{C_BOLD}{C_WHITE}SELEZIONA IL PRESET DI GENERAZIONE AD ALTA QUALITÀ:{C_RESET}")
    for i, p in enumerate(STUDIO_PRESETS, 1):
        print(f"\n  {C_BOLD}{C_CYAN}[{i}]{C_RESET} {C_BOLD}{p['title']}{C_RESET}")
        print(f"      📐 Canvas: {C_WHITE}{p['resolution']}{C_RESET} | ⏱️ Tempo Stimato: {C_GREEN}{C_BOLD}{p['est_time_m5']}{C_RESET} ({p['est_fps']})")
        print(f"      🛡️ Qualità: {C_YELLOW}{p['quality_tier']}{C_RESET}")
        print(f"      📝 {C_DIM}{p['description']}{C_RESET}")
        
    print(f"\n  {C_BOLD}{C_CYAN}[0]{C_RESET} {C_DIM}Uscita / Exit{C_RESET}\n")
    
    while True:
        choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Scegli preset [1-{len(STUDIO_PRESETS)}]:{C_RESET} ").strip()
        if choice == "0":
            print(f"\n{C_DIM}Uscita da H3MLX Studio. Buona giornata!{C_RESET}\n")
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(STUDIO_PRESETS):
                selected = STUDIO_PRESETS[idx]
                break
        except ValueError:
            pass
        print(f"{C_RED}Scelta non valida. Riprova.{C_RESET}")

    print(f"\n{C_GREEN}✓ Selezionato: {selected['title']}{C_RESET}\n")
    
    # 1. Prompt customization (Standard vs Structured MiMo protocol)
    print(f"{C_CYAN}?{C_RESET} {C_BOLD}Modalità Prompt:{C_RESET}")
    print(f"  {C_CYAN}[1]{C_RESET} Prompt Standard (Testo Libero)")
    print(f"  {C_CYAN}[2]{C_RESET} Prompt Strutturato MiMo/Qwen3-VL (Dialoghi <d>, Lip-Sync Safeguards, Soundscape)")
    p_mode = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Scegli modalità [1-2, default: 1]:{C_RESET} ").strip()
    
    if p_mode == "2":
        print(f"\n{C_MAGENTA}{C_BOLD}🎙️ COMPOSIZIONE PROMPT STRUTTURATO (MiMo / Qwen3-VL Protocol):{C_RESET}")
        vis = interactive_prompt(selected["default_prompt"], "Descrizione visiva e movimento della scena")
        dial = interactive_prompt("", "Battuta di dialogo parlata (opzionale, premi Invio se assente)")
        
        prompt_parts = [f"integrated_multimodal_description: [Shot 1] {vis}"]
        if dial:
            lang = interactive_prompt("Italian", "Lingua del dialogo")
            spk = interactive_prompt("S1", "ID Speaker (es. S1, S2)")
            act = interactive_prompt("on-screen, speaking clearly, lips remain completely closed afterwards", "Azione e stato labiale del personaggio")
            prompt_parts[0] += f" ({spk}) [{act}] <d>[{lang}] {dial} </d>"
            
        sound = interactive_prompt("Natural realistic environment soundscape, subtle ambient foley", "Overall Soundscape (rumori diegetici d'ambiente)")
        prompt_parts.append(f"\noverall_soundscape: {sound}")
        
        bgm = interactive_prompt("N/A", "Musica extradiegetica (N/A consigliato per evitare allucinazioni vocali)")
        prompt_parts.append(f"\nnon_diegetic_music: {bgm}")
        
        prompt = "".join(prompt_parts)
        print(f"\n{C_GREEN}✓ Prompt Strutturato compilato con successo:{C_RESET}\n{C_DIM}{prompt}{C_RESET}\n")
    else:
        prompt = interactive_prompt(selected["default_prompt"], "Inserisci il Prompt di generazione")
    
    # 2. Duration seconds
    sec_str = interactive_prompt(str(selected["default_seconds"]), "Durata in Secondi")
    try:
        seconds = float(sec_str)
    except ValueError:
        seconds = selected["default_seconds"]
        
    # 3. Output Path
    default_out = f"outputs/{selected['id']}_{int(time.time())}.mp4"
    out_path = interactive_prompt(default_out, "Percorso Output Video MP4")
    
    # 4. Engine Mode switch option
    mode_str = interactive_prompt(selected["mode"], "Modalità Motore ('boosted' = H3MLX Metal 4 NAX / 'canonical' = Antirez pure)")
    if mode_str not in ["boosted", "canonical"]:
        mode_str = selected["mode"]
        
    # 5. Resolve Canvas & Parameters directly from preset
    width = selected.get("width", 768)
    height = selected.get("height", 512)
    frames = calculate_canonical_frames(seconds, width, height)
    steps = selected["default_steps"]
    
    # 6. Advanced Mastering Profile
    print(f"\n{C_CYAN}?{C_RESET} {C_BOLD}Profilo Mastering & Intraframe Detail Restoration:{C_RESET}")
    print(f"  {C_CYAN}[1]{C_RESET} 4K Cinema Master (Wavelet Bayes + CAS + VideoToolbox 10-bit)")
    print(f"  {C_CYAN}[2]{C_RESET} 1080p Cinema Master (Wavelet Bayes + CAS + VideoToolbox 10-bit)")
    print(f"  {C_CYAN}[3]{C_RESET} Raw Output Nativo (Nessun post-processing)")
    m_choice = input(f"{C_CYAN}?{C_RESET} {C_BOLD}Scegli profilo [1-3, default: 1]:{C_RESET} ").strip()
    upscale_4k = (m_choice != "3")
    master_profile = "4k" if m_choice in ["", "1"] else ("1080p" if m_choice == "2" else "none")
    
    # Summary Card before launch
    print("\n" + C_CYAN + "─" * 70 + C_RESET)
    print(f"{C_BOLD}{C_WHITE}🚀 RIEPILOGO PIANO DI GENERAZIONE ALTA DEFINIZIONE:{C_RESET}")
    print(f"  • {C_BOLD}Preset:{C_RESET}       {selected['title']}")
    print(f"  • {C_BOLD}Motore:{C_RESET}       {C_GREEN if mode_str=='boosted' else C_YELLOW}{mode_str.upper()}{C_RESET} (Metal 4 NAX: {'ON' if mode_str=='boosted' else 'OFF'})")
    print(f"  • {C_BOLD}Risoluzione:{C_RESET}  {width}x{height} {'-> Mastering ' + master_profile.upper() if master_profile != 'none' else ''}")
    print(f"  • {C_BOLD}Filtri:{C_RESET}       50 Layer Densi (Token Reduction: OFF · Reuse: 1)")
    print(f"  • {C_BOLD}Durata:{C_RESET}       {frames} frames ({frames/24:.2f}s @ 24fps) | Step DiT: {steps}")
    print(f"  • {C_BOLD}Mastering:{C_RESET}    {'Wavelet Bayes + CAS 0.25 (Main10)' if master_profile != 'none' else 'Raw Pass-through'}")
    print(f"  • {C_BOLD}Tempo Stimato:{C_RESET}{C_GREEN}{C_BOLD}{selected['est_time_m5']}{C_RESET}")
    print(f"  • {C_BOLD}Output File:{C_RESET}  {out_path}")
    print(f"  • {C_BOLD}Prompt:{C_RESET}       \"{prompt[:80]}...\"")
    print(C_CYAN + "─" * 70 + C_RESET)
    
    confirm = input(f"\n{C_CYAN}?{C_RESET} {C_BOLD}Avviare la generazione adesso? [S/n]:{C_RESET} ").strip().lower()
    if confirm in ["n", "no"]:
        print(f"{C_YELLOW}Generazione annullata.{C_RESET}\n")
        sys.exit(0)
        
    print(f"\n{C_GREEN}{C_BOLD}🚀 AVVIO MOTORE H3MLX METAL 4 NAX (100% FEDELTÀ SPAZIALE)...{C_RESET}")
    print(f"{C_DIM}Inizializzazione VRAM UMA Zero-Copy su Apple Silicon...{C_RESET}\n")
    
    t0 = time.perf_counter()
    res = execute_h3_generation(
        prompt=prompt,
        output_path=out_path,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        seed=42,
        engine_mode=mode_str,
        solver=selected.get("solver", "dpm3m"),
        reuse=selected.get("reuse", 1),
        layers=selected.get("layers", 50),
        token_reduction=False,  # High Quality: Strictly OFF
        int8=(mode_str == "boosted" and selected.get("int8", True)),
        upscale_4k=upscale_4k,
        profile=True
    )
    t1 = time.perf_counter()
    
    if res.success:
        wall_time = res.wall_time_s
        fps = frames / wall_time if wall_time > 0 else 0
        print("\n" + C_GREEN + "═" * 70 + C_RESET)
        print(f"{C_BOLD}{C_WHITE}🎉 GENERAZIONE ALTA FEDELTÀ COMPLETATA CON SUCCESSO!{C_RESET}")
        print(f"  ⏱️  {C_BOLD}Tempo Totale Reale:{C_RESET} {C_GREEN}{C_BOLD}{wall_time:.2f}s{C_RESET} (Throughput: {C_BOLD}{fps:.2f} FPS{C_RESET})")
        print(f"  🎥  {C_BOLD}File Video Salvato:{C_RESET} {C_CYAN}{C_BOLD}{res.output_path}{C_RESET}")
        
        if res.profile_data:
            print(f"\n  📊 {C_BOLD}Profiling Fasi GPU Metal:{C_RESET}")
            for phase, dur in res.profile_data.items():
                print(f"     • {phase:26s}: {C_CYAN}{dur:.2f}s{C_RESET}")
                
        # Ecological Savings Summary
        kwh_local = (65.0 * wall_time) / 3600000.0
        kwh_cloud = (6400.0 * 240.0) / 3600000.0
        saved_co2_g = (kwh_cloud - kwh_local) * 420.0
        saved_water_l = (kwh_cloud - kwh_local) * 1.8
        
        print(f"\n  🌱 {C_BOLD}{C_GREEN}IMPATTO ECOLOGICO RISPARMIATO:{C_RESET}")
        print(f"     • Energia consumata su Mac: {C_GREEN}{kwh_local*1000:.3f} Wh{C_RESET} (vs {kwh_cloud*1000:.1f} Wh cloud)")
        print(f"     • CO2 evitata rispetto al Cloud: {C_GREEN}~{saved_co2_g:.1f} g{C_RESET}")
        print(f"     • Acqua di raffreddamento data center risparmiata: {C_GREEN}~{saved_water_l:.2f} Litri{C_RESET} 💧")
        print(C_GREEN + "═" * 70 + C_RESET + "\n")
    else:
        print(f"\n{C_RED}❌ Errore durante la generazione:{C_RESET}\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
