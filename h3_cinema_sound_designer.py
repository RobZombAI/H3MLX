#!/usr/bin/env python3
"""
🔊 H3 CINEMA SOUND DESIGNER & FOLEY SYNTHESIS ENGINE
Generates rich, dynamic, multi-layered 48 kHz stereo Foley soundscapes
and impact sound effects synchronized with video action events.
"""

import os
import sys
import math
import struct
import wave
import subprocess
from pathlib import Path

SAMPLE_RATE = 48000

def generate_silence(duration: float):
    samples = int(duration * SAMPLE_RATE)
    return [0.0] * samples

def synthesize_gunshot(duration: float = 0.5):
    """Sintetizza uno sparo tattico (Crack transiente + Sub-Bass 60Hz + Eiezione metallica)."""
    import random
    samples = int(duration * SAMPLE_RATE)
    data = [0.0] * samples
    for i in range(samples):
        t = i / SAMPLE_RATE
        # Transiente crack (rumore bianco con decadimento esponenziale rapidissimo)
        crack = (random.random() * 2.0 - 1.0) * math.exp(-t * 60.0) * 1.5
        # Sub-bass thud (65 Hz decrescente)
        freq = 65.0 * math.exp(-t * 12.0)
        thud = math.sin(2.0 * math.pi * freq * t) * math.exp(-t * 15.0) * 1.2
        # Timbro metallico
        metal = math.sin(2.0 * math.pi * 2400.0 * t) * math.exp(-t * 40.0) * 0.3
        
        sample_val = crack + thud + metal
        # Soft clipping
        sample_val = math.tanh(sample_val * 1.2)
        data[i] = sample_val
    return data

def synthesize_sword_clash(duration: float = 0.8):
    """Sintetizza l'impatto e il risonatore armonico di una lama d'acciaio / katana."""
    import random
    samples = int(duration * SAMPLE_RATE)
    data = [0.0] * samples
    for i in range(samples):
        t = i / SAMPLE_RATE
        impact = (random.random() * 2.0 - 1.0) * math.exp(-t * 80.0) * 0.8
        # Armoniche dell'acciaio temperato (1200Hz, 2850Hz, 4900Hz)
        ring1 = math.sin(2.0 * math.pi * 1250.0 * t) * math.exp(-t * 6.0) * 0.6
        ring2 = math.sin(2.0 * math.pi * 2850.0 * t) * math.exp(-t * 10.0) * 0.4
        ring3 = math.sin(2.0 * math.pi * 4900.0 * t) * math.exp(-t * 16.0) * 0.25
        
        data[i] = math.tanh((impact + ring1 + ring2 + ring3) * 1.1)
    return data

def synthesize_body_slam(duration: float = 0.7):
    """Sintetizza un violento impatto corpo a corpo / wrestling body slam sul mat."""
    import random
    samples = int(duration * SAMPLE_RATE)
    data = [0.0] * samples
    for i in range(samples):
        t = i / SAMPLE_RATE
        thud_noise = (random.random() * 2.0 - 1.0) * math.exp(-t * 30.0) * 0.6
        # Sub-bass mat resonance (45 Hz)
        sub = math.sin(2.0 * math.pi * 45.0 * t) * math.exp(-t * 8.0) * 1.4
        mat_crack = math.sin(2.0 * math.pi * 180.0 * t) * math.exp(-t * 25.0) * 0.7
        data[i] = math.tanh((thud_noise + sub + mat_crack) * 1.2)
    return data

def synthesize_rain_ambience(duration: float):
    """Sintetizza il suono continuo della pioggia battente su asfalto."""
    import random
    samples = int(duration * SAMPLE_RATE)
    data = [0.0] * samples
    last = 0.0
    for i in range(samples):
        # Pink / Brown noise filter per pioggia
        white = random.random() * 2.0 - 1.0
        brown = (last + (0.05 * white)) / 1.05
        last = brown
        # Micro gocce casuali
        droplet = (random.random() * 0.3) if random.random() < 0.005 else 0.0
        data[i] = (brown * 0.35 + droplet) * 0.6
    return data

def create_foley_track(prompt: str, duration: float, out_wav: Path):
    """Analizza il prompt e assembla una colonna sonora Foley 48 kHz stereo su misura."""
    prompt_lower = prompt.lower()
    samples = int(duration * SAMPLE_RATE)
    left = [0.0] * samples
    right = [0.0] * samples
    
    # Livello 1: Ambiente
    if "rain" in prompt_lower or "tokyo" in prompt_lower or "osaka" in prompt_lower:
        rain = synthesize_rain_ambience(duration)
        for i in range(samples):
            left[i] += rain[i] * 0.7
            right[i] += rain[i] * 0.75
    else:
        # Rumore di fondo neutro
        rain = synthesize_rain_ambience(duration)
        for i in range(samples):
            left[i] += rain[i] * 0.2
            right[i] += rain[i] * 0.2
            
    # Livello 2: Effetti d'Azione Specifici
    if "gun" in prompt_lower or "shoot" in prompt_lower or "muzzle" in prompt_lower or "wick" in prompt_lower:
        # Doppio colpo Gun-Fu (a t = 0.4s e t = 0.9s)
        shot1 = synthesize_gunshot(0.6)
        shot2 = synthesize_gunshot(0.6)
        
        idx1 = int(0.4 * SAMPLE_RATE)
        for j, s in enumerate(shot1):
            if idx1 + j < samples:
                left[idx1 + j] += s * 0.9
                right[idx1 + j] += s * 0.75
                
        idx2 = int(0.9 * SAMPLE_RATE)
        for j, s in enumerate(shot2):
            if idx2 + j < samples:
                left[idx2 + j] += s * 0.75
                right[idx2 + j] += s * 0.95
                
    elif "katana" in prompt_lower or "sword" in prompt_lower or "blade" in prompt_lower:
        clash1 = synthesize_sword_clash(0.7)
        clash2 = synthesize_sword_clash(0.7)
        idx1 = int(0.5 * SAMPLE_RATE)
        for j, s in enumerate(clash1):
            if idx1 + j < samples:
                left[idx1 + j] += s * 0.85
                right[idx1 + j] += s * 0.7
        idx2 = int(1.2 * SAMPLE_RATE)
        for j, s in enumerate(clash2):
            if idx2 + j < samples:
                left[idx2 + j] += s * 0.7
                right[idx2 + j] += s * 0.9
                
    elif "wrestl" in prompt_lower or "brawl" in prompt_lower or "slam" in prompt_lower or "ring" in prompt_lower:
        slam = synthesize_body_slam(0.8)
        idx = int(0.7 * SAMPLE_RATE)
        for j, s in enumerate(slam):
            if idx + j < samples:
                left[idx + j] += s * 0.95
                right[idx + j] += s * 0.95

    # Mastering Stereo 16-bit PCM
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        
        raw_bytes = bytearray()
        for i in range(samples):
            l_val = max(-1.0, min(1.0, math.tanh(left[i])))
            r_val = max(-1.0, min(1.0, math.tanh(right[i])))
            
            l_int = int(l_val * 32767.0)
            r_int = int(r_val * 32767.0)
            raw_bytes.extend(struct.pack("<hh", l_int, r_int))
        wf.writeframes(raw_bytes)
    return True

def inject_foley_to_video(video_in: Path, prompt: str, video_out: Path):
    """Genera la traccia Foley e la muxa nel video MP4 a 48 kHz."""
    # Ottieni la durata del video con ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_in)
    ]
    res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True)
    duration = float(res.stdout.strip()) if res.returncode == 0 else 4.0
    
    temp_wav = Path("/tmp/h3_custom_foley.wav")
    create_foley_track(prompt, duration, temp_wav)
    
    # Muxing FFmpeg con mastering dinamico
    mux_cmd = [
        "ffmpeg", "-y", "-i", str(video_in), "-i", str(temp_wav),
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-af", "stereowiden=crossfeed=0.3:drymix=0.85,bass=g=6.0:f=80,loudnorm=I=-14:TP=-1.0",
        str(video_out)
    ]
    subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_wav.exists():
        temp_wav.unlink()
    return video_out.exists()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 h3_cinema_sound_designer.py <video_in.mp4> <prompt> [video_out.mp4]")
        sys.exit(1)
    v_in = Path(sys.argv[1])
    p = sys.argv[2]
    v_out = Path(sys.argv[3]) if len(sys.argv) > 3 else v_in.with_name(v_in.stem + "_with_foley.mp4")
    
    print(f"🔊 Generazione Sound Design Foley per: {v_in.name}")
    if inject_foley_to_video(v_in, p, v_out):
        print(f"✅ Video con Audio Foley Cinematografico salvato in: {v_out}")
    else:
        print("❌ Errore durante l'iniezione audio.")
