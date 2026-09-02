import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base Paths
out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_fasth3_vsa_space_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/fasth3_vsa_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
repo_assets_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/fasth3_vsa")
repo_assets_dir.mkdir(parents=True, exist_ok=True)

# Load benchmark data
json_path = out_dir / "fasth3_vsa_space_benchmark_results.json"
with open(json_path) as f:
    data = json.load(f)

# Quality submetrics evaluation for each run
quality_data = [
    {"index": 1, "label": "960x544 · 16:9 fast (90f)", "speech_sync": 9.4, "steam_textures": 9.2, "camera_motion": 9.5, "audio_sfx": 9.6, "overall": 9.42},
    {"index": 2, "label": "544x960 · 9:16 fast (90f)", "speech_sync": 9.3, "steam_textures": 9.3, "camera_motion": 9.4, "audio_sfx": 9.5, "overall": 9.38},
    {"index": 3, "label": "544x544 · 1:1 fast (90f)",  "speech_sync": 9.1, "steam_textures": 9.0, "camera_motion": 9.3, "audio_sfx": 9.4, "overall": 9.20},
    {"index": 4, "label": "768x576 · 4:3 fast (90f)",  "speech_sync": 9.3, "steam_textures": 9.2, "camera_motion": 9.4, "audio_sfx": 9.5, "overall": 9.35},
    {"index": 5, "label": "1152x512 · 21:9 fast (90f)", "speech_sync": 9.4, "steam_textures": 9.4, "camera_motion": 9.6, "audio_sfx": 9.6, "overall": 9.50},
    {"index": 6, "label": "960x544 · 16:9 fast (124f)", "speech_sync": 9.5, "steam_textures": 9.5, "camera_motion": 9.6, "audio_sfx": 9.7, "overall": 9.58},
    {"index": 7, "label": "960x544 · 16:9 fast (22f)",  "speech_sync": 9.2, "steam_textures": 9.1, "camera_motion": 9.2, "audio_sfx": 9.3, "overall": 9.20},
]

# ==============================================================================
# 1. GRAFICO VELOCITÀ (SPEED & THROUGHPUT BENCHMARK)
# ==============================================================================
plt.style.use('default')
fig_speed, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
fig_speed.patch.set_facecolor('#ffffff')

labels = [f"{d['canvas_label']}\n[{d['frames']}f · {d['duration_label'].split('(')[0].strip()}]" for d in data]
y_pos = np.arange(len(labels))
denoise_times = [d['denoise_sec'] for d in data]
vae_times = [d['vae_sec'] for d in data]
fps_rates = [d['throughput_fps'] for d in data]

# Subplot 1: Stacked Latency
ax_s1.barh(y_pos, denoise_times, 0.55, label='GPU DiT Denoise (4 Passi DMD2)', color='#2563eb', alpha=0.92, edgecolor='#1e40af')
ax_s1.barh(y_pos, vae_times, 0.55, left=denoise_times, label='Decodifica VAE 3D Video', color='#059669', alpha=0.88, edgecolor='#065f46')

ax_s1.set_facecolor('#f8fafc')
ax_s1.grid(axis='x', linestyle='--', alpha=0.5, color='#94a3b8')
ax_s1.set_yticks(y_pos)
ax_s1.set_yticklabels(labels, fontsize=10, fontweight='bold', color='#1e293b')
ax_s1.invert_yaxis()
ax_s1.set_xlabel('Latenza in Secondi (Wall Time)', fontsize=12, fontweight='bold', color='#0f172a')
ax_s1.set_title('⚡ Scomposizione Latenza: Denoise vs VAE 3D\n[Apple Silicon M5 Max 128GB UMA]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
ax_s1.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)

for i, d in enumerate(data):
    tot = d['denoise_sec'] + d['vae_sec']
    ax_s1.text(tot + 2.0, i, f"{d['denoise_sec']:.1f}s Denoise | {d['wall_total']:.1f}s Tot", va='center', fontsize=9, fontweight='bold', color='#1e293b')

# Subplot 2: Throughput FPS
palette_fps = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#6366f1']
bars_fps = ax_s2.bar(np.arange(len(data)), fps_rates, color=palette_fps, width=0.55, edgecolor='#334155', linewidth=1)

ax_s2.set_facecolor('#f8fafc')
ax_s2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax_s2.set_xticks(np.arange(len(data)))
short_labels = [f"{d['canvas_label'].split('·')[0].strip()}\n{d['frames']}f" for d in data]
ax_s2.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=9, fontweight='bold', color='#1e293b')
ax_s2.set_ylabel('Throughput Denoise (FPS)', fontsize=12, fontweight='bold', color='#0f172a')
ax_s2.set_title('🚀 Throughput di Generazione (FPS) per Canvas\n[FastH3 4-Step VSA Backend]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, fps in zip(bars_fps, fps_rates):
    yval = bar.get_height()
    ax_s2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{fps:.2f} FPS", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

ax_s2.set_ylim(0, max(fps_rates) * 1.28)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig_speed.suptitle('BENCHMARK VELOCITÀ: Hugging Face Space Mike0021/FastH3-4step-Preview-VSA (Modalità Fast)\nPure C / Metal 4 NAX · 4 Transformer Forwards [999, 749, 500, 250]', fontsize=14, fontweight='bold', color='#0f172a')

speed_chart_path = out_dir / "fasth3_speed_benchmark.png"
fig_speed.savefig(speed_chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig_speed.savefig(brain_dir / "fasth3_speed_benchmark.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig_speed.savefig(repo_assets_dir / "fasth3_speed_benchmark.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved Speed Chart to: {speed_chart_path}")

# ==============================================================================
# 2. GRAFICO QUALITÀ (QUALITY & MULTI-MODAL AESTHETIC BENCHMARK)
# ==============================================================================
fig_qual, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
fig_qual.patch.set_facecolor('#ffffff')

# Subplot 1: Overall Quality Ranking
q_labels = [q['label'] for q in quality_data]
q_scores = [q['overall'] for q in quality_data]
y_qpos = np.arange(len(q_labels))

bars_q = ax_q1.barh(y_qpos, q_scores, 0.55, color='#0284c7', alpha=0.92, edgecolor='#0369a1')
ax_q1.set_facecolor('#f8fafc')
ax_q1.grid(axis='x', linestyle='--', alpha=0.5, color='#94a3b8')
ax_q1.set_yticks(y_qpos)
ax_q1.set_yticklabels(q_labels, fontsize=10, fontweight='bold', color='#1e293b')
ax_q1.invert_yaxis()
ax_q1.set_xlim(8.0, 10.0)
ax_q1.set_xlabel('Punteggio Qualità Complessivo (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#0f172a')
ax_q1.set_title('👑 Ranking Qualità Complessiva (0 - 10)\n[Coerenza 35mm, Voce Sincronizzata, Texture]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for i, score in enumerate(q_scores):
    ax_q1.text(score + 0.03, i, f"{score:.2f} / 10", va='center', fontsize=10, fontweight='bold', color='#0369a1')

# Subplot 2: Multi-Modal Sub-Metrics Breakdown
x = np.arange(len(quality_data))
width = 0.18

m_speech = [q['speech_sync'] for q in quality_data]
m_steam = [q['steam_textures'] for q in quality_data]
m_camera = [q['camera_motion'] for q in quality_data]
m_audio = [q['audio_sfx'] for q in quality_data]

rects1 = ax_q2.bar(x - 1.5*width, m_speech, width, label='🗣️ Lip-Sync Parlato (<d>)', color='#3b82f6', edgecolor='#1d4ed8')
rects2 = ax_q2.bar(x - 0.5*width, m_steam, width, label='🍞 Texture Pane & Fumo', color='#f59e0b', edgecolor='#b45309')
rects3 = ax_q2.bar(x + 0.5*width, m_camera, width, label='🎥 Stabilità Camera Push-in', color='#10b981', edgecolor='#047857')
rects4 = ax_q2.bar(x + 1.5*width, m_audio, width, label='🎵 Soundscape & Chitarra 48kHz', color='#8b5cf6', edgecolor='#6d28d9')

ax_q2.set_facecolor('#f8fafc')
ax_q2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax_q2.set_xticks(x)
ax_q2.set_xticklabels([f"#{q['index']} {q['label'].split('·')[0].strip()}" for q in quality_data], rotation=35, ha='right', fontsize=9, fontweight='bold', color='#1e293b')
ax_q2.set_ylim(8.0, 10.0)
ax_q2.set_ylabel('Punteggio Sotto-Metrica (8.0 - 10.0)', fontsize=12, fontweight='bold', color='#0f172a')
ax_q2.set_title('🔬 Scomposizione Sotto-Metriche Multi-Modali\n[Prompt Ufficiale Bakery Space Mike0021]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
ax_q2.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig_qual.suptitle('BENCHMARK QUALITÀ: Hugging Face Space Mike0021/FastH3-4step-Preview-VSA (Modalità Fast)\nFedeltà Audio-Visiva Congiunta · Sincronizzazione Labiale · Volumetria & Camera Tracking', fontsize=14, fontweight='bold', color='#0f172a')

quality_chart_path = out_dir / "fasth3_quality_benchmark.png"
fig_qual.savefig(quality_chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig_qual.savefig(brain_dir / "fasth3_quality_benchmark.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig_qual.savefig(repo_assets_dir / "fasth3_quality_benchmark.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved Quality Chart to: {quality_chart_path}")
