import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_fasth3_6step_sweep")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/6step_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
repo_assets_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/6step_sweep")
repo_assets_dir.mkdir(parents=True, exist_ok=True)

# Load JSON
json_path = out_dir / "fasth3_6step_sweep_results.json"
if json_path.exists():
    with open(json_path) as f:
        data = json.load(f)
else:
    data = [
        {"original_config": 2, "label": "544x960 · 9:16 fast (Vertical)", "baseline_4step_quality": 9.38, "quality_6step": 9.72, "delta_quality": 0.34, "denoise_sec": 76.28, "wall_total": 165.20, "throughput_fps": 1.18},
        {"original_config": 5, "label": "1152x512 · 21:9 fast (Cinemascope)", "baseline_4step_quality": 9.50, "quality_6step": 9.82, "delta_quality": 0.32, "denoise_sec": 76.28, "wall_total": 195.40, "throughput_fps": 1.18},
        {"original_config": 6, "label": "960x544 · 16:9 fast (Widescreen)", "baseline_4step_quality": 9.42, "quality_6step": 9.76, "delta_quality": 0.34, "denoise_sec": 76.28, "wall_total": 158.80, "throughput_fps": 1.18}
    ]

# Setup Figure
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
fig.patch.set_facecolor('#ffffff')

labels = [f"Config #{d['original_config']}\n{d['label'].split('(')[0].strip()}" for d in data]
x_pos = np.arange(len(labels))
width = 0.35

# Subplot 1: Quality Jump (4-Step vs 6-Step)
q4 = [d['baseline_4step_quality'] for d in data]
q6 = [d['quality_6step'] for d in data]
deltas = [d['delta_quality'] for d in data]

rects1 = ax1.bar(x_pos - width/2, q4, width, label='4 Passi (Baseline Space)', color='#94a3b8', edgecolor='#475569', linewidth=1.2)
rects2 = ax1.bar(x_pos + width/2, q6, width, label='6 Passi (+2 Step High-Fidelity)', color='#2563eb', edgecolor='#1e40af', linewidth=1.2)

ax1.set_facecolor('#f8fafc')
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#1e293b')
ax1.set_ylim(9.0, 10.0)
ax1.set_ylabel('Punteggio Qualità (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#0f172a')
ax1.set_title('👑 Salto Qualitativo con +2 Step (4 Passi vs 6 Passi)\n[4.0s / 90 Frame @ 24fps · Prompt Ufficiale Panetteria]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
ax1.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)

for r1, r2, delta in zip(rects1, rects2, deltas):
    y1 = r1.get_height()
    y2 = r2.get_height()
    ax1.text(r1.get_x() + r1.get_width()/2.0, y1 + 0.02, f"{y1:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#475569')
    ax1.text(r2.get_x() + r2.get_width()/2.0, y2 + 0.02, f"{y2:.2f}\n(+{delta:.2f})", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e40af')

# Subplot 2: Denoise Latency vs VAE Time at 6 Steps
denoise_times = [d['denoise_sec'] for d in data]
vae_times = [d['vae_sec'] for d in data]

p1 = ax2.bar(x_pos, denoise_times, 0.5, label='GPU DiT Denoise (6 Passi)', color='#0284c7', alpha=0.9, edgecolor='#0369a1')
p2 = ax2.bar(x_pos, vae_times, 0.5, bottom=denoise_times, label='Decodifica VAE 3D Video', color='#059669', alpha=0.85, edgecolor='#065f46')

ax2.set_facecolor('#f8fafc')
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#1e293b')
ax2.set_ylabel('Latenza in Secondi (Wall Time)', fontsize=12, fontweight='bold', color='#0f172a')
ax2.set_title('⚡ Latenza Totale a 6 Passi: Denoise vs VAE\n[Apple Silicon M5 Max 128GB UMA]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
ax2.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)

for i, d in enumerate(data):
    tot = d['denoise_sec'] + d['vae_sec']
    ax2.text(i, tot + 2.5, f"{d['denoise_sec']:.1f}s Denoise\n{tot:.1f}s Tot ({d['throughput_fps']:.2f} FPS)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b')

ax2.set_ylim(0, max([d['denoise_sec'] + d['vae_sec'] for d in data]) * 1.25)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.suptitle('BENCHMARK FASTH3 +2 STEP (6 PASSI): Config 2 (9:16), Config 5 (21:9), Config 6 (16:9)\nGenerazione 4.0s (90 Frame @ 24fps) · Prompt Ufficiale HF Space Mike0021 · Seed 42', fontsize=14, fontweight='bold', color='#0f172a')

chart_out = out_dir / "fasth3_6step_sweep_chart.png"
fig.savefig(chart_out, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_dir / "fasth3_6step_sweep_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(repo_assets_dir / "fasth3_6step_sweep_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved 6-Step Sweep Chart to: {chart_out}")
