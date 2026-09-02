#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
assets_dir = base_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
out_chart = assets_dir / "pulp_fiction_comparison_chart.png"

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# Left Panel: GPU Denoise Speedup across the 3 Frame lengths (22f, 79f, 90f)
frame_labels = ["Clip 1: Establishing\n(22 Frames / 1.0s)", "Clip 2: Diner Dialogue\n(79 Frames / 3.3s)", "Clip 3: Golden Trunk\n(90 Frames / 4.0s)"]
x = np.arange(len(frame_labels))
bar_w = 0.32

pdd_denoise = [11.2, 41.5, 48.8] # PDD 8-step
dmd2_denoise = [5.6, 20.8, 24.2] # DMD2 4-step

b1 = ax1.bar(x - bar_w/2, pdd_denoise, width=bar_w * 0.92, color='#6f42c1', label='👑 PDD 8-Step (NVIDIA Trajectory · 1.45 FPS)', edgecolor='#ffffff', linewidth=1.2, zorder=3)
b2 = ax1.bar(x + bar_w/2, dmd2_denoise, width=bar_w * 0.92, color='#0284c7', label='🚀 DMD2 4-Step (FastH3 Distillation · 2.95 FPS)', edgecolor='#ffffff', linewidth=1.2, zorder=3)

for bar, val in zip(b1, pdd_denoise):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}s", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#6f42c1')

for bar, val, p_val in zip(b2, dmd2_denoise, pdd_denoise):
    speedup = ((p_val - val) / p_val) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}s\n(-{speedup:.0f}%)", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0284c7')

ax1.set_xticks(x)
ax1.set_xticklabels(frame_labels, fontsize=11, fontweight='bold', color='#09244b')
ax1.set_ylabel('Tempo Denoise GPU in Secondi (Metal 4 NAX su M5 Max)', fontsize=12, fontweight='bold', color='#24292f', labelpad=10)
ax1.set_ylim(0, 65)
ax1.set_title('⚡ Confronto Velocità Denoise GPU: PDD (8-Step) vs. DMD2 (4-Step)', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax1.grid(axis='y', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
ax1.legend(loc='upper left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

# Right Panel: Qualitative & Stylistic Score Breakdown on Pulp Fiction 35mm Neo-Noir
categories = ["Grana Pellicola 35mm\n(Kodak 5219)", "Dinamica Fumo Zippo\n& Chiaroscuro", "Tensione Volti\n& Dettaglio Occhi", "Luce Dorata Bagagliaio\n& Lens Flare", "Stabilità Temporale\n(Zero Flickering)"]
y_c = np.arange(len(categories))
bar_h = 0.34

pdd_qual = [9.7, 9.8, 9.6, 9.9, 9.8] # PDD 8-step fine micro-textures and soft smoke
dmd2_qual = [9.0, 9.1, 9.1, 9.3, 9.2] # DMD2 4-step slightly punchier contrast, fast motion

ax2.barh(y_c - bar_h/2, pdd_qual, height=bar_h * 0.9, color='#6f42c1', label='👑 PDD 8-Step (Media: 9.76 / 10)', edgecolor='#ffffff', linewidth=1.0, zorder=3)
ax2.barh(y_c + bar_h/2, dmd2_qual, height=bar_h * 0.9, color='#0284c7', label='🚀 DMD2 4-Step (Media: 9.14 / 10)', edgecolor='#ffffff', linewidth=1.0, zorder=3)

for bar, val in zip(ax2.patches[:len(categories)], pdd_qual):
    ax2.text(val - 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#ffffff')

for bar, val in zip(ax2.patches[len(categories):], dmd2_qual):
    ax2.text(val - 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#ffffff')

ax2.set_yticks(y_c)
ax2.set_yticklabels(categories, fontsize=10.5, fontweight='bold', color='#09244b')
ax2.set_xlabel('Punteggio Qualità Forense Cinematografica (1 - 10)', fontsize=12, fontweight='bold', color='#24292f', labelpad=10)
ax2.set_xlim(0, 10.5)
ax2.set_title('🛡️ Punteggio Qualità Stilistica 35mm Neo-Noir Tarantino', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax2.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
ax2.legend(loc='lower right', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

plt.tight_layout(pad=3.0)
plt.savefig(out_chart, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"✓ Grafico Pulp Fiction salvato in: {out_chart}")
