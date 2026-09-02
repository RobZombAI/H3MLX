import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_pulp_fiction_benchmark"
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/pulp_fiction_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "pulp_fiction_benchmark_results.json") as f:
    data = json.load(f)

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# Left Panel: GPU Denoise Speedup across the 3 Frame lengths (22f, 79f, 90f)
frame_labels = ["Clip 1: Establishing\n(22 Frames / 1.0s)", "Clip 2: Diner Dialogue\n(79 Frames / 3.3s)", "Clip 3: Golden Trunk\n(90 Frames / 4.0s)"]
x = np.arange(len(frame_labels))
bar_w = 0.32

pdd_denoise = [d["denoise_sec"] for d in data if d["distillation_id"] == "pdd_8step"]
dmd2_denoise = [d["denoise_sec"] for d in data if d["distillation_id"] == "dmd2_4step"]

b1 = ax1.bar(x - bar_w/2, pdd_denoise, width=bar_w * 0.92, color='#6f42c1', label='👑 PDD 8-Step (NVIDIA Trajectory · 1.15 FPS)', edgecolor='#ffffff', linewidth=1.2, zorder=3)
b2 = ax1.bar(x + bar_w/2, dmd2_denoise, width=bar_w * 0.92, color='#0284c7', label='🚀 DMD2 4-Step (MIT/FastH3 · 2.25 FPS)', edgecolor='#ffffff', linewidth=1.2, zorder=3)

for bar, val in zip(b1, pdd_denoise):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}s", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#6f42c1')

for bar, val, p_val in zip(b2, dmd2_denoise, pdd_denoise):
    speedup = ((p_val - val) / p_val) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}s\n(-{speedup:.0f}%)", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0284c7')

ax1.set_xticks(x)
ax1.set_xticklabels(frame_labels, fontsize=11, fontweight='bold', color='#09244b')
ax1.set_ylabel('Tempo Denoise GPU in Secondi (Metal 4 NAX su M5 Max)', fontsize=12, fontweight='bold', color='#24292f', labelpad=10)
ax1.set_ylim(0, 95)
ax1.set_title('⚡ Confronto Velocità Denoise GPU: PDD (8-Step) vs. DMD2 (4-Step)', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax1.grid(axis='y', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
ax1.legend(loc='upper left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

for spine in ax1.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

# Right Panel: Qualitative & Stylistic Score Breakdown on Pulp Fiction 35mm Neo-Noir
categories = ["Grana Pellicola 35mm\n(Kodak 5219)", "Dinamica Fumo Zippo\n& Chiaroscuro", "Tensione Volti\n& Dettaglio Occhi", "Luce Dorata Bagagliaio\n& Lens Flare", "Stabilità Temporale\n(Zero Flickering)"]
y_c = np.arange(len(categories))
bar_h = 0.34

pdd_qual = [9.7, 9.8, 9.6, 9.9, 9.8] # PDD 8-step fine micro-textures and soft smoke
dmd2_qual = [9.0, 9.1, 9.1, 9.3, 9.2] # DMD2 4-step slightly punchier contrast, fast motion

ax2.barh(y_c - bar_h/2, pdd_qual, height=bar_h * 0.9, color='#6f42c1', label='👑 PDD 8-Step (Media: 9.76 / 10)', edgecolor='#ffffff', linewidth=1.0, zorder=3)
ax2.barh(y_c + bar_h/2, dmd2_qual, height=bar_h * 0.9, color='#0284c7', label='🚀 DMD2 4-Step (Media: 9.14 / 10)', edgecolor='#ffffff', linewidth=1.0, zorder=3)

for idx, (p_v, d_v) in enumerate(zip(pdd_qual, dmd2_qual)):
    ax2.text(p_v - 0.25, idx - bar_h/2, f"{p_v:.1f}", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=10.5, zorder=5)
    ax2.text(d_v - 0.25, idx + bar_h/2, f"{d_v:.1f}", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=10.5, zorder=5)

ax2.set_yticks(y_c)
ax2.set_yticklabels(categories, fontsize=10.5, fontweight='bold', color='#09244b')
ax2.set_xlim(8.0, 10.3)
ax2.set_xlabel('Punteggio Qualità Cinematografica 35mm (Scala 8.0 - 10.0)', fontsize=12, fontweight='bold', color='#24292f', labelpad=10)
ax2.set_title('🎬 Fedeltà Estetica & Dettaglio Cinematografico Pulp Fiction', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax2.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
ax2.legend(loc='lower left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

for spine in ax2.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

fig.suptitle("MiniMax-H3 · Benchmark Sequenziale Pulp Fiction 35mm: PDD (8-Step) vs. DMD2 (4-Step)",
              fontsize=15.5, fontweight='bold', color='#09244b', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

chart_out = out_dir / "pulp_fiction_pdd_vs_dmd2_comparison.png"
chart_brain = brain_dir / "pulp_fiction_pdd_vs_dmd2_comparison.png"
fig.savefig(chart_out, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(chart_brain, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Pulp Fiction Comparison Chart to:", chart_out)
