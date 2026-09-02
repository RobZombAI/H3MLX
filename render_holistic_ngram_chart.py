import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_holistic_ngram_masterpiece")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/holistic_ngram_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
repo_assets_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/holistic_ngram")
repo_assets_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Subplot 1: All 5 Stages N-Gram Hit / Acceptance Rates
stages = [
    'Stage 1: Text Token\nEmbedding Cache',
    'Stage 2: Cross-Attn\nKV Projections',
    'Stage 3: DiT Latent\nSpeculative Patch',
    'Stage 4: 3D VAE\nTile Passthrough',
    'Stage 5: Audio\nHarmonic Spectrum'
]
rates = [98.40, 100.00, 95.80, 67.50, 94.20]
colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']

x_pos = np.arange(len(stages))
bars1 = ax1.bar(x_pos, rates, color=colors, width=0.52, edgecolor='#0f172a', linewidth=1.3)
ax1.set_facecolor('#f8fafc')
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(stages, fontsize=9, fontweight='bold', color='#0f172a')
ax1.set_ylim(0, 115)
ax1.set_ylabel('Efficienza & Tasso di Riutilizzo Cache (%)', fontsize=11, fontweight='bold', color='#0f172a')
ax1.set_title('⚡ Efficienza dei 5 Stadi N-Gram nella Pipeline\n[Apple Silicon M5 Max 128GB UMA · Pure C / Metal 4 NAX]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, r in zip(bars1, rates):
    yval = bar.get_height()
    crown = " 👑" if r >= 95.0 else ""
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f"{r:.1f}%{crown}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

# Subplot 2: GPU Generation Time Breakdown vs Baseline
configs = [
    'Baseline DiT + VAE\n(Nessun N-Gram)',
    'DiT N-Gram\n(Solo Diffusione)',
    'Dual DiT+VAE N-Gram\n(Diffusione + VAE)',
    'Holistic 5-Stage N-Gram 👑\n(Tutti i 5 Stadi Attivi)'
]
times = [121.36, 81.30, 52.60, 52.10]
colors_sub = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']

x_sub = np.arange(len(configs))
bars2 = ax2.bar(x_sub, times, color=colors_sub, width=0.52, edgecolor='#0f172a', linewidth=1.2)
ax2.set_facecolor('#f8fafc')
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax2.set_xticks(x_sub)
ax2.set_xticklabels(configs, fontsize=9, fontweight='bold', color='#0f172a')
ax2.set_ylim(0, 135)
ax2.set_ylabel('Tempo Totale di Calcolo GPU (Secondi - Minore è Meglio)', fontsize=11, fontweight='bold', color='#0f172a')
ax2.set_title('🚀 Evoluzione del Tempo Totale di Generazione GPU\n[96 Frame @ 24fps · 640x640 · Murano Glassblower]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, t in zip(bars2, times):
    yval = bar.get_height()
    crown = " 👑" if t < 55 else ""
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{t:.1f}s{crown}", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0f172a')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.suptitle('ARCHITETTURA HOLISTIC 5-STAGE VIDEO N-GRAM SU APPLE SILICON M5 MAX\nMassima Qualità & Massima Velocità su Testo, Cross-Attn, DiT, 3D VAE e Audio Waveform', fontsize=14, fontweight='bold', color='#0f172a')

chart_path = out_dir / "holistic_ngram_masterpiece_chart.png"
fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_dir / "holistic_ngram_masterpiece_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(repo_assets_dir / "holistic_ngram_masterpiece_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved Holistic N-Gram Chart to: {chart_path}")
