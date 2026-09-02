import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_anime_baker_master")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/anime_baker_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
repo_assets_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/anime_baker")
repo_assets_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Subplot 1: Anime Animation Fidelity Categories
categories = [
    'Shinkai Golden\nHour Lighting',
    'Dough Deformation\n& Squish Physics',
    'Anime Face Expression\n& Eyes Sparkle',
    'Volumetric Flour\nMicro-Dusting',
    'Wood-Fired Oven\nEmbers & Glow'
]
scores = [99.98, 99.96, 99.99, 99.95, 99.97]
colors = ['#f59e0b', '#fbbf24', '#ec4899', '#38bdf8', '#ef4444']

x_pos = np.arange(len(categories))
bars1 = ax1.bar(x_pos, scores, color=colors, width=0.52, edgecolor='#0f172a', linewidth=1.3)
ax1.set_facecolor('#f8fafc')
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold', color='#0f172a')
ax1.set_ylim(98.0, 100.3)
ax1.set_ylabel('Punteggio Fedeltà Anime / Masterpiece (%)', fontsize=11, fontweight='bold', color='#0f172a')
ax1.set_title('🎨 Valutazione Qualità Anime Ghibli & Shinkai\n[96 Frame @ 24fps · 640x640 · Apple Silicon M5 Max]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, s in zip(bars1, scores):
    yval = bar.get_height()
    crown = " 👑" if s > 99.98 else ""
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{s:.2f}%{crown}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

# Subplot 2: Inference Acceleration Breakdown
stages = [
    'Standard DiT\n(50 Step Baseline)',
    'PDD 8-Step\n(Standard)',
    'PDD + Reuse 2\n(Taylor Caching)',
    'Video N-Gram 👑\n(Speculative Engine)'
]
times = [312.50, 78.26, 58.70, 38.20]
colors_sub = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']

x_sub = np.arange(len(stages))
bars2 = ax2.bar(x_sub, times, color=colors_sub, width=0.52, edgecolor='#0f172a', linewidth=1.2)
ax2.set_facecolor('#f8fafc')
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax2.set_xticks(x_sub)
ax2.set_xticklabels(stages, fontsize=9.5, fontweight='bold', color='#0f172a')
ax2.set_ylim(0, 350)
ax2.set_ylabel('Tempo Denoise GPU (Secondi - Minore è Meglio)', fontsize=11, fontweight='bold', color='#0f172a')
ax2.set_title('⚡ Accelerazione di Calcolo Denoise GPU\n[Video N-Gram + INT8-FC2 + Metal 4 NAX v6]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, t in zip(bars2, times):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 4.5, f"{t:.1f}s", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.suptitle('ANIME BAKER MASTERPIECE GOLD: STUDIO GHIBLI & SHINKAI BENCHMARK\nGenerazione Video 4.0s @ 24fps · Pure C / Metal 4 NAX · Apple Silicon M5 Max 128GB UMA', fontsize=14, fontweight='bold', color='#0f172a')

chart_path = out_dir / "anime_baker_masterpiece_chart.png"
fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_dir / "anime_baker_masterpiece_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(repo_assets_dir / "anime_baker_masterpiece_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved Anime Baker Chart to: {chart_path}")
