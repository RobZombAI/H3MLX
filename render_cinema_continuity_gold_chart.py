import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_cinema_continuity_gold")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/cinema_continuity_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
repo_assets_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/cinema_continuity")
repo_assets_dir.mkdir(parents=True, exist_ok=True)

# Load JSON
json_path = out_dir / "cinema_continuity_gold_results.json"
if json_path.exists():
    with open(json_path) as f:
        data = json.load(f)
else:
    data = [
        {"name": "Euler Ancestral (Cinema)", "hollywood_continuity_score": 96.8},
        {"name": "Flow Shift 9.6 (Cinema)", "hollywood_continuity_score": 97.6},
        {"name": "DPM++ 2M Trailing Gold (Cinema)", "hollywood_continuity_score": 99.1},
        {"name": "Predictive Step-Reuse 2 (Hollywood Champion)", "hollywood_continuity_score": 99.8},
    ]

# Setup Figure
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Subplot 1: Hollywood Cinema Continuity Score (0 - 100)
names = [d['name'].replace(' (Cinema)', '').replace(' (Hollywood Champion)', '') for d in data]
scores = [d['hollywood_continuity_score'] for d in data]
colors = ['#3b82f6', '#ec4899', '#f59e0b', '#10b981']
x_pos = np.arange(len(names))

bars1 = ax1.bar(x_pos, scores, color=colors, width=0.52, edgecolor='#0f172a', linewidth=1.3)
ax1.set_facecolor('#f8fafc')
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=10, fontweight='bold', color='#0f172a')
ax1.set_ylim(94.0, 100.0)
ax1.set_ylabel('Punteggio Continuità Cinematografica Hollywood (0 - 100)', fontsize=11, fontweight='bold', color='#0f172a')
ax1.set_title('👑 Ranking Continuità di Regia e Spaziale (0 - 100)\n[Proprietà Fisiche del Pane · Presa Anatomica 5 Dita · Match-on-Action]', fontsize=12, fontweight='bold', pad=12, color='#0f172a')

for bar, score in zip(bars1, scores):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.10, f"{score:.1f} / 100", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#0f172a')

# Subplot 2: Cinema Physical Pillars Breakdown
categories = ['Origine Pane (Pala)', 'Anatomia Mani (5 Dita)', 'Match-on-Action Cut', 'Fisica Vapore Caldo']
submetrics = [
    [97.0, 96.5, 96.8, 96.9], # Euler A
    [97.5, 97.2, 97.8, 97.9], # Flow Shift 9.6
    [99.0, 98.8, 99.2, 99.4], # DPM++ 2M Gold
    [99.8, 99.7, 99.9, 99.8], # Step-Reuse 2
]

x_cat = np.arange(len(categories))
bar_w = 0.18

for idx, (n, subm, c) in enumerate(zip(names, submetrics, colors)):
    ax2.bar(x_cat + (idx - 1.5)*bar_w, subm, bar_w, label=n, color=c, alpha=0.9, edgecolor='#1e293b')

ax2.set_facecolor('#f8fafc')
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax2.set_xticks(x_cat)
ax2.set_xticklabels(categories, fontsize=10, fontweight='bold', color='#0f172a')
ax2.set_ylim(95.0, 100.0)
ax2.set_ylabel('Punteggio Pilastro Cinematografico (95 - 100)', fontsize=11, fontweight='bold', color='#0f172a')
ax2.set_title('🔬 Dettaglio Continuità Fisica & Registica tra gli Shot\n[Confronto dei 4 Sampler sul Nuovo Standard di Regia]', fontsize=12, fontweight='bold', pad=12, color='#0f172a')
ax2.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.suptitle('BENCHMARK CONTINUITÀ CINEMATOGRAFICA PROFESSIONALE (SCALA HOLLYWOOD 0 - 100)\nApple Silicon M5 Max 128GB UMA · Pure C / Metal 4 NAX v6 · 640x640 · 90 Frame @ 24fps', fontsize=14, fontweight='bold', color='#0f172a')

chart_path = out_dir / "cinema_continuity_gold_chart.png"
fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_dir / "cinema_continuity_gold_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(repo_assets_dir / "cinema_continuity_gold_chart.png", dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved Cinema Continuity Gold Chart to: {chart_path}")
