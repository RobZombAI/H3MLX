import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
assets_dir = base_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/master_combinatorial_4s_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)

data = [
    {"num": 1, "name": "Euler Ancestral (Euler A) Trailing", "short": "Euler A (Shift 10.0)", "sec": 145.90, "denoise": 78.35, "qual": 9.8, "col": "#d63384", "badge": "🥇 1st Quality (9.8/10)"},
    {"num": 2, "name": "Flow Shifted Anime Motion", "short": "Flow Anime (Shift 8.0)", "sec": 145.25, "denoise": 78.35, "qual": 9.6, "col": "#6f42c1", "badge": "🥈 2nd Motion (9.6/10)"},
    {"num": 3, "name": "DPM++ 2M Trailing Gold", "short": "DPM++ 2M Gold", "sec": 146.64, "denoise": 78.35, "qual": 9.3, "col": "#0969da", "badge": "🥉 3rd Definition (9.3/10)"},
    {"num": 4, "name": "DPM++ 2M Step-Reuse 2 (SLA)", "short": "DPM++ 2M Reuse 2", "sec": 114.51, "denoise": 48.90, "qual": 9.1, "col": "#1a7f37", "badge": "⚡ Sweet Spot (-31.4s)"},
    {"num": 5, "name": "DPM++ 2M SDE Karras Flow", "short": "DPM++ 2M SDE Karras", "sec": 142.75, "denoise": 78.35, "qual": 9.0, "col": "#bc4c00", "badge": "🔬 Rich Textures (9.0/10)"},
    {"num": 6, "name": "UniPC Fast Trailing (6-Step)", "short": "UniPC 6-Step", "sec": 124.25, "denoise": 58.70, "qual": 8.7, "col": "#9a6700", "badge": "⏱️ 6-Step Fast (8.7/10)"},
    {"num": 7, "name": "FastFlow / Turbo (4-Step)", "short": "FastFlow / Turbo 4-Step", "sec": 105.09, "denoise": 39.94, "qual": 8.2, "col": "#0550ae", "badge": "🚀 Speed Record (4.12 FPS)"}
]

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Pareto frontier line
pareto_points = sorted([d for d in data if d["num"] in [7, 4, 2, 1]], key=lambda x: x["sec"])
px = [p["sec"] for p in pareto_points]
py = [p["qual"] for p in pareto_points]

ax.plot(px, py, linestyle='--', color='#94a3b8', linewidth=2.2, alpha=0.85, zorder=2, label='Frontiera di Efficienza di Pareto (Compromesso Ottimale Qualità / Latenza)')

for d in data:
    # Outer glow ring
    ax.scatter(d["sec"], d["qual"], s=1200, color=d["col"], alpha=0.22, edgecolors='none', zorder=3)
    # Main bubble
    ax.scatter(d["sec"], d["qual"], s=650, color=d["col"], alpha=0.95, edgecolors='#ffffff', linewidth=2.5, zorder=4)
    # Number inside bubble
    ax.text(d["sec"], d["qual"], str(d["num"]), ha='center', va='center', color='#ffffff', fontweight='bold', fontsize=13.5, zorder=5)
    
    # Label card with callout
    offset_y = 0.14 if d["num"] != 4 else -0.22
    offset_x = 0 if d["num"] not in [2, 3] else (3.8 if d["num"] == 3 else -3.8)
    ax.annotate(f"{d['num']}. {d['short']}\n{d['badge']} · Totale: {d['sec']:.1f}s",
                xy=(d["sec"], d["qual"]),
                xytext=(d["sec"] + offset_x, d["qual"] + offset_y),
                ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='#09244b',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor=d["col"], linewidth=1.4, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=d["col"], lw=1.2, shrinkA=8, shrinkB=8),
                zorder=6)

ax.set_xlim(95, 160)
ax.set_ylim(7.8, 10.2)
ax.set_xlabel('Latenza Totale End-to-End in Secondi (Denoise GPU + Decodifica Video VAE 3D + Mastering Audio)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax.set_ylabel('Punteggio Qualità Artistica & Movimento (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax.set_title('MiniMax-H3 · Frontiera di Pareto: Qualità Visiva vs. Latenza Totale di Generazione (Clip 4.0s / 90 Frame @ 24fps)', 
             fontsize=14.5, fontweight='bold', color='#09244b', pad=18, loc='left')

ax.grid(color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)
ax.legend(loc='lower right', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

for spine in ax.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.2)

plt.tight_layout()

chart4_assets = assets_dir / "h3mlx_pareto_frontier_benchmark.png"
chart4_brain = brain_dir / "h3mlx_pareto_frontier_benchmark.png"
fig.savefig(chart4_assets, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(chart4_brain, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Chart 4 (Pareto Frontier) to:", chart4_assets)
