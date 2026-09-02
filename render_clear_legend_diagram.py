import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

samplers = [
    {
        "num": 1,
        "name": "Euler Ancestral (Euler A)",
        "badge": "🥇 1° Posto: Massima Bellezza Ghibli",
        "total_sec": 145.90,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.8,
        "color": "#f778ba",  # Pink
        "marker": "o"
    },
    {
        "num": 2,
        "name": "Flow Shifted Anime (Shift 8.0)",
        "badge": "🥈 2° Posto: Massima Stabilità Movimento",
        "total_sec": 145.25,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.6,
        "color": "#a371f7",  # Purple
        "marker": "o"
    },
    {
        "num": 3,
        "name": "DPM++ 2M Trailing Gold",
        "badge": "🏆 Gold Standard Champion (Riferimento)",
        "total_sec": 146.64,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.3,
        "color": "#58a6ff",  # Blue
        "marker": "o"
    },
    {
        "num": 4,
        "name": "DPM++ 2M Step-Reuse 2 (SLA)",
        "badge": "⚡ 3° Posto: Sweet Spot Efficienza (-31s)",
        "total_sec": 114.51,
        "denoise_sec": 48.90,
        "vae_sec": 43.05,
        "quality": 9.1,
        "color": "#39d353",  # Green
        "marker": "o"
    },
    {
        "num": 5,
        "name": "DPM++ 2M SDE Karras Flow",
        "badge": "🔬 SDE Stocastico ad Alta Definizione",
        "total_sec": 142.75,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.0,
        "color": "#ffa657",  # Orange
        "marker": "o"
    },
    {
        "num": 6,
        "name": "UniPC Fast Trailing (6-Step)",
        "badge": "⏱️ Multistep Unificato Bilanciato",
        "total_sec": 124.25,
        "denoise_sec": 58.70,
        "vae_sec": 43.05,
        "quality": 8.7,
        "color": "#e3b341",  # Yellow
        "marker": "o"
    },
    {
        "num": 7,
        "name": "FastFlow / Turbo (4-Step)",
        "badge": "🚀 Record Velocità Assoluta (<106s)",
        "total_sec": 105.09,
        "denoise_sec": 39.94,
        "vae_sec": 43.05,
        "quality": 8.2,
        "color": "#00d2ff",  # Cyan
        "marker": "o"
    }
]

plt.style.use('dark_background')

# Layout: Upper scatter plot, Lower structured legend box
fig = plt.figure(figsize=(16, 12), dpi=220)
fig.patch.set_facecolor('#0b0e14')

# Create two subplots with custom height ratios (top: plot, bottom: legend table)
gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.28)
ax = fig.add_subplot(gs[0])
ax_legend = fig.add_subplot(gs[1])

ax.set_facecolor('#161b22')
ax_legend.set_facecolor('#161b22')

# -------------------------------------------------------------
# TOP SCATTER PLOT
# -------------------------------------------------------------
ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
ax.axhline(9.0, color='#30363d', linestyle=':', alpha=0.8)
ax.axvline(125.0, color='#30363d', linestyle=':', alpha=0.8)

# Quadrant Annotations
ax.text(102, 9.75, '🏆 ZONA SWEET SPOT\n(Velocità Elevata & Alta Qualità)', color='#39d353', fontsize=11, fontweight='bold', alpha=0.8, ha='left')
ax.text(152, 9.75, '🎨 ZONA MASTER ARTISTICO\n(Massima Fedeltà Ghibli)', color='#f778ba', fontsize=11, fontweight='bold', alpha=0.8, ha='right')
ax.text(102, 8.0, '⚡ ZONA VELOCITÀ RECORD\n(Denoise < 40s / Bozze)', color='#00d2ff', fontsize=11, fontweight='bold', alpha=0.8, ha='left')

# Pareto frontier curve
pareto_x = [105.09, 114.51, 145.90]
pareto_y = [8.2, 9.1, 9.8]
ax.plot(pareto_x, pareto_y, color='#00d2ff', linestyle='--', linewidth=2.2, alpha=0.6, label='Frontiera di Efficienza Pareto', zorder=3)

# Scatter points with big numbered badges
for s in samplers:
    x = s["total_sec"]
    y = s["quality"]
    c = s["color"]
    n = s["num"]
    
    # Outer Glow Ring
    ax.scatter(x, y, s=550, color=c, edgecolor='#ffffff', linewidth=2.0, zorder=5, alpha=0.95)
    
    # Number inside point
    ax.text(x, y, str(n), color='#000000', fontweight='heavy', fontsize=12, ha='center', va='center', zorder=6)
    
    # Clean Direct Pointer
    if n == 7: # FastFlow
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x + 1.2, y - 0.22),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 4: # Step-Reuse 2
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x + 1.2, y + 0.08),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 6: # UniPC
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x + 1.2, y - 0.22),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 5: # SDE Karras
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x - 24.5, y - 0.20),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 2: # Flow Shift Anime
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x - 25.5, y + 0.08),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 1: # Euler A
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x + 1.2, y + 0.08),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)
    elif n == 3: # DPM++ 2M Gold
        ax.annotate(f"#{n} {s['name']}\n{s['total_sec']:.1f}s | Q: {s['quality']}/10", xy=(x, y), xytext=(x + 1.2, y - 0.22),
                    fontsize=10, fontweight='bold', color=c, bbox=dict(boxstyle="round,pad=0.3", facecolor='#21262d', edgecolor=c, linewidth=1.2), zorder=7)

ax.set_xlabel('Secondi Totali di Generazione Reale (Tempo Wall-Clock: Qwen + Denoise GPU + Video VAE 3D)', fontsize=11.5, fontweight='bold', color='#c9d1d9', labelpad=10)
ax.set_ylabel('Qualità Artistica Cel-Shaded Ghibli (0 - 10)', fontsize=11.5, fontweight='bold', color='#c9d1d9', labelpad=10)
ax.set_title('MiniMax-H3 · Mappa di Efficienza: Qualità Visiva vs Secondi Totali di Calcolo (Clip 4.0s / 90 Frame)', 
             fontsize=14.5, fontweight='bold', color='#ffffff', pad=14, loc='left')

ax.set_xlim(98, 155)
ax.set_ylim(7.7, 10.2)

for spine in ax.spines.values():
    spine.set_color('#30363d')

# -------------------------------------------------------------
# BOTTOM STRUCTURED LEGEND BOX (CLEAR MAPPING TABLE)
# -------------------------------------------------------------
ax_legend.axis('off')

# Title for legend box
ax_legend.text(0.01, 0.95, "📋 LEGENDA DETTAGLIATA: COLLEGAMENTO NUMERI, COLORI, SAMPLER E TEMPI", 
               fontsize=12, fontweight='bold', color='#58a6ff', transform=ax_legend.transAxes)

# Render a beautiful 2-column legend
col1_samplers = [s for s in samplers if s["num"] in [1, 2, 3, 4]]
col2_samplers = [s for s in samplers if s["num"] in [5, 6, 7]]

y_start = 0.76
y_step = 0.22

# Column 1
for i, s in enumerate(col1_samplers):
    y_pos = y_start - i * y_step
    # Swatch badge
    ax_legend.scatter(0.02, y_pos + 0.02, s=260, color=s["color"], edgecolor='#ffffff', linewidth=1.5, transform=ax_legend.transAxes)
    ax_legend.text(0.02, y_pos + 0.02, str(s["num"]), color='#000000', fontweight='heavy', fontsize=10.5, ha='center', va='center', transform=ax_legend.transAxes)
    
    text_line = f"#{s['num']} {s['name']}  —  Totale: {s['total_sec']:.1f}s (Denoise: {s['denoise_sec']:.1f}s | VAE: {s['vae_sec']:.1f}s)  |  Qualità: {s['quality']}/10"
    ax_legend.text(0.045, y_pos + 0.02, text_line, color='#f0f6fc', fontsize=10.2, fontweight='bold', transform=ax_legend.transAxes)
    ax_legend.text(0.045, y_pos - 0.07, f"   ↳ {s['badge']}", color=s["color"], fontsize=9.2, fontweight='semibold', transform=ax_legend.transAxes)

# Column 2
for i, s in enumerate(col2_samplers):
    y_pos = y_start - i * y_step
    ax_legend.scatter(0.53, y_pos + 0.02, s=260, color=s["color"], edgecolor='#ffffff', linewidth=1.5, transform=ax_legend.transAxes)
    ax_legend.text(0.53, y_pos + 0.02, str(s["num"]), color='#000000', fontweight='heavy', fontsize=10.5, ha='center', va='center', transform=ax_legend.transAxes)
    
    text_line = f"#{s['num']} {s['name']}  —  Totale: {s['total_sec']:.1f}s (Denoise: {s['denoise_sec']:.1f}s | VAE: {s['vae_sec']:.1f}s)  |  Qualità: {s['quality']}/10"
    ax_legend.text(0.555, y_pos + 0.02, text_line, color='#f0f6fc', fontsize=10.2, fontweight='bold', transform=ax_legend.transAxes)
    ax_legend.text(0.555, y_pos - 0.07, f"   ↳ {s['badge']}", color=s["color"], fontsize=9.2, fontweight='semibold', transform=ax_legend.transAxes)

# Border for legend box
rect = plt.Rectangle((0.005, 0.02), 0.99, 0.95, fill=False, edgecolor='#30363d', linewidth=1.2, transform=ax_legend.transAxes)
ax_legend.add_patch(rect)

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_4s_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery")

chart_path = out_dir / "ghibli_4s_quality_vs_seconds_clean_legend.png"
brain_path = brain_dir / "ghibli_4s_quality_vs_seconds_clean_legend.png"

fig.savefig(chart_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
fig.savefig(brain_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
print("✓ Clean Legend Diagram successfully saved!")
