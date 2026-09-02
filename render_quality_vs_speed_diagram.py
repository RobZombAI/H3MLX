import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data for 4.0s Benchmark
samplers = [
    {
        "name": "Euler Ancestral (Euler A)",
        "tier": "Tier S · Top Arte Ghibli",
        "fps": 1.15,
        "denoise": 78.35,
        "quality": 9.8,
        "color": "#f778ba",
        "desc": "Texture acquerello, luce dorata e volti perfetti"
    },
    {
        "name": "Flow Shift Anime (Shift 8.0)",
        "tier": "Tier S · Top Movimento",
        "fps": 1.15,
        "denoise": 78.35,
        "quality": 9.6,
        "color": "#a371f7",
        "desc": "Massima stabilità nel salto della bambina e del cane"
    },
    {
        "name": "DPM++ 2M Trailing Gold",
        "tier": "Tier S · Riferimento",
        "fps": 1.15,
        "denoise": 78.35,
        "quality": 9.3,
        "color": "#58a6ff",
        "desc": "Standard 8-step nitido e definito"
    },
    {
        "name": "DPM++ 2M Step-Reuse 2",
        "tier": "Tier A+ · Sweet Spot",
        "fps": 1.84,
        "denoise": 48.90,
        "quality": 9.1,
        "color": "#39d353",
        "desc": "Velocità quasi raddoppiata con qualità 8-step"
    },
    {
        "name": "DPM++ 2M SDE Karras",
        "tier": "Tier A · SDE Stocastico",
        "fps": 1.15,
        "denoise": 78.35,
        "quality": 9.0,
        "color": "#ffa657",
        "desc": "Ricchezza organica su pelliccia e fondali"
    },
    {
        "name": "UniPC Fast Trailing (6-Step)",
        "tier": "Tier A · Bilanciato",
        "fps": 1.53,
        "denoise": 58.70,
        "quality": 8.7,
        "color": "#e3b341",
        "desc": "Ottima convergenza in soli 6 passi"
    },
    {
        "name": "FastFlow / Turbo (4-Step)",
        "tier": "Tier A · Velocità Pura",
        "fps": 2.25,
        "denoise": 39.94,
        "quality": 8.2,
        "color": "#00d2ff",
        "desc": "Denoise in meno di 40 secondi (record velocità)"
    }
]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 9.5), dpi=220)
fig.patch.set_facecolor('#0b0e14')
ax.set_facecolor('#161b22')

# Plot grid and zones
ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
ax.axhline(9.0, color='#30363d', linestyle=':', alpha=0.8)
ax.axvline(1.5, color='#30363d', linestyle=':', alpha=0.8)

# Background Quadrant Annotations
ax.text(2.15, 9.6, '🏆 ZONA SWEET SPOT\n(Alta Velocità & Alta Qualità)', color='#39d353', fontsize=11, fontweight='bold', alpha=0.7, ha='right')
ax.text(1.20, 9.6, '🎨 ZONA MASTER ARTISTICO\n(Massima Fedeltà Cel-Shaded)', color='#f778ba', fontsize=11, fontweight='bold', alpha=0.7, ha='left')
ax.text(2.15, 8.3, '⚡ ZONA VELOCITÀ PURA\n(Bozze Rapide & Storyboard)', color='#00d2ff', fontsize=11, fontweight='bold', alpha=0.7, ha='right')

# Scatter points and labels
for s in samplers:
    x = s["fps"]
    y = s["quality"]
    c = s["color"]
    
    # Bubble
    ax.scatter(x, y, s=280, color=c, edgecolor='#ffffff', linewidth=1.5, zorder=5, alpha=0.9)
    
    # Text Annotation
    offset_x = 0.02
    offset_y = 0.08 if y < 9.7 else -0.12
    if s["name"] == "Flow Shift Anime (Shift 8.0)":
        offset_y = -0.10
    elif s["name"] == "DPM++ 2M Trailing Gold":
        offset_y = -0.12
    elif s["name"] == "DPM++ 2M SDE Karras":
        offset_y = -0.14
        
    ax.annotate(
        f"{s['name']}\nQualità: {s['quality']}/10 · Denoise: {s['denoise']:.1f}s ({s['fps']:.2f} FPS)",
        xy=(x, y), xytext=(x + offset_x, y + offset_y),
        fontsize=9.5, fontweight='bold', color='#f0f6fc',
        bbox=dict(boxstyle="round,pad=0.35", facecolor='#21262d', edgecolor=c, alpha=0.9, linewidth=1.2),
        zorder=6
    )

# Connect Pareto Frontier (FastFlow -> Step-Reuse 2 -> Flow Shift Anime -> Euler Ancestral)
pareto_x = [2.25, 1.84, 1.15]
pareto_y = [8.2, 9.1, 9.8]
ax.plot(pareto_x, pareto_y, color='#00d2ff', linestyle='--', linewidth=2, alpha=0.5, label='Frontiera di Efficienza Pareto (Speed vs Quality)')

ax.set_xlabel('Velocità di Calcolo GPU: Throughput (FPS = 90 Frame / Denoise Secondi)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=12)
ax.set_ylabel('Punteggio Qualità Visiva Artistica Ghibli (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=12)
ax.set_title('MiniMax-H3 · Classifica Efficienza: Qualità Artistica vs Velocità GPU (Clip 4.0s / 90 Frame)', 
             fontsize=15, fontweight='bold', color='#ffffff', pad=18, loc='left')

ax.set_xlim(1.0, 2.35)
ax.set_ylim(7.8, 10.1)

legend = ax.legend(loc='lower left', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=10.5)

for spine in ax.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_4s_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery")

fig.savefig(out_dir / "ghibli_4s_quality_vs_speed_chart.png", dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
fig.savefig(brain_dir / "ghibli_4s_quality_vs_speed_chart.png", dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
print("✓ Quality vs Speed Diagram successfully saved!")
