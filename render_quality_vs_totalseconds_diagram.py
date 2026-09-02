import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

samplers = [
    {
        "name": "FastFlow / Turbo (4-Step)",
        "tier": "Tier A · Velocità Record",
        "total_sec": 105.09,
        "denoise_sec": 39.94,
        "vae_sec": 43.05,
        "quality": 8.2,
        "color": "#00d2ff",
        "desc": "Sub-106s Totale (Denoise 39.9s)"
    },
    {
        "name": "DPM++ 2M Step-Reuse 2",
        "tier": "Tier A+ · Sweet Spot",
        "total_sec": 114.51,
        "denoise_sec": 48.90,
        "vae_sec": 43.05,
        "quality": 9.1,
        "color": "#39d353",
        "desc": "114.5s Totale (Denoise 48.9s)"
    },
    {
        "name": "UniPC Fast Trailing (6-Step)",
        "tier": "Tier A · Bilanciato",
        "total_sec": 124.25,
        "denoise_sec": 58.70,
        "vae_sec": 43.05,
        "quality": 8.7,
        "color": "#e3b341",
        "desc": "124.3s Totale (Denoise 58.7s)"
    },
    {
        "name": "DPM++ 2M SDE Karras",
        "tier": "Tier A · SDE Stocastico",
        "total_sec": 142.75,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.0,
        "color": "#ffa657",
        "desc": "142.8s Totale (Denoise 78.4s)"
    },
    {
        "name": "Flow Shift Anime (Shift 8.0)",
        "tier": "Tier S · Top Movimento",
        "total_sec": 145.25,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.6,
        "color": "#a371f7",
        "desc": "145.3s Totale (Denoise 78.4s)"
    },
    {
        "name": "Euler Ancestral (Euler A)",
        "tier": "Tier S · Top Arte Ghibli",
        "total_sec": 145.90,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.8,
        "color": "#f778ba",
        "desc": "145.9s Totale (Denoise 78.4s)"
    },
    {
        "name": "DPM++ 2M Trailing Gold",
        "tier": "Tier S · Riferimento Standard",
        "total_sec": 146.64,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.3,
        "color": "#58a6ff",
        "desc": "146.6s Totale (Denoise 78.4s)"
    }
]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 9.5), dpi=220)
fig.patch.set_facecolor('#0b0e14')
ax.set_facecolor('#161b22')

ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
ax.axhline(9.0, color='#30363d', linestyle=':', alpha=0.8)
ax.axvline(125.0, color='#30363d', linestyle=':', alpha=0.8)

# Quadrant Annotations
ax.text(100, 9.75, '🏆 ZONA SWEET SPOT\n(Minori Secondi Totali & Altissima Qualità)', color='#39d353', fontsize=11, fontweight='bold', alpha=0.8, ha='left')
ax.text(152, 9.75, '🎨 ZONA MASTER ARTISTICO\n(Massima Fedeltà Visiva)', color='#f778ba', fontsize=11, fontweight='bold', alpha=0.8, ha='right')
ax.text(100, 8.0, '⚡ ZONA VELOCITÀ RECORD\n(Denoise < 40s / Storyboard)', color='#00d2ff', fontsize=11, fontweight='bold', alpha=0.8, ha='left')

for s in samplers:
    x = s["total_sec"]
    y = s["quality"]
    c = s["color"]
    
    ax.scatter(x, y, s=300, color=c, edgecolor='#ffffff', linewidth=1.5, zorder=5, alpha=0.9)
    
    offset_x = 1.2
    offset_y = 0.08
    if s["name"] == "Euler Ancestral (Euler A)":
        offset_x = -22.5
        offset_y = 0.07
    elif s["name"] == "Flow Shift Anime (Shift 8.0)":
        offset_x = -23.0
        offset_y = -0.15
    elif s["name"] == "DPM++ 2M Trailing Gold":
        offset_x = 1.2
        offset_y = -0.12
    elif s["name"] == "DPM++ 2M SDE Karras":
        offset_x = -22.0
        offset_y = -0.13
        
    label_text = f"{s['name']}\nTotale: {s['total_sec']:.1f}s | Denoise: {s['denoise_sec']:.1f}s\nQualità: {s['quality']}/10"
    ax.annotate(
        label_text,
        xy=(x, y), xytext=(x + offset_x, y + offset_y),
        fontsize=9.5, fontweight='bold', color='#f0f6fc',
        bbox=dict(boxstyle="round,pad=0.35", facecolor='#21262d', edgecolor=c, alpha=0.9, linewidth=1.2),
        zorder=6
    )

# Pareto curve in seconds
pareto_x = [105.09, 114.51, 145.90]
pareto_y = [8.2, 9.1, 9.8]
ax.plot(pareto_x, pareto_y, color='#00d2ff', linestyle='--', linewidth=2, alpha=0.5, label='Frontiera di Efficienza Pareto (Secondi Totali vs Qualità)')

ax.set_xlabel('Secondi Totali di Generazione (Tempo Reale Wall-Clock: Qwen + Denoise + VAE)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=12)
ax.set_ylabel('Punteggio Qualità Visiva Artistica Ghibli (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=12)
ax.set_title('MiniMax-H3 · Classifica Efficienza: Qualità Artistica vs Secondi Totali Reali (Clip 4.0s / 90 Frame)', 
             fontsize=15, fontweight='bold', color='#ffffff', pad=18, loc='left')

ax.set_xlim(98, 154)
ax.set_ylim(7.8, 10.1)

legend = ax.legend(loc='lower right', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=10.5)

for spine in ax.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_4s_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery")

fig.savefig(out_dir / "ghibli_4s_quality_vs_totalseconds_chart.png", dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
fig.savefig(brain_dir / "ghibli_4s_quality_vs_totalseconds_chart.png", dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
print("✓ Quality vs Total Seconds Diagram successfully saved!")
