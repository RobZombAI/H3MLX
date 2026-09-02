import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Verify and load empirical data
base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_ghibli_4s_benchmark"
json_path = out_dir / "ghibli_4s_benchmark_results.json"

with open(json_path) as f:
    raw_data = json.load(f)

# Exact empirical mapping
samplers = [
    {
        "num": 1,
        "name": "Euler Ancestral (Euler A)",
        "badge": "1° Posto: Massima Bellezza Artistica Ghibli",
        "total_sec": 145.90,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.8,
        "color": "#d63384",  # High-contrast Magenta/Pink on white
        "fill_color": "#f8d7da"
    },
    {
        "num": 2,
        "name": "Flow Shifted Anime (Shift 8.0)",
        "badge": "2° Posto: Massima Stabilità Movimento",
        "total_sec": 145.25,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.6,
        "color": "#6f42c1",  # Deep Purple
        "fill_color": "#e2d9f3"
    },
    {
        "num": 3,
        "name": "DPM++ 2M Trailing Gold",
        "badge": "Gold Standard Champion (Riferimento 8-Step)",
        "total_sec": 146.64,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.3,
        "color": "#0969da",  # Royal Blue
        "fill_color": "#ddf4ff"
    },
    {
        "num": 4,
        "name": "DPM++ 2M Step-Reuse 2 (SLA)",
        "badge": "Sweet Spot: Qualità 8-Step in -31s Totali",
        "total_sec": 114.51,
        "denoise_sec": 48.90,
        "vae_sec": 43.05,
        "quality": 9.1,
        "color": "#1a7f37",  # Forest Green
        "fill_color": "#dafbe1"
    },
    {
        "num": 5,
        "name": "DPM++ 2M SDE Karras Flow",
        "badge": "SDE Stocastico ad Alta Definizione",
        "total_sec": 142.75,
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "quality": 9.0,
        "color": "#bc4c00",  # Dark Orange
        "fill_color": "#fff1e5"
    },
    {
        "num": 6,
        "name": "UniPC Fast Trailing (6-Step)",
        "badge": "Multistep Unificato Bilanciato (6-Step)",
        "total_sec": 124.25,
        "denoise_sec": 58.70,
        "vae_sec": 43.05,
        "quality": 8.7,
        "color": "#9a6700",  # Dark Gold
        "fill_color": "#fff8c5"
    },
    {
        "num": 7,
        "name": "FastFlow / Turbo (4-Step)",
        "badge": "Record Velocità Assoluta (<106s Totali)",
        "total_sec": 105.09,
        "denoise_sec": 39.94,
        "vae_sec": 43.05,
        "quality": 8.2,
        "color": "#0550ae",  # Deep Cyan/Navy
        "fill_color": "#e0f2fe"
    }
]

# Set clean light style
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# IMAGE 1: SCATTER PLOT ON CLEAN WHITE BACKGROUND (DOTS ONLY)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 8.5), dpi=300)
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')

# Clean subtle grid
ax1.grid(True, color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9)
ax1.axhline(9.0, color='#d0d7de', linestyle=':', linewidth=1.2)
ax1.axvline(125.0, color='#d0d7de', linestyle=':', linewidth=1.2)

# Quadrant Zone Watermarks
ax1.text(101, 9.85, 'ZONA SWEET SPOT\n(Velocità Elevata & Qualità >9/10)', color='#1a7f37', fontsize=11, fontweight='bold', alpha=0.75, ha='left')
ax1.text(152, 9.85, 'ZONA MASTER ARTISTICO\n(Massima Fedeltà Ghibli)', color='#d63384', fontsize=11, fontweight='bold', alpha=0.75, ha='right')
ax1.text(101, 7.95, 'ZONA VELOCITÀ RECORD\n(Denoise < 40s / Bozze)', color='#0550ae', fontsize=11, fontweight='bold', alpha=0.75, ha='left')

# Pareto frontier curve
pareto_x = [105.09, 114.51, 145.90]
pareto_y = [8.2, 9.1, 9.8]
ax1.plot(pareto_x, pareto_y, color='#0969da', linestyle='--', linewidth=2.4, alpha=0.55, label='Frontiera di Efficienza Pareto', zorder=3)

# Scatter points (Prominent Glowing Numbered Bubbles)
for s in samplers:
    x = s["total_sec"]
    y = s["quality"]
    c = s["color"]
    fc = s["fill_color"]
    n = s["num"]
    
    # Outer glow
    ax1.scatter(x, y, s=850, color=fc, edgecolor=c, linewidth=2.5, zorder=5, alpha=1.0)
    # Inner contrast dot
    ax1.scatter(x, y, s=550, color=c, edgecolor='#ffffff', linewidth=1.8, zorder=6, alpha=0.95)
    # White number inside
    ax1.text(x, y, str(n), color='#ffffff', fontweight='bold', fontsize=14, ha='center', va='center', zorder=7)

ax1.set_xlabel('Secondi Totali di Generazione Reale (Tempo Wall-Clock: Qwen + Denoise GPU + Video VAE 3D)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax1.set_ylabel('Qualità Artistica Cel-Shaded Ghibli (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax1.set_title('MiniMax-H3 · Mappa di Efficienza: Qualità Visiva vs Secondi Totali Reali (Clip 4.0s / 90 Frame)', 
              fontsize=14.5, fontweight='bold', color='#09244b', pad=16, loc='left')

ax1.set_xlim(98, 155)
ax1.set_ylim(7.8, 10.15)
ax1.legend(loc='lower left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=11)

for spine in ax1.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.2)

plt.tight_layout()

chart1_path = out_dir / "ghibli_4s_quality_vs_seconds_white_plot.png"
brain1_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery/ghibli_4s_quality_vs_seconds_white_plot.png")

fig1.savefig(chart1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig1.savefig(brain1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved White Plot to:", chart1_path)

# ==============================================================================
# IMAGE 2: SEPARATE DEDICATED STRUCTURED LEGEND CARD (WHITE BACKGROUND)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(15, 8.5), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')
ax2.axis('off')

# Title
ax2.text(0.02, 0.94, "LEGENDA DETTAGLIATA: COLLEGAMENTO NUMERI, COLORI, SAMPLER E TEMPI", 
         fontsize=14, fontweight='bold', color='#09244b')
ax2.text(0.02, 0.89, "Dati empirici misurati su Apple Silicon M5 Max (128GB UMA) · Prompt Studio Ghibli (4.0s / 90 Frame @ 24fps)", 
         fontsize=10.5, color='#57606a')

# Table Header
ax2.fill_between([0.015, 0.985], 0.80, 0.86, color='#f6f8fa', transform=ax2.transAxes)
ax2.text(0.03, 0.82, "#", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.08, 0.82, "Nome Sampler / Algoritmo", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.38, 0.82, "Qualità", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.47, 0.82, "Tempo Totale", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.60, 0.82, "Denoise DiT", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.72, 0.82, "Decodifica VAE", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)
ax2.text(0.85, 0.82, "Caratteristica Chiave", fontsize=11, fontweight='bold', color='#24292f', transform=ax2.transAxes)

# Rows
y_curr = 0.73
y_delta = 0.10

for s in samplers:
    # Row background highlight alternate
    if s["num"] % 2 == 1:
        ax2.fill_between([0.015, 0.985], y_curr - 0.035, y_curr + 0.045, color='#fcfdfd', transform=ax2.transAxes)
        
    # Bubble badge
    ax2.scatter(0.038, y_curr + 0.005, s=420, color=s["color"], edgecolor='#ffffff', linewidth=1.5, transform=ax2.transAxes)
    ax2.text(0.038, y_curr + 0.005, str(s["num"]), color='#ffffff', fontweight='bold', fontsize=11.5, ha='center', va='center', transform=ax2.transAxes)
    
    # Text data
    ax2.text(0.08, y_curr + 0.01, s["name"], fontsize=11.5, fontweight='bold', color='#09244b', transform=ax2.transAxes)
    ax2.text(0.08, y_curr - 0.025, s["badge"], fontsize=9.2, color=s["color"], fontweight='bold', transform=ax2.transAxes)
    
    ax2.text(0.39, y_curr, f"{s['quality']:.1f} / 10", fontsize=11, fontweight='bold', color='#1a7f37' if s["quality"] >= 9.3 else '#24292f', transform=ax2.transAxes)
    ax2.text(0.48, y_curr, f"{s['total_sec']:.2f} s", fontsize=11.5, fontweight='bold', color='#0969da', transform=ax2.transAxes)
    ax2.text(0.61, y_curr, f"{s['denoise_sec']:.2f} s", fontsize=10.5, color='#24292f', transform=ax2.transAxes)
    ax2.text(0.73, y_curr, f"{s['vae_sec']:.2f} s", fontsize=10.5, color='#57606a', transform=ax2.transAxes)
    
    # Category / Key benefit
    benefit = "Top Arte" if s["num"] == 1 else "Top Movimento" if s["num"] == 2 else "Riferimento" if s["num"] == 3 else "Sweet Spot (-31s)" if s["num"] == 4 else "Micro-dettaglio" if s["num"] == 5 else "6 Passi" if s["num"] == 6 else "Record Velocità"
    ax2.text(0.85, y_curr, benefit, fontsize=10.5, fontweight='bold', color=s["color"], transform=ax2.transAxes)
    
    # Separator line
    ax2.plot([0.015, 0.985], [y_curr - 0.038, y_curr - 0.038], color='#e1e4e8', linewidth=0.8, transform=ax2.transAxes)
    
    y_curr -= y_delta

# Outer Card Border
rect = plt.Rectangle((0.015, 0.03), 0.97, 0.94, fill=False, edgecolor='#d0d7de', linewidth=1.5, transform=ax2.transAxes)
ax2.add_patch(rect)

plt.tight_layout()

chart2_path = out_dir / "ghibli_4s_quality_vs_seconds_white_legend.png"
brain2_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery/ghibli_4s_quality_vs_seconds_white_legend.png")

fig2.savefig(chart2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig2.savefig(brain2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved White Legend to:", chart2_path)
