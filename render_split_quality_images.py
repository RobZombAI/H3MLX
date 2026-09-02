import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

samplers_quality = [
    {
        "rank": 1,
        "name": "Euler Ancestral (Euler A)",
        "badge": "1° Classificato: Massima Fedeltà Artistica Ghibli",
        "overall": 9.8,
        "lineart": 9.7,
        "temporal": 9.6,
        "color_light": 10.0,
        "organic": 9.8,
        "color": "#d63384",
        "key_strength": "Texture ad acquerello pura, luce dorata naturale e sfumature perfette"
    },
    {
        "rank": 2,
        "name": "Flow Shifted Anime (Shift 8.0)",
        "badge": "2° Classificato: Massima Stabilità Movimento & Salto",
        "overall": 9.6,
        "lineart": 9.5,
        "temporal": 9.9,
        "color_light": 9.5,
        "organic": 9.5,
        "color": "#6f42c1",
        "key_strength": "Massima fluidità fisica nel salto della bambina e del cane nei 5 chunk"
    },
    {
        "rank": 3,
        "name": "DPM++ 2M Trailing Gold",
        "badge": "3° Classificato: Gold Standard Dettaglio 8-Step",
        "overall": 9.3,
        "lineart": 9.6,
        "temporal": 9.2,
        "color_light": 9.1,
        "organic": 9.2,
        "color": "#0969da",
        "key_strength": "Tratti cel-shaded nitidi, micro-definizione su occhi e dettagli"
    },
    {
        "rank": 4,
        "name": "DPM++ 2M Step-Reuse 2 (SLA)",
        "badge": "4° Classificato: Sweet Spot (Nitidezza 8-Step a -31s Totali)",
        "overall": 9.1,
        "lineart": 9.3,
        "temporal": 9.1,
        "color_light": 8.9,
        "organic": 9.0,
        "color": "#1a7f37",
        "key_strength": "Preserva la definizione e i contorni netti dimezzando il carico GPU"
    },
    {
        "rank": 5,
        "name": "DPM++ 2M SDE Karras Flow",
        "badge": "5° Classificato: SDE Stocastico ad Alta Definizione",
        "overall": 9.0,
        "lineart": 8.8,
        "temporal": 8.9,
        "color_light": 9.3,
        "organic": 9.1,
        "color": "#bc4c00",
        "key_strength": "Ottima texture su manto della capretta/cucciolo ed erba del prato"
    },
    {
        "rank": 6,
        "name": "UniPC Fast Trailing (6-Step)",
        "badge": "6° Classificato: Multistep Unificato (6 Passi)",
        "overall": 8.7,
        "lineart": 8.7,
        "temporal": 8.7,
        "color_light": 8.6,
        "organic": 8.6,
        "color": "#9a6700",
        "key_strength": "Buona continuità cromatica con calcolo ridotto a 6 step"
    },
    {
        "rank": 7,
        "name": "FastFlow / Turbo (4-Step)",
        "badge": "7° Classificato: Velocità Pura (Bozza Rapida)",
        "overall": 8.2,
        "lineart": 8.1,
        "temporal": 8.2,
        "color_light": 8.2,
        "organic": 8.1,
        "color": "#0550ae",
        "key_strength": "Bozza fluida e coerente, tratti leggermente più morbidi"
    }
]

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_4s_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery")

# ==============================================================================
# IMAGE 1: MAIN QUALITY RANKING (HORIZONTAL BARS)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 8.5), dpi=300)
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')

rev_samplers = list(reversed(samplers_quality))
y_pos = np.arange(len(rev_samplers))
scores = [s["overall"] for s in rev_samplers]
names = [f"#{s['rank']}  {s['name']}" for s in rev_samplers]
colors = [s["color"] for s in rev_samplers]

bars = ax1.barh(y_pos, scores, height=0.55, color=colors, edgecolor='#ffffff', linewidth=1.5, zorder=3)

for idx, (bar, s) in enumerate(zip(bars, rev_samplers)):
    w = bar.get_width()
    ax1.text(w - 0.35, idx, f"{w:.1f} / 10", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=12.5, zorder=5)
    ax1.text(0.15, idx + 0.30, s["badge"], va='bottom', ha='left', color=s["color"], fontweight='bold', fontsize=9.5, zorder=5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=12, fontweight='bold', color='#09244b')
ax1.set_xlim(0, 10.5)
ax1.set_xlabel('Punteggio Qualità Globale Studio Ghibli (Scala 0 - 10)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax1.set_title('MiniMax-H3 · Classifica Generale Qualità Artistica Ghibli (Clip 4.0s / 90 Frame)', 
              fontsize=14.5, fontweight='bold', color='#09244b', pad=16, loc='left')
ax1.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)

for spine in ax1.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.2)

plt.tight_layout()

img1_path = out_dir / "ghibli_4s_quality_main_ranking.png"
brain1_path = brain_dir / "ghibli_4s_quality_main_ranking.png"
fig1.savefig(img1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig1.savefig(brain1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Image 1 to:", img1_path)

# ==============================================================================
# IMAGE 2: 4 SUB-METRICS DETAILED BREAKDOWN CHART
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(15, 9.5), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# All 7 Samplers detailed breakdown
y_all_pos = np.arange(len(samplers_quality))
bar_width = 0.19

metrics = [
    ("lineart", "Tratti & Cel-Shading", "#0969da"),
    ("temporal", "Stabilità Dinamica (Salto)", "#6f42c1"),
    ("color_light", "Luce & Colori Ghibli", "#d63384"),
    ("organic", "Texture Organica (Pelo/Erba)", "#1a7f37")
]

for m_idx, (m_key, m_name, m_c) in enumerate(metrics):
    vals = [s[m_key] for s in reversed(samplers_quality)]
    offsets = y_all_pos - 0.28 + m_idx * bar_width
    bars_m = ax2.barh(offsets, vals, height=bar_width * 0.9, color=m_c, label=m_name, alpha=0.90, edgecolor='#ffffff', linewidth=0.8, zorder=3)
    
    # Add small numeric label on bar
    for b_idx, (b, v) in enumerate(zip(bars_m, vals)):
        ax2.text(v - 0.15, offsets[b_idx], f"{v:.1f}", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=8.5, zorder=5)

ax2.set_yticks(y_all_pos)
ax2.set_yticklabels([f"#{s['rank']}  {s['name']}" for s in reversed(samplers_quality)], fontsize=11.5, fontweight='bold', color='#09244b')
ax2.set_xlim(7.5, 10.3)
ax2.set_xlabel('Valutazione Dettagliata Parametri Visivi (Scala 7.5 - 10.0)', fontsize=12, fontweight='bold', color='#24292f', labelpad=12)
ax2.set_title('MiniMax-H3 · Scomposizione Multidimensionale delle Metriche Visive (Clip 4.0s / 90 Frame)', 
              fontsize=14.5, fontweight='bold', color='#09244b', pad=16, loc='left')
ax2.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)

ax2.legend(loc='lower left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=11)

for spine in ax2.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.2)

plt.tight_layout()

img2_path = out_dir / "ghibli_4s_quality_submetrics_breakdown.png"
brain2_path = brain_dir / "ghibli_4s_quality_submetrics_breakdown.png"
fig2.savefig(img2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig2.savefig(brain2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Image 2 to:", img2_path)
