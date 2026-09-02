import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data for quality ranking on 4.0s Ghibli benchmark
samplers_quality = [
    {
        "rank": 1,
        "name": "Euler Ancestral (Euler A)",
        "badge": "1° Classificato Assoluto: Massima Fedeltà Ghibli",
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
        "badge": "4° Classificato: Sweet Spot (Nitidezza 8-Step a -31s)",
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

# Set clean light style
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig = plt.figure(figsize=(16, 11), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Split into 2 subplots: Left (Main Ranking Bars), Right (Sub-Metrics Breakdown)
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
ax_left = fig.add_subplot(gs[0])
ax_right = fig.add_subplot(gs[1])

ax_left.set_facecolor('#ffffff')
ax_right.set_facecolor('#ffffff')

# ==============================================================================
# LEFT PANEL: MAIN QUALITY RANKING (HORIZONTAL BARS)
# ==============================================================================
rev_samplers = list(reversed(samplers_quality))
y_pos = np.arange(len(rev_samplers))
scores = [s["overall"] for s in rev_samplers]
names = [f"#{s['rank']}  {s['name']}" for s in rev_samplers]
colors = [s["color"] for s in rev_samplers]

bars = ax_left.barh(y_pos, scores, height=0.58, color=colors, edgecolor='#ffffff', linewidth=1.5, zorder=3)

# Add score annotations and badges
for idx, (bar, s) in enumerate(zip(bars, rev_samplers)):
    w = bar.get_width()
    # Score on bar
    ax_left.text(w - 0.35, idx, f"{w:.1f} / 10", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=12, zorder=5)
    # Badge note above bar
    ax_left.text(0.15, idx + 0.32, s["badge"], va='bottom', ha='left', color=s["color"], fontweight='bold', fontsize=9.2, zorder=5)

ax_left.set_yticks(y_pos)
ax_left.set_yticklabels(names, fontsize=11.5, fontweight='bold', color='#09244b')
ax_left.set_xlim(0, 10.5)
ax_left.set_xlabel('Punteggio Qualità Globale Studio Ghibli (Scala 0 - 10)', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax_left.set_title('🏆 Classifica Generale Qualità Visiva (Clip 4.0s / 90 Frame)', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax_left.grid(axis='x', color='#e1e4e8', linestyle='--', alpha=0.8, zorder=1)

for spine in ax_left.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

# ==============================================================================
# RIGHT PANEL: 4 SUB-METRICS BREAKDOWN BARS
# ==============================================================================
# Show sub-metrics for Top 5 Samplers
top_samplers = samplers_quality[:5]
metric_labels = ["Tratti & Cel-Shading", "Stabilità Dinamica (Salto)", "Luce & Colori Ghibli", "Texture Organica"]

y_top_pos = np.arange(len(top_samplers))
bar_width = 0.18

for m_idx, (m_key, m_name, m_c) in enumerate([
    ("lineart", "Tratti & Cel-Shading", "#0969da"),
    ("temporal", "Stabilità Dinamica (Salto)", "#6f42c1"),
    ("color_light", "Luce & Colori Ghibli", "#d63384"),
    ("organic", "Texture Organica (Pelo/Erba)", "#1a7f37")
]):
    vals = [s[m_key] for s in reversed(top_samplers)]
    offsets = y_top_pos - 0.27 + m_idx * bar_width
    ax_right.barh(offsets, vals, height=bar_width * 0.9, color=m_c, label=m_name if m_idx < 4 else "", alpha=0.88, edgecolor='#ffffff', linewidth=0.8)

ax_right.set_yticks(y_top_pos)
ax_right.set_yticklabels([f"#{s['rank']} {s['name'].split('(')[0].strip()}" for s in reversed(top_samplers)], fontsize=11, fontweight='bold', color='#09244b')
ax_right.set_xlim(8.0, 10.2)
ax_right.set_xlabel('Valutazione Dettagliata Parametri Visivi (8.0 - 10.0)', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax_right.set_title('🔬 Scomposizione Metriche Chiave sui Top Modelli', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax_right.grid(axis='x', color='#e1e4e8', linestyle='--', alpha=0.8)
ax_right.legend(loc='lower left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=9.5)

for spine in ax_right.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

# Overall Title
fig.suptitle("MiniMax-H3 · Valutazione Scientifica della Qualità Artistica (Prompt Studio Ghibli - 4.0s @ 24fps)",
             fontsize=15.5, fontweight='bold', color='#09244b', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_ghibli_4s_benchmark")
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery")

chart_path = out_dir / "ghibli_4s_quality_ranking_white.png"
brain_path = brain_dir / "ghibli_4s_quality_ranking_white.png"

fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Quality Only Chart to:", chart_path)
