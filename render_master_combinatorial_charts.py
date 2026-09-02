import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all 126 results
base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_master_combinatorial_4s"
json_path = out_dir / "master_combinatorial_benchmark_4s_results.json"

with open(json_path) as f:
    data = json.load(f)

# Extract presets and samplers
preset_order = ["draft", "turbo", "champion", "cinema", "reel", "quality", "oracle"]
preset_names = {
    "draft": "Ultra Draft (45L/Reuse 2)",
    "turbo": "FastVideo Turbo (4-Step)",
    "champion": "Fast Master Champion (8-Step)",
    "cinema": "Cinema 16:9 (960x544)",
    "reel": "Vertical Reel 9:16 (544x960)",
    "quality": "High Quality (20-Step)",
    "oracle": "Full Oracle Ground-Truth (50-Step)"
}

sampler_order = [
    "fastflow_taylor_skip",
    "fastvideo_turbo_ladder",
    "dpm2m_reuse2_sla",
    "unipc_fast_trailing",
    "heun_2nd_order",
    "euler_trailing",
    "euler_a_trailing",
    "euler_a_dual_clock",
    "flow_anime_s8",
    "dpm2m_trailing_s12",
    "dpm2m_karras_s14",
    "dpm2m_sde_trailing",
    "dpm2m_sde_karras",
    "er_sde_flow",
    "lq_flow_schedule",
    "deis_trailing",
    "fastvideo_8step_ladder",
    "cfg_zero_rescaled"
]

sampler_short_names = [
    "FastFlow (Taylor)",
    "FastVideo (4-Step)",
    "DPM++ 2M Reuse 2",
    "UniPC (6-Step)",
    "Heun 2nd-Order",
    "Euler Direct",
    "Euler Ancestral",
    "Euler-A DualClock",
    "Flow Shift Anime",
    "DPM++ 2M Gold",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "ER-SDE Flow",
    "LQ-Flow Schedule",
    "DEIS Integrator",
    "FastVideo 8-Step",
    "CFG-Zero* Rescaled"
]

# Build 7x18 matrix of denoise times
matrix = np.zeros((len(preset_order), len(sampler_order)))

for d in data:
    p_idx = preset_order.index(d["preset_id"])
    s_idx = sampler_order.index(d["sampler_id"])
    matrix[p_idx, s_idx] = d["denoise_sec"]

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# CHART 1: MASTER HEATMAP MATRIX (7 PRESETS x 18 SAMPLERS)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(20, 9.5), dpi=300)
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')

cax = ax1.imshow(matrix, cmap='YlGnBu_r', aspect='auto')

# Colorbar
cbar = fig1.colorbar(cax, ax=ax1, pad=0.02, shrink=0.85)
cbar.set_label('Tempo Denoise GPU in Secondi (Metal 4 NAX su M5 Max)', fontsize=11, fontweight='bold', color='#24292f')

# X & Y ticks
ax1.set_xticks(np.arange(len(sampler_order)))
ax1.set_xticklabels(sampler_short_names, rotation=35, ha='right', fontsize=10, fontweight='bold', color='#09244b')

ax1.set_yticks(np.arange(len(preset_order)))
ax1.set_yticklabels([preset_names[p] for p in preset_order], fontsize=11, fontweight='bold', color='#09244b')

# Text inside cells
for i in range(len(preset_order)):
    for j in range(len(sampler_order)):
        val = matrix[i, j]
        text_color = '#ffffff' if val > 150 else '#09244b'
        ax1.text(j, i, f"{val:.1f}s", ha='center', va='center', color=text_color, fontweight='bold', fontsize=9.5)

ax1.set_title('MiniMax-H3 · Super-Matrice Denoise GPU: 7 Preset H3MLX × 18 Sampler (Clip 4.0s / 90 Frame)', 
              fontsize=14.5, fontweight='bold', color='#09244b', pad=18, loc='left')

plt.tight_layout()

chart1_path = out_dir / "master_sampler_heatmap_matrix.png"
brain1_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/master_combinatorial_4s_gallery/master_sampler_heatmap_matrix.png")

fig1.savefig(chart1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig1.savefig(brain1_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Master Heatmap to:", chart1_path)

# ==============================================================================
# CHART 2: COMPARISON OF KEY SAMPLER SPEEDS ACROSS THE 7 PRESETS
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(16, 9), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# Select 5 representative samplers
rep_samplers = [
    ("fastflow_taylor_skip", "FastFlow Taylor / Turbo", "#0550ae"),
    ("dpm2m_reuse2_sla", "DPM++ 2M Step-Reuse 2 (SLA)", "#1a7f37"),
    ("unipc_fast_trailing", "UniPC Fast Trailing (6-Step)", "#9a6700"),
    ("flow_anime_s8", "Flow Shift Anime (Shift 8.0)", "#6f42c1"),
    ("euler_a_trailing", "Euler Ancestral (Euler A)", "#d63384")
]

x_presets = np.arange(len(preset_order))
bar_w = 0.16

for idx, (s_id, s_name, s_col) in enumerate(rep_samplers):
    s_col_idx = sampler_order.index(s_id)
    vals = matrix[:, s_col_idx]
    offsets = x_presets - 0.32 + idx * bar_w
    bars = ax2.bar(offsets, vals, width=bar_w * 0.92, color=s_col, label=s_name, edgecolor='#ffffff', linewidth=0.8, zorder=3)
    
    for b, v in zip(bars, vals):
        if v < 400: # only annotate manageable numbers
            ax2.annotate(f"{v:.0f}s", xy=(b.get_x() + b.get_width()/2, v),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                         fontsize=8, fontweight='bold', color=s_col)

ax2.set_xticks(x_presets)
ax2.set_xticklabels([preset_names[p].split('(')[0].strip() for p in preset_order], fontsize=11, fontweight='bold', color='#09244b')
ax2.set_ylabel('Tempo Denoise GPU in Secondi (Clip 4.0s / 90 Frame)', fontsize=12, fontweight='bold', color='#24292f', labelpad=10)
ax2.set_title('MiniMax-H3 · Confronto Denoise dei Principali Sampler attraverso i 7 Preset H3MLX', 
              fontsize=14.5, fontweight='bold', color='#09244b', pad=16, loc='left')
ax2.grid(axis='y', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)
ax2.legend(loc='upper left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

for spine in ax2.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.2)

plt.tight_layout()

chart2_path = out_dir / "master_preset_comparison_speed_chart.png"
brain2_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/master_combinatorial_4s_gallery/master_preset_comparison_speed_chart.png")

fig2.savefig(chart2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig2.savefig(brain2_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Master Comparison Chart to:", chart2_path)
