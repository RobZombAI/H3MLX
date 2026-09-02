import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors

# Load all 126 results
base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_master_combinatorial_4s"
json_path = out_dir / "master_combinatorial_benchmark_4s_results.json"

with open(json_path) as f:
    data = json.load(f)

# Preset ordering
preset_order = ["draft", "turbo", "champion", "cinema", "reel", "quality", "oracle"]
preset_labels = [
    "👀 Ultra Draft\n(640x640 · 45L · Reuse 2)",
    "⚡ FastVideo Turbo\n(640x640 · 50L · 4-Step)",
    "🏆 Fast Master Champion\n(640x640 · 50L · 8-Step)",
    "🎬 Cinema 16:9\n(960x544 · 50L · 8-Step)",
    "📱 Vertical Reel 9:16\n(544x960 · 50L · 8-Step)",
    "💎 High Quality Master\n(640x640 · 50L · 20-Step)",
    "👑 Full Oracle Baseline\n(640x640 · 50L · 50-Step BF16)"
]

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

sampler_labels = [
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

# Build 7x18 matrix
matrix = np.zeros((len(preset_order), len(sampler_order)))
for d in data:
    p_idx = preset_order.index(d["preset_id"])
    s_idx = sampler_order.index(d["sampler_id"])
    matrix[p_idx, s_idx] = d["denoise_sec"]

# High-contrast custom colormap: Mint -> Cyan -> Azure -> Cobalt -> Deep Navy
colors = [
    "#d1fae5",  # 0-30s: Fresh Mint (Ultra Fast)
    "#e0f2fe",  # 30-50s: Light Sky
    "#bae6fd",  # 50-80s: Soft Azure
    "#7dd3fc",  # 80-120s: Bright Cyan
    "#38bdf8",  # 120-180s: Vivid Blue
    "#0284c7",  # 180-260s: Deep Cobalt
    "#0369a1",  # 260-380s: Dark Ocean
    "#0f172a"   # >380s: Midnight Navy
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("HighVisibilityMap", colors, N=256)

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig, ax = plt.subplots(figsize=(22, 10.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Display heatmap
norm = mcolors.LogNorm(vmin=18, vmax=550)
cax = ax.imshow(matrix, cmap=custom_cmap, norm=norm, aspect='auto')

# Grid lines between cells for absolute crispness
ax.set_xticks(np.arange(len(sampler_order)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(preset_order)) - 0.5, minor=True)
ax.grid(which="minor", color="#94a3b8", linestyle='-', linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Ticks
ax.set_xticks(np.arange(len(sampler_order)))
ax.set_xticklabels(sampler_labels, rotation=35, ha='right', fontsize=10.5, fontweight='bold', color='#0f172a')

ax.set_yticks(np.arange(len(preset_order)))
ax.set_yticklabels(preset_labels, fontsize=11, fontweight='bold', color='#0f172a')

# Contrast text inside cells
for i in range(len(preset_order)):
    for j in range(len(sampler_order)):
        val = matrix[i, j]
        # Text color based on background luminance
        text_color = '#ffffff' if val > 140 else '#0f172a'
        ax.text(j, i, f"{val:.1f}s", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10.5)

# Colorbar with clear ticks
cbar = fig.colorbar(cax, ax=ax, pad=0.018, shrink=0.88)
cbar.set_label('Tempo Denoise GPU in Secondi (Metal 4 NAX su Apple Silicon M5 Max)', fontsize=11.5, fontweight='bold', color='#0f172a', labelpad=12)
cbar.set_ticks([20, 40, 60, 80, 100, 150, 200, 300, 500])
cbar.set_ticklabels(['20s', '40s', '60s', '80s', '100s', '150s', '200s', '300s', '500s'])
cbar.ax.tick_params(labelsize=10.5)

# Category grouping brackets / titles on top
ax.text(1.5, -0.75, "⚡ VELOCITÀ RECORD (4-Step)", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0284c7')
ax.text(4.0, -0.75, "⚖️ BILANCIATI & CACHE", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#16a34a')
ax.text(8.5, -0.75, "🌸 ALTA QUALITÀ & ARTE GHIBLI", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#d63384')
ax.text(14.5, -0.75, "🔬 RICERCA & FLOW MATCHING 2026", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#6f42c1')

ax.set_title('MiniMax-H3 · Super-Matrice Denoise GPU: 7 Preset H3MLX × 18 Sampler & Schedule (Clip 4.0s / 90 Frame @ 24fps)', 
             fontsize=14.5, fontweight='bold', color='#09244b', pad=32, loc='left')

for spine in ax.spines.values():
    spine.set_color('#94a3b8')
    spine.set_linewidth(1.5)

plt.tight_layout()

chart_path = out_dir / "master_sampler_heatmap_matrix_v2.png"
brain_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/master_combinatorial_4s_gallery/master_sampler_heatmap_matrix_v2.png")

fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(brain_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved High-Visibility Matrix V2 to:", chart_path)
