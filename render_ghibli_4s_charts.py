import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_ghibli_4s_benchmark"
json_path = out_dir / "ghibli_4s_benchmark_results.json"

with open(json_path) as f:
    data = json.load(f)

# Sort from fastest to slowest denoise
data_sorted = sorted(data, key=lambda x: x["denoise_sec"])

names = [
    "FastFlow / Turbo (4-Step)",
    "DPM++ 2M Step-Reuse 2",
    "UniPC Fast Trailing (6-Step)",
    "Euler Ancestral (Euler A)",
    "Flow Shifted Anime (Shift 8)",
    "DPM++ 2M Trailing Gold",
    "DPM++ 2M SDE Karras Flow"
]

denoise_vals = [d["denoise_sec"] for d in data_sorted]
vae_vals = [d["vae_sec"] for d in data_sorted]
fps_vals = [d["fps"] for d in data_sorted]

# ==============================================================================
# CHART 1: VERTICAL BAR CHART (GPU DENOISE 4.0s / 90 FRAMES)
# ==============================================================================
plt.style.use('dark_background')
fig1, ax1 = plt.subplots(figsize=(14, 8), dpi=220)
fig1.patch.set_facecolor('#0b0e14')
ax1.set_facecolor('#161b22')

x = np.arange(len(names))
colors = ['#00d2ff', '#39d353', '#e3b341', '#f778ba', '#a371f7', '#58a6ff', '#ffa657']

bars = ax1.bar(x, denoise_vals, width=0.55, color=colors, edgecolor='#30363d', linewidth=1.2)

for bar, fps in zip(bars, fps_vals):
    h = bar.get_height()
    ax1.annotate(f'{h:.2f}s\n({fps:.2f} FPS)', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                 fontsize=10.5, fontweight='bold', color='#f0f6fc')

ax1.set_ylabel('Tempo di Denoising GPU in Secondi (Metal 4 NAX su M5 Max)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=10)
ax1.set_title('MiniMax-H3 · Benchmark 4.0s (90 Frame / 5 Chunk Causali): Confronto Velocità Sampler', 
              fontsize=14.5, fontweight='bold', color='#ffffff', pad=18, loc='left')

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=10.5, fontweight='bold', color='#f0f6fc')
ax1.grid(axis='y', color='#30363d', linestyle='--', alpha=0.7)
ax1.set_ylim(0, 100)

for spine in ax1.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

chart1_path = out_dir / "ghibli_4s_speed_barchart.png"
chart1_brain = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery/ghibli_4s_speed_barchart.png")
fig1.savefig(chart1_path, dpi=220, bbox_inches='tight', facecolor=fig1.get_facecolor())
fig1.savefig(chart1_brain, dpi=220, bbox_inches='tight', facecolor=fig1.get_facecolor())
print(f"✓ Saved Chart 1 to {chart1_path}")

# ==============================================================================
# CHART 2: HORIZONTAL STACKED BAR CHART (DENOISE vs VAE 4.0s)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(15, 8.5), dpi=220)
fig2.patch.set_facecolor('#0b0e14')
ax2.set_facecolor('#161b22')

rev_names = list(reversed(names))
rev_denoise = list(reversed(denoise_vals))
rev_vae = list(reversed(vae_vals))
rev_fps = list(reversed(fps_vals))
y_pos = np.arange(len(rev_names))

b_h = 0.52
ax2.barh(y_pos, rev_denoise, height=b_h, color='#00d2ff', edgecolor='#30363d', label='Denoise GPU (DiT 50L Metal 4 NAX)')
ax2.barh(y_pos, rev_vae, left=rev_denoise, height=b_h, color='#ff8c42', edgecolor='#30363d', label='Decodifica Video VAE 3D (5 Chunk Causali)')

for idx, (d_t, v_t, fps_t) in enumerate(zip(rev_denoise, rev_vae, rev_fps)):
    ax2.text(d_t / 2, idx, f"{d_t:.1f}s", va='center', ha='center', color='#000000', fontweight='bold', fontsize=10.5)
    ax2.text(d_t + v_t / 2, idx, f"{v_t:.1f}s", va='center', ha='center', color='#000000', fontweight='bold', fontsize=10.5)
    ax2.text(d_t + v_t + 2.0, idx, f"Totale Denoise+VAE: {d_t+v_t:.1f}s | Throughput: {fps_t:.2f} FPS", va='center', ha='left', color='#58a6ff', fontweight='bold', fontsize=10.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(rev_names, fontsize=11, fontweight='bold', color='#f0f6fc')
ax2.set_title("MiniMax-H3 · Scomposizione 4.0s (90f): Denoise GPU (DiT) vs Decodifica Video VAE 3D",
              fontsize=14.5, fontweight='bold', color='#ffffff', pad=16, loc='left')
ax2.grid(axis='x', color='#30363d', linestyle='--', alpha=0.6)
ax2.set_xlabel('Secondi di Esecuzione GPU su M5 Max (128GB UMA)', fontsize=11, color='#8b949e')
ax2.set_xlim(0, 160)
ax2.legend(loc='upper right', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=11)

for spine in ax2.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

chart2_path = out_dir / "ghibli_4s_breakdown_chart.png"
chart2_brain = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/ghibli_4s_gallery/ghibli_4s_breakdown_chart.png")
fig2.savefig(chart2_path, dpi=220, bbox_inches='tight', facecolor=fig2.get_facecolor())
fig2.savefig(chart2_brain, dpi=220, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f"✓ Saved Chart 2 to {chart2_path}")
