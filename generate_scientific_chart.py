import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark matrix
json_path = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/benchmarks_matrix/benchmark_matrix_results.json")
with open(json_path) as f:
    data = json.load(f)

# Group by duration: 1s, 2s, 4s
dur_groups = {"1s": [], "2s": [], "4s": []}
for row in data:
    d = row["duration_label"]
    if d in dur_groups:
        dur_groups[d].append(row)

# Clean preset display names
name_map = {
    "draft": "Draft (Gate-Ranking 45L)",
    "turbo": "FastVideo v0.2 Turbo (DMD2)",
    "champion": "Fast Master Champion (Shift 12)",
    "cinema16x9": "Cinema 16:9 Widescreen",
    "reel9x16": "Vertical Reel 9:16",
    "quality": "High Quality Master (20L)"
}

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(3, 1, figsize=(15, 14), gridspec_kw={'height_ratios': [len(dur_groups["1s"]), len(dur_groups["2s"]), len(dur_groups["4s"])]})
fig.patch.set_facecolor('#0d1117')

colors_denoise = '#00d2ff'  # Electric Cyan
colors_vae = '#ff8c42'      # Warm Amber/Orange
colors_text = '#f0f6fc'

dur_titles = {
    "1s": "CLIP 1.0s (22 Frames @ 24fps - 1 Causal Chunk)",
    "2s": "CLIP 2.0s (39 Frames @ 24fps - 2 Causal Chunks)",
    "4s": "CLIP 4.0s (90 Frames @ 24fps - 5 Causal Chunks)"
}

for ax, (dur, rows) in zip(axes, dur_groups.items()):
    ax.set_facecolor('#161b22')
    
    # Reverse rows so fastest appears at top
    rows = list(reversed(rows))
    
    labels = [f"{name_map.get(r['preset_id'], r['preset_name'])} [{r['resolution']}]\n{r['desc']}" for r in rows]
    denoise_times = [r['denoise_sec'] for r in rows]
    vae_times = [r['vae_sec'] for r in rows]
    total_times = [r['wall_total'] for r in rows]
    
    y_pos = np.arange(len(rows))
    bar_height = 0.52
    
    # Horizontal stacked bars: Denoise + VAE
    b1 = ax.barh(y_pos, denoise_times, height=bar_height, color=colors_denoise, edgecolor='#30363d', label='Denoise GPU (DiT Metal 4 NAX)' if dur == "1s" else "")
    b2 = ax.barh(y_pos, vae_times, left=denoise_times, height=bar_height, color=colors_vae, edgecolor='#30363d', label='Video VAE Decoder 3D (Causal Tiles)' if dur == "1s" else "")
    
    # Annotations
    for idx, (d_t, v_t, tot_t) in enumerate(zip(denoise_times, vae_times, total_times)):
        # Text on denoise bar
        if d_t > 5.0:
            ax.text(d_t / 2, idx, f"{d_t:.2f}s", va='center', ha='center', color='#000000', fontweight='bold', fontsize=10)
        else:
            ax.text(d_t + 0.5, idx, f"{d_t:.2f}s", va='center', ha='left', color=colors_denoise, fontweight='bold', fontsize=9.5)
            
        # Text on VAE bar
        if v_t > 9.0:
            ax.text(d_t + (v_t / 2), idx, f"{v_t:.2f}s", va='center', ha='center', color='#000000', fontweight='bold', fontsize=10)
            
        # Stacked total latency and FPS label
        stacked_end = d_t + v_t
        ax.text(stacked_end + 1.8, idx, f"Total: {tot_t:.1f}s  |  Throughput: {rows[idx]['fps']:.2f} FPS", va='center', ha='left', color='#58a6ff', fontweight='bold', fontsize=10.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='semibold', color=colors_text)
    ax.set_title(dur_titles[dur], fontsize=13, fontweight='bold', color='#58a6ff', pad=12, loc='left')
    ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.6)
    ax.set_xlabel('Execution Seconds on Apple Silicon M5 Max (128GB UMA)', fontsize=10.5, color='#8b949e', labelpad=6)
    
    # Adjust x limits
    max_x = max([d + v for d, v in zip(denoise_times, vae_times)])
    ax.set_xlim(0, max_x * 1.30 + 5)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color('#30363d')

# Global title & legend
fig.suptitle("H3MLX · Empirical Performance Breakdown: GPU Denoise (DiT) vs 3D Video VAE Decoding", 
             fontsize=16, fontweight='bold', color='#ffffff', y=0.99)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, ['Denoise GPU (DiT Metal 4 NAX)', 'Video VAE Decoder 3D (Causal Tiles)'], 
           loc='upper right', bbox_to_anchor=(0.95, 0.99), frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_img = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/h3mlx_benchmark_chart.png")
out_img_brain = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/h3mlx_benchmark_chart.png")

plt.savefig(out_img, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.savefig(out_img_brain, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"✅ Clean high-resolution benchmark chart saved to {out_img}")
