import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Setup data
presets = [
    "Draft\n(4L Reuse 2)",
    "Turbo\n(FastVideo 4L)",
    "Champion\n(Fast Master 8L)",
    "Cinema 16:9\n(960x544)",
    "Reel 9:16\n(544x960)",
    "Quality\n(20L Master)",
    "Oracle Ref\n(50L BF16)"
]

# Denoise GPU times (seconds)
d1_times = [3.29, 6.53, 12.55, 16.41, 16.44, 30.88, 120.00]
d2_times = [6.43, 12.28, 24.11, 33.76, 33.38, 59.81, 240.00]
d4_times = [23.21, 39.94, 78.35, 113.68, 115.32, 0, 0]  # 0 for untracked 4s on Quality/Oracle

# Total times (Denoise + VAE + System)
tot1_times = [32.69, 38.56, 44.92, 50.53, 50.64, 62.80, 150.00]

# Setup plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 9.5), dpi=240)
fig.patch.set_facecolor('#0b0e14')
ax.set_facecolor('#161b22')

x = np.arange(len(presets))
bar_width = 0.26

# Colors
c1 = '#00d2ff'  # Electric Cyan (1s)
c2 = '#a371f7'  # Neon Purple (2s)
c4 = '#ff8c42'  # Amber Orange (4s)

# Create Grouped Vertical Bars
rects1 = ax.bar(x - bar_width, d1_times, bar_width, label='Clip 1.0s (22 Frame @ 24fps)', color=c1, edgecolor='#30363d', linewidth=1)
rects2 = ax.bar(x, d2_times, bar_width, label='Clip 2.0s (39 Frame @ 24fps)', color=c2, edgecolor='#30363d', linewidth=1)
rects4 = ax.bar(x + bar_width, d4_times, bar_width, label='Clip 4.0s (90 Frame @ 24fps)', color=c4, edgecolor='#30363d', linewidth=1)

# Add value labels on top of bars
def autolabel(rects, text_color, is_d4=False):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color=text_color)

autolabel(rects1, c1)
autolabel(rects2, c2)
autolabel(rects4, c4, is_d4=True)

# Customizing axes
ax.set_ylabel('Denoise GPU Time in Secondi (Metal 4 NAX su M5 Max)', fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=10)
ax.set_title('H3MLX · Confronto a Barre Verticali dei Tempi di Denoising GPU per Preset & Durata', 
             fontsize=15, fontweight='bold', color='#ffffff', pad=20, loc='left')

ax.set_xticks(x)
ax.set_xticklabels(presets, fontsize=11, fontweight='semibold', color='#f0f6fc')
ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.7)

# Set logarithmic or linear friendly view
ax.set_ylim(0, 265)

# Add hardware badge on the upper right
ax.text(0.98, 0.96, 'Apple Silicon M5 Max · 128GB UMA · 40 GPU Cores', 
        transform=ax.transAxes, fontsize=10.5, fontweight='medium', color='#8b949e', 
        ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))

# Legend styling
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=11)
legend.get_frame().set_linewidth(1.2)

# Clean spines
for spine in ax.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

out_file = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/h3mlx_vertical_barchart.png")
out_brain = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/h3mlx_vertical_barchart.png")

plt.savefig(out_file, dpi=240, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.savefig(out_brain, dpi=240, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"✅ Vertical bar chart saved to {out_file}")
