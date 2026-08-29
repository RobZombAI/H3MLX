import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set up figure
fig_width = 16.5
fig_height = 10.5
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=240)
fig.patch.set_facecolor('#0b0e14')
ax.set_facecolor('#0b0e14')
ax.axis('off')

# Data rows
headers = [
    "PRESET & PROFILE",
    "RESOLUTION",
    "STEPS & LAYERS",
    "DENOISE 1s (22f)",
    "DENOISE 2s (39f)",
    "DENOISE 4s (90f)",
    "THROUGHPUT",
    "VAE 3D (1s)"
]

rows = [
    {
        "name": "Draft (Ultra Draft)",
        "sub": "4-Step / 45L / Reuse 2 (Gate-Ranking)",
        "res": "640 x 640 (1:1)",
        "config": "4 Steps · 45 Layers · INT8",
        "d1": "3.29 s",
        "d2": "6.43 s",
        "d4": "23.21 s",
        "fps": "6.69 FPS",
        "vae": "8.82 s",
        "badge": "#39d353", # Green
        "highlight": False
    },
    {
        "name": "FastVideo v0.2 Turbo",
        "sub": "4-Step [999, 749, 500, 250] (DMD2)",
        "res": "640 x 640 (1:1)",
        "config": "4 Steps · 50 Layers · INT8",
        "d1": "6.53 s",
        "d2": "12.28 s",
        "d4": "39.94 s",
        "fps": "3.37 FPS",
        "vae": "9.98 s",
        "badge": "#00d2ff", # Cyan
        "highlight": True
    },
    {
        "name": "Fast Master Champion",
        "sub": "8-Step DPM++ / Shift 12.0 (Gold Std)",
        "res": "640 x 640 (1:1)",
        "config": "8 Steps · 50 Layers · INT8",
        "d1": "12.55 s",
        "d2": "24.11 s",
        "d4": "78.35 s",
        "fps": "1.75 FPS",
        "vae": "9.88 s",
        "badge": "#e3b341", # Gold
        "highlight": True
    },
    {
        "name": "Cinema 16:9 Widescreen",
        "sub": "Native 16:9 Anamorphic Canvas",
        "res": "960 x 544 (16:9)",
        "config": "8 Steps · 50 Layers · INT8",
        "d1": "16.41 s",
        "d2": "33.76 s",
        "d4": "113.68 s",
        "fps": "1.34 FPS",
        "vae": "11.45 s",
        "badge": "#a371f7", # Purple
        "highlight": False
    },
    {
        "name": "Vertical Reel 9:16",
        "sub": "Cross-Attention First-Frame Conditioning",
        "res": "544 x 960 (9:16)",
        "config": "8 Steps · 50 Layers · INT8",
        "d1": "16.44 s",
        "d2": "33.38 s",
        "d4": "115.32 s",
        "fps": "1.34 FPS",
        "vae": "11.38 s",
        "badge": "#f778ba", # Pink
        "highlight": False
    },
    {
        "name": "High Quality Master",
        "sub": "20 Iterations Full Convergence",
        "res": "640 x 640 (1:1)",
        "config": "20 Steps · 50 Layers · INT8",
        "d1": "30.88 s",
        "d2": "59.81 s",
        "d4": "—",
        "fps": "0.71 FPS",
        "vae": "9.58 s",
        "badge": "#58a6ff", # Blue
        "highlight": False
    },
    {
        "name": "Full Oracle (Ground-Truth)",
        "sub": "50 Steps BF16 Baseline Reference",
        "res": "640 x 640 (1:1)",
        "config": "50 Steps · 50 Layers · BF16",
        "d1": "120.00 s",
        "d2": "240.00 s",
        "d4": "—",
        "fps": "0.18 FPS",
        "vae": "9.60 s",
        "badge": "#8b949e", # Gray
        "highlight": False
    }
]

# Coordinate layout
x_cols = [0.03, 0.25, 0.38, 0.52, 0.63, 0.74, 0.85, 0.94]
y_start = 0.78
row_height = 0.088

# Draw Header Banner
title_box = patches.FancyBboxPatch((0.02, 0.88), 0.96, 0.09, boxstyle="round,pad=0.015,rounding_size=0.015",
                                    facecolor='#161b22', edgecolor='#30363d', linewidth=1.5)
ax.add_patch(title_box)

ax.text(0.04, 0.94, "H3MLX · MATRICE DEI BENCHMARK EMPIRICI", fontsize=17, fontweight='bold', color='#ffffff', va='center')
ax.text(0.04, 0.905, "Apple Silicon M5 Max (18 CPU Cores · 40 GPU Cores · Metal 4 NAX · 128 GB Unified Memory)", fontsize=11, fontweight='medium', color='#8b949e', va='center')

# Hardware Pill Badges on the right of title
pill1 = patches.FancyBboxPatch((0.74, 0.90), 0.11, 0.05, boxstyle="round,pad=0.01", facecolor='#1f242c', edgecolor='#00d2ff', linewidth=1.2)
ax.add_patch(pill1)
ax.text(0.795, 0.925, "METAL 4 NAX", fontsize=9, fontweight='bold', color='#00d2ff', ha='center', va='center')

pill2 = patches.FancyBboxPatch((0.86, 0.90), 0.11, 0.05, boxstyle="round,pad=0.01", facecolor='#1f242c', edgecolor='#39d353', linewidth=1.2)
ax.add_patch(pill2)
ax.text(0.915, 0.925, "128 GB UMA", fontsize=9, fontweight='bold', color='#39d353', ha='center', va='center')

# Draw Table Column Headers
header_box = patches.FancyBboxPatch((0.02, y_start + 0.01), 0.96, 0.045, boxstyle="round,pad=0.008,rounding_size=0.008",
                                     facecolor='#21262d', edgecolor='#30363d', linewidth=1)
ax.add_patch(header_box)

for x, h in zip(x_cols, headers):
    ax.text(x, y_start + 0.032, h, fontsize=9.5, fontweight='bold', color='#c9d1d9', va='center')

# Draw Data Rows
y_curr = y_start - 0.015

for i, r in enumerate(rows):
    y_row_top = y_curr
    
    # Row background box
    bg_color = '#1c2128' if r['highlight'] else ('#161b22' if i % 2 == 0 else '#12161c')
    border_color = r['badge'] if r['highlight'] else '#30363d'
    border_width = 1.6 if r['highlight'] else 0.8
    
    row_box = patches.FancyBboxPatch((0.02, y_curr - row_height + 0.015), 0.96, row_height - 0.01,
                                      boxstyle="round,pad=0.008,rounding_size=0.01",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=border_width)
    ax.add_patch(row_box)
    
    # Indicator strip
    strip = patches.Rectangle((0.02, y_curr - row_height + 0.015), 0.005, row_height - 0.01, facecolor=r['badge'])
    ax.add_patch(strip)
    
    # Col 1: Preset Name & Subtitle
    ax.text(x_cols[0] + 0.005, y_curr - 0.022, r['name'], fontsize=11, fontweight='bold', color='#ffffff')
    ax.text(x_cols[0] + 0.005, y_curr - 0.046, r['sub'], fontsize=8.5, color='#8b949e')
    
    # Col 2: Resolution
    ax.text(x_cols[1], y_curr - 0.032, r['res'], fontsize=10, fontweight='medium', color='#c9d1d9')
    
    # Col 3: Configuration
    ax.text(x_cols[2], y_curr - 0.032, r['config'], fontsize=9.5, color='#8b949e')
    
    # Col 4: 1s Denoise
    ax.text(x_cols[3], y_curr - 0.032, r['d1'], fontsize=11, fontweight='bold', color='#00d2ff')
    
    # Col 5: 2s Denoise
    ax.text(x_cols[4], y_curr - 0.032, r['d2'], fontsize=11, fontweight='bold', color='#00d2ff')
    
    # Col 6: 4s Denoise
    color_d4 = '#00d2ff' if r['d4'] != '—' else '#6e7681'
    ax.text(x_cols[5], y_curr - 0.032, r['d4'], fontsize=11, fontweight='bold', color=color_d4)
    
    # Col 7: FPS Throughput Badge
    fps_box = patches.FancyBboxPatch((x_cols[6] - 0.01, y_curr - 0.048), 0.075, 0.035, boxstyle="round,pad=0.006",
                                      facecolor='#1f242c', edgecolor='#39d353', linewidth=1)
    ax.add_patch(fps_box)
    ax.text(x_cols[6] + 0.027, y_curr - 0.031, r['fps'], fontsize=9.5, fontweight='bold', color='#39d353', ha='center')
    
    # Col 8: VAE 3D Decode
    ax.text(x_cols[7], y_curr - 0.032, r['vae'], fontsize=10.5, fontweight='bold', color='#ff8c42')
    
    y_curr -= row_height

# Footer bar with legend and insights
footer_box = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.065, boxstyle="round,pad=0.01,rounding_size=0.01",
                                     facecolor='#161b22', edgecolor='#30363d', linewidth=1)
ax.add_patch(footer_box)

ax.text(0.04, 0.063, "LEGENDA & METRICHE:", fontsize=9.5, fontweight='bold', color='#ffffff', va='center')

# Legend items
ax.text(0.20, 0.063, "■ Denoise GPU (Metal 4 NAX)", fontsize=9, fontweight='bold', color='#00d2ff', va='center')
ax.text(0.42, 0.063, "■ Decodifica VAE 3D (Causal Tiles)", fontsize=9, fontweight='bold', color='#ff8c42', va='center')
ax.text(0.66, 0.063, "■ Throughput Generativo (FPS)", fontsize=9, fontweight='bold', color='#39d353', va='center')
ax.text(0.86, 0.063, "★ Preset Raccomandato", fontsize=9, fontweight='bold', color='#e3b341', va='center')

plt.tight_layout()

out_file = Path("/Users/robzomb/Documents/antigravity/cool-hopper/assets/h3mlx_empirical_matrix_table.png")
out_brain = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/h3mlx_empirical_matrix_table.png")

plt.savefig(out_file, dpi=240, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.savefig(out_brain, dpi=240, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"✅ Pixel-perfect empirical matrix chart saved to {out_file}")
