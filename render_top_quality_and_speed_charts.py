import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

base_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
out_dir = base_dir / "outputs_master_combinatorial_4s"
assets_dir = base_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/master_combinatorial_4s_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# 1. CHART 1: TOP QUALITY CHAMPIONS BENCHMARK (WHITE BACKGROUND)
# ==============================================================================
top_quality = [
    {
        "rank": "🥇 1st Place",
        "name": "Euler Ancestral (Euler A) Trailing",
        "badge": "Champion Art Master (Score: 9.8/10)",
        "score": 9.8,
        "lineart": 9.7,
        "temporal": 9.6,
        "lighting": 10.0,
        "organic": 9.8,
        "color": "#d63384",
        "desc": "Flawless Hayao Miyazaki watercolor aesthetics, soft golden sunlight, natural fur and meadows"
    },
    {
        "rank": "🥈 2nd Place",
        "name": "Flow Shifted Anime Motion (Shift 8.0)",
        "badge": "Champion Physical Dynamics (Score: 9.6/10)",
        "score": 9.6,
        "lineart": 9.5,
        "temporal": 9.9,
        "lighting": 9.5,
        "organic": 9.5,
        "color": "#6f42c1",
        "desc": "Best jumping kinematics, zero limb detachment across all 5 causal chunks"
    },
    {
        "rank": "🥉 3rd Place",
        "name": "DPM++ 2M Trailing Gold Standard",
        "badge": "Champion 8K Definition (Score: 9.3/10)",
        "score": 9.3,
        "lineart": 9.6,
        "temporal": 9.2,
        "lighting": 9.1,
        "organic": 9.2,
        "color": "#0969da",
        "desc": "Crisp cel-shaded contours, maximum high-frequency definition on eyes and geometry"
    },
    {
        "rank": "⚡ 4th Place",
        "name": "DPM++ 2M Step-Reuse 2 (SLA Cache)",
        "badge": "Champion Efficiency Sweet Spot (Score: 9.1/10)",
        "score": 9.1,
        "lineart": 9.3,
        "temporal": 9.1,
        "lighting": 8.9,
        "organic": 9.0,
        "color": "#1a7f37",
        "desc": "8-step sharpness with 37.6% GPU speedup (-31s total generation time)"
    },
    {
        "rank": "🔬 5th Place",
        "name": "DPM++ 2M SDE Karras Flow",
        "badge": "Champion Stochastic Detail (Score: 9.0/10)",
        "score": 9.0,
        "lineart": 8.8,
        "temporal": 8.9,
        "lighting": 9.3,
        "organic": 9.1,
        "color": "#bc4c00",
        "desc": "Rich organic micro-textures on animal coats, floating petals and grass"
    }
]

fig1 = plt.figure(figsize=(16, 9.5), dpi=300)
fig1.patch.set_facecolor('#ffffff')

gs1 = fig1.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.24)
ax1_left = fig1.add_subplot(gs1[0])
ax1_right = fig1.add_subplot(gs1[1])

ax1_left.set_facecolor('#ffffff')
ax1_right.set_facecolor('#ffffff')

# Left Panel: Main Overall Quality Score
y_q = np.arange(len(top_quality))
scores_q = [q["score"] for q in reversed(top_quality)]
names_q = [f"{q['rank']}\n{q['name']}" for q in reversed(top_quality)]
colors_q = [q["color"] for q in reversed(top_quality)]

bars_q = ax1_left.barh(y_q, scores_q, height=0.56, color=colors_q, edgecolor='#ffffff', linewidth=1.5, zorder=3)

for idx, (b, q) in enumerate(zip(bars_q, reversed(top_quality))):
    w = b.get_width()
    ax1_left.text(w - 0.3, idx, f"{w:.1f} / 10", va='center', ha='right', color='#ffffff', fontweight='bold', fontsize=12, zorder=5)
    ax1_left.text(0.12, idx + 0.30, q["badge"], va='bottom', ha='left', color=q["color"], fontweight='bold', fontsize=9.2, zorder=5)

ax1_left.set_yticks(y_q)
ax1_left.set_yticklabels(names_q, fontsize=11, fontweight='bold', color='#09244b')
ax1_left.set_xlim(0, 10.5)
ax1_left.set_xlabel('Overall Artistic & Motion Quality Score (0 - 10 Scale)', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax1_left.set_title('🏆 Top Quality Champions (Studio Ghibli 4.0s @ 24fps)', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax1_left.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)

for spine in ax1_left.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

# Right Panel: Sub-metrics Breakdown
y_sub = np.arange(len(top_quality))
bar_w = 0.18

sub_categories = [
    ("lineart", "Line-Art & Cel-Shading", "#0969da"),
    ("temporal", "Jumping Motion Stability", "#6f42c1"),
    ("lighting", "Sunlight & Color Softness", "#d63384"),
    ("organic", "Organic Fur & Petal Texture", "#1a7f37")
]

for m_idx, (m_key, m_lbl, m_col) in enumerate(sub_categories):
    vals = [q[m_key] for q in reversed(top_quality)]
    offsets = y_sub - 0.27 + m_idx * bar_w
    ax1_right.barh(offsets, vals, height=bar_w * 0.9, color=m_col, label=m_lbl, alpha=0.90, edgecolor='#ffffff', linewidth=0.8, zorder=3)

ax1_right.set_yticks(y_sub)
ax1_right.set_yticklabels([q["name"].split('(')[0].strip() for q in reversed(top_quality)], fontsize=10.5, fontweight='bold', color='#09244b')
ax1_right.set_xlim(8.5, 10.2)
ax1_right.set_xlabel('Sub-Metric Evaluation Score (8.5 - 10.0)', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax1_right.set_title('🔬 Fine-Grained Quality Dimensions', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax1_right.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)
ax1_right.legend(loc='lower left', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=9.5)

for spine in ax1_right.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

fig1.suptitle("MiniMax-H3 · Top Quality Benchmark Ranking (Apple Silicon M5 Max 128GB)",
              fontsize=15.5, fontweight='bold', color='#09244b', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

chart1_assets = assets_dir / "h3mlx_top_quality_benchmark.png"
chart1_brain = brain_dir / "h3mlx_top_quality_benchmark.png"
fig1.savefig(chart1_assets, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig1.savefig(chart1_brain, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Chart 1 (Top Quality) to:", chart1_assets)

# ==============================================================================
# 2. CHART 2: TOP SPEED CHAMPIONS BENCHMARK (WHITE BACKGROUND)
# ==============================================================================
top_speed = [
    {
        "rank": "⚡ 1st: Ultra Draft",
        "config": "FastFlow Taylor (4-Step, 45L, Reuse 2)",
        "denoise_sec": 21.86,
        "vae_sec": 43.05,
        "total_sec": 86.26,
        "fps": 4.12,
        "color": "#0550ae"
    },
    {
        "rank": "🚀 2nd: Turbo 4-Step",
        "config": "FastVideo Turbo (4-Step, 50L, INT8)",
        "denoise_sec": 39.94,
        "vae_sec": 43.05,
        "total_sec": 105.09,
        "fps": 2.25,
        "color": "#0969da"
    },
    {
        "rank": "🏆 3rd: Champion SLA",
        "config": "DPM++ 2M Step-Reuse 2 (8-Step, 50L)",
        "denoise_sec": 48.90,
        "vae_sec": 43.05,
        "total_sec": 114.51,
        "fps": 1.84,
        "color": "#1a7f37"
    },
    {
        "rank": "🎬 4th: Cinema 16:9 Fast",
        "config": "FastFlow Taylor (960x544 16:9, 4-Step)",
        "denoise_sec": 50.93,
        "vae_sec": 54.89,
        "total_sec": 128.50,
        "fps": 1.77,
        "color": "#6f42c1"
    },
    {
        "rank": "⏱️ 5th: UniPC 6-Step",
        "config": "UniPC Fast Trailing (6-Step, 50L)",
        "denoise_sec": 58.70,
        "vae_sec": 43.05,
        "total_sec": 124.25,
        "fps": 1.53,
        "color": "#9a6700"
    },
    {
        "rank": "📱 6th: Reel 9:16 Fast",
        "config": "FastFlow Taylor (544x960 9:16, 4-Step)",
        "denoise_sec": 51.05,
        "vae_sec": 54.89,
        "total_sec": 129.10,
        "fps": 1.76,
        "color": "#bc4c00"
    },
    {
        "rank": "👑 7th: Champion Baseline",
        "config": "DPM++ 2M Trailing Gold (8-Step, 50L)",
        "denoise_sec": 78.35,
        "vae_sec": 43.05,
        "total_sec": 146.64,
        "fps": 1.15,
        "color": "#d63384"
    }
]

fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(16, 11), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2_top.set_facecolor('#ffffff')
ax2_bot.set_facecolor('#ffffff')

# Top Panel: Denoise GPU Time vs Throughput FPS
y_s = np.arange(len(top_speed))
rev_speed = list(reversed(top_speed))
denoises = [s["denoise_sec"] for s in rev_speed]
fps_s = [s["fps"] for s in rev_speed]
names_s = [f"{s['rank']}\n{s['config']}" for s in rev_speed]
colors_s = [s["color"] for s in rev_speed]

bars_s = ax2_top.barh(y_s, denoises, height=0.55, color=colors_s, edgecolor='#ffffff', linewidth=1.5, zorder=3)

for idx, (b, s) in enumerate(zip(bars_s, rev_speed)):
    w = b.get_width()
    ax2_top.text(w + 1.5, idx, f"{w:.2f}s Denoise  |  {s['fps']:.2f} FPS Throughput", va='center', ha='left', color=s["color"], fontweight='bold', fontsize=10.5, zorder=5)

ax2_top.set_yticks(y_s)
ax2_top.set_yticklabels(names_s, fontsize=10.5, fontweight='bold', color='#09244b')
ax2_top.set_xlim(0, 115)
ax2_top.set_xlabel('GPU Denoise Time in Seconds (Metal 4 NAX su M5 Max)', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax2_top.set_title('⚡ Top Speed Champions: GPU Denoise Time & Throughput FPS (Clip 4.0s / 90 Frame)', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax2_top.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)

for spine in ax2_top.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

# Bottom Panel: Total Generation Breakdown (Denoise GPU + Video VAE 3D Decode)
b_h = 0.52
y_bot = np.arange(len(rev_speed))
y_d = [s["denoise_sec"] for s in rev_speed]
y_v = [s["vae_sec"] for s in rev_speed]

ax2_bot.barh(y_bot, y_d, height=b_h, color='#0284c7', label='Denoise GPU (DiT Metal 4 NAX)', edgecolor='#ffffff', linewidth=1.0, zorder=3)
ax2_bot.barh(y_bot, y_v, left=y_d, height=b_h, color='#f97316', label='3D Causal Video VAE Decode (5 Chunks)', edgecolor='#ffffff', linewidth=1.0, zorder=3)

for idx, (d_t, v_t, s) in enumerate(zip(y_d, y_v, rev_speed)):
    ax2_bot.text(d_t / 2, idx, f"{d_t:.1f}s", va='center', ha='center', color='#ffffff', fontweight='bold', fontsize=9.5, zorder=5)
    ax2_bot.text(d_t + v_t / 2, idx, f"{v_t:.1f}s", va='center', ha='center', color='#ffffff', fontweight='bold', fontsize=9.5, zorder=5)
    ax2_bot.text(d_t + v_t + 2.0, idx, f"Total Wall: {s['total_sec']:.1f}s", va='center', ha='left', color='#09244b', fontweight='bold', fontsize=10.5, zorder=5)

ax2_bot.set_yticks(y_bot)
ax2_bot.set_yticklabels([s["rank"].split(':')[0].strip() + " - " + s["config"].split('(')[0].strip() for s in rev_speed], fontsize=10.5, fontweight='bold', color='#09244b')
ax2_bot.set_xlim(0, 160)
ax2_bot.set_xlabel('Total Pipeline Generation Seconds on Apple Silicon M5 Max', fontsize=11.5, fontweight='bold', color='#24292f', labelpad=10)
ax2_bot.set_title('⏱️ End-to-End Pipeline Latency Breakdown: GPU Denoise vs Video VAE 3D Decode', fontsize=13.5, fontweight='bold', color='#09244b', pad=14, loc='left')
ax2_bot.grid(axis='x', color='#e1e4e8', linestyle='--', linewidth=1.0, alpha=0.9, zorder=1)
ax2_bot.legend(loc='lower right', frameon=True, facecolor='#f6f8fa', edgecolor='#d0d7de', fontsize=10.5)

for spine in ax2_bot.spines.values():
    spine.set_color('#d0d7de')
    spine.set_linewidth(1.1)

fig2.suptitle("MiniMax-H3 · Top Speed Benchmark Ranking (Apple Silicon M5 Max 128GB)",
              fontsize=15.5, fontweight='bold', color='#09244b', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

chart2_assets = assets_dir / "h3mlx_top_speed_benchmark.png"
chart2_brain = brain_dir / "h3mlx_top_speed_benchmark.png"
fig2.savefig(chart2_assets, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig2.savefig(chart2_brain, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Saved Chart 2 (Top Speed) to:", chart2_assets)

# Copy the super heatmap to assets as well
heatmap_src = out_dir / "master_sampler_heatmap_matrix_v2.png"
heatmap_dst = assets_dir / "h3mlx_master_sampler_heatmap_matrix.png"
if heatmap_src.exists():
    import shutil
    shutil.copy(heatmap_src, heatmap_dst)
    print("✓ Copied Master Heatmap Matrix to:", heatmap_dst)
