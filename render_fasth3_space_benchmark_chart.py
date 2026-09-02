import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load benchmark results
json_path = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_fasth3_vsa_space_benchmark/fasth3_vsa_space_benchmark_results.json")
with open(json_path) as f:
    data = json.load(f)

# Output Paths
out_dir = Path("/Users/robzomb/Documents/antigravity/cool-hopper/outputs_fasth3_vsa_space_benchmark")
chart_out = out_dir / "fasth3_vsa_space_benchmark_chart.png"
brain_dir = Path("/Users/robzomb/.gemini/antigravity/brain/a6b18108-3903-4957-ade7-687e85eb20df/fasth3_vsa_gallery")
brain_dir.mkdir(parents=True, exist_ok=True)
chart_brain = brain_dir / "fasth3_vsa_space_benchmark_chart.png"

# Setup high quality plot
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Data extraction
labels = [f"{d['canvas_label']}\n({d['frames']}f / {d['duration_label'].split('(')[0].strip()})" for d in data]
denoise_times = [d['denoise_sec'] for d in data]
vae_times = [d['vae_sec'] for d in data]
fps_rates = [d['throughput_fps'] for d in data]
total_times = [d['wall_total'] for d in data]

y_pos = np.arange(len(labels))
bar_height = 0.55

# Subplot 1: Stacked Bar Chart (Denoise vs VAE vs Total)
p1 = ax1.barh(y_pos, denoise_times, bar_height, label='GPU DiT Denoise (4 Forwards)', color='#2563eb', alpha=0.9)
p2 = ax1.barh(y_pos, vae_times, bar_height, left=denoise_times, label='3D Video VAE Decode', color='#059669', alpha=0.85)

ax1.set_facecolor('#f8fafc')
ax1.grid(axis='x', linestyle='--', alpha=0.5, color='#94a3b8')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=10, fontweight='bold', color='#1e293b')
ax1.invert_yaxis()  # top-down
ax1.set_xlabel('Latency in Seconds (Wall Time)', fontsize=12, fontweight='bold', color='#0f172a')
ax1.set_title('FastH3 VSA (Fast Mode): Latency Breakdown\n[Apple Silicon M5 Max 128GB UMA]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
ax1.legend(loc='lower right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1')

# Add text labels on bars
for i, d in enumerate(data):
    tot = d['denoise_sec'] + d['vae_sec']
    ax1.text(tot + 2.5, i, f"{d['denoise_sec']:.1f}s Denoise | {d['wall_total']:.1f}s Tot", va='center', fontsize=9, fontweight='bold', color='#1e293b')

# Subplot 2: Throughput FPS by Aspect Ratio
colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#6366f1']
bars = ax2.bar(np.arange(len(data)), fps_rates, color=colors, width=0.55, edgecolor='#334155', linewidth=1)

ax2.set_facecolor('#f8fafc')
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94a3b8')
ax2.set_xticks(np.arange(len(data)))
short_labels = [f"{d['canvas_label'].split('·')[0].strip()}\n{d['frames']}f" for d in data]
ax2.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=9, fontweight='bold', color='#1e293b')
ax2.set_ylabel('Denoise Throughput (FPS)', fontsize=12, fontweight='bold', color='#0f172a')
ax2.set_title('Denoise Throughput (FPS) Across Fast Canvases\n[4-Step FastH3 VSA Backend]', fontsize=13, fontweight='bold', pad=12, color='#0f172a')

for bar, fps in zip(bars, fps_rates):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{fps:.2f} FPS", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

ax2.set_ylim(0, max(fps_rates) * 1.25)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle('Hugging Face Space: Mike0021/FastH3-4step-Preview-VSA Benchmark (Fast Mode)\nFrozen Bakery Prompt (Speech + Audio Sync) · Seed: 42', fontsize=14, fontweight='bold', color='#0f172a')

fig.savefig(chart_out, dpi=300, bbox_inches='tight', facecolor='#ffffff')
fig.savefig(chart_brain, dpi=300, bbox_inches='tight', facecolor='#ffffff')
print(f"✓ Saved FastH3 Space Benchmark Chart to: {chart_out}")
