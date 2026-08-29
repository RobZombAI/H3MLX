---
name: h3mlx
description: Autonomous AI agent execution skill for high-speed, photorealistic MiniMax-H3 video and 48kHz audio generation on Apple Silicon (Metal 4 / MLX / Pure C). Includes Fast Master (8-step) and FastVideo (4-step) presets, hardware auto-profiling, and broadcast mastering.
compatibility:
  platforms: [macOS, Apple Silicon M1/M2/M3/M4/M5]
  frameworks: [Hermes, Antigravity, Open-Agent, Claude, Codex, AutoGen]
---

# H3MLX: AI Agent Execution Skill for MiniMax-H3 on Apple Silicon

This skill enables any autonomous agent (such as Hermes, Antigravity, or LangChain agents) to programmatically invoke, benchmark, and orchestrate photorealistic video and synchronized 48kHz audio generation using the high-performance C/Metal 4 MiniMax-H3 engine on Apple Silicon.

---

## ⚡ Agent Capability Matrix

| Action | Preset ID | Command Pattern | Denoise Latency (1.6s) | Primary Use-Case |
| :--- | :--- | :--- | :---: | :--- |
| **8K Gold Master** | `champion` | `./h3_master_cli.sh champion "<prompt>"` | **$\approx 12.5\text{ s}$** | High-definition commercial portraits, natural skin texture, iris reflections. |
| **Sub-15s Turbo** | `turbo` | `./h3_master_cli.sh turbo "<prompt>"` | **$\approx 6.5\text{ s}$** | Fast iterations, storyboard generation, rapid social media clips. |
| **Anamorphic 16:9** | `cinema` | `./h3_master_cli.sh cinema "<prompt>"` | **$\approx 16.4\text{ s}$** | Cinematic landscape shots ($960 \times 544$). |
| **Vertical Reel 9:16** | `reel` | `./h3_master_cli.sh reel "<prompt>" 544 960 39 <img_path>` | **$\approx 16.4\text{ s}$** | Vertical social videos ($544 \times 960$) with image conditioning. |
| **Instant Bozza** | `draft` | `./h3_master_cli.sh draft "<prompt>"` | **$\approx 3.2\text{ s}$** | Prompt validation and lighting checks in sub-10s total turnaround. |

---

## 🛠️ Autonomous Agent Invocation Protocol

When an AI agent needs to generate or master video assets:

### Step 1: Tool Execution via Bash
```bash
# Execute the Master CLI with designated preset, prompt, and optional conditioning
./h3_master_cli.sh [preset_id] "[descriptive prompt]" [width] [height] [frames] [first_frame_path]
```

### Step 2: Causal Temporal Alignment Rule
Agents must align frame counts to the causal temporal lattice formula:
$$\text{Frames} = 17n + 5 \quad (n \ge 1)$$
* $n=1 \to 22 \text{ frames}$ ($0.9\text{s}$ @ 24fps)
* $n=2 \to 39 \text{ frames}$ ($1.6\text{s}$ @ 24fps)
* $n=3 \to 56 \text{ frames}$ ($2.3\text{s}$ @ 24fps)
* $n=5 \to 90 \text{ frames}$ ($3.8\text{s}$ @ 24fps)
* $n=8 \to 141 \text{ frames}$ ($6.0\text{s}$ @ 24fps)
* $n=11 \to 192 \text{ frames}$ ($8.0\text{s}$ @ 24fps)

### Step 3: Structured Output Parsing
The script outputs structured JSON metadata upon completion:
```json
{
  "status": "success",
  "preset": "champion",
  "frames": 39,
  "resolution": "640x640",
  "raw_output": "outputs/raw_champion_39f_TIMESTAMP.mp4",
  "master_output": "outputs/master_champion_39f_TIMESTAMP.mp4",
  "audio_spec": "48000Hz stereo AAC (-14 LUFS EBU R128)",
  "metrics": {
    "gpu_denoise_sec": 12.55,
    "vae_decode_sec": 9.88,
    "total_latency_sec": 44.92,
    "energy_joules": 1211.4
  }
}
```

---

## 🧠 Environment & Subagent Tool Equipping

To equip Hermes or subagents with this skill:
1. Place the repository in the agent's active workspace or skills folder (`.agents/skills/h3mlx` or `~/.agents/skills/h3mlx`).
2. The agent reads `SKILL.md` and discovers available execution binaries (`./h3_master_cli.sh`, `./h3`).
3. The agent can trigger generation, inspect outputs via `ffmpeg`, and report status back to user conversations seamlessly.
