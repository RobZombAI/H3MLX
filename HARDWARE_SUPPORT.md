# H3MLX Hardware Scaling & System Compatibility
## Unified Memory Architecture (UMA) Support across Apple Silicon (M1, M2, M3, M4, M5)

H3MLX adapts dynamically to available Unified Memory (UMA) and GPU core count on Apple Silicon, from portable MacBooks to Mac Studio and Mac Pro workstations.

---

## 🏗️ 1. Hardware Tier Matrix

| Hardware Tier | Target Models | Memory Bandwidth | Recommended Mode | VRAM Strategy |
| :--- | :--- | :---: | :--- | :--- |
| **Tier 1: Ultra Flagship** | M5 Max (128GB), M2/M3 Ultra (192GB) | `400 - 800 GB/s` | `Champion Master (3:2)` / `Cinema 4K` | **Full Resident UMA** (All 50 layers + 3D VAE kept resident) |
| **Tier 2: High Performance** | M3/M4 Max (64GB - 96GB), M1/M2 Max (64GB) | `300 - 400 GB/s` | `Champion Master` / `768x512` 14-Step | **Zero-Copy Resident UMA** (Full 50 layers) |
| **Tier 3: Mid-Range Studio** | M2/M3/M4 Pro (32GB - 48GB) | `150 - 273 GB/s` | `512x512` / `768x512` 8-Step | **Standard UMA with INT8 FC2** |
| **Tier 4: Entry Systems** | M1/M2/M3/M4 Base (16GB - 24GB) | `100 - 150 GB/s` | `512x512` with `--ssd-streaming` | **SSD Block Streaming** (Only 2 active blocks in RAM) |

---

## ⚙️ 2. Execution Flags for Systems with 16GB–36GB Memory

For Mac models with limited unified memory:
```bash
# Stream DiT blocks dynamically from fast NVMe SSD:
./h3mlx -p "A red fox in snow, cinematic" --width 512 --height 512 --ssd-streaming -o outputs/fox_light.mp4
```

---

## ⚠️ 3. Thermal Management & Fan Profiles

Under continuous batch inference, the GPU and memory bus sustain high utilization (~65W package power):
1. **High Power Mode**: On supported MacBook Pro models, enable *System Settings -> Battery -> Energy Mode -> High Power*.
2. **Forced Fan Profiles**: Using utilities such as *TG Pro* or *Macs Fan Control* set to maximum RPM helps maintain sustained boost clock frequencies and prevent thermal throttling during long batch generations.

---

## 🎬 4. Multimodal Conditioning Options

H3MLX supports multimodal conditioning inputs:
* **Text-to-Video (T2V)**: `-p "Descriptive prompt"`
* **Image-to-Video (I2V)**: `--first-frame "image.jpg"` (or `--i2v`)
* **First and Last Frame Interpolation**: `--first-frame "start.jpg" --last-frame "end.jpg"`
* **Reference Video Conditioning (V2V)**: `--ref-video "reference.mp4"`
* **Speech Dialogue Overlay**: `--speech-audio "dialogue.wav"` (synchronized at 48 kHz)
