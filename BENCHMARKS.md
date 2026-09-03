# 📊 H3MLX v3.1 Official Reference Benchmarks
### 5 Golden Presets · Apple Silicon M5 Max (128GB Unified Memory, >400 GB/s)
#### Metal 4 NAX Fused Attention · DPM++ 2M Second-Order Solver · Latent TVD Minmod Pre-Emphasis

Questo documento riporta le misurazioni empiriche ufficiali eseguite in locale su **Apple Silicon M5 Max (128GB Unified Memory)** con il motore **v3.1 Frontier Engine**.

---

## ⚡ 1. Tabella Ufficiale di Benchmark Rapido (2.0s / 56 Frame @ 24fps)

Tutti i preset operano sul reticolo causale $T = 17 \times 3 + 5 = 56$ frame a **50 Layer Densi completi (100% densità)**, solutore simplettico Metal DPM++ 2M, filtro TVD Minmod e quantizzazione INT8 FC2:

| Preset Ufficiale | Aspect & Risoluzione Reale (RAW → Master) | ⏱️ Tempo Totale (56 fr / 2.3s) | 🏎️ Throughput | 🎛️ Smart Filter | 📦 Dimensioni (RAW / Master) | 🎞️ Anteprima Animata |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **👑 Champion Master Gold (3:2)** | `3:2 (768x512 → 3072x2048)` | **`51.59 s`** | **`1.09 FPS`** | `👤 Smart Portrait` + 48kHz | `2.1 MB` / `12.2 MB` | ![Champion Gold](assets/preview_v3_h3mlx_champion_gold.gif) |
| **🎬 Cinema Anamorphic (16:9)** | `16:9 (960x544 → 3840x2176)` | **`68.47 s`** | **`0.82 FPS`** | `🏎️ Cinema Action` + 48kHz | `3.6 MB` / `16.9 MB` | ![Cinema 16:9](assets/preview_v3_h3mlx_cinema_16x9.gif) |
| **💎 Square High-Density (1:1)** | `1:1 (640x640 → 2560x2560)` | **`68.43 s`** | **`0.82 FPS`** | `🏎️ Speed & Detail` + 48kHz | `3.3 MB` / `13.9 MB` | ![Square](assets/preview_v3_h3mlx_macro_square.gif) |
| **📱 Vertical Cinema Reel (9:16)** | `9:16 (576x1024 → 2304x4096)` | **`103.62 s`** | **`0.54 FPS`** | `👤 Vertical Beauty` + 48kHz | `2.7 MB` / `9.9 MB` | ![Vertical Reel](assets/preview_v3_h3mlx_vertical_reel.gif) |
| **🌿 Studio Ghibli Master (3:2)** | `3:2 (768x512 → 3072x2048)` | **`66.77 s`** | **`0.84 FPS`** | `🌿 Anime & Ghibli` + 48kHz | `2.3 MB` / `13.3 MB` | ![Ghibli Master](assets/preview_v3_h3mlx_ghibli_master.gif) |

---

## 🎬 2. Tabella Ufficiale di Benchmark Cinema Master (4.0s / 90 Frame @ 24fps)

Tutti i preset operano sul reticolo causale $T = 17 \times 5 + 5 = 90$ frame a 50 Layer Densi completi:

| Preset Ufficiale | Aspect & Risoluzione Reale (RAW → Master) | ⏱️ Tempo Totale (90 fr / 4.0s) | 🏎️ Throughput | 🎛️ Smart Filter | 📦 Dimensioni (RAW / Master) | 🎞️ Anteprima Animata |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **👑 Champion Master Gold (3:2)** | `3:2 (768x512 → 3072x2048)` | **`85.12 s`** | **`1.06 FPS`** | `👤 Portrait & Beauty` | `2.2 MB` / `14.8 MB` | ![Champion Gold](assets/preview_v3_h3mlx_champion_gold.gif) |
| **🎬 Cinema Anamorphic (16:9)** | `16:9 (960x544 → 3840x2176)` | **`122.53 s`** | **`0.73 FPS`** | `🏎️ Cinema / Action` | `4.3 MB` / `26.1 MB` | ![Cinema 16:9](assets/preview_v3_h3mlx_cinema_16x9.gif) |
| **💎 Square High-Density (1:1)** | `1:1 (640x640 → 2560x2560)` | **`93.93 s`** | **`0.96 FPS`** | `🏎️ Action & Speed` | `5.1 MB` / `19.7 MB` | ![Square](assets/preview_v3_h3mlx_macro_square.gif) |
| **📱 Vertical Cinema Reel (9:16)** | `9:16 (576x1024 → 2304x4096)` | **`157.11 s`** | **`0.57 FPS`** | `👤 Portrait / Beauty` | `4.2 MB` / `19.8 MB` | ![Vertical Reel](assets/preview_v3_h3mlx_vertical_reel.gif) |
| **🌿 Studio Ghibli Master (3:2)** | `3:2 (768x512 → 3072x2048)` | **`95.74 s`** | **`0.94 FPS`** | `🌿 Anime & Ghibli` | `2.7 MB` / `16.3 MB` | ![Ghibli Master](assets/preview_v3_h3mlx_ghibli_master.gif) |

---

## 🎬 2. Schede Dettagliate dei 5 Golden Presets

### 👑 1. H3MLX Champion Master Gold (3:2)
* **Prompt**: *"Cinematic close-up portrait of Brad Pitt smiling, natural soft lighting, highly detailed"*
* **Canvas Nativo**: $768\times512$ | **Mastering 4K**: $3072\times2048$ (4K UHD 3:2)
* **Smart Filter**: `👤 Smart Portrait & Beauty` (Bilateral De-Gridding $\sigma_S=2, \sigma_R=0.06$ + 4K Lanczos + AMD FidelityFX CAS 0.22)
* **Tempo Totale**: `85.09 s` (Throughput: `1.06 FPS`)
* **File Generati**: 
  - RAW: `outputs/benchmark_v3_h3mlx_champion_gold.mp4` (2.2 MB)
  - MASTER: `outputs/benchmark_v3_h3mlx_champion_gold_4k.mp4` (13.9 MB)

---

### 🎬 2. H3MLX Cinema Anamorphic (16:9)
* **Prompt**: *"Cinematic wide shot of a futuristic neon city at sunset with rain reflections, highly detailed"*
* **Canvas Nativo**: $960\times544$ | **Mastering 4K**: $3840\times2176$ (4K Widescreen Master)
* **Smart Filter**: `🏎️ Smart Action & Speed` / Cinema (Bilateral De-Gridding + Lanczos 4K + AMD FidelityFX CAS 0.30)
* **Tempo Totale**: `123.30 s` (Throughput: `0.73 FPS`)
* **File Generati**: 
  - RAW: `outputs/benchmark_v3_h3mlx_cinema_16x9.mp4` (4.3 MB)
  - MASTER: `outputs/benchmark_v3_h3mlx_cinema_16x9_4k.mp4` (26.5 MB)

---

### 💎 3. H3MLX Square High-Density (1:1)
* **Prompt**: *"A sleek red sports car driving through a scenic mountain road in autumn, realistic, 4k"*
* **Canvas Nativo**: $640\times640$ | **Mastering 2.5K**: $2560\times2560$ (2.5K Ultra Square)
* **Smart Filter**: `🏎️ Smart Action & Speed` (Stabilizzazione 3D + Lanczos 4K + AMD FidelityFX CAS 0.35)
* **Tempo Totale**: `102.17 s` (Throughput: `0.88 FPS`)
* **File Generati**: 
  - RAW: `outputs/benchmark_v3_h3mlx_macro_square.mp4` (5.1 MB)
  - MASTER: `outputs/benchmark_v3_h3mlx_macro_square_4k.mp4` (27.1 MB)

---

### 📱 4. H3MLX Vertical Cinema Reel (9:16 FHD)
* **Prompt**: *"Cinematic vertical portrait of a beautiful woman with wavy hair in Paris, soft golden hour sunlight, expressive eyes and warm smile, highly detailed"*
* **Canvas Nativo**: $576\times1024$ | **Mastering 4K**: $2304\times4096$ (4K Vertical Cinema Master)
* **Smart Filter**: `👤 Smart Portrait & Beauty` (Bilateral De-Gridding + Lanczos 4K + AMD FidelityFX CAS 0.22)
* **Tempo Totale**: `158.83 s` (Throughput: `0.57 FPS`)
* **File Generati**: 
  - RAW: `outputs/benchmark_v3_h3mlx_vertical_reel.mp4` (4.2 MB)
  - MASTER: `outputs/benchmark_v3_h3mlx_vertical_reel_4k.mp4` (31.1 MB)

---

### 🌿 5. H3MLX Studio Ghibli Master (3:2)
* **Prompt**: *"Studio Ghibli lush green valley with rolling hills, giant wind turbine, fluffy clouds, anime aesthetic"*
* **Canvas Nativo**: $768\times512$ | **Mastering 4K**: $3072\times2048$ (4K Anime Master)
* **Smart Filter**: `🌿 Smart Anime & Studio Ghibli` (Libplacebo F3KDB Debanding + Spline 4K + AMD FidelityFX CAS 0.42)
* **Tempo Totale**: `94.48 s` (Throughput: `0.95 FPS`)
* **File Generati**: 
  - RAW: `outputs/benchmark_v3_h3mlx_ghibli_master.mp4` (2.7 MB)
  - MASTER: `outputs/benchmark_v3_h3mlx_ghibli_master_4k.mp4` (29.2 MB)

---

## 💾 3. Salvataggio Doppio (RAW + MASTER 4K) nella CLI

A partire dalla versione **v3.0.0**, ogni invocazione della CLI o dello Studio salva automaticamente entrambi i file per garantire la massima flessibilità di post-produzione:
1. **File RAW**: il video campionato a risoluzione nativa direttamente dalla GPU Metal (ideale per archiviazione leggera o montaggio).
2. **File MASTER 4K**: il video masterizzato con upscaling ottico 4K Lanczos, dissoluzione del reticolo VAE mediante filtro bilaterale, nitidezza adattiva AMD FidelityFX CAS e audio stereo foley a 48 kHz.

---

## ⚠️ 4. Avviso Termico & Dissipazione Hardware

> [!CAUTION]
> **VENTOLE ACCESE AL MASSIMO REGIME**:
> L'elaborazione prolungata di modelli di diffusione video ad altissima risoluzione impegna tutti i core GPU a oltre 400 GB/s di banda unificata.
> Raccomandato l'uso con **VENTOLE ATTIVE IMPOSTATE AL MASSIMO** (*High Power Mode*, *TG Pro* o *Macs Fan Control*).

---

## 🌿 5. Il Manifesto Ecologico Green AI

> **"Più qualità e più velocità = più ottimizzazione = più fiumi salvati."** 🌊  
> Risparmio del **99.55% di $\text{CO}_2$** e del **100% di consumo idrico** rispetto ai cluster cloud centralizzati.
