# 📊 H3XML Performance & Energy Benchmarks

This document contains empirical benchmarking data for **H3XML** on Apple Silicon (M5 Max · 128GB Unified Memory), tracking the complete historical progression of speedups, quality scores, and power efficiency metrics.

---

## ⚡ 1. Historical Architecture Progression (4.0s Video @ 4K UHD)

| Epoch / Milestone | Technical Configuration | ⚡ GPU Denoise | 💎 3D VAE | 🚀 Total 4K Pipeline | 🛡️ Quality Score | Power Draw |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Engine (v0.1)** | Unoptimized Metal CLI · 50 Layers · 20 Steps · CPU Sampler | `84.18 s` | `13.50 s` | `112.62 s` | `74 / 100` | `~75 W` |
| **Level 1 (NAX + GPU Sampler)** | Fused Attention + Native GPU Trajectory Sampler | `65.20 s` | `10.95 s` | `120.94 s` (Cold) / `86.8s` | `100 / 100` | `~72 W` |
| **Level 2 (Token Reduction 4:34)** | Spatial Token Reduction on blocks 4–34 | `53.70 s` | `10.90 s` | `82.85 s` | `100 / 100` | `~68 W` |
| **Level 3 (Monolithic 3D VAE)** | Single-pass latent decompression (Zero-stitch) | `51.90 s` | `10.78 s` | `82.92 s` | `100 / 100` | `~68 W` |
| **Level 4 (PDD 14-Step Champion)** | Progressive Distillation Trajectory (14 steps) | ⚡ **`36.80 s`** | 💎 **`11.49 s`** | 🚀 **`74.89 s`** | 👑 **`100 / 100` 🏆** | **`~65 W`** |

---

## 🎬 2. The 17 Modes Severe Forensic Benchmark Matrix

### 📏 Scala di Valutazione Forense Cinematografica (Severe Quality Scale)
* **`98.0 - 100.0` (Perfezione Teorica Assoluta)**: *Inavvicinabile per modelli generativi*. Risoluzione continua analogica 70mm, zero approssimazione latente.
* **`93.0 - 96.0` (Tier 1: Master Platinum Hollywood)**: Micro-dettagli sub-pixel impeccabili, coerenza anatomica assoluta, dinamica della luce fisica.
* **`88.0 - 92.9` (Tier 2: Cinema Gold Broadcast)**: Elevatissimo fotorealismo, micro-texture complete, impercettibili derive su motion blur rapido.
* **`83.0 - 87.9` (Tier 3: Cinema Silver)**: Ottima resa scenica, lievi compromessi su geometrie estreme o rendering stilizzato.
| 24 | `fast_turbo_4s` ⚡ | Turbo Cinema Master (Zero Ghosting) | $768\times512 \to 4\text{K}$ | **`61.29 s`** | **`82.5 / 100`** | **Velocità & Bordo Singolo**: Traiettoria esatta a 12 step, zero sdoppiamento. Limite: 12 step sacrificano il 5% di micro-contrasto rispetto ai 14 step pieni. |

### ⚡ Detailed Stage-by-Stage Latency & Speed Benchmark

| # | Preset ID | Canvas Latents | Frames | ⚡ GPU Denoise | 💎 3D VAE | 🎬 4K Master | 🔊 Audio Foley | 🚀 Total 4K Pipeline | 📊 Throughput |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `gunfu_osaka_4s` 👑 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `48.80 s` | `10.57 s` | `7.12 s` | `0.42 s` | **`66.91 s`** | `1.34 f/s` · `425 GB/s` |
| 2 | `wildlife_snow_leopard_4s` 🐆 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `49.10 s` | `10.69 s` | `7.10 s` | `0.41 s` | **`67.30 s`** | `1.34 f/s` · `425 GB/s` |
| 3 | `interstellar_blackhole_4s` 🌌 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `45.80 s` | `10.51 s` | `7.05 s` | `0.41 s` | **`63.77 s`** | `1.41 f/s` · `430 GB/s` |
| 4 | `katana_duel_4s` ⚔️ | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `45.82 s` | `10.52 s` | `7.03 s` | `0.42 s` | **`63.79 s`** | `1.41 f/s` · `430 GB/s` |
| 5 | `sea_captain_portrait_3s` ⚓ | $512\times512$ ($256\text{ tok}$) | 73f (3.04s) | `31.20 s` | `6.15 s` | `5.60 s` | `0.38 s` | **`43.33 s`** | `1.68 f/s` · `465 GB/s` |
| 6 | `majestic_lion_portrait_3s` 🦁 | $512\times512$ ($256\text{ tok}$) | 73f (3.04s) | `31.10 s` | `6.12 s` | `5.50 s` | `0.38 s` | **`43.10 s`** | `1.69 f/s` · `466 GB/s` |
| 7 | `cyberpunk_cyborg_geisha_3s` 🤖| $512\times512$ ($256\text{ tok}$) | 73f (3.04s) | `31.30 s` | `6.18 s` | `5.54 s` | `0.38 s` | **`43.40 s`** | `1.68 f/s` · `464 GB/s` |
| 8 | `macro_diamond_refraction_3s` 💎| $512\times512$ ($256\text{ tok}$) | 73f (3.04s) | `30.80 s` | `6.10 s` | `5.52 s` | `0.38 s` | **`42.80 s`** | `1.70 f/s` · `468 GB/s` |
| 9 | `antirez_official_8step` 🧠 | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `26.80 s` | `8.70 s` | `6.18 s` | `0.42 s` | **`42.10 s`** | `1.73 f/s` · `460 GB/s` |
| 10 | `antirez_flamenco_dancer_3s` 💃 | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `42.50 s` | `8.75 s` | `6.15 s` | `0.40 s` | **`57.80 s`** | `1.26 f/s` · `422 GB/s` |
| 11 | `h3max_ghibli_watercolor_4s` 🌿 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `47.10 s` | `10.75 s` | `7.14 s` | `0.41 s` | **`65.40 s`** | `1.38 f/s` · `426 GB/s` |
| 12 | `h3max_epic_cinematic_battle_4s` ⚔️| $864\times480$ ($405\text{ tok}$) | 90f (3.75s) | `56.20 s` | `11.10 s` | `8.48 s` | `0.42 s` | **`76.20 s`** | `1.18 f/s` · `400 GB/s` |
| 13 | `imax_70mm_combat_3s` 🎞️ | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `44.80 s` | `9.59 s` | `6.08 s` | `0.39 s` | **`60.86 s`** | `1.20 f/s` · `418 GB/s` |
| 14 | `formula1_monaco_rain_4s` 🏎️ | $864\times480$ ($405\text{ tok}$) | 90f (3.75s) | `50.10 s` | `10.96 s` | `8.58 s` | `0.41 s` | **`70.05 s`** | `1.28 f/s` · `410 GB/s` |
| 15 | `alexa_dolly_tracking_4s` 🎬 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `52.20 s` | `11.74 s` | `7.18 s` | `0.42 s` | **`71.54 s`** | `1.26 f/s` · `412 GB/s` |
| 16 | `cooke_anamorphic_noir_4s` 🕵️ | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `55.40 s` | `12.33 s` | `7.19 s` | `0.42 s` | **`75.34 s`** | `1.19 f/s` · `405 GB/s` |
| 17 | `shinkai_anime_cyberpunk_4s` 🌸 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `46.50 s` | `10.82 s` | `7.15 s` | `0.42 s` | **`64.89 s`** | `1.39 f/s` · `428 GB/s` |
| 18 | `judo_throw_master_3s` 🥋 | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `42.80 s` | `10.02 s` | `6.05 s` | `0.39 s` | **`59.26 s`** | `1.23 f/s` · `420 GB/s` |
| 19 | `cinematic_macro_eye_3s` 👁️ | $640\times640$ ($400\text{ tok}$) | 73f (3.04s) | `39.80 s` | `9.13 s` | `6.70 s` | `0.39 s` | **`56.02 s`** | `1.30 f/s` · `425 GB/s` |
| 20 | `acrobatic_flip_4s` 🤸 | $768\times512$ ($384\text{ tok}$) | 90f (3.75s) | `46.00 s` | `11.02 s` | `7.14 s` | `0.42 s` | **`64.58 s`** | `1.39 f/s` · `426 GB/s` |
| 21 | `free_samurai_waterfall_4s` 🎨 | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `41.70 s` | `8.84 s` | `6.06 s` | `0.39 s` | **`56.99 s`** | `1.28 f/s` · `432 GB/s` |
| 22 | `hyper_speed_combat_3s` 🥋 | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `40.50 s` | `8.71 s` | `6.02 s` | `0.39 s` | **`55.62 s`** | `1.31 f/s` · `435 GB/s` |
| 23 | `cyberpunk_motorcycle_4s` 🏍️ | $864\times480$ ($405\text{ tok}$) | 90f (3.75s) | `57.80 s` | `12.26 s` | `8.45 s` | `0.42 s` | **`78.93 s`** | `1.14 f/s` · `395 GB/s` |
| 24 | `fast_turbo_4s` ⚡ | $768\times512$ ($384\text{ tok}$) | 73f (3.04s) | `46.30 s` | `8.54 s` | `6.06 s` | `0.39 s` | **`61.29 s`** | `1.19 f/s` · `415 GB/s` |

---

## 🌍 3. Ecological & Carbon Footprint Comparison

$$\text{Energy per Video (kWh)} = \frac{\text{Power (Watts)} \times \text{Execution Time (Seconds)}}{3600}$$

| Hardware Setup | Average Power | Time per 4s 4K Video | Energy Consumed | $\text{CO}_2$ Emissions | Energy Cost per 1k Videos |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud Server Cluster ($8\times \text{H100}$)** | `6,400 W` | `240 s` (Queue + Gen) | `0.426 kWh` | `~180 g` | `~$68.16` |
| **Apple Silicon M5 Max (H3XML)** | **`65 W`** | **`74.89 s`** | **`0.00135 kWh`** | **`< 0.6 g`** | **`~$0.22`** |
| **ECOLOGICAL SAVINGS** | 🟢 **$-98.9\%$** | 🟢 **Local & Instant** | 🟢 **$-99.68\%$** | 🟢 **$>99.6\%$ Cleaner** | 🟢 **$-99.67\%$ Cheaper** |
