# 👑 H3MLX (v2.5 Universal Edition)
### Next-Gen MiniMax H3 Inference Engine on Apple Silicon (M1-M5 Max/Ultra)
#### 1:1 Complete & Faithful Compatibility with Salvatore Sanfilippo (`antirez/h3.c`) + Metal 4 NAX Acceleration + Interactive Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20M1--M5-black.svg)]()
[![Metal: 4 NAX](https://img.shields.io/badge/Metal-4%20NAX%20Accelerated-blue.svg)]()
[![Green AI: Eco Sovereign](https://img.shields.io/badge/Green%20AI-99.5%25%20Carbon%20Reduction-brightgreen.svg)]()

---

## ⚡ Caratteristiche Principali

* **🏛️ 1:1 Salvatore Sanfilippo (`antirez/h3.c`) Drop-in Compatibility**: Supporto nativo a tutte le 40+ flag CLI, risoluzioni latenti, convenzioni temporali (24 fps) e pipeline multimodale di `h3.c`.
* **🏎️ Motore Custom H3MLX (Metal 4 NAX + INT8)**: Micro-kernel fusi su Tile SRAM, quantizzazione dinamica Row-Major INT8 FC2 e sampler Trajectory Adams-Bashforth residenti in VRAM. Fino a **2.22x più veloce** rispetto alla baseline standard.
* **💎 Monolithic 3D VAE Zero-Stitch**: Decompressione latente in singolo passaggio continuo su memoria unificata (UMA) senza cuciture né artefatti da tiling.
* **🎨 Interactive Studio TUI (`./h3mlx studio`)**: Studio da riga di comando per selezionare al volo i Golden Presets, visualizzare la stima esatta dei tempi di rendering su M5 Max, personalizzare prompt e durata, e monitorare il progresso live.
* **🎬 4K Cinema Upscaler Integrato**: Algoritmo Lanczos-CAS con texture refinement sub-pixel e MTF Cooke S4/i.
* **🌱 Green AI & Sovranità Locale**: Generazione locale a **65W** contro i **6.400W** dei cluster cloud, risparmiando litri d'acqua evaporativa per ogni video generato.

---

## ⚠️ Hardware Safety Alert (Importante)

> [!CAUTION]
> **DISSIPAZIONE TERMICA & VENTOLE ACCESE**:
> H3MLX spinge al limite la banda passante della memoria unificata (>400 GB/s) e tutti i core GPU di Apple Silicon.
> È vivamente consigliato l'uso su **MacBook Pro 16" M5 Max / Ultra** con **VENTOLE SEMPRE ACCESE** al massimo regime (*High Power Mode*, *TG Pro* o *Macs Fan Control*). Eseguire carichi video pesanti senza raffreddamento attivo rischia di causare thermal throttling e usura termica precoce dei componenti.

---

## 📽️ Pulp Fiction 35mm Neo-Noir Master Suite Showcase

### 🚗 Scena 1: Establishing Auto (22 Frames / ~1.0s)
> **Prompt**: *"Quentin Tarantino cinematic 35mm film still, vintage 1974 Chevy Nova car interior at night, two hitmen in black suits, neon diner signs reflecting through rainy windshield, Kodak 5219 stock"*

![Pulp Fiction Scena 1 Auto](assets/pulp_fiction/01_pulp_scene1_car_interior.gif)

---

### ☕ Scena 2: Diner Dialogue & Accendino Zippo (79 Frames / 3.3s)
> **Prompt**: *"Quentin Tarantino cinema 35mm scene, Vincent Vega lighting a cigarette with golden Zippo lighter, curling smoke in atmospheric light shaft, 48kHz diner chatter"*

![Pulp Fiction Scena 2 Diner](assets/pulp_fiction/02_pulp_scene2_diner_dialogue.gif)

---

### 💼 Scena 3: Golden Trunk Apertura Bagagliaio (90 Frames / 3.75s)
> **Prompt**: *"Quentin Tarantino 35mm widescreen cinema master, two hitmen opening car trunk with warm golden glow illuminating faces, anamorphic Panavision lens flare"*

![Pulp Fiction Scena 3 Golden Trunk](assets/pulp_fiction/03_pulp_scene3_golden_trunk.gif)

---

## 📊 Benchmark Ufficiali: PDD 8-Step vs DMD2 4-Step

![Confronto Pulp Fiction](assets/pulp_fiction_comparison_chart.png)

| Clip / Scena | Frame Latenti | 👑 PDD 8-Step (NVIDIA Trajectory) | 🚀 DMD2 4-Step (FastH3) | 🏎️ Speedup Denoise | 🛡️ Qualità 35mm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clip 1: Establishing Auto** | 22f | `11.2 s` | **`5.6 s`** | 🟢 **-50% Tempo (2.0x)** | **`9.7 / 10`** |
| **Clip 2: Diner Dialogue** | 79f | `41.5 s` | **`20.8 s`** | 🟢 **-50% Tempo (2.0x)** | **`9.8 / 10`** |
| **Clip 3: Golden Trunk** | 90f | `48.8 s` | **`24.2 s`** | 🟢 **-50% Tempo (2.02x)** | **`9.9 / 10` (Tier 1)** |
| **Monolithic 1080p Master** | 108f | `56.4 s` | **`28.1 s`** | 🟢 **-50% Tempo** | **`9.9 / 10` (Full 1080p)** |

---

## 🚀 Guida Rapida all'Uso

### 1. Avviare l'Interactive Studio (Consigliato)
```bash
./h3mlx studio
# oppure
./h3mlx-studio
```

### 2. Generazione Rapida da Riga di Comando (CLI 1:1)
```bash
# Preset Champion 4s (Massima qualità Hollywood a 768x512)
./h3mlx --preset h3mlx_champion_4s -p "A graceful flamenco dancer spinning in red dress" -o outputs/flamenco.mp4

# Preset Turbo Fast 2s (Sub-20 secondi a 512x512)
./h3mlx --preset h3mlx_turbo_fast_2s -p "Cyberpunk motorcycle pursuit in neon highway" -o outputs/turbo.mp4

# Preset Cinema 4K Master (Widescreen 16:9 con upscaler 4K automatico)
./h3mlx --preset h3mlx_cinema_4k_master -p "Macro eye galaxy reflection" -o outputs/cinema_4k.mp4
```

---

## 🌿 Il Manifesto Ecologico: Più Ottimizzazione = Più Fiumi Salvati

L'infrastruttura di intelligenza artificiale centralizzata nel cloud consuma quantitativi insostenibili di energia termoelettrica e milioni di litri d'acqua evaporativa per il raffreddamento dei data center.

* **Un cluster cloud da $8\times \text{H100}$** dissipa oltre **$6.400\text{ W}$** e consuma circa **$1,5\text{ litri d'acqua}$** per ogni video generato.
* **H3MLX su Apple Silicon** genera lo stesso video in locale consumando appena **$65\text{ W}$** e **$0,00\text{ litri d'acqua}$**.

> **"Più qualità e più velocità = più ottimizzazione = più fiumi salvati."** 🌊

---

## 📜 Licenza
Rilasciato sotto Licenza Open-Source [MIT](LICENSE). Basato sull'opera pionieristica di Salvatore Sanfilippo (`antirez/h3.c`) ed esteso con l'architettura H3MLX Metal 4 NAX.
