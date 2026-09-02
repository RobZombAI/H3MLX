# 📊 H3MLX Performance, Quality & Ecological Benchmarks
## Pulp Fiction Twist Dance Benchmark Suite: 5 Golden Presets on Apple Silicon M5 Max

Questo documento contiene i dati di benchmarking empirici ufficiali misurati dal vivo su **Apple Silicon (M5 Max · 128GB Unified Memory)**, testando la celebre scena del **ballo twist di Pulp Fiction (Vincent Vega e Mia Wallace al Jack Rabbit Slim's)** attraverso i **5 Golden Presets** di H3MLX.

---

## 🎬 1. I 5 Video e Animazioni GIF del Benchmark Live

### 👑 1. `H3MLX Champion 4s` (768x512 · 14 Step PDD · 90 Frame · 3.75s)
> **Prompt**: *"Quentin Tarantino 35mm film cinema master, Vincent Vega and Mia Wallace dancing the twist contest on Jack Rabbit Slim's diner floor, bare feet, vintage retro 50s diner neon signs and trophy stage in background, rich Kodak 5219 film grain, soft warm amber cinematic lighting"*
> **Tempo Reale**: **`82.54 s`** (1.09 FPS) | **Qualità Forense**: **`91.33 / 100` (Tier 2 Cinema Gold)**

![H3MLX Champion 4s Dance](assets/pulp_fiction_5_presets/01_h3mlx_champion_4s_dance.gif)

---

### ⚡ 2. `H3MLX Turbo Fast 2s` (512x512 · 8 Step INT8 · 48 Frame · 2.0s)
> **Prompt**: *"Quentin Tarantino 35mm film cinema master, Vincent Vega and Mia Wallace dancing the twist contest on Jack Rabbit Slim's diner floor, bare feet, vintage retro 50s diner neon signs"*
> **Tempo Reale**: **`31.01 s`** (**`1.55 FPS`**) | **Qualità Forense**: **`94.79 / 100` (Tier 1 Platinum Fast 🏆)**

![H3MLX Turbo Fast 2s Dance](assets/pulp_fiction_5_presets/02_h3mlx_turbo_fast_2s_dance.gif)

---

### 🎬 3. `H3MLX Cinema 4K Master` (864x480 $\to$ 4K UHD · 14 Step PDD · 90 Frame · 3.75s)
> **Prompt**: *"Quentin Tarantino 35mm widescreen cinema master, Vincent Vega and Mia Wallace dancing the twist contest, shallow depth of field, anamorphic Panavision lens flare, 4K UHD Master"*
> **Tempo Reale**: **`94.94 s`** (0.95 FPS) | **Qualità Forense**: **`90.31 / 100` (Tier 1 Platinum 4K)**

![H3MLX Cinema 4K Master Dance](assets/pulp_fiction_5_presets/03_h3mlx_cinema_4k_master_dance.gif)

---

### 💃 4. `Antirez Canonical 8-Step` (768x512 · 8 Step BF16 · 73 Frame · 3.0s)
> **Prompt**: *"Quentin Tarantino 35mm film cinema master, Vincent Vega and Mia Wallace dancing the twist contest on Jack Rabbit Slim's diner floor"*
> **Tempo Reale**: **`66.59 s`** (1.10 FPS) | **Qualità Forense**: **`79.80 / 100` (Baseline Standard)**

![Antirez Canonical 8-Step Dance](assets/pulp_fiction_5_presets/04_antirez_canonical_8step_dance.gif)

---

### 🌿 5. `Studio Ghibli Aesthetic` (768x512 · 14 Step DPM3M · 90 Frame · 3.75s)
> **Prompt**: *"Studio Ghibli aesthetic watercolor master, Vincent Vega and Mia Wallace dancing the twist gracefully in a cozy vintage diner under warm glowing lanterns, soft hand-drawn textures, lush animated lighting"*
> **Tempo Reale**: **`92.81 s`** (0.97 FPS) | **Qualità Forense**: **`92.75 / 100` (Tier 1 Anime Master)**

![Studio Ghibli Aesthetic Dance](assets/pulp_fiction_5_presets/05_studio_ghibli_aesthetic_dance.gif)

---

## 📈 2. Grafico Comparativo Ufficiale (4 Pannelli ad Alta Risoluzione)

![Grafico Ufficiale Pulp Fiction 5 Presets](assets/pulp_fiction_5_presets_chart.png)

---

## ⚡ 3. Tabella Dati Empirici Misurati dal Vivo

| # | Preset Testato | Risoluzione Latenti | Frames | Tempo Totale Reale | Throughput GPU | Qualità Forense Severa (0-100) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 👑 **H3MLX Champion 4s** | $768\times512$ | 90f | **`82.54 s`** | `1.09 FPS` | **`91.33 / 100` (Gold)** |
| 2 | ⚡ **H3MLX Turbo Fast 2s** | $512\times512$ | 48f | **`31.01 s`** | **`1.55 FPS`** | **`94.79 / 100` (Platinum 🏆)** |
| 3 | 🎬 **H3MLX Cinema 4K Master** | $864\times480 \to 4\text{K}$ | 90f | **`94.94 s`** | `0.95 FPS` | **`90.31 / 100` (Platinum 4K)** |
| 4 | 💃 **Antirez Canonical 8-Step** | $768\times512$ | 73f | **`66.59 s`** | `1.10 FPS` | **`79.80 / 100` (Baseline)** |
| 5 | 🌿 **Studio Ghibli Aesthetic** | $768\times512$ | 90f | **`92.81 s`** | `0.97 FPS` | **`92.75 / 100` (Anime Master)** |

---

## ⚠️ 4. Hardware Safety & Thermal Fan Alert

> [!CAUTION]
> **AVVISO IMPORTANTE SULLA DISSIPAZIONE TERMICA**:
> L'esecuzione di H3MLX a piena banda unificata (>400 GB/s) su Apple Silicon (consigliato **MacBook Pro 16" M5 Max / Ultra**) sfrutta al 100% i core GPU e la SRAM.
> **UTILIZZARE SEMPRE CON LE VENTOLE ACCESE** impostate al massimo (es. *High Power Mode*, *TG Pro*, *Macs Fan Control*). Eseguire carichi video prolungati senza ventilazione attiva rischia di causare thermal throttling e deterioramento dell'hardware.

---

## 🌍 5. Il Manifesto Ecologico: Perché l'AI Locale Salva i Fiumi

$$\text{Energia per Video (kWh)} = \frac{\text{Potenza (Watt)} \times \text{Tempo (Secondi)}}{3600}$$

| Piattaforma | Potenza Media | Tempo Video 4s 4K | Energia Assorbita | $\text{CO}_2$ Emessa | Consumo Acqua Evaporativa Data Center |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cluster Cloud ($8\times \text{H100}$)** | `6.400 W` | `240 s` (Coda + Gen) | `0,426 kWh` | `~180 g` | `~1,5 Litri d'Acqua / Video` 💧 |
| **Apple Silicon M5 Max (H3MLX)** | **`65 W`** | **`82,54 s`** | **`0,00149 kWh`** | **`< 0,6 g`** | **`0,00 Litri (Zero Acqua)` 🌿** |
| **RISPARMIO ECOLOGICO NETTO** | 🟢 **-98.9%** | 🟢 **Locale & Immediato** | 🟢 **-99.65%** | 🟢 **>99.6% Meno CO2** | 🟢 **100% Acqua Risparmiata** |

> **"Più qualità e più velocità = più ottimizzazione = più fiumi salvati."** 🌊
