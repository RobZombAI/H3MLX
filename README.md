# 👑 H3MLX (v3.0.0 Universal Edition)
### Next-Gen MiniMax H3 Inference Engine on Apple Silicon (M1–M5 Max/Ultra)
#### Pure C/Metal 4 NAX Fused Attention · Content-Aware Smart Mastering · Dual Video Output (RAW + 4K Master)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20M1--M5-black.svg)]()
[![Metal: 4 NAX](https://img.shields.io/badge/Metal-4%20NAX%20Accelerated-blue.svg)]()
[![Version: 3.0.0](https://img.shields.io/badge/Version-3.0.0--Universal-blue.svg)]()
[![Green AI: Eco Sovereign](https://img.shields.io/badge/Green%20AI-99.5%25%20Carbon%20Reduction-brightgreen.svg)]()

---

## ⚡ 1. I 5 Golden Presets Ufficiali (Benchmark da 4.0s / 90 Frame @ 24fps)

La Versione 3.0 adotta esclusivamente i **5 Golden Presets ad altissima fedeltà**, ciascuno calibrato matematicamente sul reticolo temporale causale ($T = 17n + 5 = 90$ frame @ 24fps) con **50 Layer Densi completi (100% densità spaziale)**, solutore simplettico DPM++ 3M e quantizzazione dinamica Row-Major INT8 FC2 su Apple Silicon M5 Max:

| Preset Ufficiale | Risoluzione & 4K | ⏱️ Tempo Totale (90 fr / 4.0s) | 🏎️ Throughput | 🎛️ Smart Filter | 📦 Dimensioni (RAW / Master) | 🎞️ Anteprima Animata |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **👑 Champion Master Gold (3:2)** | `768x512 → 3072x2048` | **`85.09 s`** | **`1.06 FPS`** | `👤 Portrait & Beauty` | `2.2 MB` / `13.9 MB` | ![Champion Gold](assets/preview_v3_h3mlx_champion_gold.gif) |
| **🎬 Cinema Anamorphic (16:9)** | `960x544 → 3840x2176` | **`123.30 s`** | **`0.73 FPS`** | `🏎️ Cinema / Action` | `4.3 MB` / `26.5 MB` | ![Cinema 16:9](assets/preview_v3_h3mlx_cinema_16x9.gif) |
| **💎 Square High-Density (1:1)** | `640x640 → 2560x2560` | **`102.17 s`** | **`0.88 FPS`** | `🏎️ Action & Speed` | `5.1 MB` / `27.1 MB` | ![Square](assets/preview_v3_h3mlx_macro_square.gif) |
| **📱 Vertical Cinema Reel (9:16)** | `576x1024 → 2304x4096` | **`158.83 s`** | **`0.57 FPS`** | `👤 Portrait / Beauty` | `4.2 MB` / `31.1 MB` | ![Vertical Reel](assets/preview_v3_h3mlx_vertical_reel.gif) |
| **🌿 Studio Ghibli Master (3:2)** | `768x512 → 3072x2048` | **`94.48 s`** | **`0.95 FPS`** | `🌿 Anime & Ghibli` | `2.7 MB` / `29.2 MB` | ![Ghibli Master](assets/preview_v3_h3mlx_ghibli_master.gif) |

---

## 💾 2. Novità v3.0: Salvataggio Doppio (RAW Nativo + MASTER Smart 4K)

In H3MLX v3.0, ogni generazione da CLI o da Studio genera e conserva **entrambi i file video**:
* 🎬 **Video RAW (Nativo)**: il video non compresso campionato a risoluzione nativa direttamente dalla GPU Metal.
* 💎 **Video MASTER (Smart 4K UHD)**: il master broadcast con de-gridding bilaterale edge-preserving, upscaling ottico 4K Lanczos, sharpening adattivo AMD FidelityFX CAS e traccia audio Foley a 48 kHz.

---

## 🧠 3. Smart Mastering Filter Engine & X-MinimaxH3 Innovations

Integrazione nativa delle migliori tecnologie di post-produzione open-source e della suite algoritmica di **X-MinimaxH3**:
1. **Wavelet Bayesian Denoising (`vaguedenoiser`)**: Scomposizione su 7 piani wavelet con soglia bayesiana Garrote. Elimina completamente il rumore di quantizzazione e la grana del VAE su cieli, pelle e sfondi sfocati.
2. **AMD FidelityFX CAS (Contrast Adaptive Sharpening 0.25)**: GPUOpen MIT. Aumenta la nitidezza locale e il micro-contrasto sub-pixel (iridi, pori, singoli fili d'erba e peli di barba) senza artefatti o aloni bianchi (zero ringing / haloing).
3. **Apple VideoToolbox Hardware 10-Bit (`hevc_videotoolbox` Main 10 `p010le`)**: Mastering 4K a 10-bit con oltre 1.07 miliardi di colori in appena **~3 secondi** grazie ai Media Engine hardware di Apple Silicon, con normalizzazione broadcast EBU R128 a 48 kHz.
4. **Terminal Latent Guard (`h3_terminal_latent_guard.py`)**: Algoritmo statistico MAD (Median Absolute Deviation) per prevenire il collasso energetico nella metà inferiore degli ultimi fotogrammi, tipico della periodicità temporale a 5 fasi del VAE.
5. **Native Latent 3D Upscaler (`h3_latent_upscaler_3d.py`)**: Architettura neurale 3D ResNet/TemporalConv calibrata sui 24 canali latenti di MiniMax H3 per scalare i latenti prima del second sampling DiT.
6. **Structured Prompting Engine (MiMo / Qwen3-VL Protocol)**: Supporto completo nel TUI Studio a dialoghi delimitati `<d>[Lang]...</d>`, speaker IDs `(S1)`, lip-sync safeguards per eliminare movimenti labiali fuori battuta e isolamento `overall_soundscape:`.

---

## 🚀 Guida Rapida Turnkey (Pronto all'Uso)

### 1. Clona ed esegui il setup automatico
```bash
git clone https://github.com/RobZombAI/H3MLX.git
cd H3MLX
./setup.sh
```

### 2. Download pesi (se non presenti)
```bash
./download_models.sh
```

### 3. Genera subito con un Golden Preset
```bash
# Esegui il Champion Master Gold (Brad Pitt) salvando sia RAW che 4K Master:
./h3mlx --preset h3mlx_champion_gold

# Oppure il Vertical Reel 9:16 per Instagram / TikTok:
./h3mlx --preset h3mlx_vertical_reel

# Oppure con un prompt personalizzato e Smart Filter automatico:
./h3mlx -p "Cinematic portrait of a cyberpunk hacker in Tokyo, neon reflections" --preset h3mlx_cinema_16x9
```

### 4. Studio Interattivo
```bash
./h3mlx studio
```

---

## 📊 Report di Velocità della Singola Generazione sulla CLI

Al termine di ogni run, la CLI stampa un report analitico dettagliato:

```text
======================================================================
🎉 GENERAZIONE ALTA FEDELTÀ COMPLETATA CON SUCCESSO!
⏱️  Tempo Totale Reale:       85.09s  (Throughput: 1.06 FPS)
🎬  Video RAW (Nativo 768x512): outputs/video.mp4 (2.20 MB)
💎  Video MASTER (Smart 4K):   outputs/video_4k.mp4 (13.90 MB)
📐  Risoluzione & Frame:      768x512 -> 4K UHD | 90 frames (3.75s @ 24fps)

📊 Profiling GPU Metal & Smart Mastering:
   • denoise_s                : 64.12s
   • vae_decode_s             : 19.85s
======================================================================
```

---

## ⚠️ Avviso Termico & Dissipazione Hardware

> [!CAUTION]
> **VENTOLE ACCESE AL MASSIMO REGIME**:
> L'elaborazione prolungata di modelli di diffusione video ad altissima risoluzione impegna tutti i core GPU a oltre 400 GB/s di banda unificata.
> Raccomandato l'uso con **VENTOLE ATTIVE IMPOSTATE AL MASSIMO** (*High Power Mode*, *TG Pro* o *Macs Fan Control*).

---

## 🌿 Il Manifesto Green AI

$$\text{Energia per Video (kWh)} = \frac{\text{Potenza (Watt)} \times \text{Tempo (Secondi)}}{3600}$$

* **Cluster Cloud ($8\times \text{H100}$)**: `6.400 W` $\times$ `240 s` = `0,426 kWh` | `~180 g CO2` | **~1,5 Litri d'Acqua Evaporativa** 💧
* **Apple Silicon M5 Max (H3MLX)**: `65 W` $\times$ `85,09 s` = `0,00153 kWh` | `< 0,6 g CO2` | **0,00 Litri d'Acqua (Zero Consumo Idrico)** 🌿

> **"Più qualità e più velocità = più ottimizzazione = più fiumi salvati."** 🌊

---

## 📜 Licenza
Rilasciato sotto Licenza Open-Source [MIT](LICENSE). Basato sull'opera pionieristica di Salvatore Sanfilippo (`antirez/h3.c`) ed esteso con l'architettura H3MLX Metal 4 NAX.
