# 💻 H3MLX Hardware Support & Scaling Matrix
## Supporto Scalabile Completo su Tutta la Famiglia Apple Silicon (M1, M2, M3, M4, M5)

H3MLX è progettato per adattarsi dinamicamente alla memoria unificata (UMA) e al numero di core GPU del tuo Mac, garantendo prestazioni ottimali da MacBook portatili fino ai Mac Studio e Mac Pro Ultra.

---

## 🏗️ 1. Matrice di Scalabilità Hardware UMA

| Categoria Hardware | Modelli di Riferimento | Banda di Memoria | Configurazione Consigliata | Modalità VRAM Ottimale |
| :--- | :--- | :---: | :--- | :--- |
| **Tier 1: Ultra Flagship** | M5 Max (128GB), M2/M3 Ultra (192GB) | `400 - 800 GB/s` | `h3mlx_champion_4s` / `h3mlx_cinema_4k_master` | **UMA Zero-Copy Resident** (Tutti i 50 layer + 3D VAE in VRAM) |
| **Tier 2: High Performance** | M3/M4 Max (64GB - 96GB), M1/M2 Max (64GB) | `300 - 400 GB/s` | `h3mlx_champion_4s` / `768x512` 14 Step | **UMA Zero-Copy** (Piena velocità, 50 layer) |
| **Tier 3: Mid-Range Studio** | M2/M3/M4 Pro (32GB - 48GB) | `150 - 273 GB/s` | `h3mlx_turbo_fast_2s` / `512x512` | **Standard UMA + INT8** |
| **Tier 4: Entry & Lightweight**| M1/M2/M3 Base (16GB - 24GB) | `100 - 150 GB/s` | `512x512` 8 Step con `--ssd-streaming` | **SSD Streaming DiT** (Solo 2 blocchi residenti) |

---

## ⚙️ 2. Flag di Adattamento per Mac con 16GB–36GB di Memoria

Se utilizzi un MacBook Air o un Mac con memoria unificata limitata:
```bash
# Abilita lo streaming dei blocchi DiT da SSD NVMe integrato:
./h3mlx -p "A cute red fox in snow" --width 512 --height 512 --ssd-streaming -o outputs/fox_light.mp4
```

---

## ⚠️ 3. Guida alla Gestione Termica & Ventole (Thermal Safety Guide)

Sui portatili ad alte prestazioni (specialmente **MacBook Pro 16" M5 Max / M4 Max / M3 Max**):
1. **High Power Mode**: Su macOS, apri *Impostazioni di Sistema -> Batteria -> Modalità Energetica* e seleziona **"Prestazioni elevate"** (High Power).
2. **Controllo Forzato delle Ventole**: Consigliamo l'uso di utility come **TG Pro** o **Macs Fan Control** impostando le ventole al 100% prima di sessioni prolungate di generazione video continua.
3. **Perché è vitale**: La GPU Metal e la SRAM interna operano a frequenze di picco consumando fino a 65W; una ventilazione attiva previene il thermal throttling e allunga la vita operativa della scheda logica.

---

## 🎬 4. Supporto Multimodale Completo (T2V, I2V, V2V, Audio)

H3MLX supporta nativamente tutte le modalità di condizionamento:
* **Text-to-Video (T2V)**: `-p "Prompt descrittivo"`
* **Image-to-Video (I2V)**: `--first-frame "immagine.jpg"` (oppure `--i2v`)
* **Interpolation Frame (I2V Loop)**: `--first-frame "start.jpg" --last-frame "end.jpg"`
* **Video Reference Conditioning (V2V)**: `--ref-video "video_guida.mp4"`
* **Audio-Conditioned Motion**: `--ref-audio "traccia.wav"`
* **Speech Overlay Nativo**: `--speech-audio "dialogo.wav"` (mixato in C a 48 kHz senza perdita di sync)
