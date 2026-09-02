#!/usr/bin/env python3
"""
🎬 H3MLX Systematic High-Quality Presets & Mathematical Lattice Catalog
Exclusively calibrated for Apple Silicon (M1-M5 Max/Ultra) and MiniMax H3.
Systematic Architecture Guarantees:
  - 100% Dense Spatial Sampling (50 Full DiT Blocks, Token Reduction: OFF)
  - Zero-Loss Step Trajectories (Reuse: 1, Exact Neural Flow)
  - Causal Temporal Lattice Invariance: T = 17n + 5 (n >= 1) at 24 fps
  - High-Token Spatial Canvas (All canvas resolutions >= 1500 latent tokens)
"""

from typing import Dict, Any, Tuple

# Canonical Antirez Canvas & High-Token Latent Definitions (No sub-1500 token canvas)
CANONICAL_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "3:2_standard": (768, 512),       # Gold Standard: 1536 latent tokens (Balanced portrait/cinema)
    "16:9_cinema": (960, 544),        # Anamorphic Widescreen (2040 latent tokens)
    "16:9_standard": (864, 480),      # Standard Widescreen (1620 latent tokens)
    "1:1_high_density": (640, 640),   # High-Density Square (1600 latent tokens, fine macro/faces)
    "9:16_portrait": (544, 960),      # Vertical Cinema Reel (2040 latent tokens)
    "21:9_ultrawide": (1008, 432)     # Epic Anamorphic Ultra-Widescreen (1701 latent tokens)
}

# Systematic Studio-Quality Video Presets (All Tier 1 Platinum / Forensic Grade)
PRESETS: Dict[str, Dict[str, Any]] = {
    # 1. Champion Master Gold (The absolute benchmark for realistic portraits and Hollywood cinema)
    "h3mlx_champion_gold": {
        "name": "👑 H3MLX Champion Master Gold (3:2)",
        "description": "Massima fedeltà assoluta: 50 layer densi al 100%, senza potature spaziali, iride e pori sub-pixel.",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": False,
        "int8": True,
        "prompt": "Masterpiece award-winning cinematic close-up portrait of Brad Pitt with sharp detailed blue eyes with iris reflections, natural weathered skin texture with authentic pores, golden hour rim lighting, shot on Arri Alexa with Cooke Anamorphic lens, 4k master"
    },

    # 2. Cinema Anamorphic Widescreen (16:9 Hollywood format)
    "h3mlx_cinema_16x9": {
        "name": "🎬 H3MLX Cinema Anamorphic (16:9)",
        "description": "Formato panoramico widescreen 16:9 (960x544) con ottica Cooke S4/i e coerenza di fase ottica.",
        "width": 960,
        "height": 544,
        "seconds": 3.75,
        "frames": 90,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": False,
        "int8": True,
        "upscale_4k": True,
        "prompt": "Cinematic wide anamorphic shot of a futuristic neon metropolis in heavy rain, reflections in wet asphalt, steam rising from grates, 35mm film grain, 4k 24fps master"
    },

    # 3. Macro High-Density Square (1:1 640x640 - replaces blurry 512x512)
    "h3mlx_macro_square": {
        "name": "💎 H3MLX Macro High-Density (1:1 640x640)",
        "description": "Quadrato ad altissima densità (1600 token) ottimizzato per macro ottiche, gioielli, orologi e dettagli estremi.",
        "width": 640,
        "height": 640,
        "seconds": 3.75,
        "frames": 90,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": False,
        "int8": True,
        "prompt": "Intricate macro close-up of a luxury mechanical tourbillon watch movement, polished steel gears and ruby bearings in motion, dramatic side studio lighting, 8k uhd photorealistic"
    },

    # 4. Vertical Cinema Reel (9:16 High Definition)
    "h3mlx_vertical_reel": {
        "name": "📱 H3MLX Vertical Cinema Reel (9:16)",
        "description": "Cinematografia verticale ad altissima definizione (544x960) per reel e ritratti a figura intera.",
        "width": 544,
        "height": 960,
        "seconds": 3.75,
        "frames": 90,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": False,
        "int8": True,
        "prompt": "Editorial fashion runway full body shot of a graceful model walking confidently in flowing silk haute couture gown, dramatic spotlight, slow motion 24fps master"
    },

    # 5. Studio Ghibli Aesthetic Master
    "h3mlx_ghibli_master": {
        "name": "🌿 H3MLX Studio Ghibli Master (3:2)",
        "description": "Estetica Hayao Miyazaki con dinamica del vento, texture ad acquerello soffici e cieli dipinti a mano.",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": False,
        "int8": True,
        "prompt": "Studio Ghibli style lush green rolling hills with blooming colorful wildflowers, giant wind turbine spinning gently under fluffy summer cumulus clouds, hand-painted watercolor aesthetic"
    },

    # 6. Antirez Canonical Pure Baseline (BF16 Reference)
    "antirez_canonical_bf16": {
        "name": "💃 Antirez Canonical 1:1 Pure (BF16)",
        "description": "Configurazione 1:1 originale di Salvatore Sanfilippo (antirez) in pura virgola mobile BF16 senza quantizzazione.",
        "width": 768,
        "height": 512,
        "seconds": 3.0,
        "frames": 73,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "canonical",
        "solver": "euler",
        "token_reduction": False,
        "int8": False,
        "prompt": "A graceful flamenco dancer in vibrant red dress spinning energetically, dramatic studio spotlighting, highly detailed fabric texture"
    }
}

# Aliases for backwards compatibility with previous preset names
PRESETS["h3mlx_champion_4s"] = PRESETS["h3mlx_champion_gold"]
PRESETS["h3mlx_livello1"] = PRESETS["h3mlx_champion_gold"]
PRESETS["h3mlx_cinema_4k_master"] = PRESETS["h3mlx_cinema_16x9"]
PRESETS["h3mlx_ghibli_watercolor_4s"] = PRESETS["h3mlx_ghibli_master"]
PRESETS["antirez_canonical_8step"] = PRESETS["antirez_canonical_bf16"]

def calculate_canonical_frames(seconds: float, width: int = 768, height: int = 512, *args, **kwargs) -> int:
    """
    Calculate the exact causal temporal lattice frame count: T = 17*n + 5 (n >= 1) at 24 fps.
    Guarantees 100% mathematical temporal synchronization with MiniMax H3 3D VAE.
    """
    raw_frames = max(22, int(round(seconds * 24)))
    n = max(1, int(round((raw_frames - 5) / 17.0)))
    return 17 * n + 5

def get_preset(preset_id: str) -> Dict[str, Any]:
    """Retrieve a preset configuration or raise ValueError if not found."""
    if preset_id not in PRESETS:
        available = ", ".join(list(PRESETS.keys())[:6])
        raise ValueError(f"Unknown preset '{preset_id}'. High-quality presets available: {available}")
    return PRESETS[preset_id].copy()
