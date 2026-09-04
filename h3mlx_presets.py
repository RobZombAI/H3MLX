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
    "9:16_reel_fhd": (576, 1024),      # Exact 9:16 Mathematical Vertical (2304 latent tokens)
    "9:16_portrait": (576, 1024),      # Vertical Cinema Reel (2304 latent tokens)
    "21:9_ultrawide": (1008, 432)     # Epic Anamorphic Ultra-Widescreen (1701 latent tokens)
}

# Systematic Studio-Quality Video Presets (All Tier 1 Platinum / Forensic Grade)
PRESETS: Dict[str, Dict[str, Any]] = {
    # 1. Champion Master Gold (Standard 3:2 canvas for realistic portraits and cinematography)
    "h3mlx_champion_gold": {
        "name": "Champion Master (3:2)",
        "description": "Standard 3:2 canvas (768x512). 50 dense DiT layers with DPM++ 2M flow matching.",
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
        "upscale_4k": True,
        "prompt": "Cinematic close-up portrait of a person smiling, natural soft lighting, highly detailed"
    },

    # 2. Cinema Anamorphic Widescreen (16:9 widescreen format)
    "h3mlx_cinema_16x9": {
        "name": "Cinema Widescreen (16:9)",
        "description": "Widescreen 16:9 format (960x544) with spatial optical consistency.",
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
        "prompt": "Cinematic wide shot of a futuristic neon city at sunset with rain reflections, highly detailed"
    },

    # 3. Macro High-Density Square (1:1 640x640)
    "h3mlx_macro_square": {
        "name": "Square High-Density (1:1)",
        "description": "High-density square canvas (640x640, 1600 tokens) for fine detail and 1:1 compositions.",
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
        "upscale_4k": True,
        "prompt": "A sleek red sports car driving through a scenic mountain road in autumn, realistic, 4k"
    },

    # 4. Vertical Cinema Reel (9:16 High Definition)
    "h3mlx_vertical_reel": {
        "name": "Vertical Cinema Reel (9:16)",
        "description": "Exact 9:16 vertical ratio (576x1024, 2304 tokens) optimized for mobile fullscreen portraits.",
        "width": 576,
        "height": 1024,
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
        "prompt": "Cinematic vertical portrait of a beautiful woman with wavy hair in Paris, soft golden hour sunlight, expressive eyes and warm smile, highly detailed"
    },

    # 5. Studio Ghibli Aesthetic Master
    "h3mlx_ghibli_master": {
        "name": "Stylized / Anime (3:2)",
        "description": "Hand-painted aesthetic with fluid wind dynamics, soft watercolor textures, and vibrant skies.",
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
        "upscale_4k": True,
        "prompt": "Studio Ghibli lush green valley with rolling hills, giant wind turbine, fluffy clouds, anime aesthetic"
    }
}

# Aliases for backwards compatibility with previous preset names
PRESETS["h3mlx_champion_4s"] = PRESETS["h3mlx_champion_gold"]
PRESETS["h3mlx_livello1"] = PRESETS["h3mlx_champion_gold"]
PRESETS["h3mlx_cinema_4k_master"] = PRESETS["h3mlx_cinema_16x9"]
PRESETS["h3mlx_ghibli_watercolor_4s"] = PRESETS["h3mlx_ghibli_master"]

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
