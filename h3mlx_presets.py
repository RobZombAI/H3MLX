#!/usr/bin/env python3
"""
🎬 H3MLX Presets & Aspect Ratio Catalog
Co-designed for Salvatore Sanfilippo (antirez) h3.c full compatibility
and H3MLX Metal 4 NAX accelerated engine.
"""

from typing import Dict, Any, Tuple

# Canonical Antirez Canvas & Latent Definitions
CANONICAL_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "16:9_cinema": (864, 480),     # Default canonical antirez resolution
    "16:9_sd": (768, 432),
    "16:9_hd": (1024, 576),
    "16:9_720p": (1280, 720),
    "3:2_standard": (768, 512),    # Most balanced token layout (384 tokens)
    "1:1_square": (512, 512),      # Fast square canvas (256 tokens)
    "1:1_large": (640, 640),       # High-density square (400 tokens)
    "9:16_portrait": (480, 864),   # Vertical mobile format
    "9:16_short": (512, 768),      # Portrait 2:3
    "21:9_anamorphic": (1008, 432) # Ultra-widescreen cinematic
}

# Standard Antirez & H3MLX Video Presets
PRESETS: Dict[str, Dict[str, Any]] = {
    # 1. Antirez Canonical Presets (Pure 1:1 baseline)
    "antirez_canonical_8step": {
        "name": "Antirez Canonical 8-Step",
        "description": "Salvatore Sanfilippo (antirez) official 8-step baseline setting",
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
        "prompt": "A graceful flamenco dancer in red dress spinning energetically, studio lighting, highly detailed"
    },
    "antirez_cinema_standard_20step": {
        "name": "Antirez Cinema Standard 20-Step",
        "description": "Standard 20-step canonical full flow setting",
        "width": 864,
        "height": 480,
        "seconds": 4.0,
        "frames": 90,
        "steps": 20,
        "reuse": 1,
        "layers": 50,
        "mode": "canonical",
        "solver": "euler",
        "token_reduction": False,
        "int8": False,
        "prompt": "Cinematic wide shot of an epic medieval battle on a misty morning, 35mm film grain"
    },
    "antirez_fast_square_2s": {
        "name": "Antirez Fast Square 2s",
        "description": "Quick draft 2.0s 512x512 video generation",
        "width": 512,
        "height": 512,
        "seconds": 2.0,
        "frames": 48,
        "steps": 8,
        "reuse": 1,
        "layers": 50,
        "mode": "canonical",
        "solver": "euler",
        "token_reduction": False,
        "int8": False,
        "prompt": "A cute red panda eating fresh bamboo leaves, macro photography, natural lighting"
    },

    # 2. H3MLX Engine Boosted Presets (Accelerated with NAX, INT8 & Monolithic VAE)
    "h3mlx_champion_4s": {
        "name": "H3MLX Champion 4s (Master Gold)",
        "description": "14-step PDD optimal trajectory with Metal 4 NAX fused kernels & Monolithic 3D VAE",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 14,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": True,
        "int8": True,
        "prompt": "Osaka gunfu neon rooftop sword fight in heavy rain, cinematic shallow depth of field, anamorphic lens flare"
    },
    "h3mlx_turbo_fast_2s": {
        "name": "H3MLX Turbo Fast 2s",
        "description": "Sub-15s ultra fast preview with row-major INT8 and predictive step reuse",
        "width": 512,
        "height": 512,
        "seconds": 2.0,
        "frames": 48,
        "steps": 8,
        "reuse": 2,
        "layers": 45,
        "mode": "boosted",
        "solver": "euler",
        "token_reduction": True,
        "int8": True,
        "prompt": "Cyberpunk high-speed motorcycle pursuit through glowing neon highway, motion blur, sharp focus"
    },
    "h3mlx_cinema_4k_master": {
        "name": "H3MLX Cinema 4K Master",
        "description": "Full 50 layers with Cooke S4/i MTF prompt conditioning and 4K detailer upscaling",
        "width": 864,
        "height": 480,
        "seconds": 4.0,
        "frames": 90,
        "steps": 14,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": True,
        "int8": True,
        "upscale_4k": True,
        "prompt": "Intricate macro close-up of a human eye with galaxy reflections in the iris, 8k uhd photorealistic"
    },
    "h3mlx_ghibli_watercolor_4s": {
        "name": "H3MLX Ghibli Watercolor 4s",
        "description": "Studio Ghibli aesthetic with soft watercolor textures and wind dynamics",
        "width": 768,
        "height": 512,
        "seconds": 3.75,
        "frames": 90,
        "steps": 14,
        "reuse": 1,
        "layers": 50,
        "mode": "boosted",
        "solver": "dpm3m",
        "token_reduction": True,
        "int8": True,
        "prompt": "Studio Ghibli style lush green valley with blooming flowers and wind turbine under fluffy summer clouds"
    }
}

def calculate_canonical_frames(seconds: float, width: int = 768, height: int = 512, *args, **kwargs) -> int:
    """Calculate the canonical frame count using antirez lattice rounding (24 fps)."""
    raw_frames = int(round(seconds * 24))
    if raw_frames <= 22:
        return 22
    elif raw_frames <= 39:
        return 39
    elif raw_frames <= 48:
        return 48
    elif raw_frames <= 73:
        return 73
    elif raw_frames <= 90:
        return 90
    elif raw_frames <= 96:
        return 96
    return raw_frames

def get_preset(preset_id: str) -> Dict[str, Any]:
    """Retrieve a preset configuration or raise ValueError if not found."""
    if preset_id not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_id}'. Available presets: {available}")
    return PRESETS[preset_id].copy()
