#!/usr/bin/env python3
"""
h3_spatial_matrix.py - Spatial-Temporal Activity & Kinetic Saliency Matrix Generator
=====================================================================================
Analyzes the scene / anchor frame to partition the canvas into:
  1. Active Dynamic Zones (M >= 0.5): Subject in motion (limbs, torso, face).
  2. Transition Zones (0.15 <= M < 0.5): Motion boundary, shadows, hair envelope.
  3. Static / Completed Zones (M < 0.15): Background architecture, pavement, sky.

Provides:
  - Latent-resolution activity mask [H_lat, W_lat] for DiT selective token guidance.
  - VAE tile activity map for zero-redundancy stationary tile caching.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageFilter

def compute_activity_matrix(
    image_path: str,
    target_width: int = 576,
    target_height: int = 1024,
    output_bin_path: str = "",
    static_threshold: float = 0.15,
    active_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes a normalized spatial activity matrix M in [0.0, 1.0].
    """
    img = Image.open(image_path).convert("RGB")
    if img.size != (target_width, target_height):
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert to float numpy array
    rgb = np.array(img, dtype=np.float32) / 255.0
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    # 1. High-frequency spatial detail / edge energy
    laplacian = np.abs(np.array(img.convert("L").filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0])), dtype=np.float32) / 255.0)

    # 2. Color & Contrast deviation from perimeter (Background Prior)
    # Estimate background distribution from top, left, and right perimeter borders
    border_pixels = np.concatenate([
        rgb[:32, :, :].reshape(-1, 3),          # Top border (sky / buildings)
        rgb[:, :24, :].reshape(-1, 3),          # Left border
        rgb[:, -24:, :].reshape(-1, 3),         # Right border
    ], axis=0)
    bg_mean = np.median(border_pixels, axis=0)
    color_dist = np.linalg.norm(rgb - bg_mean, axis=-1)
    color_dist /= (color_dist.max() + 1e-6)

    # 3. Dynamic Central Subject Prior (vertical reel centering)
    y_coords, x_coords = np.ogrid[:target_height, :target_width]
    center_x = target_width / 2.0
    center_y = target_height * 0.55 # Slightly lower for standing full-body / 3-quarter
    sigma_x = target_width * 0.32
    sigma_y = target_height * 0.38
    spatial_prior = np.exp(-(((x_coords - center_x) ** 2) / (2 * sigma_x ** 2) +
                             ((y_coords - center_y) ** 2) / (2 * sigma_y ** 2)))

    # 4. Fused Saliency Energy
    fused_energy = 0.45 * color_dist + 0.35 * spatial_prior + 0.20 * (laplacian / (laplacian.max() + 1e-6))
    
    # Smooth with morphological / Gaussian relaxation
    fused_img = Image.fromarray((fused_energy * 255.0).astype(np.uint8))
    smoothed = np.array(fused_img.filter(ImageFilter.GaussianBlur(radius=12)), dtype=np.float32) / 255.0

    # Normalize between 0.0 and 1.0
    pmin, pmax = np.percentile(smoothed, 5), np.percentile(smoothed, 95)
    matrix = np.clip((smoothed - pmin) / (pmax - pmin + 1e-6), 0.0, 1.0)

    # Latent resolution (spatial downsampling 8x)
    latent_h = target_height // 8
    latent_w = target_width // 8
    latent_img = Image.fromarray((matrix * 255.0).astype(np.uint8)).resize((latent_w, latent_h), Image.Resampling.BILINEAR)
    latent_matrix = np.array(latent_img, dtype=np.float32) / 255.0

    # VAE Tile Map (assuming typical 512px tiling)
    tile_h = min(512, target_height)
    tile_w = min(512, target_width)
    tiles_y = (target_height + tile_h - 1) // tile_h
    tiles_x = (target_width + tile_w - 1) // tile_w
    tile_activity = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * (target_height // tiles_y)
            y1 = (ty + 1) * (target_height // tiles_y)
            x0 = tx * (target_width // tiles_x)
            x1 = (tx + 1) * (target_width // tiles_x)
            tile_activity[ty, tx] = matrix[y0:y1, x0:x1].mean()

    static_mask = matrix < static_threshold
    active_mask = matrix >= active_threshold
    static_pct = (static_mask.sum() / matrix.size) * 100.0
    active_pct = (active_mask.sum() / matrix.size) * 100.0

    result = {
        "matrix_pixel": matrix,
        "matrix_latent": latent_matrix,
        "tile_activity": tile_activity,
        "static_coverage_pct": float(static_pct),
        "active_coverage_pct": float(active_pct),
        "latent_shape": (latent_h, latent_w),
        "tile_shape": (tiles_y, tiles_x)
    }

    if output_bin_path:
        out_p = Path(output_bin_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        latent_matrix.astype(np.float32).tofile(str(out_p))
        preview_p = out_p.with_suffix(".png")
        Image.fromarray((matrix * 255.0).astype(np.uint8)).save(str(preview_p))
        result["bin_path"] = str(out_p)
        result["preview_png"] = str(preview_p)

    return result

if __name__ == "__main__":
    import sys
    test_img = "outputs/breakdance_level123/anchors/breakdance_l123_anchor_frame.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    res = compute_activity_matrix(test_img, 576, 1024, "outputs/test_matrix/activity.bin")
    print(f"✅ Activity Matrix Computed:")
    print(f"   • Static Coverage:  {res['static_coverage_pct']:.1f}% (Zone già fatte)")
    print(f"   • Active Coverage:  {res['active_coverage_pct']:.1f}% (Soggetto dinamico)")
    print(f"   • Latent Shape:     {res['latent_shape']}")
    print(f"   • Tile Activity:\n{res['tile_activity']}")
    print(f"   • Visual Preview:   {res['preview_png']}")
