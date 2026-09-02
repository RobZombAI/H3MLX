#ifndef H3_VIDEOGRAM_3D_AND_SPECULATIVE_VAE_H
#define H3_VIDEOGRAM_3D_AND_SPECULATIVE_VAE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// 1. 3D SPATIO-TEMPORAL VIDEO-GRAM ENGINE (FRAME-SKIP & OPTICAL FLOW)
// =========================================================================

typedef struct {
    size_t num_tokens_total;
    size_t num_tokens_skipped;
    size_t num_tokens_computed;
    float temporal_sparsity_ratio; // (num_tokens_skipped / num_tokens_total)
    float speedup_multiplier;      // e.g. 1.35x - 1.45x
} H3VideoGram3DStats;

/**
 * Evaluates 3D Spatio-Temporal Video-Gram optical motion vector across frames F_{t-1} -> F_t -> F_{t+1}.
 * For tokens with stationary background or linear optical flow, generates a speculative draft and flags skip_mask.
 */
void h3_videogram_3d_predict_motion(
    const float *latents_curr_frame,     // [C x H x W]
    const float *latents_prev_frame,     // [C x H x W]
    float *latents_next_pred,            // [C x H x W] Output speculative draft
    uint8_t *token_skip_mask,            // [H x W] (1 = Skip DiT, 0 = Compute DiT)
    size_t channels,
    size_t height,
    size_t width,
    float motion_threshold,              // Default ~0.015f
    H3VideoGram3DStats *stats
);

// =========================================================================
// 2. SPECULATIVE VIDEO VAE DECODER ENGINE (SUB-8S TILE INTERPOLATION)
// =========================================================================

typedef struct {
    size_t total_tiles;
    size_t full_decoded_tiles;
    size_t speculative_interpolated_tiles;
    double vae_decode_wall_seconds;
    float vae_acceleration_factor; // e.g. 3.4x (28.9s -> 8.5s)
} H3SpeculativeVAEStats;

/**
 * Speculative Tile-Variance VAE Decoder:
 * Classifies latent spatial tiles (e.g. 2x4 grid at 304px) into Active (face, motion) vs Stationary (background).
 * Stationary tiles use SIMD NEON sub-pixel optical projection; Active tiles run full 3D Transposed Convolutions.
 */
void h3_speculative_vae_tile_classify(
    const float *latent_tile_curr,
    const float *latent_tile_prev,
    size_t tile_elements,
    float variance_threshold,           // Default ~0.008f
    bool *out_requires_full_decode,
    float *out_tile_confidence
);

/**
 * Fast ARM NEON SIMD 128-bit Bilinear/Bicubic Residual Spatial Projection for stationary VAE tiles.
 */
void h3_speculative_vae_simd_project_tile(
    const float *prev_decoded_pixels,
    const float *curr_latent_residual,
    float *out_predicted_pixels,
    size_t pixel_channels,
    size_t pixel_height,
    size_t pixel_width
);

#ifdef __cplusplus
}
#endif

#endif // H3_VIDEOGRAM_3D_AND_SPECULATIVE_VAE_H
