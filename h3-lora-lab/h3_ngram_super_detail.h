#ifndef H3_NGRAM_SUPER_DETAIL_H
#define H3_NGRAM_SUPER_DETAIL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_DETAIL_BANK_SIZE 65536
#define H3_PATCH_DIM 16

// High-Frequency Micro-Texture Descriptor
typedef struct {
    uint64_t signature_hash;
    float high_pass_residual[H3_PATCH_DIM]; // High-frequency spatial Laplacian gradient
    float subpixel_motion_vector[3];        // 3D subpixel flow dx, dy, dt
    float edge_sharpness_score;             // Sharpness factor (0.0 to 2.0)
    uint32_t frequency_occurrences;
} H3DetailPatchEntry;

typedef struct {
    H3DetailPatchEntry bank[H3_DETAIL_BANK_SIZE];
    uint64_t total_injections;
    uint64_t micro_edges_enhanced;
    float global_sharpness_boost;           // e.g. 1.25x sharpness multiplier
    bool enable_temporal_supersampling;    // Multi-frame subpixel anti-aliasing
    bool enable_adaptive_focal_denoise;    // 16-step focal denoise on hands/faces
} H3SuperDetailEngine;

// Lifecycle
void h3_super_detail_init(H3SuperDetailEngine *engine, float sharpness_boost);
void h3_super_detail_reset(H3SuperDetailEngine *engine);

// 1. High-Frequency Micro-Texture Injection
// Injects sub-pixel texture details (pores, fibers, metallic caustics) into latent patches
void h3_super_detail_inject_latent_patch(
    H3SuperDetailEngine *engine,
    float *latent_patch,
    int dim,
    int x, int y, int frame_idx
);

// 2. Multi-Frame Temporal Super-Sampling & Anti-Aliasing (TSSAA)
// Stabilizes ultra-fine lines (eyelashes, gear teeth, steam wisps) across N=5 frames
void h3_super_detail_temporal_supersample(
    H3SuperDetailEngine *engine,
    float *current_frame_latents,
    const float *history_latents,
    int total_elements,
    int window_size
);

// 3. 3D VAE Sub-Pixel Micro-Sharpener & Full Frame Texture Harmonizer
// Applies high-order unsharp Laplacian reconstruction during RGB decode
void h3_super_detail_refine_rgb_tile(
    H3SuperDetailEngine *engine,
    float *rgb_tile_64x64,
    int tile_x, int tile_y
);

void h3_super_detail_refine_rgb_interleaved(
    H3SuperDetailEngine *engine,
    float *rgb,
    int frames,
    int height,
    int width
);

// Telemetry Report
void h3_super_detail_print_telemetry(const H3SuperDetailEngine *engine);

#ifdef __cplusplus
}
#endif

#endif // H3_NGRAM_SUPER_DETAIL_H
