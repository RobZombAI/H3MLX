#include "h3_ngram_super_detail.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Fast MurmurHash3 mixer
static inline uint64_t dmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static uint64_t hash_detail_patch(const float *patch, int dim) {
    uint64_t h = 0x27d4eb2f165667c5ULL;
    for (int i = 0; i < dim; i++) {
        int32_t q = (int32_t)lrintf(patch[i] * 256.0f);
        h ^= dmix64((uint64_t)(uint32_t)q + 0x9e3779b97f4a7c15ULL);
        h = (h << 13) | (h >> 51);
    }
    return dmix64(h);
}

void h3_super_detail_init(H3SuperDetailEngine *engine, float sharpness_boost) {
    if (!engine) return;
    memset(engine, 0, sizeof(H3SuperDetailEngine));
    engine->global_sharpness_boost = (sharpness_boost > 0.0f) ? sharpness_boost : 1.35f;
    engine->enable_temporal_supersampling = true;
    engine->enable_adaptive_focal_denoise = true;
}

void h3_super_detail_reset(H3SuperDetailEngine *engine) {
    if (!engine) return;
    memset(engine->bank, 0, sizeof(engine->bank));
    engine->total_injections = 0;
    engine->micro_edges_enhanced = 0;
}

// Injects sub-pixel micro-texture details into DiT latent patches
void h3_super_detail_inject_latent_patch(
    H3SuperDetailEngine *engine,
    float *latent_patch,
    int dim,
    int x, int y, int frame_idx
) {
    (void)x;
    (void)y;
    (void)frame_idx;
    if (!engine || !latent_patch) return;

    uint64_t h = hash_detail_patch(latent_patch, dim);
    uint32_t idx = (uint32_t)(h % H3_DETAIL_BANK_SIZE);

    H3DetailPatchEntry *entry = &engine->bank[idx];
    engine->total_injections++;

    if (entry->signature_hash == h && entry->frequency_occurrences >= 2) {
        // High-frequency detail injection: sharpen latent edge gradients
        float boost = 0.15f * engine->global_sharpness_boost * entry->edge_sharpness_score;
        for (int i = 0; i < dim; i++) {
            latent_patch[i] += boost * entry->high_pass_residual[i];
        }
        engine->micro_edges_enhanced++;
    } else {
        // Compute Laplacian high-pass spatial gradient
        entry->signature_hash = h;
        entry->frequency_occurrences++;
        entry->edge_sharpness_score = 1.20f;
        for (int i = 0; i < dim; i++) {
            float prev = (i > 0) ? latent_patch[i - 1] : latent_patch[i];
            float next = (i + 1 < dim) ? latent_patch[i + 1] : latent_patch[i];
            entry->high_pass_residual[i] = 2.0f * latent_patch[i] - prev - next;
        }
    }
}

// Multi-Frame Temporal Super-Sampling & Anti-Aliasing (TSSAA)
void h3_super_detail_temporal_supersample(
    H3SuperDetailEngine *engine,
    float *current_frame_latents,
    const float *history_latents,
    int total_elements,
    int window_size
) {
    if (!engine || !engine->enable_temporal_supersampling || !current_frame_latents || !history_latents || window_size <= 1) {
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        float cur = current_frame_latents[i];
        float hist = history_latents[i];
        // Adaptive exponential moving average with edge preservation
        float delta = fabsf(cur - hist);
        if (delta < 0.08f) {
            // Smooth static micro-details to eliminate temporal flicker
            current_frame_latents[i] = 0.75f * cur + 0.25f * hist;
        }
    }
}

// Refines decoded RGB tiles via sub-pixel Laplacian sharpening
void h3_super_detail_refine_rgb_tile(
    H3SuperDetailEngine *engine,
    float *rgb_tile_64x64,
    int tile_x, int tile_y
) {
    (void)tile_x;
    (void)tile_y;
    if (!engine || !rgb_tile_64x64) return;

    float boost = 0.08f * (engine->global_sharpness_boost - 1.0f);
    if (boost <= 0.001f) return;

    int size = 64;
    for (int c = 0; c < 3; c++) {
        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                int idx = c * (size * size) + y * size + x;
                float center = rgb_tile_64x64[idx];
                float top    = rgb_tile_64x64[c * (size * size) + (y - 1) * size + x];
                float bottom = rgb_tile_64x64[c * (size * size) + (y + 1) * size + x];
                float left   = rgb_tile_64x64[c * (size * size) + y * size + (x - 1)];
                float right  = rgb_tile_64x64[c * (size * size) + y * size + (x + 1)];

                float laplacian = 4.0f * center - top - bottom - left - right;
                rgb_tile_64x64[idx] = fminf(1.0f, fmaxf(0.0f, center + boost * laplacian));
            }
        }
    }
}

void h3_super_detail_refine_rgb_interleaved(
    H3SuperDetailEngine *engine,
    float *rgb,
    int frames,
    int height,
    int width
) {
    if (!engine || !rgb || frames <= 0 || height <= 2 || width <= 2) return;

    float boost = 0.14f * engine->global_sharpness_boost;
    if (boost <= 0.001f) return;

    size_t frame_pixels = (size_t)height * (size_t)width;
    size_t frame_stride = frame_pixels * 3;

    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < frames; f++) {
        float *cur_frame = rgb + (size_t)f * frame_stride;
        float *src_copy = malloc(frame_stride * sizeof(float));
        if (!src_copy) continue;
        memcpy(src_copy, cur_frame, frame_stride * sizeof(float));

        for (int y = 2; y < height - 2; y++) {
            for (int x = 2; x < width - 2; x++) {
                size_t c_idx = ((size_t)y * (size_t)width + (size_t)x) * 3;

                // Center RGB & Luminance
                float r0 = src_copy[c_idx];
                float g0 = src_copy[c_idx + 1];
                float b0 = src_copy[c_idx + 2];
                float y0 = 0.299f * r0 + 0.587f * g0 + 0.114f * b0;

                if (y0 < 0.001f) continue;

                // Level 1: 3x3 8-neighborhood luminance
                size_t top_idx = (((size_t)(y - 1) * (size_t)width) + (size_t)x) * 3;
                size_t bot_idx = (((size_t)(y + 1) * (size_t)width) + (size_t)x) * 3;
                size_t lft_idx = (((size_t)y * (size_t)width) + (size_t)(x - 1)) * 3;
                size_t rgt_idx = (((size_t)y * (size_t)width) + (size_t)(x + 1)) * 3;
                size_t tl_idx  = (((size_t)(y - 1) * (size_t)width) + (size_t)(x - 1)) * 3;
                size_t tr_idx  = (((size_t)(y - 1) * (size_t)width) + (size_t)(x + 1)) * 3;
                size_t bl_idx  = (((size_t)(y + 1) * (size_t)width) + (size_t)(x - 1)) * 3;
                size_t br_idx  = (((size_t)(y + 1) * (size_t)width) + (size_t)(x + 1)) * 3;

                float yt = 0.299f * src_copy[top_idx] + 0.587f * src_copy[top_idx + 1] + 0.114f * src_copy[top_idx + 2];
                float yb = 0.299f * src_copy[bot_idx] + 0.587f * src_copy[bot_idx + 1] + 0.114f * src_copy[bot_idx + 2];
                float yl = 0.299f * src_copy[lft_idx] + 0.587f * src_copy[lft_idx + 1] + 0.114f * src_copy[lft_idx + 2];
                float yr = 0.299f * src_copy[rgt_idx] + 0.587f * src_copy[rgt_idx + 1] + 0.114f * src_copy[rgt_idx + 2];
                float ytl = 0.299f * src_copy[tl_idx] + 0.587f * src_copy[tl_idx + 1] + 0.114f * src_copy[tl_idx + 2];
                float ytr = 0.299f * src_copy[tr_idx] + 0.587f * src_copy[tr_idx + 1] + 0.114f * src_copy[tr_idx + 2];
                float ybl = 0.299f * src_copy[bl_idx] + 0.587f * src_copy[bl_idx + 1] + 0.114f * src_copy[bl_idx + 2];
                float ybr = 0.299f * src_copy[br_idx] + 0.587f * src_copy[br_idx + 1] + 0.114f * src_copy[br_idx + 2];

                // Level 2: 5x5 pyramidal luminance for multi-subject contour crispness
                size_t top2_idx = (((size_t)(y - 2) * (size_t)width) + (size_t)x) * 3;
                size_t bot2_idx = (((size_t)(y + 2) * (size_t)width) + (size_t)x) * 3;
                size_t lft2_idx = (((size_t)y * (size_t)width) + (size_t)(x - 2)) * 3;
                size_t rgt2_idx = (((size_t)y * (size_t)width) + (size_t)(x + 2)) * 3;

                float yt2 = 0.299f * src_copy[top2_idx] + 0.587f * src_copy[top2_idx + 1] + 0.114f * src_copy[top2_idx + 2];
                float yb2 = 0.299f * src_copy[bot2_idx] + 0.587f * src_copy[bot2_idx + 1] + 0.114f * src_copy[bot2_idx + 2];
                float yl2 = 0.299f * src_copy[lft2_idx] + 0.587f * src_copy[lft2_idx + 1] + 0.114f * src_copy[lft2_idx + 2];
                float yr2 = 0.299f * src_copy[rgt2_idx] + 0.587f * src_copy[rgt2_idx + 1] + 0.114f * src_copy[rgt2_idx + 2];

                // Multi-Scale Pyramidal Laplacian high-frequency gradient
                float lap1 = 0.55f * (8.0f * y0 - yt - yb - yl - yr - ytl - ytr - ybl - ybr);
                float lap2 = 0.25f * (4.0f * y0 - yt2 - yb2 - yl2 - yr2);
                float lap = lap1 + 0.60f * lap2;

                // Local micro-texture variance
                float max_y = fmaxf(y0, fmaxf(fmaxf(yt, yb), fmaxf(fmaxf(yl, yr), fmaxf(fmaxf(ytl, ytr), fmaxf(ybl, ybr)))));
                float min_y = fminf(y0, fminf(fminf(yt, yb), fminf(fminf(yl, yr), fminf(fminf(ytl, ytr), fminf(ybl, ybr)))));
                float contrast = max_y - min_y;

                // Raised Cosine Luminance Mask G(Y0): soft window between Y = 0.05 and Y = 0.95
                float lum_mask = 0.0f;
                if (y0 >= 0.05f && y0 <= 0.95f) {
                    float norm_y = (y0 - 0.05f) / 0.90f;
                    lum_mask = sinf(3.14159265358979323846f * norm_y);
                    lum_mask *= lum_mask; // Raised cosine ^ 2
                }

                // Anti-ringing coring & highlight/shadow soft limit
                if (lum_mask > 0.01f && contrast > 0.015f && contrast < 0.65f) {
                    /* Higher focal boost for skin pores, irises, catchlights, and sheer fabric */
                    float focal_weight = expf(-powf(contrast - 0.22f, 2.0f) / (2.0f * 0.16f * 0.16f));
                    float adaptive_gain = boost * lum_mask * (0.40f + 0.60f * focal_weight);
                    
                    // Coring: suppress low-amplitude sensor noise
                    float coring_threshold = 0.006f;
                    if (fabsf(lap) > coring_threshold) {
                        float delta_y = adaptive_gain * (lap > 0 ? (lap - coring_threshold) : (lap + coring_threshold));
                        float y_new = fminf(1.0f, fmaxf(0.001f, y0 + delta_y));
                        float ratio = y_new / y0;

                        // Preserve natural chromaticity without color noise
                        cur_frame[c_idx]     = fminf(1.0f, fmaxf(0.0f, r0 * ratio));
                        cur_frame[c_idx + 1] = fminf(1.0f, fmaxf(0.0f, g0 * ratio));
                        cur_frame[c_idx + 2] = fminf(1.0f, fmaxf(0.0f, b0 * ratio));

                        #pragma omp atomic
                        engine->micro_edges_enhanced++;
                    }
                }
                #pragma omp atomic
                engine->total_injections++;
            }
        }
        free(src_copy);
    }
}

void h3_super_detail_print_telemetry(const H3SuperDetailEngine *engine) {
    if (!engine) return;
    printf("\n  💎 [N-GRAM SUPER-DETAIL & MICRO-FIDELITY TELEMETRY]\n");
    printf("     • Total Pixels Processed:           %llu\n", (unsigned long long)engine->total_injections);
    printf("     • Micro-Textures & Pores Enhanced:  %llu\n", (unsigned long long)engine->micro_edges_enhanced);
    printf("     • Optical Sharpness Multiplier:     %.2fx\n\n", engine->global_sharpness_boost);
}
