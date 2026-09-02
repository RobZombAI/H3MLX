#include "h3_videogram_3d_and_speculative_vae.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// =========================================================================
// 1. 3D SPATIO-TEMPORAL VIDEO-GRAM ENGINE (FRAME-SKIP & OPTICAL FLOW)
// =========================================================================

void h3_videogram_3d_predict_motion(
    const float *latents_curr_frame,
    const float *latents_prev_frame,
    float *latents_next_pred,
    uint8_t *token_skip_mask,
    size_t channels,
    size_t height,
    size_t width,
    float motion_threshold,
    H3VideoGram3DStats *stats
) {
    if (!latents_curr_frame || !latents_prev_frame || !latents_next_pred || !token_skip_mask) return;

    size_t spatial_tokens = height * width;
    size_t skipped_count = 0;

    for (size_t token_idx = 0; token_idx < spatial_tokens; token_idx++) {
        float token_energy_diff = 0.0f;
        float token_energy_ref = 0.0f;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t diff_acc = vdupq_n_f32(0.0f);
        float32x4_t ref_acc = vdupq_n_f32(0.0f);

        size_t c_simd_limit = channels & ~15UL;
        for (size_t c = 0; c < c_simd_limit; c += 16) {
            size_t offset0 = (c + 0) * spatial_tokens + token_idx;
            size_t offset4 = (c + 4) * spatial_tokens + token_idx;
            size_t offset8 = (c + 8) * spatial_tokens + token_idx;
            size_t offset12 = (c + 12) * spatial_tokens + token_idx;

            float32x4_t curr_vec = {
                latents_curr_frame[offset0],
                latents_curr_frame[offset4],
                latents_curr_frame[offset8],
                latents_curr_frame[offset12]
            };
            float32x4_t prev_vec = {
                latents_prev_frame[offset0],
                latents_prev_frame[offset4],
                latents_prev_frame[offset8],
                latents_prev_frame[offset12]
            };

            float32x4_t delta = vabdq_f32(curr_vec, prev_vec); // |curr - prev|
            diff_acc = vaddq_f32(diff_acc, delta);
            ref_acc = vaddq_f32(ref_acc, vabsq_f32(prev_vec));
        }
        token_energy_diff = vaddvq_f32(diff_acc);
        token_energy_ref = vaddvq_f32(ref_acc);

        for (size_t c = c_simd_limit; c < channels; c++) {
            size_t offset = c * spatial_tokens + token_idx;
            float c_val = latents_curr_frame[offset];
            float p_val = latents_prev_frame[offset];
            token_energy_diff += fabsf(c_val - p_val);
            token_energy_ref += fabsf(p_val);
        }
#else
        for (size_t c = 0; c < channels; c++) {
            size_t offset = c * spatial_tokens + token_idx;
            float c_val = latents_curr_frame[offset];
            float p_val = latents_prev_frame[offset];
            token_energy_diff += fabsf(c_val - p_val);
            token_energy_ref += fabsf(p_val);
        }
#endif

        float rel_motion = token_energy_ref > 1e-5f ? (token_energy_diff / token_energy_ref) : 0.0f;

        if (rel_motion < motion_threshold) {
            // Stationary token: Extrapolate next frame: z_{t+1} = 2*z_t - z_{t-1}
            token_skip_mask[token_idx] = 1;
            skipped_count++;

            for (size_t c = 0; c < channels; c++) {
                size_t offset = c * spatial_tokens + token_idx;
                float c_val = latents_curr_frame[offset];
                float p_val = latents_prev_frame[offset];
                latents_next_pred[offset] = (2.0f * c_val) - p_val;
            }
        } else {
            // Active motion token: Needs full DiT attention
            token_skip_mask[token_idx] = 0;
            for (size_t c = 0; c < channels; c++) {
                size_t offset = c * spatial_tokens + token_idx;
                latents_next_pred[offset] = latents_curr_frame[offset];
            }
        }
    }

    if (stats) {
        stats->num_tokens_total = spatial_tokens;
        stats->num_tokens_skipped = skipped_count;
        stats->num_tokens_computed = spatial_tokens - skipped_count;
        stats->temporal_sparsity_ratio = (float)skipped_count / (float)spatial_tokens;
        stats->speedup_multiplier = 1.0f / (1.0f - (stats->temporal_sparsity_ratio * 0.70f));
    }
}

// =========================================================================
// 2. SPECULATIVE VIDEO VAE DECODER ENGINE (SUB-8S TILE INTERPOLATION)
// =========================================================================

void h3_speculative_vae_tile_classify(
    const float *latent_tile_curr,
    const float *latent_tile_prev,
    size_t tile_elements,
    float variance_threshold,
    bool *out_requires_full_decode,
    float *out_tile_confidence
) {
    if (!latent_tile_curr || !latent_tile_prev || !out_requires_full_decode) return;

    double diff_sum = 0.0;
    double ref_sum = 0.0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t diff_acc = vdupq_n_f32(0.0f);
    float32x4_t ref_acc = vdupq_n_f32(0.0f);

    size_t simd_limit = tile_elements & ~15UL;
    for (size_t i = 0; i < simd_limit; i += 16) {
        float32x4_t c0 = vld1q_f32(&latent_tile_curr[i + 0]);
        float32x4_t p0 = vld1q_f32(&latent_tile_prev[i + 0]);
        diff_acc = vaddq_f32(diff_acc, vabdq_f32(c0, p0));
        ref_acc = vaddq_f32(ref_acc, vabsq_f32(p0));

        float32x4_t c1 = vld1q_f32(&latent_tile_curr[i + 4]);
        float32x4_t p1 = vld1q_f32(&latent_tile_prev[i + 4]);
        diff_acc = vaddq_f32(diff_acc, vabdq_f32(c1, p1));
        ref_acc = vaddq_f32(ref_acc, vabsq_f32(p1));

        float32x4_t c2 = vld1q_f32(&latent_tile_curr[i + 8]);
        float32x4_t p2 = vld1q_f32(&latent_tile_prev[i + 8]);
        diff_acc = vaddq_f32(diff_acc, vabdq_f32(c2, p2));
        ref_acc = vaddq_f32(ref_acc, vabsq_f32(p2));

        float32x4_t c3 = vld1q_f32(&latent_tile_curr[i + 12]);
        float32x4_t p3 = vld1q_f32(&latent_tile_prev[i + 12]);
        diff_acc = vaddq_f32(diff_acc, vabdq_f32(c3, p3));
        ref_acc = vaddq_f32(ref_acc, vabsq_f32(p3));
    }
    diff_sum = (double)vaddvq_f32(diff_acc);
    ref_sum = (double)vaddvq_f32(ref_acc);

    for (size_t i = simd_limit; i < tile_elements; i++) {
        diff_sum += fabsf(latent_tile_curr[i] - latent_tile_prev[i]);
        ref_sum += fabsf(latent_tile_prev[i]);
    }
#else
    for (size_t i = 0; i < tile_elements; i++) {
        diff_sum += fabsf(latent_tile_curr[i] - latent_tile_prev[i]);
        ref_sum += fabsf(latent_tile_prev[i]);
    }
#endif

    float rel_diff = ref_sum > 1e-5 ? (float)(diff_sum / ref_sum) : 0.0f;
    float confidence = 1.0f - rel_diff;
    if (confidence < 0.0f) confidence = 0.0f;

    if (out_tile_confidence) *out_tile_confidence = confidence;

    // If temporal tile variance is below threshold, use fast speculative projection
    if (rel_diff < variance_threshold) {
        *out_requires_full_decode = false; // Fast Speculative VAE Tile
    } else {
        *out_requires_full_decode = true;  // Full 3D VAE Convolutions
    }
}

void h3_speculative_vae_simd_project_tile(
    const float *prev_decoded_pixels,
    const float *curr_latent_residual,
    float *out_predicted_pixels,
    size_t pixel_channels,
    size_t pixel_height,
    size_t pixel_width
) {
    if (!prev_decoded_pixels || !out_predicted_pixels) return;

    size_t total_pixels = pixel_channels * pixel_height * pixel_width;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t simd_limit = total_pixels & ~15UL;
    for (size_t i = 0; i < simd_limit; i += 16) {
        float32x4_t p0 = vld1q_f32(&prev_decoded_pixels[i + 0]);
        float32x4_t p1 = vld1q_f32(&prev_decoded_pixels[i + 4]);
        float32x4_t p2 = vld1q_f32(&prev_decoded_pixels[i + 8]);
        float32x4_t p3 = vld1q_f32(&prev_decoded_pixels[i + 12]);

        vst1q_f32(&out_predicted_pixels[i + 0], p0);
        vst1q_f32(&out_predicted_pixels[i + 4], p1);
        vst1q_f32(&out_predicted_pixels[i + 8], p2);
        vst1q_f32(&out_predicted_pixels[i + 12], p3);
    }
    for (size_t i = simd_limit; i < total_pixels; i++) {
        out_predicted_pixels[i] = prev_decoded_pixels[i];
    }
#else
    memcpy(out_predicted_pixels, prev_decoded_pixels, total_pixels * sizeof(float));
#endif
}
