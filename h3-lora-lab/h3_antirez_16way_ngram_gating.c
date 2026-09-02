#include "h3_antirez_16way_ngram_gating.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// 16 orthogonal primes for 16-way independent hashing
static const uint64_t ORTHOGONAL_PRIMES[H3_NGRAM_NUM_HASH_WAYS] = {
    31ULL, 37ULL, 41ULL, 43ULL, 47ULL, 53ULL, 59ULL, 61ULL,
    67ULL, 71ULL, 73ULL, 79ULL, 83ULL, 89ULL, 97ULL, 101ULL
};

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void h3_antirez_ngram_init(H3Antirez16WayNGramTable *table) {
    if (!table) return;
    memset(table, 0, sizeof(*table));

    for (int way = 0; way < H3_NGRAM_NUM_HASH_WAYS; way++) {
        table->hash_primes[way] = ORTHOGONAL_PRIMES[way];
        size_t slice_bytes = (size_t)H3_NGRAM_TABLE_ENTRIES * H3_NGRAM_SLICE_DIM * sizeof(float);
        table->table_slices[way] = (float *)malloc(slice_bytes);
        if (table->table_slices[way]) {
            // Initialize with structured pseudo-random normalized orthogonal bases
            for (size_t i = 0; i < (size_t)H3_NGRAM_TABLE_ENTRIES * H3_NGRAM_SLICE_DIM; i++) {
                float val = ((float)(rand() % 2000) - 1000.0f) / 10000.0f;
                table->table_slices[way][i] = val;
            }
        }
    }
    table->initialized = true;
}

void h3_antirez_ngram_free(H3Antirez16WayNGramTable *table) {
    if (!table) return;
    for (int way = 0; way < H3_NGRAM_NUM_HASH_WAYS; way++) {
        if (table->table_slices[way]) {
            free(table->table_slices[way]);
            table->table_slices[way] = NULL;
        }
    }
    table->initialized = false;
}

void h3_antirez_ngram_async_lookup(
    const H3Antirez16WayNGramTable *table,
    uint32_t token_id_prev2,
    uint32_t token_id_prev1,
    uint32_t token_id_curr,
    H3Layer2GatingContext *ctx
) {
    if (!table || !table->initialized || !ctx) return;

    // 1. Compute 16 distinct hash coordinates for Bigram and Trigram
    for (int way = 0; way < H3_NGRAM_NUM_HASH_WAYS; way++) {
        uint64_t prime = table->hash_primes[way];
        
        // Order-preserving hash polynomial: (w_{t-2}*P^2 + w_{t-1}*P + w_t)
        uint64_t h = (uint64_t)token_id_curr + 
                     ((uint64_t)token_id_prev1 * prime) + 
                     ((uint64_t)token_id_prev2 * prime * prime);
        
        // Avalanche bit-mixing
        h ^= h >> 16;
        h *= 0x85ebca6bULL;
        h ^= h >> 13;
        h *= 0xc2b2ae35ULL;
        h ^= h >> 16;

        uint32_t slot = (uint32_t)(h % H3_NGRAM_TABLE_ENTRIES);

        // 2. Fetch 160-dim slice into reconstructed 2560-dim output vector
        float *dest = &ctx->fetched_ngram_vector[way * H3_NGRAM_SLICE_DIM];
        const float *src = &table->table_slices[way][(size_t)slot * H3_NGRAM_SLICE_DIM];
        memcpy(dest, src, H3_NGRAM_SLICE_DIM * sizeof(float));
    }

    ctx->lookup_ready = true;
}

void h3_antirez_ngram_layer2_inject_gating(
    const H3Layer2GatingContext *ctx,
    const float *layer1_hidden_state,
    float *layer2_hidden_state,
    size_t embed_dim,
    float gate_bias
) {
    if (!ctx || !ctx->lookup_ready || !layer1_hidden_state || !layer2_hidden_state) return;

    size_t dim = embed_dim < H3_NGRAM_EMBED_DIM ? embed_dim : H3_NGRAM_EMBED_DIM;
    float scale = 1.0f / sqrtf((float)dim);

    float dot_sim = 0.0f;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // 4-way unrolled ARM NEON SIMD dot product
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec3 = vdupq_n_f32(0.0f);

    size_t simd_limit = dim & ~15UL;
    for (size_t i = 0; i < simd_limit; i += 16) {
        float32x4_t h0 = vld1q_f32(&layer1_hidden_state[i + 0]);
        float32x4_t v0 = vld1q_f32(&ctx->fetched_ngram_vector[i + 0]);
        sum_vec0 = vfmaq_f32(sum_vec0, h0, v0);

        float32x4_t h1 = vld1q_f32(&layer1_hidden_state[i + 4]);
        float32x4_t v1 = vld1q_f32(&ctx->fetched_ngram_vector[i + 4]);
        sum_vec1 = vfmaq_f32(sum_vec1, h1, v1);

        float32x4_t h2 = vld1q_f32(&layer1_hidden_state[i + 8]);
        float32x4_t v2 = vld1q_f32(&ctx->fetched_ngram_vector[i + 8]);
        sum_vec2 = vfmaq_f32(sum_vec2, h2, v2);

        float32x4_t h3 = vld1q_f32(&layer1_hidden_state[i + 12]);
        float32x4_t v3 = vld1q_f32(&ctx->fetched_ngram_vector[i + 12]);
        sum_vec3 = vfmaq_f32(sum_vec3, h3, v3);
    }
    float32x4_t sum_comb = vaddq_f32(vaddq_f32(sum_vec0, sum_vec1), vaddq_f32(sum_vec2, sum_vec3));
    dot_sim = vaddvq_f32(sum_comb);

    for (size_t i = simd_limit; i < dim; i++) {
        dot_sim += layer1_hidden_state[i] * ctx->fetched_ngram_vector[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dot_sim += layer1_hidden_state[i] * ctx->fetched_ngram_vector[i];
    }
#endif

    float gate_val = fast_sigmoid((dot_sim * scale) + gate_bias);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // 4-way unrolled ARM NEON SIMD Gated Residual Injection: h_2 = h_1 + gate * v_ngram
    float32x4_t gate_vec = vdupq_n_f32(gate_val);
    for (size_t i = 0; i < simd_limit; i += 16) {
        float32x4_t h0 = vld1q_f32(&layer1_hidden_state[i + 0]);
        float32x4_t v0 = vld1q_f32(&ctx->fetched_ngram_vector[i + 0]);
        vst1q_f32(&layer2_hidden_state[i + 0], vfmaq_f32(h0, gate_vec, v0));

        float32x4_t h1 = vld1q_f32(&layer1_hidden_state[i + 4]);
        float32x4_t v1 = vld1q_f32(&ctx->fetched_ngram_vector[i + 4]);
        vst1q_f32(&layer2_hidden_state[i + 4], vfmaq_f32(h1, gate_vec, v1));

        float32x4_t h2 = vld1q_f32(&layer1_hidden_state[i + 8]);
        float32x4_t v2 = vld1q_f32(&ctx->fetched_ngram_vector[i + 8]);
        vst1q_f32(&layer2_hidden_state[i + 8], vfmaq_f32(h2, gate_vec, v2));

        float32x4_t h3 = vld1q_f32(&layer1_hidden_state[i + 12]);
        float32x4_t v3 = vld1q_f32(&ctx->fetched_ngram_vector[i + 12]);
        vst1q_f32(&layer2_hidden_state[i + 12], vfmaq_f32(h3, gate_vec, v3));
    }
    for (size_t i = simd_limit; i < dim; i++) {
        layer2_hidden_state[i] = layer1_hidden_state[i] + (gate_val * ctx->fetched_ngram_vector[i]);
    }
#else
    for (size_t i = 0; i < dim; i++) {
        layer2_hidden_state[i] = layer1_hidden_state[i] + (gate_val * ctx->fetched_ngram_vector[i]);
    }
#endif
}

void h3_antirez_dit_patch_ngram_gate(
    const float *prev_velocity,
    const float *curr_velocity,
    float *interpolated_velocity,
    size_t num_elements,
    float threshold
) {
    if (!prev_velocity || !curr_velocity || !interpolated_velocity) return;

    double diff = 0.0;
    double ref = 0.0;
    size_t stride = 32;

    for (size_t i = 0; i < num_elements; i += stride) {
        diff += fabsf(curr_velocity[i] - prev_velocity[i]);
        ref += fabsf(prev_velocity[i]);
    }

    float delta = ref > 1e-5 ? (float)(diff / ref) : 0.0f;

    if (delta < threshold) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t c_prev = vdupq_n_f32(0.7f);
        float32x4_t c_curr = vdupq_n_f32(0.3f);
        size_t simd_limit = num_elements & ~15UL;

        for (size_t i = 0; i < simd_limit; i += 16) {
            float32x4_t p0 = vld1q_f32(&prev_velocity[i + 0]);
            float32x4_t c0 = vld1q_f32(&curr_velocity[i + 0]);
            vst1q_f32(&interpolated_velocity[i + 0], vfmaq_f32(vmulq_f32(p0, c_prev), c0, c_curr));

            float32x4_t p1 = vld1q_f32(&prev_velocity[i + 4]);
            float32x4_t c1 = vld1q_f32(&curr_velocity[i + 4]);
            vst1q_f32(&interpolated_velocity[i + 4], vfmaq_f32(vmulq_f32(p1, c_prev), c1, c_curr));

            float32x4_t p2 = vld1q_f32(&prev_velocity[i + 8]);
            float32x4_t c2 = vld1q_f32(&curr_velocity[i + 8]);
            vst1q_f32(&interpolated_velocity[i + 8], vfmaq_f32(vmulq_f32(p2, c_prev), c2, c_curr));

            float32x4_t p3 = vld1q_f32(&prev_velocity[i + 12]);
            float32x4_t c3 = vld1q_f32(&curr_velocity[i + 12]);
            vst1q_f32(&interpolated_velocity[i + 12], vfmaq_f32(vmulq_f32(p3, c_prev), c3, c_curr));
        }
        for (size_t i = simd_limit; i < num_elements; i++) {
            interpolated_velocity[i] = (prev_velocity[i] * 0.7f) + (curr_velocity[i] * 0.3f);
        }
#else
        for (size_t i = 0; i < num_elements; i++) {
            interpolated_velocity[i] = (prev_velocity[i] * 0.7f) + (curr_velocity[i] * 0.3f);
        }
#endif
    } else {
        memcpy(interpolated_velocity, curr_velocity, num_elements * sizeof(float));
    }
}
