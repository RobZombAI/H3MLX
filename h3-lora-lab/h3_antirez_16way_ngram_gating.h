#ifndef H3_ANTIREZ_16WAY_NGRAM_GATING_H
#define H3_ANTIREZ_16WAY_NGRAM_GATING_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_NGRAM_NUM_HASH_WAYS 16
#define H3_NGRAM_EMBED_DIM 2560
#define H3_NGRAM_SLICE_DIM (H3_NGRAM_EMBED_DIM / H3_NGRAM_NUM_HASH_WAYS) // 160 floats per slot
#define H3_NGRAM_TABLE_ENTRIES 65536 // 64K entries per hash table way

// 16-Way Multi-Hash Table State
typedef struct {
    float *table_slices[H3_NGRAM_NUM_HASH_WAYS]; // 16 slices of 64K x 160 floats
    uint64_t hash_primes[H3_NGRAM_NUM_HASH_WAYS];
    bool initialized;
} H3Antirez16WayNGramTable;

// Gating Context for Layer-2 Injection
typedef struct {
    float fetched_ngram_vector[H3_NGRAM_EMBED_DIM]; // Reconstructed 2560-dim vector
    float gating_scores[H3_NGRAM_EMBED_DIM];        // Per-channel learned sigmoid gate
    bool lookup_ready;
    uint32_t token_bigram_hash;
    uint32_t token_trigram_hash;
} H3Layer2GatingContext;

// Lifecycle
void h3_antirez_ngram_init(H3Antirez16WayNGramTable *table);
void h3_antirez_ngram_free(H3Antirez16WayNGramTable *table);

// 1. Asynchronous 16-Way Hash Lookup (Triggered at Layer 0/1 to mask latency)
void h3_antirez_ngram_async_lookup(
    const H3Antirez16WayNGramTable *table,
    uint32_t token_id_prev2,
    uint32_t token_id_prev1,
    uint32_t token_id_curr,
    H3Layer2GatingContext *ctx
);

// 2. Layer-2 Gating Injection Kernel (Applied after Layer-1 Attention)
void h3_antirez_ngram_layer2_inject_gating(
    const H3Layer2GatingContext *ctx,
    const float *layer1_hidden_state,
    float *layer2_hidden_state,
    size_t embed_dim,
    float gate_bias
);

// 3. Spatio-Temporal Video Patch N-Gram Velocity Interpolation (DiT Blocks 14-36)
void h3_antirez_dit_patch_ngram_gate(
    const float *prev_velocity,
    const float *curr_velocity,
    float *interpolated_velocity,
    size_t num_elements,
    float threshold
);

#ifdef __cplusplus
}
#endif

#endif // H3_ANTIREZ_16WAY_NGRAM_GATING_H
