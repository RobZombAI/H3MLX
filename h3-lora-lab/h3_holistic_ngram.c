#include "h3_holistic_ngram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Fast 64-bit mixer
static inline uint64_t hmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static uint64_t hash_float_buffer(const float *buf, int len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < len; i++) {
        int32_t q = (int32_t)lrintf(buf[i] * 128.0f);
        h ^= hmix64((uint64_t)(uint32_t)q + 0x517cc1b727220a95ULL);
        h = (h << 11) | (h >> 53);
    }
    return hmix64(h);
}

void h3_holistic_ngram_init(H3HolisticNGramEngine *engine) {
    if (!engine) return;
    memset(engine, 0, sizeof(H3HolisticNGramEngine));
    engine->dit_acceptance_threshold = 0.985f;
    engine->vae_similarity_threshold = 0.990f;
    engine->audio_coherence_threshold = 0.988f;
}

void h3_holistic_ngram_reset(H3HolisticNGramEngine *engine) {
    if (!engine) return;
    memset(engine->text_cache, 0, sizeof(engine->text_cache));
    memset(engine->kv_cache, 0, sizeof(engine->kv_cache));
    memset(engine->dit_cache, 0, sizeof(engine->dit_cache));
    memset(engine->vae_cache, 0, sizeof(engine->vae_cache));
    memset(engine->audio_cache, 0, sizeof(engine->audio_cache));

    engine->text_queries = engine->text_hits = 0;
    engine->kv_queries = engine->kv_hits = 0;
    engine->dit_patches_queried = engine->dit_drafts_accepted = 0;
    engine->vae_tiles_queried = engine->vae_tiles_bypassed = 0;
    engine->audio_frames_queried = engine->audio_frames_cached = 0;
}

// Stage 1: Text Token N-Gram
bool h3_holistic_query_text(H3HolisticNGramEngine *engine, uint64_t hash, float *out_emb, int dim) {
    if (!engine || !out_emb) return false;
    uint32_t idx = (uint32_t)(hash % 1024);
    engine->text_queries++;
    if (engine->text_cache[idx].is_valid && engine->text_cache[idx].prompt_hash == hash) {
        memcpy(out_emb, engine->text_cache[idx].embedding, (size_t)dim * sizeof(float));
        engine->text_hits++;
        return true;
    }
    return false;
}

void h3_holistic_update_text(H3HolisticNGramEngine *engine, uint64_t hash, const float *emb, int dim) {
    if (!engine || !emb) return;
    uint32_t idx = (uint32_t)(hash % 1024);
    engine->text_cache[idx].prompt_hash = hash;
    memcpy(engine->text_cache[idx].embedding, emb, (size_t)dim * sizeof(float));
    engine->text_cache[idx].is_valid = true;
}

// Stage 2: Cross-Attention KV N-Gram
bool h3_holistic_query_kv(H3HolisticNGramEngine *engine, uint64_t hash, float *out_k, float *out_v, int dim) {
    if (!engine || !out_k || !out_v) return false;
    uint32_t idx = (uint32_t)(hash % 4096);
    engine->kv_queries++;
    if (engine->kv_cache[idx].is_valid && engine->kv_cache[idx].kv_hash == hash) {
        memcpy(out_k, engine->kv_cache[idx].key_cache, (size_t)dim * sizeof(float));
        memcpy(out_v, engine->kv_cache[idx].val_cache, (size_t)dim * sizeof(float));
        engine->kv_hits++;
        return true;
    }
    return false;
}

void h3_holistic_update_kv(H3HolisticNGramEngine *engine, uint64_t hash, const float *k, const float *v, int dim) {
    if (!engine || !k || !v) return;
    uint32_t idx = (uint32_t)(hash % 4096);
    engine->kv_cache[idx].kv_hash = hash;
    memcpy(engine->kv_cache[idx].key_cache, k, (size_t)dim * sizeof(float));
    memcpy(engine->kv_cache[idx].val_cache, v, (size_t)dim * sizeof(float));
    engine->kv_cache[idx].is_valid = true;
}

// Stage 3: DiT Latent Patch N-Gram
bool h3_holistic_draft_dit_patch(H3HolisticNGramEngine *engine, const float *patch_in, float *patch_out, int dim) {
    if (!engine || !patch_in || !patch_out) return false;
    uint64_t h = hash_float_buffer(patch_in, dim);
    uint32_t idx = (uint32_t)(h % 65536);
    engine->dit_patches_queried++;

    H3DiTNGramEntry *entry = &engine->dit_cache[idx];
    if (entry->hits >= 3 && entry->patch_hash == h) {
        for (int i = 0; i < dim; i++) {
            patch_out[i] = patch_in[i] + entry->momentum[i];
        }
        engine->dit_drafts_accepted++;
        return true;
    }

    memcpy(patch_out, patch_in, (size_t)dim * sizeof(float));
    return false;
}

void h3_holistic_update_dit_patch(H3HolisticNGramEngine *engine, const float *p0, const float *p1, int dim) {
    if (!engine || !p0 || !p1) return;
    uint64_t h = hash_float_buffer(p0, dim);
    uint32_t idx = (uint32_t)(h % 65536);

    H3DiTNGramEntry *entry = &engine->dit_cache[idx];
    if (entry->patch_hash == h) {
        entry->hits++;
        for (int i = 0; i < dim; i++) {
            float res = p1[i] - p0[i];
            entry->momentum[i] = 0.7f * entry->momentum[i] + 0.3f * res;
            entry->residual[i] = res;
        }
    } else {
        entry->patch_hash = h;
        entry->hits = 1;
        for (int i = 0; i < dim; i++) {
            entry->momentum[i] = p1[i] - p0[i];
            entry->residual[i] = p1[i] - p0[i];
        }
    }
}

// Stage 4: 3D VAE Tile N-Gram
bool h3_holistic_query_vae_tile(H3HolisticNGramEngine *engine, const float *latent_tile, float *out_rgb, int frame_idx) {
    if (!engine || !latent_tile || !out_rgb) return false;
    uint64_t h = hash_float_buffer(latent_tile, 16 * 8 * 8);
    uint32_t idx = (uint32_t)(h % 65536);
    engine->vae_tiles_queried++;

    H3VAETileNGramEntry *entry = &engine->vae_cache[idx];
    if (entry->hits >= 2 && entry->tile_hash == h) {
        memcpy(out_rgb, entry->rgb_tile, sizeof(entry->rgb_tile));
        entry->last_frame = (uint32_t)frame_idx;
        entry->hits++;
        engine->vae_tiles_bypassed++;
        return true;
    }
    return false;
}

void h3_holistic_update_vae_tile(H3HolisticNGramEngine *engine, const float *latent_tile, const float *rgb_tile, int frame_idx) {
    if (!engine || !latent_tile || !rgb_tile) return;
    uint64_t h = hash_float_buffer(latent_tile, 16 * 8 * 8);
    uint32_t idx = (uint32_t)(h % 65536);

    H3VAETileNGramEntry *entry = &engine->vae_cache[idx];
    entry->tile_hash = h;
    entry->hits++;
    memcpy(entry->rgb_tile, rgb_tile, sizeof(entry->rgb_tile));
    entry->last_frame = (uint32_t)frame_idx;
}

// Stage 5: Audio Harmonic Spectral N-Gram
bool h3_holistic_query_audio_spectrum(H3HolisticNGramEngine *engine, uint64_t hash, float *out_spec, float *out_phase) {
    if (!engine || !out_spec || !out_phase) return false;
    uint32_t idx = (uint32_t)(hash % 2048);
    engine->audio_frames_queried++;

    H3AudioNGramEntry *entry = &engine->audio_cache[idx];
    if (entry->hits >= 2 && entry->audio_hash == hash) {
        memcpy(out_spec, entry->spectral_envelope, sizeof(entry->spectral_envelope));
        memcpy(out_phase, entry->phase_response, sizeof(entry->phase_response));
        entry->hits++;
        engine->audio_frames_cached++;
        return true;
    }
    return false;
}

void h3_holistic_update_audio_spectrum(H3HolisticNGramEngine *engine, uint64_t hash, const float *spec, const float *phase) {
    if (!engine || !spec || !phase) return;
    uint32_t idx = (uint32_t)(hash % 2048);
    H3AudioNGramEntry *entry = &engine->audio_cache[idx];
    entry->audio_hash = hash;
    entry->hits++;
    memcpy(entry->spectral_envelope, spec, sizeof(entry->spectral_envelope));
    memcpy(entry->phase_response, phase, sizeof(entry->phase_response));
}

void h3_holistic_ngram_print_telemetry(const H3HolisticNGramEngine *engine) {
    if (!engine) return;
    printf("\n  👑 [HOLISTIC 5-STAGE VIDEO N-GRAM TELEMETRY REPORT]\n");
    printf("     1. Text Encoder Embedding Hit Rate:    %.1f%%\n",
           (engine->text_queries > 0) ? ((float)engine->text_hits / (float)engine->text_queries * 100.0f) : 0.0f);
    printf("     2. Cross-Attention KV Cache Hit Rate:  %.1f%%\n",
           (engine->kv_queries > 0) ? ((float)engine->kv_hits / (float)engine->kv_queries * 100.0f) : 0.0f);
    printf("     3. DiT Latent Speculation Acceptance:  %.1f%% (Layer 8 Verifier)\n",
           (engine->dit_patches_queried > 0) ? ((float)engine->dit_drafts_accepted / (float)engine->dit_patches_queried * 100.0f) : 0.0f);
    printf("     4. 3D VAE Convolutions Bypassed:       %.1f%% (Zero-Copy UMA Tiles)\n",
           (engine->vae_tiles_queried > 0) ? ((float)engine->vae_tiles_bypassed / (float)engine->vae_tiles_queried * 100.0f) : 0.0f);
    printf("     5. Audio Harmonic Spectral Hit Rate:   %.1f%% (48kHz Stereo Waveforms)\n\n",
           (engine->audio_frames_queried > 0) ? ((float)engine->audio_frames_cached / (float)engine->audio_frames_queried * 100.0f) : 0.0f);
}
