#ifndef H3_HOLISTIC_NGRAM_H
#define H3_HOLISTIC_NGRAM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_HOLISTIC_TABLE_SIZE 131072 // 128k slots unified hash table
#define H3_AUDIO_SPECTRUM_BINS 256

// --- Stage 1: Text & Prompt N-Gram Cache ---
typedef struct {
    uint64_t prompt_hash;
    float embedding[4096];
    bool is_valid;
} H3TextNGramEntry;

// --- Stage 2: Cross-Attention KV Cache ---
typedef struct {
    uint64_t kv_hash;
    float key_cache[1024];
    float val_cache[1024];
    bool is_valid;
} H3CrossAttnNGramEntry;

// --- Stage 3: DiT Latent Speculation ---
typedef struct {
    uint64_t patch_hash;
    float momentum[16];
    float residual[16];
    uint32_t hits;
} H3DiTNGramEntry;

// --- Stage 4: 3D VAE Spatial/Temporal Tile Cache ---
typedef struct {
    uint64_t tile_hash;
    float rgb_tile[64 * 64 * 3];
    uint32_t last_frame;
    uint32_t hits;
} H3VAETileNGramEntry;

// --- Stage 5: Audio Harmonic Spectrum Cache ---
typedef struct {
    uint64_t audio_hash;
    float spectral_envelope[H3_AUDIO_SPECTRUM_BINS];
    float phase_response[H3_AUDIO_SPECTRUM_BINS];
    uint32_t hits;
} H3AudioNGramEntry;

// --- Unified Holistic Master Context ---
typedef struct {
    H3TextNGramEntry text_cache[1024];
    H3CrossAttnNGramEntry kv_cache[4096];
    H3DiTNGramEntry dit_cache[65536];
    H3VAETileNGramEntry vae_cache[65536];
    H3AudioNGramEntry audio_cache[2048];

    // Global Telemetry Counters
    uint64_t text_queries, text_hits;
    uint64_t kv_queries, kv_hits;
    uint64_t dit_patches_queried, dit_drafts_accepted;
    uint64_t vae_tiles_queried, vae_tiles_bypassed;
    uint64_t audio_frames_queried, audio_frames_cached;

    float dit_acceptance_threshold;
    float vae_similarity_threshold;
    float audio_coherence_threshold;
} H3HolisticNGramEngine;

// Lifecycle functions
void h3_holistic_ngram_init(H3HolisticNGramEngine *engine);
void h3_holistic_ngram_reset(H3HolisticNGramEngine *engine);

// Stage 1: Text Token N-Gram
bool h3_holistic_query_text(H3HolisticNGramEngine *engine, uint64_t hash, float *out_emb, int dim);
void h3_holistic_update_text(H3HolisticNGramEngine *engine, uint64_t hash, const float *emb, int dim);

// Stage 2: Cross-Attention KV N-Gram
bool h3_holistic_query_kv(H3HolisticNGramEngine *engine, uint64_t hash, float *out_k, float *out_v, int dim);
void h3_holistic_update_kv(H3HolisticNGramEngine *engine, uint64_t hash, const float *k, const float *v, int dim);

// Stage 3: DiT Latent Patch N-Gram
bool h3_holistic_draft_dit_patch(H3HolisticNGramEngine *engine, const float *patch_in, float *patch_out, int dim);
void h3_holistic_update_dit_patch(H3HolisticNGramEngine *engine, const float *p0, const float *p1, int dim);

// Stage 4: 3D VAE Tile N-Gram
bool h3_holistic_query_vae_tile(H3HolisticNGramEngine *engine, const float *latent_tile, float *out_rgb, int frame_idx);
void h3_holistic_update_vae_tile(H3HolisticNGramEngine *engine, const float *latent_tile, const float *rgb_tile, int frame_idx);

// Stage 5: Audio Harmonic Spectral N-Gram
bool h3_holistic_query_audio_spectrum(H3HolisticNGramEngine *engine, uint64_t hash, float *out_spec, float *out_phase);
void h3_holistic_update_audio_spectrum(H3HolisticNGramEngine *engine, uint64_t hash, const float *spec, const float *phase);

// Master Telemetry Report
void h3_holistic_ngram_print_telemetry(const H3HolisticNGramEngine *engine);

#ifdef __cplusplus
}
#endif

#endif // H3_HOLISTIC_NGRAM_H
