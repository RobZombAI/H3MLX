#ifndef H3_VAE_NGRAM_SPECULATIVE_H
#define H3_VAE_NGRAM_SPECULATIVE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VAE_HASH_SIZE 65536
#define H3_VAE_TILE_LATENT_CH 16
#define H3_VAE_TILE_LATENT_SIZE 8    // 8x8 latent patch -> 64x64 pixel tile
#define H3_VAE_TILE_RGB_SIZE 64      // 64x64 RGB tile

typedef struct {
    uint64_t tile_hash;
    float rgb_cache[H3_VAE_TILE_RGB_SIZE * H3_VAE_TILE_RGB_SIZE * 3];
    uint32_t last_frame_seen;
    uint32_t hit_count;
} H3VAENGramTile;

typedef struct {
    H3VAENGramTile tiles[H3_VAE_HASH_SIZE];
    uint64_t total_tiles_queried;
    uint64_t total_tiles_cached;
    uint64_t total_tiles_interpolated;
    float similarity_threshold;
} H3VAENGramContext;

// Initialize VAE N-Gram context
void h3_vae_ngram_init(H3VAENGramContext *ctx, float similarity_threshold);

// Reset cache per sequence
void h3_vae_ngram_reset(H3VAENGramContext *ctx);

// Compute 64-bit spatial hash of an 8x8x16 latent tile
uint64_t h3_vae_ngram_hash_tile(const float *latent_tile, int channels, int size);

// Speculatively check if a 64x64 tile can be bypassed or interpolated
bool h3_vae_ngram_query_tile(
    H3VAENGramContext *ctx,
    const float *latent_tile,
    float *output_rgb_tile,
    int tile_x, int tile_y,
    int frame_idx
);

// Update VAE N-Gram cache with ground-truth decoded RGB tile
void h3_vae_ngram_update_tile(
    H3VAENGramContext *ctx,
    const float *latent_tile,
    const float *rgb_tile,
    int frame_idx
);

// Print VAE telemetry statistics
void h3_vae_ngram_print_telemetry(const H3VAENGramContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // H3_VAE_NGRAM_SPECULATIVE_H
