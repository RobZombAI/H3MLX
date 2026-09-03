#ifndef H3_SPATIOTEMPORAL_H
#define H3_SPATIOTEMPORAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NAX-Spatiotemporal Layout Descriptor
 * Manages the multimodal decomposition for MiniMax-H3 on Apple Silicon:
 * - Global Multimodal Anchors: Text (512) and Audio (512) tokens remain 100% visible
 * - Local Spatiotemporal Video Chunks: Fast AMX batched attention (e.g. 4 frames, 6144 tokens)
 * - Keyframe Anchors: Periodic temporal anchor frames (stride K) for narrative stability
 */
typedef struct {
    uint32_t text_rows;         /* 512 global text prompt tokens */
    uint32_t audio_rows;        /* Global audio tokens */
    uint32_t video_rows;        /* Total video latent tokens */
    uint32_t total_sequence;    /* text_rows + video_rows + audio_rows */
    uint32_t latent_t;          /* Number of temporal latent frames */
    uint32_t spatial_tokens;    /* Tokens per latent frame (e.g. 32x48 = 1536) */
    uint32_t chunk_frames;      /* Frames per local temporal chunk (default: 4) */
    uint32_t chunk_tokens;      /* chunk_frames * spatial_tokens */
    uint32_t num_chunks;        /* (latent_t + chunk_frames - 1) / chunk_frames */
    uint32_t keyframe_stride;   /* Anchor stride across frames (default: 4) */
    int enabled;
} h3_nax_st_layout;

/* Initialize layout descriptor from DiT sequence dimensions */
int h3_nax_st_init(h3_nax_st_layout *layout,
                   uint32_t sequence,
                   uint32_t text_rows,
                   uint32_t audio_rows,
                   uint32_t latent_t,
                   uint32_t latent_h,
                   uint32_t latent_w,
                   uint32_t chunk_frames,
                   uint32_t keyframe_stride);

#ifdef __cplusplus
}
#endif

#endif /* H3_SPATIOTEMPORAL_H */
