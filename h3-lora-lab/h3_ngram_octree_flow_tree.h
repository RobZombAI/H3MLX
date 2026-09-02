#ifndef H3_NGRAM_OCTREE_FLOW_TREE_H
#define H3_NGRAM_OCTREE_FLOW_TREE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_OCTREE_MAX_DEPTH 3          // Level 0: 32x32, Level 1: 16x16, Level 2: 8x8, Level 3: 4x4
#define H3_FLOW_BANK_SIZE 65536
#define H3_SPECULATIVE_TREE_DEPTH 3    // Tri-Gram speculative tree (Frames F_t, F_{t+1}, F_{t+2})

// 1. Spatio-Temporal Octree Node
typedef struct H3OctreeNode {
    uint8_t depth;                     // 0 = 32x32, 1 = 16x16, 2 = 8x8, 3 = 4x4
    int16_t x, y, z;                   // Spatial coordinates & temporal frame
    float spatial_variance;            // High-frequency energy
    bool is_cached;
    uint64_t hash;
    float *latent_data;                // Pointer to UMA resident latent buffer
    struct H3OctreeNode *children[4];  // 2D Quad-split / 3D Octree
} H3OctreeNode;

// 2. Motion-Aware Optical Flow Vector Descriptor
typedef struct {
    uint64_t tile_hash;
    float flow_dx;                     // Sub-pixel horizontal displacement
    float flow_dy;                     // Sub-pixel vertical displacement
    float flow_dt;                     // Temporal acceleration
    float confidence_score;            // 0.0 to 1.0
    uint32_t lifetime_ticks;
} H3OpticalFlowTile;

// 3. Multi-Frame Speculative Tree State (Tri-Gram)
typedef struct {
    float *speculative_latents[H3_SPECULATIVE_TREE_DEPTH]; // F_t, F_{t+1}, F_{t+2}
    float cosine_verification[H3_SPECULATIVE_TREE_DEPTH];
    bool branch_accepted[H3_SPECULATIVE_TREE_DEPTH];
    int accepted_lookahead_steps;
} H3SpeculativeTreeState;

// Master Scalable Engine Definition
typedef struct {
    H3OctreeNode *octree_root;
    H3OpticalFlowTile flow_bank[H3_FLOW_BANK_SIZE];
    H3SpeculativeTreeState tree_state;

    // Global Statistics & Telemetry
    uint64_t total_octree_tiles;
    uint64_t octree_macro_skipped;     // 32x32 tiles skipped
    uint64_t octree_micro_sharpened;   // 4x4 tiles sharpened
    uint64_t flow_warps_applied;       // Deformed cache hits under camera motion
    uint64_t tree_frames_accelerated;  // Parallel tri-gram frames verified
    float effective_speedup_factor;
} H3ScalableNGramEngine;

// Lifecycle
void h3_scalable_ngram_init(H3ScalableNGramEngine *engine);
void h3_scalable_ngram_free(H3ScalableNGramEngine *engine);
void h3_scalable_ngram_reset(H3ScalableNGramEngine *engine);

// 1. Hierarchical Octree Subdivision & Adaptive Lookup
H3OctreeNode* h3_octree_build(H3ScalableNGramEngine *engine, const float *latent_frame, int width, int height, int frame_idx);
bool h3_octree_lookup_or_subdivide(H3ScalableNGramEngine *engine, H3OctreeNode *node, float *out_patch);

// 2. Neural Optical Flow Warping
void h3_flow_estimate_and_warp(
    H3ScalableNGramEngine *engine,
    const float *prev_frame,
    float *curr_frame,
    int width, int height, int channels
);

// 3. Multi-Frame Speculative Tree Evaluation (Tri-Gram Parallel Verifier)
int h3_speculative_tree_step(
    H3ScalableNGramEngine *engine,
    float **latent_frames,
    int num_frames,
    int dit_layer_verify
);

// Scientific Telemetry
void h3_scalable_ngram_print_telemetry(const H3ScalableNGramEngine *engine);

#ifdef __cplusplus
}
#endif

#endif // H3_NGRAM_OCTREE_FLOW_TREE_H
