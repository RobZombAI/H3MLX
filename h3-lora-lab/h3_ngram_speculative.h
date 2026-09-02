#ifndef H3_NGRAM_SPECULATIVE_H
#define H3_NGRAM_SPECULATIVE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_NGRAM_TABLE_SIZE 65536
#define H3_NGRAM_PATCH_DIM 96
#define H3_NGRAM_MAX_ORDER 3

typedef struct {
    uint64_t spatial_hash;
    float momentum[H3_NGRAM_PATCH_DIM];
    float last_residual[H3_NGRAM_PATCH_DIM];
    uint32_t hit_count;
    uint32_t last_step_seen;
} H3NGramEntry;

typedef struct {
    H3NGramEntry entries[H3_NGRAM_TABLE_SIZE];
    float *momentum_buffer;
    size_t buffer_count;
    uint64_t total_lookups;
    uint64_t total_drafts_generated;
    uint64_t total_drafts_accepted;
    float acceptance_threshold;
} H3NGramSpeculativeContext;

// Initialize the N-gram speculative context
void h3_ngram_init(H3NGramSpeculativeContext *ctx, float acceptance_threshold);

// Reset table per video generation sequence
void h3_ngram_reset(H3NGramSpeculativeContext *ctx);

// Compute 64-bit spatial hash of a latent patch (16 floats)
uint64_t h3_ngram_hash_patch(const float *patch, int dim);

// Speculatively draft latent updates for recurring patches with smooth ratio scaling
bool h3_ngram_draft_step(
    H3NGramSpeculativeContext *ctx,
    const float *last_velocity,
    const float *previous_velocity,
    float *draft_velocity,
    size_t count,
    float ratio,
    int step_idx
);

// Verify draft at early-exit layer (Layer 8) via cosine similarity
bool h3_ngram_verify_draft(
    H3NGramSpeculativeContext *ctx,
    const float *draft_features,
    const float *target_features,
    int feature_dim
);

// Manifold-Projected Non-Local Optimal Transport (Sinkhorn-Wasserstein) Token Recovery
void h3_ngram_sinkhorn_manifold_recovery(
    float *draft_velocity,
    const float *last_velocity,
    const float *previous_velocity,
    size_t count,
    int patch_dim,
    int width_patches,
    int height_patches,
    float ratio
);

// SO(3) Lie Algebra Rotational Kinematics & Jacobian Incompressibility Anti-Collapse
void h3_ngram_so3_rotational_kinematics_recovery(
    float *draft_velocity,
    const float *last_velocity,
    const float *previous_velocity,
    size_t count,
    int patch_dim,
    int width_patches,
    int height_patches,
    float ratio
);

// Record ground-truth residual into the N-gram table
void h3_ngram_update_table(
    H3NGramSpeculativeContext *ctx,
    const float *velocity_t0,
    const float *velocity_t1,
    size_t count,
    int step_idx
);

// Print telemetry statistics
void h3_ngram_print_telemetry(const H3NGramSpeculativeContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // H3_NGRAM_SPECULATIVE_H
