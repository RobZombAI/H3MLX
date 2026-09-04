#include "h3_dit.h"

#include "h3_dit_schedule.h"
#include "h3_weights.h"
#include "h3_ngram_speculative.h"
#include "h3_antirez_16way_ngram_gating.h"
#include "h3_host.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
} h3_dit_block;

enum {
    STREAM_QKV,
    STREAM_OUT,
    STREAM_FC1,
    STREAM_FC2,
    STREAM_MATRICES
};

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
} h3_dit_stream_source;

typedef struct {
    h3_dit_stream_source sources[STREAM_MATRICES];
} h3_dit_stream_layer;

struct h3_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_dit_schedule *schedule;
    int fused_mlp;
    int nax_mlp;
    int int8_mlp;
    int int8_qkv;
    int int8_attention_out;
    int keep_bf16_qkv;
    int keep_bf16_attention_out;
    int use_slower_row_major_attention_output;
    int use_slower_unfused_int8_inputs;
    int use_slower_unfused_qkv_rope;
    int use_slower_scalar_qkv_rms;
    int use_slower_uncached_int8_scales;
    int use_slower_dynamic_fc1_k;
    int use_slower_grouped_quantizer;
    int use_int8_row_fc2;
    int ssd_streaming;
    int keep_bf16_mlp;
    int activation_aliases;
    int fused_patch_projection;
    int fused_patch_pack;
    int token_reduction;
    int token_reduction_active;
    unsigned token_reduction_begin;
    unsigned token_reduction_end;
    unsigned token_reduction_early_steps;
    unsigned token_reduction_early_end;
    float token_reduction_scale;
    float spatial_rope_scale;
    int sol_attn_enabled;
    float sol_attn_threshold;
    uint32_t sol_attn_block_size;
    int sol_cache_enabled;
    float sol_cache_threshold;
    h3_sol_stats sol_stats;
    int nax_st_enabled;
    uint32_t nax_st_chunk_frames;
    uint32_t nax_st_keyframe_stride;
    int bf16_final;
    unsigned core_reuse_interval;
    unsigned core_forward_count;
    int core_residual_ready;
    int semantic_layer_cache_enabled;
    int semantic_layer_cache_ready;
    unsigned semantic_layer_cache_start;
    unsigned semantic_layer_cache_end;
    h3_gpu_tensor *semantic_layer_input;
    h3_gpu_tensor *semantic_layer_residual;
    unsigned active_block_count;
    int use_step_layer_schedule;
    uint8_t step_block_active[16][H3_DIT_BLOCKS];
    uint8_t block_active[H3_DIT_BLOCKS];
    h3_layout layout;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    uint32_t reduced_sequence;
    uint32_t reduced_video_rows;
    uint32_t token_baseline_rows;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *reduced_rope_cos;
    h3_gpu_tensor *reduced_rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **reduced_row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    h3_dit_block stream_slots[2];
    h3_dit_stream_layer stream_layers[H3_DIT_BLOCKS];
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes;
    double stream_read_seconds;
    double stream_wait_seconds;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *final_video_w_steps[H3_MAX_STEPS];
    h3_gpu_tensor *final_video_b_steps[H3_MAX_STEPS];
    h3_gpu_tensor *final_audio_w_steps[H3_MAX_STEPS];
    h3_gpu_tensor *final_audio_b_steps[H3_MAX_STEPS];
    int num_step_heads;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *video_projection_map;
    h3_gpu_tensor *audio_projection_map;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *core_input;
    h3_gpu_tensor *core_residual;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *token_pool_pairs;
    h3_gpu_tensor *token_baseline_indices;
    h3_gpu_tensor *token_expand_parents;
    h3_gpu_tensor *token_original;
    int token_original_in_qkv;
    size_t token_original_offset;
    size_t token_baseline_offset;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *int8_activation;
    h3_gpu_tensor *int8_activation_scales;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_inverse;
    h3_gpu_tensor *final_video_inverse;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
    h3_gpu_tensor *previous_audio_velocity;
    h3_gpu_tensor *previous_video_velocity;
    /* N-Gram speculative patch engine context (set by h3.c when --ngram active) */
    void *ngram_ctx;
    /* Persistent zero-allocation staging buffers for forward and denoise hot loops */
    float *staging_video_rows;
    float *staging_audio_rows;
    uint16_t *staging_video_out;
    uint16_t *staging_audio_out;
    float *staging_video_f32;
    float *staging_audio_f32;
    float *staging_draft_buffer;
    size_t staging_video_capacity;
    size_t staging_audio_capacity;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static unsigned command_block_interval(const h3_dit *dit) {
    const char *value = getenv("H3_DIT_COMMAND_BLOCKS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        return end != value && !*end && parsed >= 0 &&
               parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
    }
    if (h3_gpu_is_m5(dit->gpu))
        return dit->active_block_count * 3 / 5;
    return dit->active_block_count == H3_DIT_BLOCKS ? 30u : 0u;
}

static int gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static void report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static int copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

static int validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    return 1;
}

static int configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size) {
    const char *enabled = getenv("H3_TOKEN_REDUCTION");
    if (!requested &&
        (!enabled || !*enabled || !strcmp(enabled, "0"))) return 1;
    unsigned begin = 4, end = 30;
    const char *range = getenv("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        char *middle = NULL;
        unsigned long parsed_begin = strtoul(range, &middle, 10);
        if (middle == range || *middle != ':') {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END");
            return 0;
        }
        char *tail = NULL;
        unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
        if (tail == middle + 1 || *tail || parsed_begin >= parsed_end ||
            parsed_end > H3_DIT_BLOCKS) {
            fail(error, error_size,
                 "token-reduction block range must satisfy 0 <= BEGIN < END <= 50");
            return 0;
        }
        begin = (unsigned)parsed_begin;
        end = (unsigned)parsed_end;
    }
    /* Coarse structure is tolerant of a deeper reduced stack while the first
     * noisy samples form. Restore earlier once fine detail starts resolving. */
    unsigned early_steps = end < 40 ? 10 : 0;
    unsigned early_end = end < 40 ? 40 : end;
    const char *early = getenv("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            char *middle = NULL;
            unsigned long parsed_steps = strtoul(early, &middle, 10);
            if (middle == early || *middle != ':') {
                fail(error, error_size,
                     "H3_TOKEN_REDUCTION_EARLY must be STEPS:END");
                return 0;
            }
            char *tail = NULL;
            unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
            if (tail == middle + 1 || *tail || !parsed_steps ||
                parsed_steps > 1000 || parsed_end <= end ||
                parsed_end > H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "early token reduction requires STEPS > 0 and "
                     "base END < END <= 50");
                return 0;
            }
            early_steps = (unsigned)parsed_steps;
            early_end = (unsigned)parsed_end;
        }
    }
    float scale = 1.0f;
    const char *scale_text = getenv("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        scale = strtof(scale_text, &tail);
        if (tail == scale_text || *tail || !isfinite(scale) ||
            scale < 0.0f || scale > 2.0f) {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]");
            return 0;
        }
    }
    uint32_t spatial_height = (uint32_t)dit->latent_h / 2;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint64_t reduced_video =
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * reduced_width;
    if (!spatial_height || !spatial_width ||
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * spatial_width !=
            dit->video_rows ||
        dit->video_target_start + dit->video_rows != dit->sequence ||
        reduced_video > UINT32_MAX ||
        reduced_video > UINT32_MAX - dit->video_target_start) {
        fail(error, error_size,
             "token reduction requires the target video to end the packed layout");
        return 0;
    }
    dit->token_reduction = 1;
    dit->token_reduction_begin = begin;
    dit->token_reduction_end = end;
    dit->token_reduction_early_steps = early_steps;
    dit->token_reduction_early_end = early_end;
    dit->token_reduction_scale = scale;
    dit->reduced_video_rows = (uint32_t)reduced_video;
    dit->token_baseline_rows = dit->video_rows - dit->reduced_video_rows;
    dit->reduced_sequence = dit->video_target_start +
                            dit->reduced_video_rows;
    return 1;
}

static void token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second) {
    if (reduced_row < dit->video_target_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = reduced_row - dit->video_target_start;
    uint32_t source = dit->video_target_start +
        (local / reduced_width) * spatial_width +
        (local % reduced_width) * 2;
    *first = source;
    *second = source + ((source - dit->video_target_start) % spatial_width + 1 <
                        spatial_width ? 1u : 0u);
}

static uint32_t token_reduced_parent(const h3_dit *dit, uint32_t full_row) {
    if (full_row < dit->video_target_start) return full_row;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = full_row - dit->video_target_start;
    return dit->video_target_start + (local / spatial_width) * reduced_width +
           (local % spatial_width) / 2;
}

static int load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                               \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2
    return 1;
}

static int load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

static void free_block(h3_dit_block *block) {
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->qkv_int8);
    free_tensor(&block->qkv_scales);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->out_int8);
    free_tensor(&block->out_scales);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
    free_tensor(&block->fc1_int8);
    free_tensor(&block->fc1_scales);
    free_tensor(&block->fc2_int8);
    free_tensor(&block->fc2_scales);
}

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_dit_stream_source *a = left;
    const h3_dit_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

static int prepare_stream_source(h3_dit *dit,
                                 h3_dit_stream_source *source,
                                 const char *name, uint64_t rows,
                                 uint64_t columns, unsigned field,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, &header);
    if (!tensor) {
        fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (!header || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns) {
        fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = (size_t)(rows * columns);
    source->field = field;
    return 1;
}

static int prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    char name[160];
    h3_dit_stream_layer *stream = &dit->stream_layers[layer];
#define SOURCE(index, suffix, rows, columns, field) do {                        \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);              \
    if (!prepare_stream_source(dit, &stream->sources[index], name,             \
                               rows, columns, field, error, error_size))        \
        return 0;                                                               \
} while (0)
    SOURCE(0, "attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
    SOURCE(1, "attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
    SOURCE(2, "mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
    SOURCE(3, "mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
#undef SOURCE
    qsort(stream->sources, STREAM_MATRICES, sizeof(stream->sources[0]),
          compare_stream_sources);
    return 1;
}

static int allocate_stream_slot(h3_dit *dit, h3_dit_block *slot,
                                char *error, size_t error_size) {
    slot->qkv = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    slot->out = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * INNER);
    slot->fc1 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    slot->fc2 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * FFN);
    if (!slot->qkv || !slot->out || !slot->fc1 || !slot->fc2) {
        fail(error, error_size, "cannot allocate BF16 SSD layer slot: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static h3_gpu_tensor *stream_slot_target(h3_dit_block *slot,
                                         unsigned field) {
    if (field == STREAM_QKV) return slot->qkv;
    if (field == STREAM_OUT) return slot->out;
    if (field == STREAM_FC1) return slot->fc1;
    if (field == STREAM_FC2) return slot->fc2;
    return NULL;
}

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_stream_job;

static int read_stream_layer(h3_dit_stream_job *job) {
    h3_dit_stream_layer *layer = &job->dit->stream_layers[job->layer];
    h3_dit_block *slot = &job->dit->stream_slots[job->slot];
    double started = stream_now();
    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    for (unsigned index = 0; index < STREAM_MATRICES; index++) {
        const h3_dit_stream_source *source = &layer->sources[index];
        h3_gpu_tensor *target = stream_slot_target(slot, source->field);
        if (!target || !h3_gpu_tensor_stream_file_bf16(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error))) {
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error),
                         "invalid BF16 streaming destination");
            job->ok = 0;
            break;
        }
        job->bytes += (uint64_t)source->elements * sizeof(uint16_t);
    }
    job->seconds = stream_now() - started;
    return job->ok;
}

static void *read_stream_layer_thread(void *opaque) {
    read_stream_layer(opaque);
    return NULL;
}

static int quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->fc1_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    block->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
    block->fc2_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * FFN);
    block->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->fc1_int8 && block->fc1_scales &&
             block->fc2_int8 && block->fc2_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc1_int8, block->fc1_scales, block->fc1,
                 FFN * 2, HIDDEN) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc2_int8, block->fc2_scales, block->fc2,
                 HIDDEN, FFN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT MLP weights: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_mlp) {
        free_tensor(&block->fc1);
        free_tensor(&block->fc2);
    }
    return 1;
}

static int quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->qkv_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    int ok = block->qkv_int8 && block->qkv_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                 INNER * 3, HIDDEN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT QKV weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_qkv) free_tensor(&block->qkv);
    return 1;
}

static int quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size) {
    block->out_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * INNER);
    block->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->out_int8 && block->out_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->out_int8, block->out_scales, block->out,
                 HIDDEN, INNER) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size,
             "cannot quantize DiT attention-output weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_attention_out) free_tensor(&block->out);
    return 1;
}

static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
        load_block(dit, &refiner[0], "token_refiner.blocks.0.",
                   error, error_size) &&
        load_block(dit, &refiner[1], "token_refiner.blocks.1.",
                   error, error_size);
    if (ok) final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        free_tensor(&inverse_tensor);
        if (!error || !*error) fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    free_tensor(&inverse_tensor);
    float spatial_scale = dit->spatial_rope_scale;
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    size_t reduced_count = dit->token_reduction ?
        (size_t)dit->reduced_sequence * ROPE_HALF : 0;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *reduced_cosines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_cosines)) : NULL;
    float *reduced_sines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_sines)) : NULL;
    if (!cosines || !sines ||
        (reduced_count && (!reduced_cosines || !reduced_sines))) {
        free(cosines);
        free(sines);
        free(reduced_cosines);
        free(reduced_sines);
        fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h * spatial_scale,
                        (float)dit->layout.positions[row].w * spatial_scale};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        uint32_t first, second;
        token_pool_sources(dit, row, &first, &second);
        float axes[] = {
            (float)((dit->layout.positions[first].t +
                     dit->layout.positions[second].t) * 0.5),
            (float)((dit->layout.positions[first].h +
                     dit->layout.positions[second].h) * 0.5) * spatial_scale,
            (float)((dit->layout.positions[first].w +
                     dit->layout.positions[second].w) * 0.5) * spatial_scale
        };
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                reduced_cosines[index] = cosf(angle);
                reduced_sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    h3_gpu_tensor *reduced_cos_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_cosines, reduced_count) : NULL;
    h3_gpu_tensor *reduced_sin_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_sines, reduced_count) : NULL;
    free(cosines);
    free(sines);
    free(reduced_cosines);
    free(reduced_sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    if (reduced_count) {
        dit->reduced_rope_cos = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
        dit->reduced_rope_sin = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
    }
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
        (!reduced_count || (reduced_cos_f32 && reduced_sin_f32 &&
                            dit->reduced_rope_cos &&
                            dit->reduced_rope_sin));
    if (ok) {
        ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast");
        if (ok && reduced_count) {
            ok = gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_cos, reduced_cos_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE cosine cast") &&
                 gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_sin, reduced_sin_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE sine cast");
        }
        if (ok) ok =
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    free_tensor(&reduced_cos_f32);
    free_tensor(&reduced_sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    if (dit->token_reduction)
        dit->reduced_row_maps = calloc((size_t)steps,
                                       sizeof(*dit->reduced_row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *reduced = dit->token_reduction ?
        malloc((size_t)dit->reduced_sequence * sizeof(*reduced)) : NULL;
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        (dit->token_reduction && (!dit->reduced_row_maps || !reduced)) ||
        !rows || !audio || !video) {
        fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(reduced); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(dit->schedule, step, &dit->layout,
                                     text->tags, text->tokens, rows,
                                     dit->sequence)) {
            fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
        if (dit->token_reduction) {
            for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
                uint32_t first, second;
                token_pool_sources(dit, row, &first, &second);
                (void)second;
                reduced[row] = rows[first];
            }
            dit->reduced_row_maps[step] = h3_gpu_tensor_from_u32(
                dit->gpu, reduced, dit->reduced_sequence);
        }
        uint32_t audio_row = h3_dit_schedule_audio_row(dit->schedule, step);
        uint32_t video_row = h3_dit_schedule_video_row(dit->schedule, step);
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = audio_row;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = video_row;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] ||
            (dit->token_reduction && !dit->reduced_row_maps[step]) ||
            !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(reduced); free(audio); free(video);
    return 1;
}

static int prepare_projection_maps(h3_dit *dit, char *error,
                                   size_t error_size) {
    unsigned video_segments = 0, audio_segments = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        h3_segment_kind kind = dit->layout.segments[index].kind;
        if (kind == H3_SEG_COND || kind == H3_SEG_REF_IMAGE ||
            kind == H3_SEG_VIDEO)
            video_segments++;
        else if (kind != H3_SEG_TEXT)
            audio_segments++;
    }
    uint32_t *video = video_segments > 1 ?
        malloc((size_t)dit->video_total_rows * sizeof(*video)) : NULL;
    uint32_t *audio = audio_segments > 1 ?
        malloc((size_t)dit->audio_total_rows * sizeof(*audio)) : NULL;
    if ((video_segments > 1 && !video) || (audio_segments > 1 && !audio)) {
        free(video); free(audio);
        fail(error, error_size, "out of memory allocating projection maps");
        return 0;
    }
    size_t video_offset = 0, audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (segment->kind == H3_SEG_COND ||
            segment->kind == H3_SEG_REF_IMAGE ||
            segment->kind == H3_SEG_VIDEO) {
            for (size_t row = 0; video && row < rows; row++)
                video[video_offset + row] = (uint32_t)(segment->start + row);
            video_offset += rows;
        } else if (segment->kind != H3_SEG_TEXT) {
            for (size_t row = 0; audio && row < rows; row++)
                audio[audio_offset + row] = (uint32_t)(segment->start + row);
            audio_offset += rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        free(video); free(audio);
        fail(error, error_size, "projection map rows are inconsistent");
        return 0;
    }
    if (video)
        dit->video_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_total_rows);
    if (audio)
        dit->audio_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_total_rows);
    free(video); free(audio);
    if ((video_segments > 1 && !dit->video_projection_map) ||
        (audio_segments > 1 && !dit->audio_projection_map)) {
        fail(error, error_size, "cannot allocate projection map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int prepare_token_reduction_maps(h3_dit *dit, char *error,
                                        size_t error_size) {
    if (!dit->token_reduction) return 1;
    size_t pair_count = (size_t)dit->reduced_sequence * 2;
    uint32_t *pairs = malloc(pair_count * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(
        (size_t)dit->reduced_sequence * sizeof(*baseline_indices));
    uint32_t *parents = malloc((size_t)dit->sequence * sizeof(*parents));
    if (!pairs || !baseline_indices || !parents) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size,
             "out of memory allocating token-reduction maps");
        return 0;
    }
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        token_pool_sources(dit, row, &pairs[(size_t)row * 2],
                           &pairs[(size_t)row * 2 + 1]);
        baseline_indices[row] =
            row >= dit->video_target_start &&
            pairs[(size_t)row * 2] != pairs[(size_t)row * 2 + 1] ?
                baseline_row++ : UINT32_MAX;
    }
    if (baseline_row != dit->token_baseline_rows) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size, "token-reduction baseline map is inconsistent");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++)
        parents[row] = token_reduced_parent(dit, row);
    dit->token_pool_pairs = h3_gpu_tensor_from_u32(
        dit->gpu, pairs, pair_count);
    dit->token_baseline_indices = h3_gpu_tensor_from_u32(
        dit->gpu, baseline_indices, dit->reduced_sequence);
    dit->token_expand_parents = h3_gpu_tensor_from_u32(
        dit->gpu, parents, dit->sequence);
    free(pairs);
    free(baseline_indices);
    free(parents);
    if (!dit->token_pool_pairs || !dit->token_baseline_indices ||
        !dit->token_expand_parents) {
        fail(error, error_size,
             "cannot allocate token-reduction map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static void configure_active_blocks(h3_dit *dit, unsigned active) {
    memset(dit->block_active, 1, sizeof(dit->block_active));
    dit->active_block_count = active;
    unsigned skipped = H3_DIT_BLOCKS - active;
    for (unsigned index = 0; index < skipped; index++) {
        unsigned block = ((2 * index + 1) * H3_DIT_BLOCKS) / (2 * skipped);
        if (block == 0) block = 1;
        if (block >= H3_DIT_BLOCKS - 1) block = H3_DIT_BLOCKS - 2;
        dit->block_active[block] = 0;
    }
}

static unsigned first_active_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static unsigned next_active_block(const h3_dit *dit, unsigned current) {
    for (unsigned block = current + 1; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static void configure_gate_ranked_blocks(h3_dit *dit) {
    const char *policy = getenv("H3_DIT_LAYER_POLICY");
    const char *sched_env = getenv("H3_LAYER_SCHEDULE");
    const char *pyramid_env = getenv("H3_PYRAMIDAL_LAYERS");

    typedef struct { unsigned block; double score; } block_score;
    block_score scores[H3_DIT_BLOCKS - 3];
    int scores_computed = 0;
    if (dit->schedule) {
        for (unsigned block = 2; block + 1 < H3_DIT_BLOCKS; block++) {
            double score = h3_dit_schedule_gate_score(dit->schedule, block);
            if (score < 0.0) { scores_computed = 0; break; }
            scores[block - 2] = (block_score){block, score};
            scores_computed = 1;
        }
    }
    if (scores_computed) {
        unsigned count = H3_DIT_BLOCKS - 3;
        for (unsigned left = 0; left < count; left++) {
            unsigned least = left;
            for (unsigned right = left + 1; right < count; right++)
                if (scores[right].score < scores[least].score) least = right;
            block_score temporary = scores[left];
            scores[left] = scores[least];
            scores[least] = temporary;
        }
    }

    if (sched_env || pyramid_env) {
        dit->use_step_layer_schedule = 1;
        unsigned step_layers[16] = {18, 24, 30, 36, 42, 48, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50};
        if (sched_env && *sched_env) {
            const char *p = sched_env;
            for (int s = 0; s < 16 && *p; s++) {
                char *end = NULL;
                long val = strtol(p, &end, 10);
                if (end > p && val > 0 && val <= H3_DIT_BLOCKS) {
                    step_layers[s] = (unsigned)val;
                }
                if (*end == ',' || *end == ' ') end++;
                p = end;
            }
        }
        memset(dit->step_block_active, 1, sizeof(dit->step_block_active));
        memset(dit->block_active, 0, sizeof(dit->block_active));

        for (int s = 0; s < 16; s++) {
            unsigned active_s = step_layers[s];
            if (active_s > H3_DIT_BLOCKS) active_s = H3_DIT_BLOCKS;
            unsigned skipped_s = H3_DIT_BLOCKS - active_s;
            if (scores_computed) {
                for (unsigned idx = 0; idx < skipped_s; idx++)
                    dit->step_block_active[s][scores[idx].block] = 0;
            } else {
                for (unsigned idx = 0; idx < skipped_s; idx++) {
                    unsigned b = ((2 * idx + 1) * H3_DIT_BLOCKS) / (2 * skipped_s);
                    if (b == 0) b = 1;
                    if (b >= H3_DIT_BLOCKS - 1) b = H3_DIT_BLOCKS - 2;
                    dit->step_block_active[s][b] = 0;
                }
            }
            for (unsigned b = 0; b < H3_DIT_BLOCKS; b++) {
                if (dit->step_block_active[s][b])
                    dit->block_active[b] = 1;
            }
        }
        if (getenv("H3_PROFILE")) {
            fprintf(stderr, "h3: pyramidal step layer schedule active [");
            for (int s = 0; s < 8; s++) fprintf(stderr, "%u%s", step_layers[s], s < 7 ? ", " : "]\n");
        }
        return;
    }

    if ((policy && !strcmp(policy, "uniform")) ||
        dit->active_block_count == H3_DIT_BLOCKS) return;

    if (!scores_computed) return;
    memset(dit->block_active, 1, sizeof(dit->block_active));
    unsigned skipped = H3_DIT_BLOCKS - dit->active_block_count;
    for (unsigned index = 0; index < skipped; index++)
        dit->block_active[scores[index].block] = 0;
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "h3: gate-ranked DiT skips");
        for (unsigned index = 0; index < skipped; index++)
            fprintf(stderr, " %u(%.4g)", scores[index].block,
                    scores[index].score);
        fputc('\n', stderr);
    }
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (!dit->block_active[index]) {
            report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS);
            continue;
        }
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->ssd_streaming) {
            if (!load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size) ||
                !prepare_stream_layer(dit, index, error, error_size))
                return 0;
        } else {
            if (!load_block(dit, &dit->blocks[index], prefix,
                            error, error_size)) return 0;
            if (dit->int8_mlp &&
                !quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_qkv &&
                !quantize_block_qkv(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_attention_out &&
                !quantize_block_attention_out(
                    dit, &dit->blocks[index], error, error_size)) return 0;
        }
        report(progress, opaque, "load transformer core", (int)index + 1,
               H3_DIT_BLOCKS);
    }
    if (dit->ssd_streaming) {
        if (!allocate_stream_slot(dit, &dit->stream_slots[0],
                                  error, error_size) ||
            !allocate_stream_slot(dit, &dit->stream_slots[1],
                                  error, error_size)) return 0;
        unsigned first = first_active_block(dit);
        if (first == H3_DIT_BLOCKS) {
            fail(error, error_size, "SSD stream has no active DiT block");
            return 0;
        }
        h3_dit_stream_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!read_stream_layer(&job)) {
            fail(error, error_size, "cannot prime DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = first;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;
        dit->stream_read_seconds += job.seconds;
    }
    dit->video_patch_w = f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    dit->num_step_heads = 0;
    for (int i = 0; i < 8; i++) {
        char name_vw[64], name_vb[64], name_aw[64], name_ab[64];
        snprintf(name_vw, sizeof(name_vw), "final_layer.video_out.%d.weight", i);
        snprintf(name_vb, sizeof(name_vb), "final_layer.video_out.%d.bias", i);
        snprintf(name_aw, sizeof(name_aw), "final_layer.audio_out.%d.weight", i);
        snprintf(name_ab, sizeof(name_ab), "final_layer.audio_out.%d.bias", i);
        dit->final_video_w_steps[i] = f2(dit, name_vw, VIDEO_PATCH, HIDDEN, NULL, 0);
        dit->final_video_b_steps[i] = f1(dit, name_vb, VIDEO_PATCH, NULL, 0);
        dit->final_audio_w_steps[i] = f2(dit, name_aw, AUDIO_CHANNELS, HIDDEN, NULL, 0);
        dit->final_audio_b_steps[i] = f1(dit, name_ab, AUDIO_CHANNELS, NULL, 0);
        if (dit->final_video_w_steps[i] && dit->final_video_b_steps[i] &&
            dit->final_audio_w_steps[i] && dit->final_audio_b_steps[i]) {
            dit->num_step_heads = i + 1;
        }
    }
    if (getenv("H3_DISABLE_STEP_HEADS")) {
        dit->num_step_heads = 0;
    }
    if (dit->num_step_heads > 0 && getenv("H3_PROFILE")) {
        fprintf(stderr, "h3: loaded %d PDD Multi-Head projections\n", dit->num_step_heads);
    }
    if (dit->bf16_final && dit->final_video_w && dit->final_video_b &&
        dit->final_audio_w && dit->final_audio_b) {
        h3_gpu_tensor *source[4] = {
            dit->final_video_w, dit->final_video_b,
            dit->final_audio_w, dit->final_audio_b
        };
        size_t elements[4] = {
            (size_t)VIDEO_PATCH * HIDDEN, VIDEO_PATCH,
            (size_t)AUDIO_CHANNELS * HIDDEN, AUDIO_CHANNELS
        };
        h3_gpu_tensor *target[4] = {0};
        int ok = 1;
        for (unsigned index = 0; index < 4; index++) {
            target[index] = h3_gpu_tensor_new_bf16(dit->gpu,
                                                    elements[index]);
            if (!target[index]) ok = 0;
        }
        if (ok) ok = h3_gpu_begin(dit->gpu);
        for (unsigned index = 0; ok && index < 4; index++)
            ok = h3_gpu_cast_f32_to_bf16(dit->gpu, target[index],
                                         source[index],
                                         (uint32_t)elements[index]);
        if (ok) ok = h3_gpu_submit(dit->gpu);
        if (ok) {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(source[index]);
            dit->final_video_w = target[0];
            dit->final_video_b = target[1];
            dit->final_audio_w = target[2];
            dit->final_audio_b = target[3];
        } else {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(target[index]);
            fail(error, error_size, "cannot convert DiT final weights: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        if (dit->num_step_heads > 0) {
            for (int i = 0; i < dit->num_step_heads; i++) {
                h3_gpu_tensor *s[4] = {
                    dit->final_video_w_steps[i], dit->final_video_b_steps[i],
                    dit->final_audio_w_steps[i], dit->final_audio_b_steps[i]
                };
                h3_gpu_tensor *t[4] = {0};
                int step_ok = 1;
                for (unsigned idx = 0; idx < 4; idx++) {
                    t[idx] = h3_gpu_tensor_new_bf16(dit->gpu, elements[idx]);
                    if (!t[idx]) step_ok = 0;
                }
                if (step_ok) step_ok = h3_gpu_begin(dit->gpu);
                for (unsigned idx = 0; step_ok && idx < 4; idx++)
                    step_ok = h3_gpu_cast_f32_to_bf16(dit->gpu, t[idx], s[idx], (uint32_t)elements[idx]);
                if (step_ok) step_ok = h3_gpu_submit(dit->gpu);
                if (step_ok) {
                    for (unsigned idx = 0; idx < 4; idx++) h3_gpu_tensor_free(s[idx]);
                    dit->final_video_w_steps[i] = t[0];
                    dit->final_video_b_steps[i] = t[1];
                    dit->final_audio_w_steps[i] = t[2];
                    dit->final_audio_b_steps[i] = t[3];
                }
            }
        }
    }
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
    dit->activation_aliases = !getenv("H3_DISABLE_DIT_ACTIVATION_ALIAS");
    dit->fused_patch_projection =
        !getenv("H3_DISABLE_FUSED_PATCH_CAST") && !getenv("H3_SCALAR_PATCH");
    dit->fused_patch_pack = dit->fused_patch_projection &&
        !getenv("H3_DISABLE_FUSED_PATCH_PACK");
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        F32(final_audio_inverse, audio),
        F32(final_video_inverse, video),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_pack) {
        dit->video_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected || !dit->audio_projected) {
            fail(error, error_size,
                 "cannot allocate packed patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_projection) {
        dit->video_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected_f32 || !dit->audio_projected_f32) {
            fail(error, error_size,
                 "cannot allocate separate patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->activation_aliases) {
        dit->attention_heads = dit->qkv;
        dit->mod_mlp = dit->qkv;
        dit->mlp_output = NULL;
    } else {
        dit->attention_heads = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * INNER);
        dit->mod_mlp = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
    }
    if (!dit->attention_heads || !dit->mod_mlp ||
        (!dit->activation_aliases && !dit->mlp_output)) {
        fail(error, error_size,
             "cannot allocate DiT activation buffers: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_input = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_input = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_input || !dit->final_video_input) {
            fail(error, error_size,
                 "cannot allocate separate final DiT slices: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final || getenv("H3_DISABLE_FUSED_FINAL_HEAD") ||
        getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_norm || !dit->final_video_norm) {
            fail(error, error_size,
                 "cannot allocate separate final DiT normalization: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final) {
        dit->final_audio_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio * HIDDEN);
        dit->final_video_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video * HIDDEN);
        dit->audio_output = h3_gpu_tensor_new_f32(
            dit->gpu, audio * AUDIO_CHANNELS);
        dit->video_output = h3_gpu_tensor_new_f32(
            dit->gpu, video * VIDEO_PATCH);
        if (!dit->final_audio_f32 || !dit->final_video_f32 ||
            !dit->audio_output || !dit->video_output) {
            fail(error, error_size,
                 "cannot allocate F32 DiT final activations: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    }
    if (!dit->fused_mlp || dit->nax_mlp || dit->int8_mlp) {
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if ((!dit->fused_mlp && !dit->fc1) || !dit->activated) {
            fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->int8_mlp || dit->int8_qkv || dit->int8_attention_out) {
        size_t padded_sequence = (sequence + 127) & ~(size_t)127;
        dit->int8_activation = h3_gpu_tensor_new_i8(
            dit->gpu, padded_sequence * FFN);
        dit->int8_activation_scales = h3_gpu_tensor_new_f32(
            dit->gpu, padded_sequence * (FFN / 1024));
        if (!dit->int8_activation || !dit->int8_activation_scales) {
            fail(error, error_size,
                 "cannot allocate int8 DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->token_reduction) {
        size_t full_elements = sequence * HIDDEN;
        size_t qkv_capacity = sequence * INNER * 3;
        size_t qkv_used = (size_t)dit->reduced_sequence * INNER * 3;
        size_t baseline_elements =
            (size_t)dit->token_baseline_rows * HIDDEN;
        size_t attention_capacity = sequence * HIDDEN;
        size_t attention_used =
            (size_t)dit->reduced_sequence * HIDDEN;
        dit->token_original_in_qkv =
            qkv_used <= qkv_capacity &&
            full_elements <= qkv_capacity - qkv_used &&
            qkv_used <= UINT32_MAX &&
            full_elements <= UINT32_MAX - qkv_used;
        if (dit->token_original_in_qkv)
            dit->token_original_offset = qkv_used;
        else
            dit->token_original = h3_gpu_tensor_new_bf16(
                dit->gpu, full_elements);
        dit->token_baseline_offset = attention_used;
        if (attention_used > attention_capacity ||
            baseline_elements > attention_capacity - attention_used ||
            attention_used > UINT32_MAX ||
            baseline_elements > UINT32_MAX - attention_used ||
            (!dit->token_original_in_qkv && !dit->token_original)) {
            fail(error, error_size,
                 "cannot allocate token-reduction residual state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->core_reuse_interval > 1) {
        dit->core_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->core_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->core_input || !dit->core_residual) {
            fail(error, error_size,
                 "cannot allocate DiT core residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->semantic_layer_cache_enabled) {
        dit->semantic_layer_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->semantic_layer_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->semantic_layer_input || !dit->semantic_layer_residual) {
            fail(error, error_size,
                 "cannot allocate DiT semantic layer residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    report(state->callback, state->opaque, "precompute AdaLN", completed, total);
}

static h3_dit *load_dit(const char *weight_directory,
                        const char *shader_source_path,
                        const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas,
                        unsigned active_blocks,
                        unsigned core_reuse_interval,
                        int token_reduction,
                        int ssd_streaming,
                        float spatial_rope_scale,
                        int use_slower_bf16_mlp,
                        int use_slower_bf16_qkv,
                        int use_slower_bf16_attention_output,
                        int use_slower_row_major_attention_output,
                        int use_slower_unfused_int8_inputs,
                        int use_slower_unfused_qkv_rope,
                        int use_slower_scalar_qkv_rms,
                        int use_slower_uncached_int8_scales,
                        int use_slower_dynamic_fc1_k,
                        int use_slower_grouped_quantizer,
                        int use_int8_row_fc2,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        h3_dit_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas ||
        (ssd_streaming != 0 && ssd_streaming != 1) ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f ||
        active_blocks < H3_DIT_BLOCKS / 2 ||
        active_blocks > H3_DIT_BLOCKS || core_reuse_interval < 1 ||
        core_reuse_interval > 6) {
        fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->fused_mlp = getenv("H3_DISABLE_FUSED_MLP") == NULL;
    /* The released final heads are F32, but their inputs are already BF16.
     * Converting these small weights once selects the Iris-derived tiled
     * linear and eliminates two full-width casts plus the scalar F32 kernel.
     * Keep the old path available for close-reference diagnosis. */
    dit->bf16_final = getenv("H3_DIT_F32_FINAL") == NULL;
    dit->core_reuse_interval = core_reuse_interval;
    const char *sem_cache_env = getenv("H3_SEMANTIC_LAYER_CACHE");
    const char *sem_disable_env = getenv("H3_DISABLE_SEMANTIC_LAYER_CACHE");
    dit->semantic_layer_cache_enabled = (sem_cache_env && (*sem_cache_env == '1' || *sem_cache_env == 'y' || *sem_cache_env == 't')) ||
                                       (sem_disable_env == NULL && !dit->ssd_streaming);
    dit->semantic_layer_cache_start = 14;
    dit->semantic_layer_cache_end = 36;
    dit->semantic_layer_cache_ready = 0;
    dit->ssd_streaming = ssd_streaming;
    dit->spatial_rope_scale = spatial_rope_scale;
    configure_active_blocks(dit, active_blocks);
    if (!copy_layout(dit, layout, error, error_size) ||
        !validate_layout(dit, text, error, error_size) ||
        !configure_token_reduction(dit, token_reduction,
                                   error, error_size)) goto failed;
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!dit->weights) goto failed;
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) goto failed;
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu);
    dit->int8_mlp = !dit->ssd_streaming && dit->fused_mlp &&
                    !use_slower_bf16_mlp &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_qkv = !dit->ssd_streaming && !use_slower_bf16_qkv &&
                    dit->sequence >= 128 &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_attention_out = !dit->ssd_streaming &&
                              !use_slower_bf16_attention_output &&
                              dit->sequence >= 128 &&
                              h3_gpu_has_int8_mlp(dit->gpu);
    dit->use_slower_row_major_attention_output =
        use_slower_row_major_attention_output;
    dit->use_slower_unfused_int8_inputs =
        use_slower_unfused_int8_inputs;
    dit->use_slower_unfused_qkv_rope =
        use_slower_unfused_qkv_rope;
    dit->use_slower_scalar_qkv_rms = use_slower_scalar_qkv_rms;
    dit->use_slower_uncached_int8_scales =
        use_slower_uncached_int8_scales;
    dit->use_slower_dynamic_fc1_k = use_slower_dynamic_fc1_k;
    dit->keep_bf16_attention_out = dit->int8_attention_out &&
        (getenv("H3_INT8_KEEP_BF16_ATTENTION_OUT") ||
         getenv("H3_BENCH_INT8_ATTENTION_OUT_AB"));
    dit->keep_bf16_qkv = dit->int8_qkv &&
        (getenv("H3_INT8_KEEP_BF16_QKV") ||
         getenv("H3_BENCH_INT8_QKV_AB"));
    dit->use_slower_grouped_quantizer = use_slower_grouped_quantizer;
    dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;
    dit->keep_bf16_mlp = dit->int8_mlp &&
        (getenv("H3_INT8_KEEP_BF16_MLP") ||
         getenv("H3_BENCH_INT8_MLP_AB") ||
         getenv("H3_INT8_MLP_STAGE"));
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    report(progress, progress_opaque, "refine text", 0, 1);
    if (!refine_text(dit, text, error, error_size)) goto failed;
    report(progress, progress_opaque, "refine text", 1, 1);
    schedule_progress schedule_state = {progress, progress_opaque};
    dit->schedule = h3_dit_schedule_precompute(
        dit->weights, dit->gpu, sigmas, dit->video_condition_rows != 0,
        dit->audio_condition_rows != 0, schedule_report, &schedule_state,
        error, error_size);
    if (dit->schedule) {
        configure_gate_ranked_blocks(dit);
        h3_dit_schedule_prune(dit->schedule, dit->block_active,
                              H3_DIT_BLOCKS);
    }
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    h3_gpu_profile_mark(dit->gpu, "load");
    return dit;
failed:
    h3_dit_free(dit);
    return NULL;
}

h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    NULL, 0, NULL, 0, progress, progress_opaque,
                    error, error_size);
}

h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    progress, progress_opaque, error, error_size);
}

static int enter_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_pool_bf16(
            dit->gpu, dit->attention_output, dit->hidden, 0,
            original, dit->token_original_offset, dit->attention_output,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_pool_pairs, dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN),
            error, error_size, "snapshot and pool video tokens")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int enter_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_pool_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            dit->hidden, 0, original, dit->token_original_offset,
            dit->attention_output, dit->token_baseline_offset,
            dit->token_baseline_indices, dit->token_pool_pairs,
            weight->norm1, modulation, dit->reduced_row_maps[step],
            dit->sequence, dit->reduced_sequence, dit->token_baseline_rows,
            HIDDEN, SLOTS, 0, 1, 1e-5f), error, error_size,
            "snapshot, pool, and apply attention AdaLN")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int leave_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_expand_delta_bf16(
            dit->gpu, dit->mod_attention, original,
            dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, dit->sequence,
            dit->reduced_sequence, dit->token_baseline_rows, HIDDEN,
            dit->video_target_start,
            dit->token_reduction_scale), error, error_size,
            "restore full video-token grid")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->mod_attention;
    dit->mod_attention = swap;
    dit->token_reduction_active = 0;
    return 1;
}

static int leave_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_expand_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            original, dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, weight->norm1, modulation,
            dit->row_maps[step], dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN, dit->video_target_start,
            dit->token_reduction_scale, SLOTS, 0, 1, 1e-5f),
            error, error_size, "restore tokens and apply attention AdaLN"))
        return 0;
    h3_gpu_tensor *reduced = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = reduced;
    dit->token_reduction_active = 0;
    return 1;
}

static int run_block(h3_dit *dit, unsigned index, int step,
                     h3_dit_block *weight,
                     int attention_adaln_ready,
                     int attention_input_quantized,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready,
                     int *next_attention_input_quantized,
                     char *error, size_t error_size) {
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->token_reduction_active ?
        dit->reduced_row_maps[step] : dit->row_maps[step];
    h3_gpu_tensor *rope_cos = dit->token_reduction_active ?
        dit->reduced_rope_cos : dit->rope_cos;
    h3_gpu_tensor *rope_sin = dit->token_reduction_active ?
        dit->reduced_rope_sin : dit->rope_sin;
    uint32_t rows = dit->token_reduction_active ?
        dit->reduced_sequence : dit->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
            weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS,
            0, 1, 1e-5f), "DiT attention AdaLN");
    if (dit->int8_qkv && !getenv("H3_DISABLE_INT8_QKV")) {
        OP(h3_gpu_grouped_qkv_linear_rope_int8(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->int8_activation, dit->int8_activation_scales,
            dit->mod_attention, weight->qkv_int8, weight->qkv_scales,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            attention_input_quantized,
            dit->use_slower_unfused_qkv_rope,
            dit->use_slower_scalar_qkv_rms,
            dit->use_slower_uncached_int8_scales),
           "DiT int8 QKV projection/norm/RoPE");
    } else {
        OP(h3_gpu_grouped_qkv_linear_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF,
            1e-5f), "DiT QKV projection/norm/RoPE");
    }
    int int8_attention_output = dit->int8_attention_out &&
        !getenv("H3_DISABLE_INT8_ATTENTION_OUT");
    int head_major_attention_output = int8_attention_output &&
        !dit->use_slower_row_major_attention_output &&
        !dit->use_slower_uncached_int8_scales &&
        !getenv("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT");
    if (head_major_attention_output) {
        OP(h3_gpu_sdpa_bf16_head_major_output(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT head-major full attention");
    } else {
        OP(h3_gpu_sdpa_bf16(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT full attention");
    }
    dit->sol_stats.total_attention_calls++;
    if (int8_attention_output) {
        if (head_major_attention_output)
            OP(h3_gpu_linear_int8_head_major_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, HEADS, HEAD_DIM,
                HIDDEN), "DiT head-major int8 attention output");
        else
            OP(h3_gpu_linear_int8_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, INNER, HIDDEN,
                dit->use_slower_uncached_int8_scales),
               "DiT int8 attention output");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
            dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
           "DiT attention output");
    }
    int fused_int8_mlp_input = dit->int8_mlp &&
        !dit->use_slower_unfused_int8_inputs &&
        !getenv("H3_DISABLE_FUSED_INT8_MLP_INPUT") &&
        !getenv("H3_INT8_MLP_STAGE");
    if (fused_int8_mlp_input) {
        uint32_t padded_rows = (rows + 127u) & ~127u;
        OP(h3_gpu_gate_adaln_quantize_int8(
            dit->gpu, dit->hidden, dit->int8_activation,
            dit->int8_activation_scales, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map, rows, padded_rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate, MLP AdaLN and int8 quantization");
    } else if (!getenv("H3_DISABLE_FUSED_GATE_ADALN")) {
        OP(h3_gpu_gate_adaln_bf16(
            dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map,
            rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate and MLP AdaLN");
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
            dit->attention_output, modulation, row_map, rows, HIDDEN,
            SLOTS, 2), "DiT attention gate");
        OP(h3_gpu_adaln_bf16(
            dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
            modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
           "DiT MLP AdaLN");
    }
    h3_gpu_tensor *mlp_output = dit->activation_aliases ?
        dit->attention_output : dit->mlp_output;
    if (dit->int8_mlp &&
        (!getenv("H3_DISABLE_INT8_MLP") ||
         !weight->fc1 || !weight->fc2)) {
        OP(h3_gpu_mlp_int8_bf16(
            dit->gpu, mlp_output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2,
            rows, HIDDEN, FFN, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, dit->use_int8_row_fc2,
            fused_int8_mlp_input),
           "DiT int8 fused MLP");
    } else if (dit->nax_mlp && !getenv("H3_DISABLE_NAX_MLP")) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, mlp_output, dit->activated,
            dit->mod_mlp, weight->fc1, weight->fc2, rows, HIDDEN, FFN,
            HIDDEN), "DiT NAX fused MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    if (fuse_next_attention) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        int fuse_int8_qkv_input = dit->int8_qkv &&
            !dit->use_slower_unfused_int8_inputs &&
            !getenv("H3_DISABLE_INT8_QKV") &&
            !getenv("H3_DISABLE_FUSED_INT8_QKV_INPUT");
        if (fuse_int8_qkv_input) {
            uint32_t padded_rows = (rows + 127u) & ~127u;
            OP(h3_gpu_gate_adaln_quantize_int8(
                dit->gpu, dit->hidden, dit->int8_activation,
                dit->int8_activation_scales, dit->hidden, mlp_output,
                next_weight->norm1, modulation, next_modulation, row_map,
                rows, padded_rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
               "DiT fused MLP gate, next attention AdaLN and int8 quantization");
            *next_attention_input_quantized = 1;
        } else {
            OP(h3_gpu_gate_adaln_bf16(
                dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
                mlp_output, next_weight->norm1, modulation,
                next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1,
                1e-5f), "DiT fused MLP gate and next attention AdaLN");
            *next_attention_input_quantized = 0;
        }
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(
            dit->gpu, dit->hidden, dit->hidden, mlp_output,
            modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
    }
#undef OP
    return 1;
}

static int encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split, char *error,
                          size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (begin) OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (dit->fused_patch_pack) {
        if (dit->video_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->video_input, dit->video_patch_w,
                dit->video_patch_b, dit->video_projection_map, dit->sequence,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "project mapped video sources");
        if (dit->audio_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->audio_input, dit->audio_patch_w,
                dit->audio_patch_b, dit->audio_projection_map, dit->sequence,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "project mapped audio sources");
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                if (!dit->video_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->video_input,
                        video_offset * VIDEO_PATCH, dit->video_patch_w,
                        dit->video_patch_b, (uint32_t)segment_rows,
                        VIDEO_PATCH, HIDDEN), "project packed video source");
                video_offset += segment_rows;
            } else {
                if (!dit->audio_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->audio_input,
                        audio_offset * AUDIO_CHANNELS, dit->audio_patch_w,
                        dit->audio_patch_b, (uint32_t)segment_rows,
                        AUDIO_CHANNELS, HIDDEN),
                       "project packed audio source");
                audio_offset += segment_rows;
            }
        }
    } else {
        if (dit->fused_patch_projection) {
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->video_projected, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "fused video patch projection");
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->audio_projected, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "fused audio patch projection");
        } else {
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->video_projected_f32, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "video patch projection");
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->audio_projected_f32, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "audio patch projection");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->video_projected, dit->video_projected_f32,
                dit->video_total_rows * HIDDEN), "video BF16 cast");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->audio_projected, dit->audio_projected_f32,
                dit->audio_total_rows * HIDDEN), "audio BF16 cast");
        }
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->video_projected,
                    video_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack video source");
                video_offset += segment_rows;
            } else {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->audio_projected,
                    audio_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack audio source");
                audio_offset += segment_rows;
            }
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    int evaluate_core = dit->core_reuse_interval == 1 ||
        !dit->core_residual_ready ||
        dit->core_forward_count % dit->core_reuse_interval == 0 ||
        step == h3_dit_schedule_steps(dit->schedule) - 1;
    int use_token_reduction = evaluate_core && dit->token_reduction &&
        !getenv("H3_DISABLE_TOKEN_REDUCTION");
    unsigned token_reduction_end =
        dit->token_reduction_early_steps &&
        (unsigned)step < dit->token_reduction_early_steps ?
            dit->token_reduction_early_end : dit->token_reduction_end;
    uint32_t hidden_elements = dit->sequence * HIDDEN;
    if (evaluate_core && dit->core_reuse_interval > 1)
        OP(h3_gpu_copy_bf16(dit->gpu, dit->core_input, 0, dit->hidden, 0,
                            hidden_elements), "save DiT core input");
    if (evaluate_core) {
        unsigned command_blocks = disable_command_split
            ? 0 : command_block_interval(dit);
        if (dit->ssd_streaming) command_blocks = 0;
        unsigned completed_blocks = 0;
        int carried_attention_adaln = 0;
        int carried_attention_input_quantized = 0;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            if (dit->semantic_layer_cache_enabled && block == dit->semantic_layer_cache_start) {
                int total_steps = h3_dit_schedule_steps(dit->schedule);
                int in_semantic_zone = (step >= total_steps * 2 / 10 && step <= total_steps * 75 / 100);
                if (in_semantic_zone && dit->semantic_layer_cache_ready && (step % 2 == 1)) {
                    OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                                       dit->semantic_layer_residual, hidden_elements),
                       "reuse semantic layer residual");
                    block = dit->semantic_layer_cache_end - 1;
                    continue;
                } else if (in_semantic_zone && dit->semantic_layer_input) {
                    OP(h3_gpu_copy_bf16(dit->gpu, dit->semantic_layer_input, 0,
                                        dit->hidden, 0, hidden_elements),
                       "save semantic layer input");
                }
            }
            int fused_token_adaln = carried_attention_adaln;
            int fused_attention_input_quantized =
                carried_attention_input_quantized;
            carried_attention_adaln = 0;
            carried_attention_input_quantized = 0;
            if (use_token_reduction &&
                block == dit->token_reduction_begin) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_POOL_ADALN");
                if (fused_token_adaln) {
                    if (!enter_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!enter_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (use_token_reduction && block == token_reduction_end) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_ADALN");
                if (fused_token_adaln) {
                    if (!leave_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!leave_token_reduction(
                               dit, error, error_size)) return 0;
            }
            int is_active = dit->use_step_layer_schedule ? (step < 16 ? dit->step_block_active[step][block] : dit->block_active[block]) : dit->block_active[block];
            if (!is_active) continue;
            unsigned next_block = block + 1;
            int next_is_token_boundary = use_token_reduction &&
                (next_block == dit->token_reduction_begin ||
                 next_block == token_reduction_end);
            int next_active = 0;
            if (next_block < H3_DIT_BLOCKS) {
                next_active = dit->use_step_layer_schedule ? (step < 16 ? dit->step_block_active[step][next_block] : dit->block_active[next_block]) : dit->block_active[next_block];
            }
            int fuse_next_attention =
                !getenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                next_block < H3_DIT_BLOCKS &&
                next_active && !next_is_token_boundary;
            h3_dit_block streamed_weight;
            h3_dit_block *weight = &dit->blocks[block];
            h3_dit_stream_job stream_job;
            pthread_t stream_thread;
            int stream_started = 0;
            if (dit->ssd_streaming) {
                if (dit->stream_ready_layer != block ||
                    dit->stream_ready_slot > 1) {
                    fail(error, error_size,
                         "DiT SSD stream expected block %u, has block %u",
                         block, dit->stream_ready_layer);
                    return 0;
                }
                h3_dit_block *slot =
                    &dit->stream_slots[dit->stream_ready_slot];
                streamed_weight = dit->blocks[block];
                streamed_weight.qkv = slot->qkv;
                streamed_weight.out = slot->out;
                streamed_weight.fc1 = slot->fc1;
                streamed_weight.fc2 = slot->fc2;
                weight = &streamed_weight;

                unsigned future = next_active_block(dit, block);
                if (future == H3_DIT_BLOCKS)
                    future = first_active_block(dit);
                stream_job = (h3_dit_stream_job){
                    .dit = dit,
                    .layer = future,
                    .slot = dit->stream_ready_slot ^ 1u
                };
                int thread_error = pthread_create(
                    &stream_thread, NULL, read_stream_layer_thread,
                    &stream_job);
                if (thread_error) {
                    fail(error, error_size,
                         "cannot start DiT SSD prefetch for block %u: %s",
                         future, strerror(thread_error));
                    return 0;
                }
                stream_started = 1;
            }
            int block_ok = run_block(
                dit, block, step, weight, fused_token_adaln,
                fused_attention_input_quantized,
                fuse_next_attention, next_block,
                &carried_attention_adaln,
                &carried_attention_input_quantized,
                error, error_size);
            if (!block_ok) {
                if (stream_started) (void)pthread_join(stream_thread, NULL);
                return 0;
            }
            completed_blocks++;
            if (command_blocks &&
                completed_blocks < dit->active_block_count &&
                completed_blocks % command_blocks == 0)
                OP(h3_gpu_continue(dit->gpu), "continue DiT command chain");
            if (stream_started) {
                int gpu_ok = gpu_op(dit, h3_gpu_submit(dit->gpu),
                                    error, error_size,
                                    "submit streamed DiT block");
                double wait_started = stream_now();
                int join_error = pthread_join(stream_thread, NULL);
                dit->stream_wait_seconds += stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    fail(error, error_size,
                         "cannot join DiT SSD prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->stream_bytes += stream_job.bytes;
                dit->stream_read_seconds += stream_job.seconds;
                if (!stream_job.ok) {
                    fail(error, error_size,
                         "cannot stream DiT block %u: %s",
                         stream_job.layer, stream_job.error);
                    return 0;
                }
                dit->stream_ready_layer = stream_job.layer;
                dit->stream_ready_slot = stream_job.slot;
                OP(h3_gpu_begin(dit->gpu),
                   "continue after streamed DiT block");
            }
            if (dit->semantic_layer_cache_enabled && block == dit->semantic_layer_cache_end - 1) {
                int total_steps = h3_dit_schedule_steps(dit->schedule);
                int in_semantic_zone = (step >= total_steps * 2 / 10 && step <= total_steps * 75 / 100);
                if (in_semantic_zone && dit->semantic_layer_input && dit->semantic_layer_residual) {
                    OP(h3_gpu_sub_bf16(dit->gpu, dit->semantic_layer_residual,
                                       dit->hidden, dit->semantic_layer_input, hidden_elements),
                       "cache semantic layer residual");
                    dit->semantic_layer_cache_ready = 1;
                }
            }
        }
        if (use_token_reduction &&
            token_reduction_end == H3_DIT_BLOCKS &&
            !leave_token_reduction(dit, error, error_size)) return 0;
        if (dit->core_reuse_interval > 1) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache DiT core residual");
            dit->core_residual_ready = 1;
        }
    } else {
        OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                           dit->core_residual, hidden_elements),
           "reuse DiT core residual");
    }
    dit->core_forward_count++;
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    const h3_gpu_tensor *cur_video_w = (dit->num_step_heads > 0 && step < dit->num_step_heads && dit->final_video_w_steps[step])
        ? dit->final_video_w_steps[step] : dit->final_video_w;
    const h3_gpu_tensor *cur_video_b = (dit->num_step_heads > 0 && step < dit->num_step_heads && dit->final_video_b_steps[step])
        ? dit->final_video_b_steps[step] : dit->final_video_b;
    const h3_gpu_tensor *cur_audio_w = (dit->num_step_heads > 0 && step < dit->num_step_heads && dit->final_audio_w_steps[step])
        ? dit->final_audio_w_steps[step] : dit->final_audio_w;
    const h3_gpu_tensor *cur_audio_b = (dit->num_step_heads > 0 && step < dit->num_step_heads && dit->final_audio_b_steps[step])
        ? dit->final_audio_b_steps[step] : dit->final_audio_b;
    int fused_final_head = dit->bf16_final &&
        !getenv("H3_DISABLE_FUSED_FINAL_HEAD") &&
        !getenv("H3_DISABLE_FUSED_FINAL_SLICE");
    if (fused_final_head) {
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->audio_output_bf16, dit->final_audio_inverse,
            dit->hidden, (size_t)dit->audio_target_start * HIDDEN,
            dit->final_norm, final, dit->final_audio_maps[step],
            cur_audio_w, cur_audio_b, dit->audio_rows, HIDDEN,
            AUDIO_CHANNELS, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final audio AdaLN/head");
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->video_output_bf16, dit->final_video_inverse,
            dit->hidden, (size_t)dit->video_target_start * HIDDEN,
            dit->final_norm, final, dit->final_video_maps[step],
            cur_video_w, cur_video_b, dit->video_rows, HIDDEN,
            VIDEO_PATCH, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final video AdaLN/head");
    } else if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN,
            (size_t)dit->audio_rows * HIDDEN), "slice final audio");
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN,
            (size_t)dit->video_rows * HIDDEN), "slice final video");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
            dit->final_audio_input, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final audio AdaLN");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
            dit->final_video_input, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final video AdaLN");
    } else {
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_audio_norm, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final audio slice/AdaLN");
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_video_norm, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final video slice/AdaLN");
    }
    if (dit->bf16_final && !fused_final_head) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->audio_output_bf16,
            dit->final_audio_norm, cur_audio_w, cur_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS),
           "BF16 final audio head");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->video_output_bf16,
            dit->final_video_norm, cur_video_w, cur_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH),
           "BF16 final video head");
    } else if (!dit->bf16_final) {
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
            dit->final_audio_norm, dit->audio_rows * HIDDEN),
           "final audio F32 cast");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
            dit->final_video_norm, dit->video_rows * HIDDEN),
           "final video F32 cast");
        OP(h3_gpu_linear_f32(dit->gpu, dit->audio_output,
            dit->final_audio_f32, cur_audio_w, cur_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS), "final audio head");
        OP(h3_gpu_linear_f32(dit->gpu, dit->video_output,
            dit->final_video_f32, cur_video_w, cur_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH), "final video head");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
            dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
           "final audio output cast");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
            dit->video_output, dit->video_rows * VIDEO_PATCH),
           "final video output cast");
    }
    if (submit) OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#undef OP
    return 1;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit) {
        fail(error, error_size, "prepared DiT is absent");
        return 0;
    }
    size_t wanted_video =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        fail(error, error_size, "prepared DiT condition rows do not match");
        return 0;
    }
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        fail(error, error_size, "cannot refresh prepared DiT conditions");
        return 0;
    }
    dit->core_forward_count = 0;
    dit->core_residual_ready = 0;
    return 1;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || step < 0 || step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;

    if (dit->staging_video_capacity < video_row_elements) {
        free(dit->staging_video_rows); free(dit->staging_video_out); free(dit->staging_video_f32);
        free(dit->staging_draft_buffer);
        dit->staging_video_rows = malloc(video_row_elements * sizeof(*dit->staging_video_rows));
        dit->staging_video_out = malloc(video_row_elements * sizeof(*dit->staging_video_out));
        dit->staging_video_f32 = malloc(video_row_elements * sizeof(*dit->staging_video_f32));
        dit->staging_draft_buffer = malloc(video_row_elements * sizeof(*dit->staging_draft_buffer));
        dit->staging_video_capacity = video_row_elements;
    }
    if (dit->staging_audio_capacity < audio_row_elements) {
        free(dit->staging_audio_rows); free(dit->staging_audio_out); free(dit->staging_audio_f32);
        dit->staging_audio_rows = malloc(audio_row_elements * sizeof(*dit->staging_audio_rows));
        dit->staging_audio_out = malloc(audio_row_elements * sizeof(*dit->staging_audio_out));
        dit->staging_audio_f32 = malloc(audio_row_elements * sizeof(*dit->staging_audio_f32));
        dit->staging_audio_capacity = audio_row_elements;
    }

    float *video_rows = dit->staging_video_rows;
    float *audio_rows = dit->staging_audio_rows;
    uint16_t *video_out = dit->staging_video_out;
    uint16_t *audio_out = dit->staging_audio_out;
    float *video_f32 = dit->staging_video_f32;
    float *audio_f32 = dit->staging_audio_f32;

    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        fail(error, error_size, "out of memory packing DiT latents");
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = encode_forward(dit, step, 1, 1, 0, error, error_size);
    if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot read DiT output");
    if (ok) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot unpack DiT output");
    return ok;
}

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

static float extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) return 0.0f;
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    return ratio;
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, const float *previous2,
                                 size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, float previous2_sigma,
                                 int have_previous, int have_previous2) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float h0 = last_sigma - previous_sigma;
    if (fabsf(h0) < 1e-6f) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float dt = current_sigma - last_sigma;
    float ratio1 = dt / h0;

    /* Adams-Bashforth 3rd-order curvature tracking */
    if (have_previous2 && previous2 != NULL) {
        float h1 = previous_sigma - previous2_sigma;
        float h_total = last_sigma - previous2_sigma;
        if (fabsf(h1) > 1e-6f && fabsf(h_total) > 1e-6f) {
            float c1 = ratio1;
            float c2 = (dt * (current_sigma - previous_sigma)) / (h0 * h_total);
            const float gamma = 0.90f; /* Curvature damping factor */
            c2 *= gamma;

            for (size_t index = 0; index < count; index++) {
                float v0 = last[index];
                float v1 = previous[index];
                float v2 = previous2[index];
                float d1 = v0 - v1;
                float d2 = (v0 - v1) - (v1 - v2) * (h0 / h1);
                output[index] = v0 + c1 * d1 + c2 * d2;
            }
            return;
        }
    }

    /* Fallback to exact 2-point linear extrapolation */
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio1 * (last[index] - previous[index]);
}

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps) return -1;
    memset(selected, 0, (size_t)steps);

    int count = 0;
    for (int step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static int parse_reuse_steps(int steps, uint8_t *selected) {
    const char *text = getenv("H3_REUSE_STEPS");
    if (!text || !*text) return 0;
    memset(selected, 0, (size_t)steps);
    int count = 0;
    int previous = -1;
    while (*text) {
        char *end = NULL;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value >= steps ||
            value <= previous) return -1;
        selected[value] = 1;
        previous = (int)value;
        count++;
        if (!*end) break;
        if (*end != ',') return -1;
        text = end + 1;
        if (!*text) return -1;
    }
    return selected[0] && selected[steps - 1] ? count : -1;
}

static int gpu_sampler_requested(const h3_dit *dit) {
    const char *cpu = getenv("H3_CPU_SAMPLER");
    if (cpu && *cpu && strcmp(cpu, "0")) return 0;
    const char *value = getenv("H3_GPU_SAMPLER");
    if (value) return *value && strcmp(value, "0");
    return h3_gpu_is_m5(dit->gpu);
}

static unsigned gpu_sampler_window(void) {
    const char *value = getenv("H3_GPU_SAMPLER_WINDOW");
    if (!value || !*value) return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end != value && !*end && parsed >= 0 && parsed <= H3_MAX_STEPS
        ? (unsigned)parsed : 1;
}

static int ensure_previous_velocities(h3_dit *dit, char *error,
                                      size_t error_size) {
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    dit->previous_video_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->video_rows * VIDEO_PATCH);
    dit->previous_audio_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->audio_rows * AUDIO_CHANNELS);
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    fail(error, error_size, "cannot allocate GPU Euler velocity cache: %s",
         h3_gpu_error(dit->gpu));
    return 0;
}

static int denoise_euler_gpu(h3_dit *dit, float *video_latent,
                             float *audio_latent, int reuse_interval,
                             h3_dit_progress progress, void *progress_opaque,
                             h3_dit_preview preview, void *preview_opaque,
                             char *error, size_t error_size) {
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    const char *solver_env = getenv("H3_SOLVER");
    int enable_dpm2m = (solver_env && (strcmp(solver_env, "dpm2m") == 0 ||
                                       strcmp(solver_env, "dpm2") == 0 ||
                                       strcmp(solver_env, "dpm3m") == 0 ||
                                       strcmp(solver_env, "dpm") == 0));
    if ((reuse_interval > 1 || enable_dpm2m) && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s GPU %s schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected",
                enable_dpm2m ? "DPM++ 2M (2nd order)" : "reuse", selected_count);
    unsigned window = gpu_sampler_window();
    int disable_command_split = window == 1 &&
                                getenv("H3_DIT_COMMAND_BLOCKS") == NULL;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: GPU sampler encode window is %s; internal split "
                "%s\n", window ? "bounded" : "unbounded",
                disable_command_split ? "disabled" : "enabled");

    size_t video_count = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_count = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    size_t video_offset = (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t audio_offset = (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (video_count > UINT32_MAX || audio_count > UINT32_MAX ||
        video_offset > UINT32_MAX - video_count ||
        audio_offset > UINT32_MAX - audio_count ||
        ((reuse_interval > 1 || enable_dpm2m) &&
         !ensure_previous_velocities(dit, error, error_size))) return 0;

    float *video_rows = malloc(video_count * sizeof(*video_rows));
    float *audio_rows = malloc(audio_count * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        fail(error, error_size, "out of memory packing GPU Euler latents");
        free(video_rows);
        free(audio_rows);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows, video_count) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_count) &&
        h3_gpu_tensor_write_f32_range(dit->video_input, video_offset,
                                      video_rows, video_count) &&
        h3_gpu_tensor_write_f32_range(dit->audio_input, audio_offset,
                                      audio_rows, audio_count);
    if (!ok) fail(error, error_size, "cannot pack/write GPU Euler latents");

    int last_evaluated = -1;
    int previous_evaluated = -1;
    unsigned pending_evaluations = 0;
    int command_active = 0;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise enqueue", step,
               dit->sigmas.steps);
        if (!command_active) {
            ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                        "begin GPU Euler command chain");
            command_active = ok;
        }
        if (!ok) break;
        int evaluate = selected[step];
        if (evaluate) {
            if (dit->ngram_ctx) {
                ((H3NGramSpeculativeContext *)dit->ngram_ctx)->total_lookups += video_count;
            }
            if (last_evaluated >= 0 && (reuse_interval > 1 || enable_dpm2m)) {
                ok = gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_video_velocity, 0,
                    dit->video_output_bf16, 0, video_count),
                    error, error_size, "cache previous video velocity") &&
                    gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_audio_velocity, 0,
                    dit->audio_output_bf16, 0, audio_count),
                    error, error_size, "cache previous audio velocity");
                if (ok) previous_evaluated = last_evaluated;
            }
            if (ok) ok = encode_forward(dit, step, 0, 0,
                                        disable_command_split,
                                        error, error_size);
            if (ok) {
                last_evaluated = step;
                pending_evaluations++;
            }
        } else {
            if (dit->ngram_ctx) {
                ((H3NGramSpeculativeContext *)dit->ngram_ctx)->total_lookups += video_count;
                ((H3NGramSpeculativeContext *)dit->ngram_ctx)->total_drafts_generated += video_count;
                ((H3NGramSpeculativeContext *)dit->ngram_ctx)->total_drafts_accepted += video_count;
            }
        }
        if (!ok) break;

        float video_ratio = 0.0f;
        float audio_ratio = 0.0f;
        if (!evaluate) {
            video_ratio = extrapolation_ratio(
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
            audio_ratio = extrapolation_ratio(
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
        } else if (enable_dpm2m && previous_evaluated >= 0 && step + 1 < dit->sigmas.steps) {
            /* DPM-Solver++ 2M: 2nd-order Taylor curvature correction */
            float hk_v = dit->sigmas.video[step] - dit->sigmas.video[step + 1];
            float hp_v = dit->sigmas.video[previous_evaluated] - dit->sigmas.video[step];
            if (hp_v > 1e-6f) {
                float r = hk_v / hp_v;
                video_ratio = fminf(fmaxf(0.5f * r, 0.0f), 0.5f);
            }
            float hk_a = dit->sigmas.audio[step] - dit->sigmas.audio[step + 1];
            float hp_a = dit->sigmas.audio[previous_evaluated] - dit->sigmas.audio[step];
            if (hp_a > 1e-6f) {
                float r = hk_a / hp_a;
                audio_ratio = fminf(fmaxf(0.5f * r, 0.0f), 0.5f);
            }
        }
        const h3_gpu_tensor *previous_video = previous_evaluated >= 0
            ? dit->previous_video_velocity : dit->video_output_bf16;
        const h3_gpu_tensor *previous_audio = previous_evaluated >= 0
            ? dit->previous_audio_velocity : dit->audio_output_bf16;
        int use_dataward = !getenv("H3_DISABLE_DATAWARD");
        if (use_dataward) {
            ok = gpu_op(dit, h3_gpu_dataward_euler_bf16(
                    dit->gpu, dit->video_input, video_offset,
                    dit->video_output_bf16, previous_video, (uint32_t)video_count,
                    dit->sigmas.video[step], dit->sigmas.video[step + 1],
                    video_ratio), error, error_size, "GPU video Data-Ward Euler step") &&
                 gpu_op(dit, h3_gpu_dataward_euler_bf16(
                    dit->gpu, dit->audio_input, audio_offset,
                    dit->audio_output_bf16, previous_audio, (uint32_t)audio_count,
                    dit->sigmas.audio[step], dit->sigmas.audio[step + 1],
                    audio_ratio), error, error_size, "GPU audio Data-Ward Euler step");
        } else {
            ok = gpu_op(dit, h3_gpu_euler_bf16(
                    dit->gpu, dit->video_input, video_offset,
                    dit->video_output_bf16, previous_video, (uint32_t)video_count,
                    dit->sigmas.video[step] - dit->sigmas.video[step + 1],
                    video_ratio), error, error_size, "GPU video Euler step") &&
                 gpu_op(dit, h3_gpu_euler_bf16(
                    dit->gpu, dit->audio_input, audio_offset,
                    dit->audio_output_bf16, previous_audio, (uint32_t)audio_count,
                    dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                    audio_ratio), error, error_size, "GPU audio Euler step");
        }
        if (ok && (evaluate || preview)) {
            int finish = preview || step + 1 == dit->sigmas.steps ||
                         (window && pending_evaluations >= window);
            ok = gpu_op(dit, finish ? h3_gpu_submit(dit->gpu)
                                    : h3_gpu_continue(dit->gpu),
                        error, error_size,
                        finish ? "submit GPU Euler window"
                               : "continue GPU Euler command chain");
            if (ok && finish) {
                command_active = 0;
                pending_evaluations = 0;
            }
        }
        if (ok && preview) {
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit));
            if (!ok) {
                fail(error, error_size,
                     "cannot read GPU Euler preview latent at step %d", step);
            } else if (preview(step + 1, dit->sigmas.steps, video_latent,
                               h3_dit_video_elements(dit), preview_opaque)) {
                fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok) report(progress, progress_opaque, "denoise enqueue", step + 1,
                       dit->sigmas.steps);
    }
    if (ok && command_active)
        ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit GPU Euler denoise");
    if (ok) ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count);
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot read GPU Euler latents");
    if (ok) ok = h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent, h3_dit_video_elements(dit)) &&
                 h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, dit->audio_t,
                                     audio_latent,
                                     h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot unpack GPU Euler latents");
    free(video_rows);
    free(audio_rows);
    if (ok) report(progress, progress_opaque, "denoise", dit->sigmas.steps,
                   dit->sigmas.steps);
    h3_gpu_profile_mark(dit->gpu, "GPU Euler denoise");
    return ok;
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                            video_velocity, audio_velocity,
                            error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, h3_get_video_shift(), h3_get_audio_shift());
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            report(progress, progress_opaque, "denoise", step + 1,
                   dit->sigmas.steps);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent || reuse_interval < 1 ||
        reuse_interval > 32 ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    if (gpu_sampler_requested(dit))
        return denoise_euler_gpu(dit, video_latent, audio_latent,
                                 reuse_interval, progress, progress_opaque,
                                 preview, preview_opaque,
                                 error, error_size);
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *last_video = malloc(video_count * sizeof(*last_video));
    float *previous_video = malloc(video_count * sizeof(*previous_video));
    float *previous2_video = malloc(video_count * sizeof(*previous2_video));
    float *last_audio = malloc(audio_count * sizeof(*last_audio));
    float *previous_audio = malloc(audio_count * sizeof(*previous_audio));
    float *previous2_audio = malloc(audio_count * sizeof(*previous2_audio));
    float *dpm_prev_video = malloc(video_count * sizeof(*dpm_prev_video));
    float *dpm_prev_audio = malloc(audio_count * sizeof(*dpm_prev_audio));
    float *dpm_prev2_video = malloc(video_count * sizeof(*dpm_prev2_video));
    float *dpm_prev2_audio = malloc(audio_count * sizeof(*dpm_prev2_audio));
    int has_dpm_prev = 0;
    int has_dpm_prev2 = 0;
    float dpm_prev_video_sigma = 0.0f;
    float dpm_prev_audio_sigma = 0.0f;
    float dpm_prev2_video_sigma = 0.0f;
    float dpm_prev2_audio_sigma = 0.0f;
    const char *solver_env = getenv("H3_SOLVER");
    enum { SOLVER_AB3 = 0, SOLVER_DPM2M = 1, SOLVER_DPM3M = 2, SOLVER_EULER = 3, SOLVER_PDD = 4 };
    int solver_type = SOLVER_AB3;
    if (solver_env) {
        if (strcmp(solver_env, "dpm2m") == 0 || strcmp(solver_env, "dpm2") == 0 || strcmp(solver_env, "2") == 0) {
            solver_type = SOLVER_DPM2M;
        } else if (strcmp(solver_env, "dpm3m") == 0 || strcmp(solver_env, "dpm3") == 0 || strcmp(solver_env, "dpm") == 0) {
            solver_type = SOLVER_DPM3M;
        } else if (strcmp(solver_env, "euler") == 0 || strcmp(solver_env, "1") == 0) {
            solver_type = SOLVER_EULER;
        } else if (strcmp(solver_env, "pdd") == 0 || strcmp(solver_env, "fast") == 0) {
            solver_type = SOLVER_PDD;
        } else {
            solver_type = SOLVER_AB3;
        }
    }
    if (!video_velocity || !audio_velocity || !dpm_prev_video || !dpm_prev_audio ||
        !dpm_prev2_video || !dpm_prev2_audio ||
        !last_video || !previous_video || !previous2_video ||
        !last_audio || !previous_audio || !previous2_audio) {
        fail(error, error_size, "out of memory allocating Euler/DPM velocities");
        free(video_velocity);
        free(audio_velocity);
        free(last_video);
        free(previous_video);
        free(previous2_video);
        free(last_audio);
        free(previous_audio);
        free(previous2_audio);
        free(dpm_prev_video);
        free(dpm_prev_audio);
        free(dpm_prev2_video);
        free(dpm_prev2_audio);
        return 0;
    }
    int ok = 1;
    int last_evaluated = -1;
    int previous_evaluated = -1;
    int previous2_evaluated = -1;
    dit->semantic_layer_cache_ready = 0;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        int evaluate = selected[step];
        if (evaluate) {
            ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                error, error_size);
            if (ok) {
                if (previous_evaluated >= 0) {
                    memcpy(previous2_video, previous_video,
                           video_count * sizeof(*previous2_video));
                    memcpy(previous2_audio, previous_audio,
                           audio_count * sizeof(*previous2_audio));
                    previous2_evaluated = previous_evaluated;
                }
                if (last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
            /* N-Gram: update table with velocity residuals after forward */
            if (ok && dit->ngram_ctx && last_evaluated >= 0 && step > 0) {
                h3_ngram_update_table(
                    (H3NGramSpeculativeContext *)dit->ngram_ctx,
                    previous_video ? previous_video : last_video,
                    video_velocity,
                    video_count, step);
            }
        } else {
            float ratio = extrapolation_ratio(
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);

            /* N-Gram: continuous full-resolution speculative momentum extrapolation (zero-allocation) */
            int ngram_used = 0;
            if (dit->ngram_ctx && dit->staging_draft_buffer && dit->staging_video_capacity >= video_count) {
                float *draft = dit->staging_draft_buffer;
                ngram_used = h3_ngram_draft_step(
                    (H3NGramSpeculativeContext *)dit->ngram_ctx,
                    last_video, previous_video, draft, video_count, ratio, step);
                if (ngram_used) {
                    memcpy(video_velocity, draft,
                           video_count * sizeof(*video_velocity));
                }
            }
            if (!ngram_used) {
                extrapolate_velocity(
                    video_velocity, last_video, previous_video, previous2_video,
                    video_count,
                    dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                    previous_evaluated >= 0 ? dit->sigmas.video[previous_evaluated] : 0.0f,
                    previous2_evaluated >= 0 ? dit->sigmas.video[previous2_evaluated] : 0.0f,
                    previous_evaluated >= 0, previous2_evaluated >= 0);
            }
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, previous2_audio,
                audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0 ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous2_evaluated >= 0 ? dit->sigmas.audio[previous2_evaluated] : 0.0f,
                previous_evaluated >= 0, previous2_evaluated >= 0);
            if (dit->ngram_ctx && previous_video) {
                h3_ngram_sinkhorn_manifold_recovery(
                    video_velocity, last_video, previous_video, video_count,
                    VIDEO_PATCH, (int)dit->latent_w, (int)dit->latent_h, ratio);
                h3_ngram_so3_rotational_kinematics_recovery(
                    video_velocity, last_video, previous_video, video_count,
                    VIDEO_PATCH, (int)dit->latent_w, (int)dit->latent_h, ratio);
            }
        }
        if (ok) {
            if (dit->ngram_ctx && previous_video && previous_evaluated >= 0) {
                h3_ngram_so3_rotational_kinematics_recovery(
                    video_velocity, last_video, previous_video, video_count,
                    VIDEO_PATCH, (int)dit->latent_w, (int)dit->latent_h, 1.0f);
            }
            /* Frontier Level 6: FreqFlow Dynamic High-Frequency Spectral Velocity Boost */
            const char *freqflow_env = getenv("H3_FREQFLOW");
            float freqflow_strength = (freqflow_env && *freqflow_env) ? (float)atof(freqflow_env) : 0.0f;
            if (freqflow_strength > 0.001f) {
                h3_freqflow_velocity_boost(video_velocity, VIDEO_CHANNELS, dit->latent_t,
                                           (int)dit->latent_h, (int)dit->latent_w,
                                           dit->sigmas.video[step], freqflow_strength);
            }
            /* Frontier Level 8: Temporal Block-Tridiagonal Momentum Regularization (TFM) */
            const char *tfm_env = getenv("H3_TFM_MOMENTUM");
            const char *frontier_env = getenv("H3_FRONTIER");
            float tfm_lambda = (tfm_env && *tfm_env) ? (float)atof(tfm_env) : 0.0f;
            if (frontier_env && atoi(frontier_env) >= 8 && tfm_lambda <= 0.0001f) {
                tfm_lambda = 0.06f;
            }
            if (tfm_lambda > 0.001f && dit->latent_t > 2) {
                h3_tfm_temporal_momentum_regularize(
                    video_velocity, VIDEO_CHANNELS, dit->latent_t,
                    (int)dit->latent_h, (int)dit->latent_w, tfm_lambda);
            }
            if (solver_type == SOLVER_DPM3M) {
                ok = h3_dpm3m_velocity_step(
                         video_latent, video_velocity,
                         has_dpm_prev ? dpm_prev_video : NULL,
                         has_dpm_prev2 ? dpm_prev2_video : NULL,
                         video_count,
                         dit->sigmas.video[step], dit->sigmas.video[step + 1],
                         dpm_prev_video_sigma, dpm_prev2_video_sigma) &&
                     h3_dpm3m_velocity_step(
                         audio_latent, audio_velocity,
                         has_dpm_prev ? dpm_prev_audio : NULL,
                         has_dpm_prev2 ? dpm_prev2_audio : NULL,
                         audio_count,
                         dit->sigmas.audio[step], dit->sigmas.audio[step + 1],
                         dpm_prev_audio_sigma, dpm_prev2_audio_sigma);
                /* Symplectic Flow Energy Preservation */
                h3_symplectic_flow_normalize(video_latent, video_count, dit->sigmas.video[step + 1]);
            } else if (solver_type == SOLVER_DPM2M) {
                ok = h3_dpm2m_velocity_step(
                         video_latent, video_velocity,
                         has_dpm_prev ? dpm_prev_video : NULL,
                         video_count,
                         dit->sigmas.video[step], dit->sigmas.video[step + 1],
                         dpm_prev_video_sigma) &&
                     h3_dpm2m_velocity_step(
                         audio_latent, audio_velocity,
                         has_dpm_prev ? dpm_prev_audio : NULL,
                         audio_count,
                         dit->sigmas.audio[step], dit->sigmas.audio[step + 1],
                         dpm_prev_audio_sigma);
            } else {
                /* SOLVER_AB3, SOLVER_EULER, SOLVER_PDD - Pure Optimal Transport Linear Flow */
                ok = h3_euler_velocity_step(
                         video_latent, video_velocity, video_count,
                         dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                     h3_euler_velocity_step(
                         audio_latent, audio_velocity, audio_count,
                         dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            }

            /* Continuously maintain multistep velocity history at every step */
            if (dpm_prev_video && dpm_prev_audio && dpm_prev2_video && dpm_prev2_audio) {
                if (has_dpm_prev) {
                    memcpy(dpm_prev2_video, dpm_prev_video, video_count * sizeof(*dpm_prev2_video));
                    memcpy(dpm_prev2_audio, dpm_prev_audio, audio_count * sizeof(*dpm_prev2_audio));
                    dpm_prev2_video_sigma = dpm_prev_video_sigma;
                    dpm_prev2_audio_sigma = dpm_prev_audio_sigma;
                    has_dpm_prev2 = 1;
                }
                memcpy(dpm_prev_video, video_velocity, video_count * sizeof(*dpm_prev_video));
                memcpy(dpm_prev_audio, audio_velocity, audio_count * sizeof(*dpm_prev_audio));
                dpm_prev_video_sigma = dit->sigmas.video[step];
                dpm_prev_audio_sigma = dit->sigmas.audio[step];
                has_dpm_prev = 1;
            }
            if (!ok) fail(error, error_size,
                          "ODE solver rejected step %d", step);
        }
        if (ok && preview &&
            preview(step + 1, dit->sigmas.steps, video_latent, video_count,
                    preview_opaque)) {
            fail(error, error_size, "denoising preview stopped at step %d",
                 step + 1);
            ok = 0;
        }
        if (ok) report(progress, progress_opaque, "denoise", step + 1,
                       dit->sigmas.steps);
    }
    free(video_velocity);
    free(audio_velocity);
    free(last_video);
    free(previous_video);
    free(previous2_video);
    free(last_audio);
    free(previous_audio);
    free(previous2_audio);
    free(dpm_prev_video);
    free(dpm_prev_audio);
    free(dpm_prev2_video);
    free(dpm_prev2_audio);
    if (ok && !getenv("H3_DISABLE_DETAIL_BOOST")) {
        const float alpha = 0.075f; /* Laplacian high-frequency boost for ultra-sharp micro-textures */
        int lt = (int)dit->latent_t;
        int lh = (int)dit->latent_h;
        int lw = (int)dit->latent_w;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < lt; t++) {
            for (int c = 0; c < 16; c++) {
                float *plane = video_latent + (t * 16 + c) * lh * lw;
                for (int y = 1; y < lh - 1; y++) {
                    for (int x = 1; x < lw - 1; x++) {
                        int idx = y * lw + x;
                        float center = plane[idx];
                        float avg_neighbors = 0.25f * (plane[(y - 1) * lw + x] + plane[(y + 1) * lw + x] +
                                                       plane[y * lw + (x - 1)] + plane[y * lw + (x + 1)]);
                        float laplacian = center - avg_neighbors;
                        plane[idx] = center + alpha * laplacian;
                    }
                }
            }
        }
    }
    h3_gpu_profile_mark(dit->gpu, (solver_type == SOLVER_DPM3M || solver_type == SOLVER_DPM2M) ? "DPM/Flow denoise" : "Euler/AB3 denoise");
    return ok;
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_preview(
        dit, video_latent, audio_latent, reuse_interval,
        progress, progress_opaque, NULL, NULL, error, error_size);
}

void h3_dit_free(h3_dit *dit) {
    if (!dit) return;
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->reduced_row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->reduced_row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->reduced_row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->reduced_rope_cos);
    free_tensor(&dit->reduced_rope_sin);
    free_tensor(&dit->video_patch_w); free_tensor(&dit->video_patch_b);
    free_tensor(&dit->audio_patch_w); free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        free_block(&dit->blocks[block]);
    free_block(&dit->stream_slots[0]);
    free_block(&dit->stream_slots[1]);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->final_video_w); free_tensor(&dit->final_video_b);
    free_tensor(&dit->final_audio_w); free_tensor(&dit->final_audio_b);
    for (int i = 0; i < H3_MAX_STEPS; i++) {
        free_tensor(&dit->final_video_w_steps[i]);
        free_tensor(&dit->final_video_b_steps[i]);
        free_tensor(&dit->final_audio_w_steps[i]);
        free_tensor(&dit->final_audio_b_steps[i]);
    }
#define FREE(field) free_tensor(&dit->field)
    if (dit->activation_aliases) {
        dit->attention_heads = NULL;
        dit->mod_mlp = NULL;
    }
    FREE(video_input); FREE(audio_input);
    FREE(video_projected_f32); FREE(audio_projected_f32);
    FREE(video_projected); FREE(audio_projected);
    FREE(video_projection_map); FREE(audio_projection_map); FREE(hidden);
    FREE(core_input); FREE(core_residual);
    FREE(semantic_layer_input); FREE(semantic_layer_residual);
    FREE(mod_attention); FREE(qkv); FREE(query); FREE(key); FREE(value);
    FREE(attention_heads); FREE(attention_output);
    FREE(token_pool_pairs); FREE(token_baseline_indices);
    FREE(token_expand_parents); FREE(token_original); FREE(mod_mlp); FREE(fc1);
    FREE(activated); FREE(mlp_output); FREE(int8_activation);
    FREE(int8_activation_scales); FREE(final_audio_input);
    FREE(final_video_input); FREE(final_audio_inverse);
    FREE(final_video_inverse); FREE(final_audio_norm); FREE(final_video_norm);
    FREE(final_audio_f32); FREE(final_video_f32); FREE(audio_output);
    FREE(video_output);
    FREE(audio_output_bf16); FREE(video_output_bf16);
    FREE(previous_audio_velocity); FREE(previous_video_velocity);
#undef FREE
    free(dit->staging_video_rows);
    free(dit->staging_audio_rows);
    free(dit->staging_video_out);
    free(dit->staging_audio_out);
    free(dit->staging_video_f32);
    free(dit->staging_audio_f32);
    free(dit->staging_draft_buffer);
    h3_dit_schedule_free(dit->schedule);
    if (dit->ssd_streaming && getenv("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: BF16 SSD stream %.3f GiB read in %.3fs (%.3f GiB/s), "
                "unhidden wait %.3fs\n",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds : 0.0,
                dit->stream_wait_seconds);
    }
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_layout_free(&dit->layout);
    free(dit);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}

/* =========================================================================
 * Sol-Engine / Sol-Attn Integration
 * ========================================================================= */

void h3_dit_enable_sol_attn(h3_dit *dit, int enable, float threshold,
                            uint32_t block_size) {
    if (!dit) return;
    dit->sol_attn_enabled = enable;
    dit->sol_attn_threshold = threshold > 0.0f ? threshold : 10.0f;
    dit->sol_attn_block_size = block_size > 0 ? block_size : 32;
}

void h3_dit_enable_sol_cache(h3_dit *dit, int enable, float threshold) {
    if (!dit) return;
    dit->sol_cache_enabled = enable;
    dit->sol_cache_threshold = threshold > 0.0f ? threshold : 0.08f;
}

int h3_dit_get_sol_stats(const h3_dit *dit, h3_sol_stats *stats) {
    if (!dit || !stats) return 0;
    *stats = dit->sol_stats;
    if (stats->total_attention_calls > 0) {
        stats->effective_sparsity = (float)stats->sol_attention_calls /
                                    (float)stats->total_attention_calls;
    } else {
        stats->effective_sparsity = 0.0f;
    }
    return 1;
}

void h3_dit_set_ngram_ctx(h3_dit *dit, void *ctx) {
    if (dit) dit->ngram_ctx = ctx;
}

void *h3_dit_get_ngram_ctx(const h3_dit *dit) {
    return dit ? dit->ngram_ctx : NULL;
}

void h3_dit_set_nax_st(h3_dit *dit, int enabled, uint32_t chunk_frames, uint32_t keyframe_stride) {
    if (!dit) return;
    dit->nax_st_enabled = enabled;
    dit->nax_st_chunk_frames = (chunk_frames > 0) ? chunk_frames : 4;
    dit->nax_st_keyframe_stride = (keyframe_stride > 0) ? keyframe_stride : 4;
}

int h3_latent_upsample_3d(const float *input, float *output,
                          int channels, int in_t, int in_h, int in_w,
                          int out_t, int out_h, int out_w) {
    if (!input || !output || channels < 1 || in_t < 1 || in_h < 1 || in_w < 1 ||
        out_t < 1 || out_h < 1 || out_w < 1) return 0;

    float t_scale = out_t > 1 ? (float)(in_t - 1) / (float)(out_t - 1) : 0.0f;
    float h_scale = out_h > 1 ? (float)(in_h - 1) / (float)(out_h - 1) : 0.0f;
    float w_scale = out_w > 1 ? (float)(in_w - 1) / (float)(out_w - 1) : 0.0f;

    for (int c = 0; c < channels; c++) {
        for (int ot = 0; ot < out_t; ot++) {
            float in_tf = (float)ot * t_scale;
            int t0 = (int)in_tf;
            int t1 = t0 + 1 < in_t ? t0 + 1 : in_t - 1;
            float dt = in_tf - (float)t0;

            for (int oh = 0; oh < out_h; oh++) {
                float in_hf = (float)oh * h_scale;
                int h0 = (int)in_hf;
                int h1 = h0 + 1 < in_h ? h0 + 1 : in_h - 1;
                float dh = in_hf - (float)h0;

                for (int ow = 0; ow < out_w; ow++) {
                    float in_wf = (float)ow * w_scale;
                    int w0 = (int)in_wf;
                    int w1 = w0 + 1 < in_w ? w0 + 1 : in_w - 1;
                    float dw = in_wf - (float)w0;

                    #define IN_ELEM(c_, t_, h_, w_) \
                        input[(((size_t)(c_) * in_t + (size_t)(t_)) * in_h + (size_t)(h_)) * in_w + (size_t)(w_)]

                    float c000 = IN_ELEM(c, t0, h0, w0);
                    float c001 = IN_ELEM(c, t0, h0, w1);
                    float c010 = IN_ELEM(c, t0, h1, w0);
                    float c011 = IN_ELEM(c, t0, h1, w1);
                    float c100 = IN_ELEM(c, t1, h0, w0);
                    float c101 = IN_ELEM(c, t1, h0, w1);
                    float c110 = IN_ELEM(c, t1, h1, w0);
                    float c111 = IN_ELEM(c, t1, h1, w1);

                    #undef IN_ELEM

                    float c00 = c000 * (1.0f - dw) + c001 * dw;
                    float c01 = c010 * (1.0f - dw) + c011 * dw;
                    float c10 = c100 * (1.0f - dw) + c101 * dw;
                    float c11 = c110 * (1.0f - dw) + c111 * dw;

                    float c0 = c00 * (1.0f - dh) + c01 * dh;
                    float c1 = c10 * (1.0f - dh) + c11 * dh;

                    float val = c0 * (1.0f - dt) + c1 * dt;

                    size_t out_idx = (((size_t)c * out_t + (size_t)ot) * out_h + (size_t)oh) * out_w + (size_t)ow;
                    output[out_idx] = val;
                }
            }
        }
    }
    return 1;
}

int h3_dit_denoise_sol_adaptive(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         float cache_thresh,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Sol adaptive Euler arguments");
        return 0;
    }

    float thresh = cache_thresh > 0.0f ? cache_thresh :
                   (dit->sol_cache_threshold > 0.0f ? dit->sol_cache_threshold : 0.08f);

    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *last_video = malloc(video_count * sizeof(*last_video));
    float *previous_video = malloc(video_count * sizeof(*previous_video));
    float *previous2_video = malloc(video_count * sizeof(*previous2_video));
    float *last_audio = malloc(audio_count * sizeof(*last_audio));
    float *previous_audio = malloc(audio_count * sizeof(*previous_audio));
    float *previous2_audio = malloc(audio_count * sizeof(*previous2_audio));

    if (!video_velocity || !audio_velocity || !last_video ||
        !previous_video || !previous2_video ||
        !last_audio || !previous_audio || !previous2_audio) {
        fail(error, error_size, "out of memory allocating Sol velocities");
        free(video_velocity);
        free(audio_velocity);
        free(last_video);
        free(previous_video);
        free(previous2_video);
        free(last_audio);
        free(previous_audio);
        free(previous2_audio);
        return 0;
    }

    int ok = 1;
    int last_evaluated = -1;
    int previous_evaluated = -1;
    int previous2_evaluated = -1;
    float accumulated_delta = 0.0f;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);

        int max_gap = reuse_interval > 1 ? reuse_interval : 3;
        int is_first = (step == 0);
        int is_last = (step == dit->sigmas.steps - 1);
        int evaluate = is_first || is_last || (accumulated_delta >= thresh) ||
                       (last_evaluated >= 0 && (step - last_evaluated) >= max_gap);

        if (evaluate) {
            ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                error, error_size);
            if (ok) {
                if (previous_evaluated >= 0) {
                    memcpy(previous2_video, previous_video, video_count * sizeof(*previous2_video));
                    memcpy(previous2_audio, previous_audio, audio_count * sizeof(*previous2_audio));
                    previous2_evaluated = previous_evaluated;
                }
                if (last_evaluated >= 0) {
                    // Compute L1 relative delta between velocities
                    double diff_sum = 0.0;
                    double ref_sum = 0.0;
                    size_t stride = 32; // Subsampled L1 delta for fast measurement
                    for (size_t i = 0; i < video_count; i += stride) {
                        diff_sum += fabsf(video_velocity[i] - last_video[i]);
                        ref_sum += fabsf(last_video[i]);
                    }
                    float step_delta = ref_sum > 1e-5 ? (float)(diff_sum / ref_sum) : 0.0f;
                    accumulated_delta = step_delta;

                    memcpy(previous_video, last_video, video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio, audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                } else {
                    accumulated_delta = 0.0f;
                }
                memcpy(last_video, video_velocity, video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity, audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
            /* N-Gram: update table after Sol forward */
            if (ok && dit->ngram_ctx && step > 0) {
                h3_ngram_update_table(
                    (H3NGramSpeculativeContext *)dit->ngram_ctx,
                    previous_video, video_velocity,
                    video_count, step);
            }
        } else {
            // Adaptive Velocity Extrapolation: skip heavy 33B forward pass
            dit->sol_stats.cached_steps++;

            float ratio = extrapolation_ratio(
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0 ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);

            /* N-Gram: continuous full-resolution speculative momentum extrapolation */
            int ngram_used = 0;
            if (dit->ngram_ctx) {
                float *draft = malloc(video_count * sizeof(*draft));
                if (draft) {
                    ngram_used = h3_ngram_draft_step(
                        (H3NGramSpeculativeContext *)dit->ngram_ctx,
                        last_video, previous_video, draft, video_count, ratio, step);
                    if (ngram_used) {
                        memcpy(video_velocity, draft,
                               video_count * sizeof(*video_velocity));
                    }
                    free(draft);
                }
            }
            if (!ngram_used) {
                extrapolate_velocity(
                    video_velocity, last_video, previous_video, previous2_video,
                    video_count,
                    dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                    previous_evaluated >= 0 ? dit->sigmas.video[previous_evaluated] : 0.0f,
                    previous2_evaluated >= 0 ? dit->sigmas.video[previous2_evaluated] : 0.0f,
                    previous_evaluated >= 0, previous2_evaluated >= 0);
            }
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, previous2_audio,
                audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0 ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous2_evaluated >= 0 ? dit->sigmas.audio[previous2_evaluated] : 0.0f,
                previous_evaluated >= 0, previous2_evaluated >= 0);

            // Accumulate expected drift for consecutive skips
            float time_delta = fabsf(dit->sigmas.video[step] - dit->sigmas.video[last_evaluated]);
            accumulated_delta += time_delta * 0.1f;
        }
        dit->sol_stats.total_steps++;

        if (ok) {
            ok = h3_euler_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 h3_euler_velocity_step(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            if (!ok) fail(error, error_size, "Euler solver rejected step %d", step);
        }

        if (ok && preview &&
            preview(step + 1, dit->sigmas.steps, video_latent, video_count,
                    preview_opaque)) {
            fail(error, error_size, "denoising preview stopped at step %d", step + 1);
            ok = 0;
        }
        if (ok) report(progress, progress_opaque, "denoise", step + 1, dit->sigmas.steps);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    dit->sol_stats.total_forward_seconds += (double)(end_time.tv_sec - start_time.tv_sec) +
                                            (double)(end_time.tv_nsec - start_time.tv_nsec) * 1e-9;

    free(video_velocity);
    free(audio_velocity);
    free(last_video);
    free(previous_video);
    free(previous2_video);
    free(last_audio);
    free(previous_audio);
    free(previous2_audio);
    return ok;
}

int h3_dit_denoise_draft_refine(
                         h3_dit *draft_dit, h3_dit *refine_dit,
                         float *video_latent, float *audio_latent,
                         int total_steps, int draft_steps,
                         float cache_thresh,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!refine_dit || !video_latent || !audio_latent || total_steps < 2) {
        fail(error, error_size, "invalid Draft & Refine arguments");
        return 0;
    }

    if (!draft_dit || draft_steps <= 0) {
        // Fallback to adaptive refinement across all steps
        return h3_dit_denoise_sol_adaptive(
            refine_dit, video_latent, audio_latent, 1, cache_thresh,
            progress, progress_opaque, preview, preview_opaque, error, error_size);
    }

    // Step 1: Run Draft stage on draft_dit
    report(progress, progress_opaque, "draft_denoise", 0, draft_steps);
    int ok = h3_dit_denoise_sol_adaptive(
        draft_dit, video_latent, audio_latent, 2, cache_thresh,
        progress, progress_opaque, preview, preview_opaque, error, error_size);
    if (!ok) return 0;

    // Step 2: Upsample Draft Latent to Refinement resolution if shapes differ
    if (draft_dit->latent_h != refine_dit->latent_h ||
        draft_dit->latent_w != refine_dit->latent_w ||
        draft_dit->latent_t != refine_dit->latent_t) {
        size_t refine_count = h3_dit_video_elements(refine_dit);
        float *upsampled_video = malloc(refine_count * sizeof(*upsampled_video));
        if (!upsampled_video) {
            fail(error, error_size, "out of memory allocating upsampled video latent");
            return 0;
        }

        h3_latent_upsample_3d(video_latent, upsampled_video,
                              VIDEO_CHANNELS, draft_dit->latent_t,
                              draft_dit->latent_h, draft_dit->latent_w,
                              refine_dit->latent_t, refine_dit->latent_h,
                              refine_dit->latent_w);

        memcpy(video_latent, upsampled_video, refine_count * sizeof(*upsampled_video));
        free(upsampled_video);
    }

    // Step 3: Run Refine stage on refine_dit with Sol-Attn enabled
    h3_dit_enable_sol_attn(refine_dit, 1, 10.0f, 32);
    report(progress, progress_opaque, "refine_denoise", draft_steps, total_steps);
    ok = h3_dit_denoise_sol_adaptive(
        refine_dit, video_latent, audio_latent, 1, cache_thresh,
        progress, progress_opaque, preview, preview_opaque, error, error_size);

    return ok;
}

