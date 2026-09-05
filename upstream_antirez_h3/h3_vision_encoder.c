#include "h3_vision_encoder.h"

#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VISION_HIDDEN = 1152,
    VISION_INTERMEDIATE = 4304,
    VISION_LAYERS = 27,
    VISION_HEADS = 16,
    VISION_HEAD_DIM = 72,
    VISION_ROPE_HALF = 36,
    PATCH = 16,
    TEMPORAL_PATCH = 2,
    MERGE = 2,
    PATCH_DIM = 3 * TEMPORAL_PATCH * PATCH * PATCH,
    MERGE_DIM = VISION_HIDDEN * MERGE * MERGE,
    POSITION_SIDE = 48,
    POSITION_COUNT = POSITION_SIDE * POSITION_SIDE,
    OUTPUT_WIDTH = H3_VISION_OUTPUT_WIDTH
};

typedef struct {
    h3_gpu_tensor *norm1_w;
    h3_gpu_tensor *norm1_b;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *norm2_w;
    h3_gpu_tensor *norm2_b;
    h3_gpu_tensor *fc1_w;
    h3_gpu_tensor *fc1_b;
    h3_gpu_tensor *fc2_w;
    h3_gpu_tensor *fc2_b;
} vision_block_weights;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void debug_tensor(const char *label, const h3_gpu_tensor *tensor,
                         size_t elements) {
    if (!getenv("H3_DEBUG_VISION") || !tensor) return;
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(tensor, values, elements)) {
        free(values);
        fprintf(stderr, "h3 vision debug %-24s unreadable\n", label);
        return;
    }
    size_t nonfinite = 0;
    float minimum = INFINITY, maximum = -INFINITY;
    for (size_t index = 0; index < elements; index++) {
        float value = bf16_to_f32(values[index]);
        if (!isfinite(value)) nonfinite++;
        else {
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
    }
    fprintf(stderr, "h3 vision debug %-24s min %g max %g nonfinite %zu\n",
            label, minimum, maximum, nonfinite);
    free(values);
}

static h3_gpu_tensor *bf1(const h3_weight_store *weights, h3_gpu *gpu,
                          const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(weights, gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(const h3_weight_store *weights, h3_gpu *gpu,
                          const char *name, uint64_t rows, uint64_t columns,
                          char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(weights, gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf5(const h3_weight_store *weights, h3_gpu *gpu,
                          const char *name, const uint64_t shape[5],
                          char *error, size_t error_size) {
    return h3_weight_load_bf16(weights, gpu, name, 5, shape,
                               error, error_size);
}

static int gpu_op(h3_gpu *gpu, int ok, const char *operation,
                  char *error, size_t error_size) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static int debug_sync(h3_gpu *gpu, const char *label,
                      const h3_gpu_tensor *tensor, size_t elements,
                      char *error, size_t error_size) {
    if (!getenv("H3_DEBUG_VISION_STEPS")) return 1;
    if (!gpu_op(gpu, h3_gpu_submit(gpu), "submit vision debug step",
                error, error_size)) return 0;
    debug_tensor(label, tensor, elements);
    return gpu_op(gpu, h3_gpu_begin(gpu), "resume vision debug step",
                  error, error_size);
}

static int prepare_patch_rows(const float *pixels, int frames,
                              int height, int width, uint16_t *rows) {
    int grid_h = height / PATCH;
    int grid_w = width / PATCH;
    size_t output = 0;
    for (int block_h = 0; block_h < grid_h / MERGE; block_h++)
        for (int block_w = 0; block_w < grid_w / MERGE; block_w++)
            for (int inner_h = 0; inner_h < MERGE; inner_h++)
                for (int inner_w = 0; inner_w < MERGE; inner_w++)
                    for (int channel = 0; channel < 3; channel++)
                        for (int temporal = 0; temporal < TEMPORAL_PATCH;
                             temporal++)
                            for (int patch_h = 0; patch_h < PATCH; patch_h++)
                                for (int patch_w = 0; patch_w < PATCH; patch_w++) {
                                    int frame = frames == 1 ? 0 : temporal;
                                    int y = (block_h * MERGE + inner_h) * PATCH +
                                            patch_h;
                                    int x = (block_w * MERGE + inner_w) * PATCH +
                                            patch_w;
                                    size_t input = (((size_t)frame * 3u +
                                        (size_t)channel) * (size_t)height +
                                        (size_t)y) * (size_t)width + (size_t)x;
                                    rows[output++] = f32_to_bf16(
                                        pixels[input] * 2.0f - 1.0f);
                                }
    return output == (size_t)grid_h * (size_t)grid_w * PATCH_DIM;
}

static int prepare_position_rows(const uint16_t *table, int grid_h,
                                 int grid_w, uint16_t *rows) {
    size_t output_row = 0;
    for (int block_h = 0; block_h < grid_h / MERGE; block_h++) {
        for (int block_w = 0; block_w < grid_w / MERGE; block_w++) {
            for (int inner_h = 0; inner_h < MERGE; inner_h++) {
                for (int inner_w = 0; inner_w < MERGE; inner_w++) {
                    int row = block_h * MERGE + inner_h;
                    int column = block_w * MERGE + inner_w;
                    double hv = (double)row * (POSITION_SIDE - 1) /
                                (double)(grid_h - 1);
                    double wv = (double)column * (POSITION_SIDE - 1) /
                                (double)(grid_w - 1);
                    int h0 = (int)hv;
                    int w0 = (int)wv;
                    int h1 = h0 + 1 < POSITION_SIDE ? h0 + 1 : h0;
                    int w1 = w0 + 1 < POSITION_SIDE ? w0 + 1 : w0;
                    float dh = (float)(hv - h0);
                    float dw = (float)(wv - w0);
                    float coefficients[] = {
                        (1.0f - dh) * (1.0f - dw),
                        (1.0f - dh) * dw,
                        dh * (1.0f - dw),
                        dh * dw
                    };
                    int indices[] = {
                        h0 * POSITION_SIDE + w0,
                        h0 * POSITION_SIDE + w1,
                        h1 * POSITION_SIDE + w0,
                        h1 * POSITION_SIDE + w1
                    };
                    for (int channel = 0; channel < VISION_HIDDEN; channel++) {
                        float sum = 0.0f;
                        for (int corner = 0; corner < 4; corner++) {
                            float coefficient = bf16_to_f32(
                                f32_to_bf16(coefficients[corner]));
                            float term = bf16_to_f32(
                                table[(size_t)indices[corner] * VISION_HIDDEN +
                                      (size_t)channel]) * coefficient;
                            sum += bf16_to_f32(f32_to_bf16(term));
                        }
                        rows[output_row * VISION_HIDDEN + (size_t)channel] =
                            f32_to_bf16(sum);
                    }
                    output_row++;
                }
            }
        }
    }
    return output_row == (size_t)grid_h * (size_t)grid_w;
}

static void prepare_rope(int grid_h, int grid_w,
                         uint16_t *cosines, uint16_t *sines) {
    size_t row = 0;
    for (int block_h = 0; block_h < grid_h / MERGE; block_h++)
        for (int block_w = 0; block_w < grid_w / MERGE; block_w++)
            for (int inner_h = 0; inner_h < MERGE; inner_h++)
                for (int inner_w = 0; inner_w < MERGE; inner_w++) {
                    float positions[] = {(float)(block_h * MERGE + inner_h),
                                         (float)(block_w * MERGE + inner_w)};
                    for (int axis = 0; axis < 2; axis++) {
                        for (int index = 0; index < VISION_ROPE_HALF / 2;
                             index++) {
                            float inverse = powf(10000.0f,
                                -(float)(index * 2) /
                                (float)VISION_ROPE_HALF);
                            float angle = positions[axis] * inverse;
                            size_t offset = row * VISION_ROPE_HALF +
                                (size_t)axis * (VISION_ROPE_HALF / 2) +
                                (size_t)index;
                            cosines[offset] = f32_to_bf16(cosf(angle));
                            sines[offset] = f32_to_bf16(sinf(angle));
                        }
                    }
                    row++;
                }
}

static void free_block_weights(vision_block_weights *weights) {
#define FREE(field) free_tensor(&weights->field)
    FREE(norm1_w); FREE(norm1_b); FREE(qkv_w); FREE(qkv_b);
    FREE(out_w); FREE(out_b); FREE(norm2_w); FREE(norm2_b);
    FREE(fc1_w); FREE(fc1_b); FREE(fc2_w); FREE(fc2_b);
#undef FREE
}

static int load_block_weights(const h3_weight_store *store, h3_gpu *gpu,
                              int layer, vision_block_weights *weights,
                              char *error, size_t error_size) {
    memset(weights, 0, sizeof(*weights));
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "model.visual.blocks.%d.", layer);
#define LOAD1(field, suffix, width) do {                                      \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    weights->field = bf1(store, gpu, name, width, error, error_size);          \
    if (!weights->field) goto failed;                                          \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                              \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    weights->field = bf2(store, gpu, name, rows, columns, error, error_size);  \
    if (!weights->field) goto failed;                                          \
} while (0)
    LOAD1(norm1_w, "norm1.weight", VISION_HIDDEN);
    LOAD1(norm1_b, "norm1.bias", VISION_HIDDEN);
    LOAD2(qkv_w, "attn.qkv.weight", VISION_HIDDEN * 3, VISION_HIDDEN);
    LOAD1(qkv_b, "attn.qkv.bias", VISION_HIDDEN * 3);
    LOAD2(out_w, "attn.proj.weight", VISION_HIDDEN, VISION_HIDDEN);
    LOAD1(out_b, "attn.proj.bias", VISION_HIDDEN);
    LOAD1(norm2_w, "norm2.weight", VISION_HIDDEN);
    LOAD1(norm2_b, "norm2.bias", VISION_HIDDEN);
    LOAD2(fc1_w, "mlp.linear_fc1.weight", VISION_INTERMEDIATE, VISION_HIDDEN);
    LOAD1(fc1_b, "mlp.linear_fc1.bias", VISION_INTERMEDIATE);
    LOAD2(fc2_w, "mlp.linear_fc2.weight", VISION_HIDDEN, VISION_INTERMEDIATE);
    LOAD1(fc2_b, "mlp.linear_fc2.bias", VISION_HIDDEN);
#undef LOAD1
#undef LOAD2
    return 1;
failed:
#undef LOAD1
#undef LOAD2
    free_block_weights(weights);
    return 0;
}

static int run_block(h3_gpu *gpu, const vision_block_weights *weights,
                     uint32_t rows, h3_gpu_tensor *hidden,
                     h3_gpu_tensor *norm, h3_gpu_tensor *qkv,
                     h3_gpu_tensor *query, h3_gpu_tensor *key,
                     h3_gpu_tensor *value, h3_gpu_tensor *heads,
                     h3_gpu_tensor *branch, h3_gpu_tensor *fc1,
                     h3_gpu_tensor *activated,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin,
                     char *error, size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!gpu_op(gpu, (call), label, error, error_size)) return 0;              \
} while (0)
#define SYNC(label, tensor, elements) do {                                     \
    if (!debug_sync(gpu, label, tensor, elements, error, error_size)) return 0;\
} while (0)
    OP(h3_gpu_begin(gpu), "begin Qwen vision block");
    OP(h3_gpu_layer_norm_bf16(gpu, norm, hidden, weights->norm1_w,
        weights->norm1_b, rows, VISION_HIDDEN, 1e-6f),
       "Qwen vision attention norm");
    SYNC("stage attention norm", norm, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_linear_bf16(gpu, qkv, norm, weights->qkv_w, weights->qkv_b,
        rows, VISION_HIDDEN, VISION_HIDDEN * 3), "Qwen vision QKV");
    SYNC("stage qkv", qkv, (size_t)rows * VISION_HIDDEN * 3);
    OP(h3_gpu_vision_qkv_rope_bf16(gpu, query, key, value, qkv,
        rope_cos, rope_sin, rows, VISION_HEADS, VISION_HEAD_DIM,
        VISION_ROPE_HALF), "Qwen vision RoPE");
    SYNC("stage query", query, (size_t)rows * VISION_HIDDEN);
    SYNC("stage key", key, (size_t)rows * VISION_HIDDEN);
    SYNC("stage value", value, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_sdpa_bf16(gpu, heads, query, key, value, rows,
        VISION_HEADS, VISION_HEAD_DIM,
        1.0f / sqrtf((float)VISION_HEAD_DIM)), "Qwen vision attention");
    SYNC("stage attention", heads, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_linear_bf16(gpu, branch, heads, weights->out_w, weights->out_b,
        rows, VISION_HIDDEN, VISION_HIDDEN), "Qwen vision attention output");
    SYNC("stage attention output", branch, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, branch,
        rows * VISION_HIDDEN), "Qwen vision attention residual");
    SYNC("stage attention residual", hidden, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_layer_norm_bf16(gpu, norm, hidden, weights->norm2_w,
        weights->norm2_b, rows, VISION_HIDDEN, 1e-6f),
       "Qwen vision MLP norm");
    SYNC("stage mlp norm", norm, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_linear_bf16(gpu, fc1, norm, weights->fc1_w, weights->fc1_b,
        rows, VISION_HIDDEN, VISION_INTERMEDIATE), "Qwen vision MLP input");
    SYNC("stage mlp input", fc1, (size_t)rows * VISION_INTERMEDIATE);
    OP(h3_gpu_gelu_bf16(gpu, activated, fc1,
        rows * VISION_INTERMEDIATE, 1), "Qwen vision GELU");
    SYNC("stage gelu", activated, (size_t)rows * VISION_INTERMEDIATE);
    OP(h3_gpu_linear_bf16(gpu, branch, activated, weights->fc2_w,
        weights->fc2_b, rows, VISION_INTERMEDIATE, VISION_HIDDEN),
       "Qwen vision MLP output");
    SYNC("stage mlp output", branch, (size_t)rows * VISION_HIDDEN);
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, branch,
        rows * VISION_HIDDEN), "Qwen vision MLP residual");
    OP(h3_gpu_submit(gpu), "submit Qwen vision block");
#undef SYNC
#undef OP
    return 1;
}

static h3_gpu_tensor *run_merger(const h3_weight_store *weights, h3_gpu *gpu,
                                 const h3_gpu_tensor *hidden,
                                 uint32_t patch_rows, int deepstack,
                                 int merger_index,
                                 char *error, size_t error_size) {
    uint32_t rows = patch_rows / (MERGE * MERGE);
    uint32_t norm_width = deepstack ? MERGE_DIM : VISION_HIDDEN;
    uint32_t norm_rows = deepstack ? rows : patch_rows;
    char prefix[128], name[192];
    if (deepstack)
        snprintf(prefix, sizeof(prefix),
                 "model.visual.deepstack_merger_list.%d.", merger_index);
    else
        snprintf(prefix, sizeof(prefix), "model.visual.merger.");
#define NAME(suffix) (snprintf(name, sizeof(name), "%s%s", prefix, suffix), name)
    h3_gpu_tensor *norm_w = bf1(weights, gpu, NAME("norm.weight"), norm_width,
                                 error, error_size);
    h3_gpu_tensor *norm_b = bf1(weights, gpu, NAME("norm.bias"), norm_width,
                                 error, error_size);
    h3_gpu_tensor *fc1_w = bf2(weights, gpu, NAME("linear_fc1.weight"),
                                MERGE_DIM, MERGE_DIM, error, error_size);
    h3_gpu_tensor *fc1_b = bf1(weights, gpu, NAME("linear_fc1.bias"),
                                MERGE_DIM, error, error_size);
    h3_gpu_tensor *fc2_w = bf2(weights, gpu, NAME("linear_fc2.weight"),
                                OUTPUT_WIDTH, MERGE_DIM, error, error_size);
    h3_gpu_tensor *fc2_b = bf1(weights, gpu, NAME("linear_fc2.bias"),
                                OUTPUT_WIDTH, error, error_size);
#undef NAME
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(
        gpu, (size_t)patch_rows * VISION_HIDDEN);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * MERGE_DIM);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * MERGE_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * OUTPUT_WIDTH);
    if (!norm_w || !norm_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b ||
        !norm || !fc1 || !activated || !output) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate Qwen vision merger: %s",
                 h3_gpu_error(gpu));
        free_tensor(&output);
        goto cleanup;
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), "begin Qwen vision merger",
                error, error_size) ||
        !gpu_op(gpu, h3_gpu_layer_norm_bf16(
            gpu, norm, hidden, norm_w, norm_b, norm_rows, norm_width, 1e-6f),
            "Qwen vision merger norm", error, error_size) ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, fc1, norm, fc1_w, fc1_b, rows, MERGE_DIM, MERGE_DIM),
            "Qwen vision merger input", error, error_size) ||
        !gpu_op(gpu, h3_gpu_gelu_bf16(
            gpu, activated, fc1, rows * MERGE_DIM, 0),
            "Qwen vision merger GELU", error, error_size) ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, output, activated, fc2_w, fc2_b, rows, MERGE_DIM,
            OUTPUT_WIDTH), "Qwen vision merger output", error, error_size) ||
        !gpu_op(gpu, h3_gpu_submit(gpu), "submit Qwen vision merger",
                error, error_size)) {
        free_tensor(&output);
    }
cleanup:
    free_tensor(&norm_w); free_tensor(&norm_b);
    free_tensor(&fc1_w); free_tensor(&fc1_b);
    free_tensor(&fc2_w); free_tensor(&fc2_b);
    free_tensor(&norm); free_tensor(&fc1); free_tensor(&activated);
    return output;
}

void h3_vision_output_free(h3_vision_output *output) {
    if (!output) return;
    free(output->merged);
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
        free(output->deepstack[index]);
    memset(output, 0, sizeof(*output));
}

int h3_vision_encode_bf16(const char *weight_directory,
                          const char *shader_source_path,
                          const float *pixels, int frames,
                          int height, int width,
                          h3_vision_progress progress, void *progress_opaque,
                          h3_vision_output *output,
                          char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !pixels || !output ||
        (frames != 1 && frames != 2) || height < 32 || width < 32 ||
        height % 32 || width % 32) {
        fail(error, error_size, "invalid Qwen vision encoder arguments");
        return 0;
    }
    size_t grid_h = (size_t)height / PATCH;
    size_t grid_w = (size_t)width / PATCH;
    if (grid_h > UINT32_MAX / grid_w) {
        fail(error, error_size, "Qwen vision patch sequence is too large");
        return 0;
    }
    uint32_t rows = (uint32_t)(grid_h * grid_w);
    uint32_t merged_rows = rows / (MERGE * MERGE);
    h3_weight_store *weights = h3_weight_store_open(
        weight_directory, error, error_size);
    h3_gpu *gpu = weights ? h3_gpu_create(
        shader_source_path, error, error_size) : NULL;
    if (!weights || !gpu) {
        h3_gpu_free(gpu);
        h3_weight_store_free(weights);
        return 0;
    }
    h3_gpu_profile_set_label(gpu, "Qwen vision encoder");

    size_t patch_elements = (size_t)rows * PATCH_DIM;
    size_t hidden_elements = (size_t)rows * VISION_HIDDEN;
    uint16_t *patch_data = malloc(patch_elements * sizeof(*patch_data));
    uint16_t *position_table = malloc(
        (size_t)POSITION_COUNT * VISION_HIDDEN * sizeof(*position_table));
    uint16_t *position_data = malloc(hidden_elements * sizeof(*position_data));
    uint16_t *rope_cos_data = malloc(
        (size_t)rows * VISION_ROPE_HALF * sizeof(*rope_cos_data));
    uint16_t *rope_sin_data = malloc(
        (size_t)rows * VISION_ROPE_HALF * sizeof(*rope_sin_data));
    h3_gpu_tensor *position_weight = bf2(weights, gpu,
        "model.visual.pos_embed.weight", POSITION_COUNT, VISION_HIDDEN,
        error, error_size);
    if (!patch_data || !position_table || !position_data || !rope_cos_data ||
        !rope_sin_data || !position_weight ||
        !h3_gpu_tensor_read_bf16(position_weight, position_table,
                                 (size_t)POSITION_COUNT * VISION_HIDDEN) ||
        !prepare_patch_rows(pixels, frames, height, width, patch_data) ||
        !prepare_position_rows(position_table, (int)grid_h, (int)grid_w,
                               position_data)) {
        if (!error || !*error)
            fail(error, error_size, "cannot prepare Qwen vision patch inputs");
        goto failed;
    }
    prepare_rope((int)grid_h, (int)grid_w, rope_cos_data, rope_sin_data);
    free(position_table); position_table = NULL;
    free_tensor(&position_weight);

    h3_gpu_tensor *patches = h3_gpu_tensor_from_bf16(
        gpu, patch_data, patch_elements);
    h3_gpu_tensor *position = h3_gpu_tensor_from_bf16(
        gpu, position_data, hidden_elements);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos_data, (size_t)rows * VISION_ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin_data, (size_t)rows * VISION_ROPE_HALF);
    free(patch_data); patch_data = NULL;
    free(position_data); position_data = NULL;
    free(rope_cos_data); rope_cos_data = NULL;
    free(rope_sin_data); rope_sin_data = NULL;
    uint64_t patch_shape[] = {VISION_HIDDEN, 3, TEMPORAL_PATCH, PATCH, PATCH};
    h3_gpu_tensor *patch_w = bf5(weights, gpu,
        "model.visual.patch_embed.proj.weight", patch_shape,
        error, error_size);
    h3_gpu_tensor *patch_b = bf1(weights, gpu,
        "model.visual.patch_embed.proj.bias", VISION_HIDDEN,
        error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    if (!patches || !position || !rope_cos || !rope_sin || !patch_w ||
        !patch_b || !hidden ||
        !gpu_op(gpu, h3_gpu_begin(gpu), "begin Qwen vision patch embed",
                error, error_size) ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, hidden, patches, patch_w, patch_b, rows, PATCH_DIM,
            VISION_HIDDEN), "Qwen vision patch embed", error, error_size) ||
        !gpu_op(gpu, h3_gpu_add_bf16(
            gpu, hidden, hidden, position, rows * VISION_HIDDEN),
            "Qwen vision position add", error, error_size) ||
        !gpu_op(gpu, h3_gpu_submit(gpu), "submit Qwen vision patch embed",
                error, error_size)) {
        if ((!error || !*error) && (!patches || !position || !rope_cos ||
                                    !rope_sin || !patch_w || !patch_b || !hidden))
            fail(error, error_size, "cannot allocate Qwen vision tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&hidden);
        free_tensor(&patches); free_tensor(&position);
        free_tensor(&rope_cos); free_tensor(&rope_sin);
        free_tensor(&patch_w); free_tensor(&patch_b);
        goto failed;
    }
    free_tensor(&patches); free_tensor(&position);
    free_tensor(&patch_w); free_tensor(&patch_b);
    debug_tensor("patch+position", hidden, hidden_elements);

    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, hidden_elements * 3);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *heads = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * VISION_INTERMEDIATE);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * VISION_INTERMEDIATE);
    h3_gpu_tensor *deepstack[H3_VISION_DEEPSTACKS] = {NULL, NULL, NULL};
    h3_gpu_tensor *merged = NULL;
    int ok = norm && qkv && query && key && value && heads && branch && fc1 &&
             activated;
    if (!ok) {
        fail(error, error_size, "cannot allocate Qwen vision activations: %s",
             h3_gpu_error(gpu));
        goto tower_cleanup;
    }

    for (int layer = 0; layer < VISION_LAYERS && ok; layer++) {
        vision_block_weights block;
        ok = load_block_weights(weights, gpu, layer, &block,
                                error, error_size);
        if (ok) ok = run_block(gpu, &block, rows, hidden, norm, qkv,
            query, key, value, heads, branch, fc1, activated,
            rope_cos, rope_sin, error, error_size);
        free_block_weights(&block);
        if (!ok) break;
        if (getenv("H3_DEBUG_VISION")) {
            char label[64];
            snprintf(label, sizeof(label), "block %d", layer);
            debug_tensor(label, hidden, hidden_elements);
        }
        if (progress) progress(layer + 1, VISION_LAYERS, progress_opaque);
        if (layer == 8 || layer == 16 || layer == 24) {
            int index = layer == 8 ? 0 : layer == 16 ? 1 : 2;
            deepstack[index] = run_merger(weights, gpu, hidden, rows, 1,
                                           index, error, error_size);
            ok = deepstack[index] != NULL;
            if (ok) {
                char label[64];
                snprintf(label, sizeof(label), "deepstack %d", index);
                debug_tensor(label, deepstack[index],
                             (size_t)merged_rows * OUTPUT_WIDTH);
            }
        }
    }
    merged = ok ? run_merger(
        weights, gpu, hidden, rows, 0, 0, error, error_size) : NULL;
    if (!merged) ok = 0;
    if (merged) debug_tensor("final merger", merged,
                             (size_t)merged_rows * OUTPUT_WIDTH);
    if (ok) {
        size_t output_elements = (size_t)merged_rows * OUTPUT_WIDTH;
        output->merged = malloc(output_elements * sizeof(*output->merged));
        for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
            output->deepstack[index] = malloc(
                output_elements * sizeof(*output->deepstack[index]));
        ok = output->merged && output->deepstack[0] && output->deepstack[1] &&
             output->deepstack[2];
        if (ok) ok = h3_gpu_tensor_read_bf16(
            merged, output->merged, output_elements);
        for (unsigned index = 0; index < H3_VISION_DEEPSTACKS && ok; index++)
            ok = h3_gpu_tensor_read_bf16(
                deepstack[index], output->deepstack[index], output_elements);
        if (ok) ok = h3_gpu_get_stats(gpu, &output->gpu_stats);
        if (!ok && (!error || !*error))
            fail(error, error_size, "cannot read Qwen vision output");
        if (ok) {
            output->grid_h = (int)grid_h;
            output->grid_w = (int)grid_w;
            output->tokens = merged_rows;
        }
    }
tower_cleanup:
    free_tensor(&merged);
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
        free_tensor(&deepstack[index]);
    free_tensor(&hidden); free_tensor(&norm); free_tensor(&qkv);
    free_tensor(&query); free_tensor(&key); free_tensor(&value);
    free_tensor(&heads); free_tensor(&branch); free_tensor(&fc1);
    free_tensor(&activated); free_tensor(&rope_cos); free_tensor(&rope_sin);
    if (!ok) goto failed;
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return 1;

failed:
    free(patch_data); free(position_table); free(position_data);
    free(rope_cos_data); free(rope_sin_data);
    free_tensor(&position_weight);
    h3_vision_output_free(output);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return 0;
}
