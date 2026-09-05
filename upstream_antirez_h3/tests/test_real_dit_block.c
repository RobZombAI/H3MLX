#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_ROWS = 6, AUDIO_ROWS = 16, VIDEO_ROWS = 2, SEQUENCE = 24,
    TEXT_DIM = 5120, HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM, FFN = 14336, TIME_INPUT = 256,
    TIME_HIDDEN = 5376, TIME_DIM = 2688, ROPE_HALF = 48,
    MODALITIES = 3, SLOTS = 6, MAX_TENSORS = 700
};

typedef struct {
    h3_st_header fixture;
    h3_weight_store *weights;
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} test_context;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} plain_block_weights;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_dit_block.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation/load failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry overflow");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void gpu_call(test_context *test, int ok, const char *operation) {
    if (ok) return;
    fprintf(stderr, "FAIL tests/test_real_dit_block.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static void *fixture_load(test_context *test, const char *name,
                          h3_dtype dtype, size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != (uint64_t)elements) {
        fprintf(stderr, "FAIL tests/test_real_dit_block.c: bad fixture tensor %s\n",
                name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *result = malloc(bytes ? bytes : 1);
    require(result != NULL, "fixture allocation failed");
    char error[512];
    if (!h3_st_read_data(&test->fixture, tensor, result, bytes, error,
                         sizeof(error))) die(error);
    return result;
}

static h3_gpu_tensor *fixture_bf16(test_context *test, const char *name,
                                   size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    require(tensor && tensor->dtype == H3_DTYPE_BF16 &&
            h3_st_tensor_elements(tensor) == (uint64_t)elements,
            "bad direct BF16 fixture tensor");
    return own(test, h3_gpu_tensor_load_bf16(test->gpu, test->fixture.path,
                                              tensor->file_offset, elements));
}

static h3_gpu_tensor *fixture_f32(test_context *test, const char *name,
                                  size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    require(tensor && tensor->dtype == H3_DTYPE_F32 &&
            h3_st_tensor_elements(tensor) == (uint64_t)elements,
            "bad direct F32 fixture tensor");
    return own(test, h3_gpu_tensor_load_f32(test->gpu, test->fixture.path,
                                             tensor->file_offset, elements));
}

static h3_gpu_tensor *new_bf16(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, elements));
}

static h3_gpu_tensor *new_f32(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_f32(test->gpu, elements));
}

static h3_gpu_tensor *weight_bf16_1d(test_context *test, const char *name,
                                     uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 1, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_bf16_2d(test_context *test, const char *name,
                                     uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 2, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_f32_1d(test_context *test, const char *name,
                                    uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_f32(
        test->weights, test->gpu, name, 1, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_f32_2d(test_context *test, const char *name,
                                    uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_f32(
        test->weights, test->gpu, name, 2, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static plain_block_weights load_plain_block(test_context *test,
                                             const char *prefix) {
    plain_block_weights result;
    char name[192];
#define BF1(field, suffix, width) do {                                          \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    result.field = weight_bf16_1d(test, name, width);                           \
} while (0)
#define BF2(field, suffix, rows, columns) do {                                  \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    result.field = weight_bf16_2d(test, name, rows, columns);                   \
} while (0)
    BF1(norm1, "norm1.weight", HIDDEN);
    BF1(norm2, "norm2.weight", HIDDEN);
    BF2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    BF1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    BF1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    BF2(out, "attn.out_proj.weight", HIDDEN, INNER);
    BF2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    BF2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef BF1
#undef BF2
    return result;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double compare_bf16(test_context *test, const char *name,
                           h3_gpu_tensor *got_tensor, size_t elements,
                           double *relative_l2, double *absolute) {
    uint16_t *got = malloc(elements * sizeof(*got));
    uint16_t *want = fixture_load(test, name, H3_DTYPE_BF16, elements);
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read BF16 result");
    double max_error = 0.0, max_value = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double actual = bf16_to_f32(got[index]);
        double expected = bf16_to_f32(want[index]);
        double delta = actual - expected;
        if (fabs(delta) > max_error) max_error = fabs(delta);
        if (fabs(expected) > max_value) max_value = fabs(expected);
        square_error += delta * delta;
        square_value += expected * expected;
    }
    free(got);
    free(want);
    *absolute = max_error;
    *relative_l2 = sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24));
    return max_error / (max_value > 1e-12 ? max_value : 1e-12);
}

static void print_compare(test_context *test, const char *name,
                          h3_gpu_tensor *tensor, size_t elements,
                          double bound) {
    double l2, absolute;
    double relative = compare_bf16(test, name, tensor, elements, &l2, &absolute);
    printf("%-26s rel-max %.5g abs %.5g rel-L2 %.5g\n",
           name, relative, absolute, l2);
    if (relative >= bound || l2 >= bound) die("MLX parity bound exceeded");
}

static void print_compare_f32(test_context *test, const char *name,
                              h3_gpu_tensor *tensor, size_t elements,
                              double bound) {
    float *got = malloc(elements * sizeof(*got));
    float *want = fixture_load(test, name, H3_DTYPE_F32, elements);
    require(got != NULL, "F32 comparison allocation failed");
    require(h3_gpu_tensor_read_f32(tensor, got, elements),
            "cannot read F32 result");
    double max_error = 0.0, max_value = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double delta = (double)got[index] - want[index];
        if (fabs(delta) > max_error) max_error = fabs(delta);
        if (fabs(want[index]) > max_value) max_value = fabs(want[index]);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative = max_error / (max_value > 1e-12 ? max_value : 1e-12);
    double l2 = sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24));
    printf("%-26s rel-max %.5g abs %.5g rel-L2 %.5g\n",
           name, relative, max_error, l2);
    free(got);
    free(want);
    if (relative >= bound || l2 >= bound) die("F32 MLX parity bound exceeded");
}

static void run_refiner_block(test_context *test,
                              const plain_block_weights *weight,
                              h3_gpu_tensor *hidden,
                              h3_gpu_tensor *norm,
                              h3_gpu_tensor *qkv,
                              h3_gpu_tensor *query,
                              h3_gpu_tensor *key,
                              h3_gpu_tensor *value,
                              h3_gpu_tensor *attention_heads,
                              h3_gpu_tensor *branch,
                              h3_gpu_tensor *fc1,
                              h3_gpu_tensor *activated,
                              h3_gpu_tensor *dummy_rope,
                              const char *label) {
    gpu_call(test, h3_gpu_rms_norm_bf16(test->gpu, norm, hidden, weight->norm1,
                                        TEXT_ROWS, HIDDEN, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, qkv, norm, weight->qkv, NULL,
                                      TEXT_ROWS, HIDDEN, INNER * 3), label);
    gpu_call(test, h3_gpu_grouped_qkv_rope_bf16(
                                        test->gpu, query, key, value, qkv,
                                        weight->q_norm, weight->k_norm,
                                        dummy_rope, dummy_rope, TEXT_ROWS,
                                        HEADS, HEAD_DIM, 0, 1e-5f), label);
    gpu_call(test, h3_gpu_sdpa_bf16(test->gpu, attention_heads, query, key,
                                    value, TEXT_ROWS, HEADS, HEAD_DIM,
                                    1.0f / sqrtf((float)HEAD_DIM)), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, branch, attention_heads,
                                      weight->out, NULL, TEXT_ROWS, INNER,
                                      HIDDEN), label);
    gpu_call(test, h3_gpu_add_bf16(test->gpu, hidden, hidden, branch,
                                   TEXT_ROWS * HIDDEN), label);
    gpu_call(test, h3_gpu_rms_norm_bf16(test->gpu, norm, hidden, weight->norm2,
                                        TEXT_ROWS, HIDDEN, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, fc1, norm, weight->fc1, NULL,
                                      TEXT_ROWS, HIDDEN, FFN * 2), label);
    gpu_call(test, h3_gpu_swiglu_bf16(test->gpu, activated, fc1,
                                      TEXT_ROWS, FFN), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, branch, activated, weight->fc2,
                                      NULL, TEXT_ROWS, FFN, HIDDEN), label);
    gpu_call(test, h3_gpu_add_bf16(test->gpu, hidden, hidden, branch,
                                   TEXT_ROWS * HIDDEN), label);
}

static void run_dit_block_inplace(test_context *test,
                                  const plain_block_weights *weight,
                                  h3_gpu_tensor *adaln_w,
                                  h3_gpu_tensor *adaln_b,
                                  h3_gpu_tensor *time_silu,
                                  h3_gpu_tensor *hidden,
                                  h3_gpu_tensor *modulation,
                                  h3_gpu_tensor *row_map,
                                  h3_gpu_tensor *mod_attention,
                                  h3_gpu_tensor *qkv,
                                  h3_gpu_tensor *query,
                                  h3_gpu_tensor *key,
                                  h3_gpu_tensor *value,
                                  h3_gpu_tensor *attention_heads,
                                  h3_gpu_tensor *attention_output,
                                  h3_gpu_tensor *mod_mlp,
                                  h3_gpu_tensor *fc1,
                                  h3_gpu_tensor *activated,
                                  h3_gpu_tensor *mlp_output,
                                  h3_gpu_tensor *rope_cos,
                                  h3_gpu_tensor *rope_sin,
                                  const char *label) {
    gpu_call(test, h3_gpu_linear_bf16(
        test->gpu, modulation, time_silu, adaln_w, adaln_b, 1, TIME_DIM,
        MODALITIES * SLOTS * HIDDEN), label);
    gpu_call(test, h3_gpu_adaln_bf16(
        test->gpu, mod_attention, hidden, weight->norm1, modulation, row_map,
        SEQUENCE, HIDDEN, SLOTS, 0, 1, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, qkv, mod_attention,
                                      weight->qkv, NULL, SEQUENCE, HIDDEN,
                                      INNER * 3), label);
    gpu_call(test, h3_gpu_grouped_qkv_rope_bf16(
        test->gpu, query, key, value, qkv, weight->q_norm, weight->k_norm,
        rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), label);
    gpu_call(test, h3_gpu_sdpa_bf16(
        test->gpu, attention_heads, query, key, value, SEQUENCE, HEADS,
        HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)), label);
    gpu_call(test, h3_gpu_linear_bf16(
        test->gpu, attention_output, attention_heads, weight->out, NULL,
        SEQUENCE, INNER, HIDDEN), label);
    gpu_call(test, h3_gpu_gate_bf16(
        test->gpu, hidden, hidden, attention_output, modulation, row_map,
        SEQUENCE, HIDDEN, SLOTS, 2), label);
    gpu_call(test, h3_gpu_adaln_bf16(
        test->gpu, mod_mlp, hidden, weight->norm2, modulation, row_map,
        SEQUENCE, HIDDEN, SLOTS, 3, 4, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, fc1, mod_mlp, weight->fc1,
                                      NULL, SEQUENCE, HIDDEN, FFN * 2), label);
    gpu_call(test, h3_gpu_swiglu_bf16(test->gpu, activated, fc1, SEQUENCE,
                                      FFN), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, mlp_output, activated,
                                      weight->fc2, NULL, SEQUENCE, FFN,
                                      HIDDEN), label);
    gpu_call(test, h3_gpu_gate_bf16(
        test->gpu, hidden, hidden, mlp_output, modulation, row_map, SEQUENCE,
        HIDDEN, SLOTS, 5), label);
}

static void cleanup(test_context *test) {
    for (size_t index = 0; index < test->owned_count; index++) {
        h3_gpu_tensor_free(test->owned[index]);
    }
    h3_gpu_free(test->gpu);
    h3_weight_store_free(test->weights);
    h3_st_free_header(&test->fixture);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_dit_block0_bf16.safetensors";
    const char *step_fixture_path = argc > 3 ? argv[3] :
        "misc/fixtures/h3_real_dit_step0_bf16.safetensors";
    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/transformer", model_root);
    test_context test;
    memset(&test, 0, sizeof(test));
    char error[512];
    if (!h3_st_read_header(fixture_path, &test.fixture, error, sizeof(error)))
        die(error);
    test.weights = h3_weight_store_open(weights_path, error, sizeof(error));
    if (!test.weights) die(error);
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) die(error);

    h3_gpu_tensor *text_qwen = fixture_bf16(&test, "x.text_qwen",
                                             TEXT_ROWS * TEXT_DIM);
    h3_gpu_tensor *condition_w = weight_bf16_2d(
        &test, "condition_proj.weight", HIDDEN, TEXT_DIM);
    h3_gpu_tensor *condition_b = weight_bf16_1d(
        &test, "condition_proj.bias", HIDDEN);
    plain_block_weights refiner0 = load_plain_block(
        &test, "token_refiner.blocks.0.");
    plain_block_weights refiner1 = load_plain_block(
        &test, "token_refiner.blocks.1.");
    h3_gpu_tensor *refiner_final = weight_bf16_1d(
        &test, "token_refiner.final_norm.weight", HIDDEN);

    h3_gpu_tensor *refined = new_bf16(&test, TEXT_ROWS * HIDDEN);
    h3_gpu_tensor *refiner_norm = new_bf16(&test, TEXT_ROWS * HIDDEN);
    h3_gpu_tensor *refiner_qkv = new_bf16(&test, TEXT_ROWS * INNER * 3);
    h3_gpu_tensor *refiner_q = new_bf16(&test, TEXT_ROWS * INNER);
    h3_gpu_tensor *refiner_k = new_bf16(&test, TEXT_ROWS * INNER);
    h3_gpu_tensor *refiner_v = new_bf16(&test, TEXT_ROWS * INNER);
    h3_gpu_tensor *refiner_heads = new_bf16(&test, TEXT_ROWS * INNER);
    h3_gpu_tensor *refiner_branch = new_bf16(&test, TEXT_ROWS * HIDDEN);
    h3_gpu_tensor *refiner_fc1 = new_bf16(&test, TEXT_ROWS * FFN * 2);
    h3_gpu_tensor *refiner_activated = new_bf16(&test, TEXT_ROWS * FFN);

    h3_gpu_tensor *video_rows = fixture_f32(&test, "x.video_rows", VIDEO_ROWS * 96);
    h3_gpu_tensor *audio_rows = fixture_f32(&test, "x.audio_rows", AUDIO_ROWS * 32);
    h3_gpu_tensor *video_w = weight_f32_2d(&test, "video_patch_proj.weight", HIDDEN, 96);
    h3_gpu_tensor *video_b = weight_f32_1d(&test, "video_patch_proj.bias", HIDDEN);
    h3_gpu_tensor *audio_w = weight_f32_2d(&test, "audio_patch_proj.weight", HIDDEN, 32);
    h3_gpu_tensor *audio_b = weight_f32_1d(&test, "audio_patch_proj.bias", HIDDEN);
    h3_gpu_tensor *video_f32 = new_f32(&test, VIDEO_ROWS * HIDDEN);
    h3_gpu_tensor *audio_f32 = new_f32(&test, AUDIO_ROWS * HIDDEN);
    h3_gpu_tensor *video_bf16 = new_bf16(&test, VIDEO_ROWS * HIDDEN);
    h3_gpu_tensor *audio_bf16 = new_bf16(&test, AUDIO_ROWS * HIDDEN);
    h3_gpu_tensor *hidden = new_bf16(&test, SEQUENCE * HIDDEN);

    h3_gpu_tensor *time_features = fixture_f32(&test, "x.time_features", TIME_INPUT);
    h3_gpu_tensor *time_in_w = weight_f32_2d(
        &test, "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT);
    h3_gpu_tensor *time_in_b = weight_f32_1d(
        &test, "time_embedder.proj_in.bias", TIME_HIDDEN);
    h3_gpu_tensor *time_out_w = weight_f32_2d(
        &test, "time_embedder.proj_out.weight", TIME_DIM, TIME_HIDDEN);
    h3_gpu_tensor *time_out_b = weight_f32_1d(
        &test, "time_embedder.proj_out.bias", TIME_DIM);
    h3_gpu_tensor *time_hidden = new_f32(&test, TIME_HIDDEN);
    h3_gpu_tensor *time_silu = new_f32(&test, TIME_HIDDEN);
    h3_gpu_tensor *time_output = new_f32(&test, TIME_DIM);
    h3_gpu_tensor *time_bf16 = new_bf16(&test, TIME_DIM);
    h3_gpu_tensor *time_bf16_silu = new_bf16(&test, TIME_DIM);

    plain_block_weights block0 = load_plain_block(&test, "blocks.0.");
    h3_gpu_tensor *adaln_w = weight_bf16_2d(
        &test, "blocks.0.adaln_proj.linear.weight",
        MODALITIES * SLOTS * HIDDEN, TIME_DIM);
    h3_gpu_tensor *adaln_b = weight_bf16_1d(
        &test, "blocks.0.adaln_proj.linear.bias", MODALITIES * SLOTS * HIDDEN);
    h3_gpu_tensor *rope_cos = fixture_bf16(&test, "x.rope_cos",
                                            SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = fixture_bf16(&test, "x.rope_sin",
                                            SEQUENCE * ROPE_HALF);
    int32_t *signed_rows = fixture_load(&test, "x.row_map", H3_DTYPE_I32,
                                        SEQUENCE);
    uint32_t rows[SEQUENCE];
    for (size_t index = 0; index < SEQUENCE; index++) {
        require(signed_rows[index] >= 0, "negative modulation row");
        rows[index] = (uint32_t)signed_rows[index];
    }
    free(signed_rows);
    h3_gpu_tensor *row_map = own(&test, h3_gpu_tensor_from_u32(
                                           test.gpu, rows, SEQUENCE));
    h3_gpu_tensor *modulation = new_bf16(
        &test, MODALITIES * SLOTS * HIDDEN);
    h3_gpu_tensor *mod_attention = new_bf16(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *qkv = new_bf16(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *query = new_bf16(&test, SEQUENCE * INNER);
    h3_gpu_tensor *key = new_bf16(&test, SEQUENCE * INNER);
    h3_gpu_tensor *value = new_bf16(&test, SEQUENCE * INNER);
    h3_gpu_tensor *attention_heads = new_bf16(&test, SEQUENCE * INNER);
    h3_gpu_tensor *attention_output = new_bf16(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *after_attention = new_bf16(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *mod_mlp = new_bf16(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *fc1 = new_bf16(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *activated = new_bf16(&test, SEQUENCE * FFN);
    h3_gpu_tensor *mlp_output = new_bf16(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *block_output = new_bf16(&test, SEQUENCE * HIDDEN);

    gpu_call(&test, h3_gpu_begin(test.gpu), "begin command stream");
    gpu_call(&test, h3_gpu_linear_bf16(test.gpu, refined, text_qwen,
                                        condition_w, condition_b, TEXT_ROWS,
                                        TEXT_DIM, HIDDEN), "condition projection");
    run_refiner_block(&test, &refiner0, refined, refiner_norm, refiner_qkv,
                      refiner_q, refiner_k, refiner_v, refiner_heads,
                      refiner_branch, refiner_fc1, refiner_activated,
                      refiner0.q_norm, "refiner block 0");
    run_refiner_block(&test, &refiner1, refined, refiner_norm, refiner_qkv,
                      refiner_q, refiner_k, refiner_v, refiner_heads,
                      refiner_branch, refiner_fc1, refiner_activated,
                      refiner1.q_norm, "refiner block 1");
    gpu_call(&test, h3_gpu_rms_norm_bf16(test.gpu, refined, refined,
                                          refiner_final, TEXT_ROWS, HIDDEN,
                                          1e-5f), "refiner final norm");

    gpu_call(&test, h3_gpu_linear_f32(test.gpu, video_f32, video_rows, video_w,
                                       video_b, VIDEO_ROWS, 96, HIDDEN),
             "video patch projection");
    gpu_call(&test, h3_gpu_linear_f32(test.gpu, audio_f32, audio_rows, audio_w,
                                       audio_b, AUDIO_ROWS, 32, HIDDEN),
             "audio patch projection");
    gpu_call(&test, h3_gpu_cast_f32_to_bf16(test.gpu, video_bf16, video_f32,
                                             VIDEO_ROWS * HIDDEN), "video cast");
    gpu_call(&test, h3_gpu_cast_f32_to_bf16(test.gpu, audio_bf16, audio_f32,
                                             AUDIO_ROWS * HIDDEN), "audio cast");
    gpu_call(&test, h3_gpu_copy_bf16(test.gpu, hidden, 0, refined, 0,
                                      TEXT_ROWS * HIDDEN), "pack text");
    gpu_call(&test, h3_gpu_copy_bf16(test.gpu, hidden, TEXT_ROWS * HIDDEN,
                                      audio_bf16, 0, AUDIO_ROWS * HIDDEN),
             "pack audio");
    gpu_call(&test, h3_gpu_copy_bf16(
        test.gpu, hidden, (TEXT_ROWS + AUDIO_ROWS) * HIDDEN,
        video_bf16, 0, VIDEO_ROWS * HIDDEN), "pack video");

    gpu_call(&test, h3_gpu_linear_f32(test.gpu, time_hidden, time_features,
                                       time_in_w, time_in_b, 1, TIME_INPUT,
                                       TIME_HIDDEN), "time projection in");
    gpu_call(&test, h3_gpu_silu_f32(test.gpu, time_silu, time_hidden,
                                     TIME_HIDDEN), "time SiLU");
    gpu_call(&test, h3_gpu_linear_f32(test.gpu, time_output, time_silu,
                                       time_out_w, time_out_b, 1, TIME_HIDDEN,
                                       TIME_DIM), "time projection out");
    gpu_call(&test, h3_gpu_cast_f32_to_bf16(test.gpu, time_bf16, time_output,
                                             TIME_DIM), "time cast");
    gpu_call(&test, h3_gpu_silu_bf16(test.gpu, time_bf16_silu, time_bf16,
                                      TIME_DIM), "AdaLN SiLU");
    gpu_call(&test, h3_gpu_linear_bf16(
        test.gpu, modulation, time_bf16_silu, adaln_w, adaln_b, 1, TIME_DIM,
        MODALITIES * SLOTS * HIDDEN), "block-0 AdaLN projection");
    gpu_call(&test, h3_gpu_adaln_bf16(
        test.gpu, mod_attention, hidden, block0.norm1, modulation, row_map,
        SEQUENCE, HIDDEN, SLOTS, 0, 1, 1e-5f), "block-0 attention AdaLN");
    gpu_call(&test, h3_gpu_linear_bf16(test.gpu, qkv, mod_attention, block0.qkv,
                                        NULL, SEQUENCE, HIDDEN, INNER * 3),
             "block-0 QKV");
    gpu_call(&test, h3_gpu_grouped_qkv_rope_bf16(
        test.gpu, query, key, value, qkv, block0.q_norm, block0.k_norm,
        rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "block-0 QK norm/RoPE");
    gpu_call(&test, h3_gpu_sdpa_bf16(
        test.gpu, attention_heads, query, key, value, SEQUENCE, HEADS,
        HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)), "block-0 attention");
    gpu_call(&test, h3_gpu_linear_bf16(
        test.gpu, attention_output, attention_heads, block0.out, NULL,
        SEQUENCE, INNER, HIDDEN), "block-0 attention output");
    gpu_call(&test, h3_gpu_gate_bf16(
        test.gpu, after_attention, hidden, attention_output, modulation,
        row_map, SEQUENCE, HIDDEN, SLOTS, 2), "block-0 attention gate");
    gpu_call(&test, h3_gpu_adaln_bf16(
        test.gpu, mod_mlp, after_attention, block0.norm2, modulation, row_map,
        SEQUENCE, HIDDEN, SLOTS, 3, 4, 1e-5f), "block-0 MLP AdaLN");
    gpu_call(&test, h3_gpu_linear_bf16(test.gpu, fc1, mod_mlp, block0.fc1,
                                        NULL, SEQUENCE, HIDDEN, FFN * 2),
             "block-0 MLP input");
    gpu_call(&test, h3_gpu_swiglu_bf16(test.gpu, activated, fc1, SEQUENCE,
                                        FFN), "block-0 SwiGLU");
    gpu_call(&test, h3_gpu_linear_bf16(test.gpu, mlp_output, activated,
                                        block0.fc2, NULL, SEQUENCE, FFN,
                                        HIDDEN), "block-0 MLP output");
    gpu_call(&test, h3_gpu_gate_bf16(
        test.gpu, block_output, after_attention, mlp_output, modulation,
        row_map, SEQUENCE, HIDDEN, SLOTS, 5), "block-0 MLP gate");
    gpu_call(&test, h3_gpu_submit(test.gpu), "submit command stream");

    print_compare(&test, "x.text_refined", refined, TEXT_ROWS * HIDDEN, 0.025);
    print_compare(&test, "x.video_embed", video_bf16, VIDEO_ROWS * HIDDEN, 0.01);
    print_compare(&test, "x.audio_embed", audio_bf16, AUDIO_ROWS * HIDDEN, 0.01);
    print_compare(&test, "x.hidden", hidden, SEQUENCE * HIDDEN, 0.025);
    print_compare(&test, "x.time_embedding", time_bf16, TIME_DIM, 0.01);
    print_compare(&test, "x.modulation", modulation,
                  MODALITIES * SLOTS * HIDDEN, 0.025);
    print_compare(&test, "x.mod_attention", mod_attention,
                  SEQUENCE * HIDDEN, 0.025);
    print_compare(&test, "x.attention_output", attention_output,
                  SEQUENCE * HIDDEN, 0.04);
    print_compare(&test, "x.mod_mlp", mod_mlp, SEQUENCE * HIDDEN, 0.04);
    print_compare(&test, "x.block_output", block_output,
                  SEQUENCE * HIDDEN, 0.05);

    h3_gpu_tensor *saved[5];
    for (size_t index = 0; index < 5; index++) {
        saved[index] = new_bf16(&test, SEQUENCE * HIDDEN);
    }
    gpu_call(&test, h3_gpu_begin(test.gpu), "begin remaining blocks");
    size_t saved_index = 0;
    for (int layer = 1; layer < 50; layer++) {
        char prefix[64];
        char name[128];
        snprintf(prefix, sizeof(prefix), "blocks.%d.", layer);
        plain_block_weights weight = load_plain_block(&test, prefix);
        snprintf(name, sizeof(name), "%sadaln_proj.linear.weight", prefix);
        h3_gpu_tensor *layer_adaln_w = weight_bf16_2d(
            &test, name, MODALITIES * SLOTS * HIDDEN, TIME_DIM);
        snprintf(name, sizeof(name), "%sadaln_proj.linear.bias", prefix);
        h3_gpu_tensor *layer_adaln_b = weight_bf16_1d(
            &test, name, MODALITIES * SLOTS * HIDDEN);
        run_dit_block_inplace(
            &test, &weight, layer_adaln_w, layer_adaln_b, time_bf16_silu,
            block_output, modulation, row_map, mod_attention, qkv, query, key,
            value, attention_heads, attention_output, mod_mlp, fc1, activated,
            mlp_output, rope_cos, rope_sin, prefix);
        if (layer == 9 || layer == 19 || layer == 29 || layer == 39 ||
            layer == 49) {
            gpu_call(&test, h3_gpu_copy_bf16(
                test.gpu, saved[saved_index++], 0, block_output, 0,
                SEQUENCE * HIDDEN), "save block checkpoint");
        }
        if (layer == 1 || (layer + 1) % 5 == 0) {
            fprintf(stderr, "native DiT command encoding: %d/50 blocks\n",
                    layer + 1);
        }
    }
    require(saved_index == 5, "did not save every block checkpoint");

    h3_gpu_tensor *final_norm = weight_bf16_1d(
        &test, "final_layer.norm.weight", HIDDEN);
    h3_gpu_tensor *final_adaln_w = weight_bf16_2d(
        &test, "final_layer.adaln_proj.linear.weight", 2 * HIDDEN, TIME_DIM);
    h3_gpu_tensor *final_adaln_b = weight_bf16_1d(
        &test, "final_layer.adaln_proj.linear.bias", 2 * HIDDEN);
    h3_gpu_tensor *final_video_w = weight_f32_2d(
        &test, "final_layer.video_out.weight", 96, HIDDEN);
    h3_gpu_tensor *final_video_b = weight_f32_1d(
        &test, "final_layer.video_out.bias", 96);
    h3_gpu_tensor *final_audio_w = weight_f32_2d(
        &test, "final_layer.audio_out.weight", 32, HIDDEN);
    h3_gpu_tensor *final_audio_b = weight_f32_1d(
        &test, "final_layer.audio_out.bias", 32);
    h3_gpu_tensor *final_modulation = new_bf16(&test, 2 * HIDDEN);
    h3_gpu_tensor *final_audio_input = new_bf16(&test, AUDIO_ROWS * HIDDEN);
    h3_gpu_tensor *final_video_input = new_bf16(&test, VIDEO_ROWS * HIDDEN);
    h3_gpu_tensor *final_audio_norm = new_bf16(&test, AUDIO_ROWS * HIDDEN);
    h3_gpu_tensor *final_video_norm = new_bf16(&test, VIDEO_ROWS * HIDDEN);
    h3_gpu_tensor *final_audio_f32 = new_f32(&test, AUDIO_ROWS * HIDDEN);
    h3_gpu_tensor *final_video_f32 = new_f32(&test, VIDEO_ROWS * HIDDEN);
    h3_gpu_tensor *audio_rows_output = new_f32(&test, AUDIO_ROWS * 32);
    h3_gpu_tensor *video_rows_output = new_f32(&test, VIDEO_ROWS * 96);
    uint32_t zero_rows[AUDIO_ROWS];
    memset(zero_rows, 0, sizeof(zero_rows));
    h3_gpu_tensor *final_rows = own(&test, h3_gpu_tensor_from_u32(
                                              test.gpu, zero_rows, AUDIO_ROWS));

    gpu_call(&test, h3_gpu_linear_bf16(
        test.gpu, final_modulation, time_bf16_silu, final_adaln_w,
        final_adaln_b, 1, TIME_DIM, 2 * HIDDEN), "final AdaLN projection");
    gpu_call(&test, h3_gpu_copy_bf16(
        test.gpu, final_audio_input, 0, block_output, TEXT_ROWS * HIDDEN,
        AUDIO_ROWS * HIDDEN), "slice final audio");
    gpu_call(&test, h3_gpu_copy_bf16(
        test.gpu, final_video_input, 0, block_output,
        (TEXT_ROWS + AUDIO_ROWS) * HIDDEN, VIDEO_ROWS * HIDDEN),
        "slice final video");
    gpu_call(&test, h3_gpu_adaln_bf16(
        test.gpu, final_audio_norm, final_audio_input, final_norm,
        final_modulation, final_rows, AUDIO_ROWS, HIDDEN, 2, 0, 1, 1e-5f),
        "final audio AdaLN");
    gpu_call(&test, h3_gpu_adaln_bf16(
        test.gpu, final_video_norm, final_video_input, final_norm,
        final_modulation, final_rows, VIDEO_ROWS, HIDDEN, 2, 0, 1, 1e-5f),
        "final video AdaLN");
    gpu_call(&test, h3_gpu_cast_bf16_to_f32(
        test.gpu, final_audio_f32, final_audio_norm, AUDIO_ROWS * HIDDEN),
        "final audio cast");
    gpu_call(&test, h3_gpu_cast_bf16_to_f32(
        test.gpu, final_video_f32, final_video_norm, VIDEO_ROWS * HIDDEN),
        "final video cast");
    gpu_call(&test, h3_gpu_linear_f32(
        test.gpu, audio_rows_output, final_audio_f32, final_audio_w,
        final_audio_b, AUDIO_ROWS, HIDDEN, 32), "final audio head");
    gpu_call(&test, h3_gpu_linear_f32(
        test.gpu, video_rows_output, final_video_f32, final_video_w,
        final_video_b, VIDEO_ROWS, HIDDEN, 96), "final video head");
    gpu_call(&test, h3_gpu_submit(test.gpu), "submit full DiT command stream");

    h3_st_free_header(&test.fixture);
    memset(&test.fixture, 0, sizeof(test.fixture));
    if (!h3_st_read_header(step_fixture_path, &test.fixture, error,
                           sizeof(error))) die(error);
    static const int checkpoint_layers[] = {9, 19, 29, 39, 49};
    for (size_t index = 0; index < 5; index++) {
        char name[32];
        snprintf(name, sizeof(name), "x.block_%d", checkpoint_layers[index]);
        print_compare(&test, name, saved[index], SEQUENCE * HIDDEN, 0.15);
    }
    print_compare_f32(&test, "x.audio_rows_output", audio_rows_output,
                      AUDIO_ROWS * 32, 0.25);
    print_compare_f32(&test, "x.video_rows_output", video_rows_output,
                      VIDEO_ROWS * 96, 0.25);
    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "cannot read GPU stats");
    printf("real full step: %.3f GiB allocated, %llu direct, %llu SDPA, "
           "%llu blits, %.3f GPU seconds, %llu submission\n",
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)stats.direct_dispatches,
           (unsigned long long)stats.mps_sdpa_dispatches,
           (unsigned long long)stats.blit_copies, stats.gpu_seconds,
           (unsigned long long)stats.submissions);
    require(stats.submissions == 2,
            "setup/block-0 plus remaining full step did not use two batches");
    cleanup(&test);
    puts("ok: real FL2VA setup and complete 50-block denoising step match MLX");
    return 0;
}
