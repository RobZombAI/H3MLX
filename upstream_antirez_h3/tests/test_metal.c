#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 32,
    HIDDEN = 256,
    HEADS = 4,
    HEAD_DIM = 32,
    INNER = HEADS * HEAD_DIM,
    FFN = 128,
    T_ROWS = 2,
    T_DIM = 32,
    MODALITIES = 3,
    MODULATION_SLOTS = 6,
    ROPE_HALF = 12,
    MAX_GPU_TENSORS = 64
};

typedef struct {
    h3_st_header fixture;
    h3_gpu *gpu;
    h3_gpu_tensor *tensors[MAX_GPU_TENSORS];
    size_t tensor_count;
} metal_test;

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_metal.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void require_gpu(metal_test *test, int condition, const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_metal.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *own(metal_test *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation failed");
    require(test->tensor_count < MAX_GPU_TENSORS, "Metal tensor registry is full");
    test->tensors[test->tensor_count++] = tensor;
    return tensor;
}

static void *load_host(metal_test *test, const char *name, h3_dtype dtype,
                       size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    if (!tensor) {
        fprintf(stderr, "FAIL tests/test_metal.c: fixture has no %s\n", name);
        exit(1);
    }
    if (tensor->dtype != dtype || h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_metal.c: unexpected type/shape for %s\n",
                name);
        exit(1);
    }
    size_t item_size = h3_dtype_size(dtype);
    require(elements <= SIZE_MAX / item_size, "fixture tensor size overflow");
    size_t bytes = elements * item_size;
    void *result = malloc(bytes ? bytes : 1);
    require(result != NULL, "host allocation failed");
    char error[256];
    if (!h3_st_read_data(&test->fixture, tensor, result, bytes, error,
                         sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_metal.c: %s\n", error);
        exit(1);
    }
    return result;
}

static h3_gpu_tensor *upload_f32(metal_test *test, const char *name,
                                 size_t elements) {
    float *values = load_host(test, name, H3_DTYPE_F32, elements);
    h3_gpu_tensor *result = own(test, h3_gpu_tensor_from_f32(test->gpu, values,
                                                             elements));
    free(values);
    return result;
}

static h3_gpu_tensor *new_f32(metal_test *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_f32(test->gpu, elements));
}

static double relative_max(const float *got, const float *want, size_t count,
                           double *maximum_absolute) {
    double error = 0.0;
    double scale = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = fabs((double)got[index] - want[index]);
        double magnitude = fabs((double)want[index]);
        if (delta > error) error = delta;
        if (magnitude > scale) scale = magnitude;
    }
    *maximum_absolute = error;
    return error / (scale > 1e-12 ? scale : 1e-12);
}

static double compare(metal_test *test, const char *name,
                      h3_gpu_tensor *got_tensor, size_t elements,
                      double *maximum_absolute) {
    float *got = malloc(elements * sizeof(*got));
    float *want = load_host(test, name, H3_DTYPE_F32, elements);
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_f32(got_tensor, got, elements),
            "cannot read completed Metal tensor");
    double relative = relative_max(got, want, elements, maximum_absolute);
    free(got);
    free(want);
    return relative;
}

static void cleanup(metal_test *test) {
    for (size_t index = 0; index < test->tensor_count; index++) {
        h3_gpu_tensor_free(test->tensors[index]);
    }
    h3_gpu_free(test->gpu);
    h3_st_free_header(&test->fixture);
}

int main(int argc, char **argv) {
    const char *fixture_path = argc > 1 ? argv[1] :
                                         "misc/fixtures/h3_dit.safetensors";
    metal_test test;
    memset(&test, 0, sizeof(test));
    char error[512];
    if (!h3_st_read_header(fixture_path, &test.fixture, error, sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_metal.c: %s\n", error);
        return 1;
    }
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) {
        fprintf(stderr, "FAIL tests/test_metal.c: %s\n", error);
        h3_st_free_header(&test.fixture);
        return 1;
    }

    h3_gpu_tensor *h_in = upload_f32(&test, "x.h_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *attn_in = upload_f32(&test, "x.attn_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *t_emb = upload_f32(&test, "x.t_emb", T_ROWS * T_DIM);
    h3_gpu_tensor *rope_cos = upload_f32(&test, "x.rope_cos",
                                         SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = upload_f32(&test, "x.rope_sin",
                                         SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *norm1 = upload_f32(&test, "norm1.weight", HIDDEN);
    h3_gpu_tensor *norm2 = upload_f32(&test, "norm2.weight", HIDDEN);
    h3_gpu_tensor *adaln_w = upload_f32(&test, "adaln_proj.linear.weight",
                                        MODALITIES * MODULATION_SLOTS * HIDDEN * T_DIM);
    h3_gpu_tensor *adaln_b = upload_f32(&test, "adaln_proj.linear.bias",
                                        MODALITIES * MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *qkv_w = upload_f32(&test, "attn.qkv_proj.weight",
                                      INNER * 3 * HIDDEN);
    h3_gpu_tensor *q_norm = upload_f32(&test, "attn.q_norm.weight", HEAD_DIM);
    h3_gpu_tensor *k_norm = upload_f32(&test, "attn.k_norm.weight", HEAD_DIM);
    h3_gpu_tensor *out_w = upload_f32(&test, "attn.out_proj.weight",
                                      HIDDEN * INNER);
    h3_gpu_tensor *fc1_w = upload_f32(&test, "mlp.fc1.weight", FFN * 2 * HIDDEN);
    h3_gpu_tensor *fc2_w = upload_f32(&test, "mlp.fc2.weight", HIDDEN * FFN);

    int32_t *runs = load_host(&test, "x.runs", H3_DTYPE_I32, 12);
    uint32_t row_map[SEQUENCE];
    memset(row_map, 0xff, sizeof(row_map));
    for (size_t run = 0; run < 4; run++) {
        int32_t start = runs[run * 3];
        int32_t stop = runs[run * 3 + 1];
        int32_t row = runs[run * 3 + 2];
        require(start >= 0 && stop >= start && stop <= SEQUENCE,
                "invalid run bounds in fixture");
        require(row >= 0 && row < T_ROWS * MODALITIES,
                "invalid modulation row in fixture");
        for (int32_t index = start; index < stop; index++) {
            require(row_map[index] == UINT32_MAX, "overlapping fixture runs");
            row_map[index] = (uint32_t)row;
        }
    }
    free(runs);
    for (size_t index = 0; index < SEQUENCE; index++) {
        require(row_map[index] != UINT32_MAX, "fixture runs do not cover sequence");
    }
    h3_gpu_tensor *gpu_row_map = own(&test, h3_gpu_tensor_from_u32(test.gpu,
                                                                   row_map,
                                                                   SEQUENCE));

    h3_gpu_tensor *plain_qkv = new_f32(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *plain_q = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_k = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_v = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_sdpa = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_attn_out = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *plain_fc1 = new_f32(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *plain_swiglu = new_f32(&test, SEQUENCE * FFN);
    h3_gpu_tensor *plain_mlp_out = new_f32(&test, SEQUENCE * HIDDEN);

    h3_gpu_tensor *t_silu = new_f32(&test, T_ROWS * T_DIM);
    h3_gpu_tensor *modulation = new_f32(&test, T_ROWS * MODALITIES *
                                               MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *mod_attn = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_qkv = new_f32(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *full_q = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_k = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_v = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_sdpa = new_f32(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_attn_out = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *after_attn = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *mod_mlp = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_fc1 = new_f32(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *full_swiglu = new_f32(&test, SEQUENCE * FFN);
    h3_gpu_tensor *full_mlp_out = new_f32(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *h_out = new_f32(&test, SEQUENCE * HIDDEN);

    require_gpu(&test, h3_gpu_begin(test.gpu), "begin command stream");

    require_gpu(&test, h3_gpu_linear_f32(test.gpu, plain_qkv, attn_in, qkv_w,
                                          NULL, SEQUENCE, HIDDEN, INNER * 3),
                "plain QKV projection");
    require_gpu(&test, h3_gpu_qkv_rope_f32(test.gpu, plain_q, plain_k, plain_v,
                                            plain_qkv, q_norm, k_norm, rope_cos,
                                            rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                            ROPE_HALF, 1e-5f),
                "plain QK norm and RoPE");
    require_gpu(&test, h3_gpu_sdpa_f32(test.gpu, plain_sdpa, plain_q, plain_k,
                                       plain_v, SEQUENCE, HEADS, HEAD_DIM,
                                       1.0f / sqrtf((float)HEAD_DIM)),
                "plain attention");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, plain_attn_out, plain_sdpa,
                                          out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "plain attention output projection");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, plain_fc1, attn_in, fc1_w,
                                          NULL, SEQUENCE, HIDDEN, FFN * 2),
                "plain MLP input projection");
    require_gpu(&test, h3_gpu_swiglu_f32(test.gpu, plain_swiglu, plain_fc1,
                                          SEQUENCE, FFN),
                "plain SwiGLU");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, plain_mlp_out, plain_swiglu,
                                          fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "plain MLP output projection");

    require_gpu(&test, h3_gpu_silu_f32(test.gpu, t_silu, t_emb, T_ROWS * T_DIM),
                "AdaLN SiLU");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, modulation, t_silu, adaln_w,
                                          adaln_b, T_ROWS, T_DIM,
                                          MODALITIES * MODULATION_SLOTS * HIDDEN),
                "AdaLN projection");
    require_gpu(&test, h3_gpu_adaln_f32(test.gpu, mod_attn, h_in, norm1,
                                         modulation, gpu_row_map, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 0, 1, 1e-5f),
                "attention AdaLN");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, full_qkv, mod_attn, qkv_w,
                                          NULL, SEQUENCE, HIDDEN, INNER * 3),
                "full QKV projection");
    require_gpu(&test, h3_gpu_qkv_rope_f32(test.gpu, full_q, full_k, full_v,
                                            full_qkv, q_norm, k_norm, rope_cos,
                                            rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                            ROPE_HALF, 1e-5f),
                "full QK norm and RoPE");
    require_gpu(&test, h3_gpu_sdpa_f32(test.gpu, full_sdpa, full_q, full_k,
                                       full_v, SEQUENCE, HEADS, HEAD_DIM,
                                       1.0f / sqrtf((float)HEAD_DIM)),
                "full attention");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, full_attn_out, full_sdpa,
                                          out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "full attention output projection");
    require_gpu(&test, h3_gpu_gate_f32(test.gpu, after_attn, h_in,
                                        full_attn_out, modulation, gpu_row_map,
                                        SEQUENCE, HIDDEN, MODULATION_SLOTS, 2),
                "attention residual gate");
    require_gpu(&test, h3_gpu_adaln_f32(test.gpu, mod_mlp, after_attn, norm2,
                                         modulation, gpu_row_map, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 3, 4, 1e-5f),
                "MLP AdaLN");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, full_fc1, mod_mlp, fc1_w,
                                          NULL, SEQUENCE, HIDDEN, FFN * 2),
                "full MLP input projection");
    require_gpu(&test, h3_gpu_swiglu_f32(test.gpu, full_swiglu, full_fc1,
                                          SEQUENCE, FFN),
                "full SwiGLU");
    require_gpu(&test, h3_gpu_linear_f32(test.gpu, full_mlp_out, full_swiglu,
                                          fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "full MLP output projection");
    require_gpu(&test, h3_gpu_gate_f32(test.gpu, h_out, after_attn,
                                        full_mlp_out, modulation, gpu_row_map,
                                        SEQUENCE, HIDDEN, MODULATION_SLOTS, 5),
                "MLP residual gate");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit command stream");

    double attn_abs, mlp_abs, block_abs;
    double attn_rel = compare(&test, "x.attn_out", plain_attn_out,
                              SEQUENCE * HIDDEN, &attn_abs);
    double mlp_rel = compare(&test, "x.mlp_out", plain_mlp_out,
                             SEQUENCE * HIDDEN, &mlp_abs);
    double block_rel = compare(&test, "x.h_out", h_out,
                               SEQUENCE * HIDDEN, &block_abs);
    printf("Metal/MLX parity: attention rel %.3g abs %.3g; "
           "MLP rel %.3g abs %.3g; block rel %.3g abs %.3g\n",
           attn_rel, attn_abs, mlp_rel, mlp_abs, block_rel, block_abs);
    require(attn_rel < 5e-3, "attention exceeds MLX error bound");
    require(mlp_rel < 5e-3, "MLP exceeds MLX error bound");
    require(block_rel < 5e-3, "full block exceeds MLX error bound");

    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "cannot read Metal statistics");
    require(stats.allocated_bytes > 0, "Metal allocation counter stayed zero");
    require(stats.live_bytes == stats.allocated_bytes,
            "live Metal tensor bytes do not match retained allocations");
    require(stats.peak_live_bytes == stats.live_bytes,
            "peak Metal tensor bytes do not match retained allocations");

    cleanup(&test);
    puts("ok: Metal toy block matches MLX fixture without CPU round trips");
    return 0;
}
