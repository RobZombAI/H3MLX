#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    SEQUENCE = 32, HIDDEN = 256, HEADS = 4, HEAD_DIM = 32,
    INNER = HEADS * HEAD_DIM, FFN = 128, T_ROWS = 2, T_DIM = 32,
    MODALITIES = 3, MODULATION_SLOTS = 6, ROPE_HALF = 12,
    MAX_TENSORS = 104
};

typedef struct {
    h3_st_header fixture;
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} test_context;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void require_gpu(test_context *test, int condition,
                        const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_bf16.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry is full");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void *load(test_context *test, const char *name, h3_dtype dtype,
                  size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_bf16.c: invalid or absent %s\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes ? bytes : 1);
    require(data != NULL, "host tensor allocation failed");
    char error[256];
    if (!h3_st_read_data(&test->fixture, tensor, data, bytes, error,
                         sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        exit(1);
    }
    return data;
}

static h3_gpu_tensor *upload(test_context *test, const char *name,
                             size_t elements) {
    uint16_t *values = load(test, name, H3_DTYPE_BF16, elements);
    h3_gpu_tensor *tensor = own(test, h3_gpu_tensor_from_bf16(test->gpu,
                                                              values, elements));
    free(values);
    return tensor;
}

static h3_gpu_tensor *fresh(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, elements));
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void test_token_reduction_kernels(test_context *test) {
    enum {
        FULL_ROWS = 6, REDUCED_ROWS = 4, BASELINE_ROWS = 2,
        WIDTH = 4, PADDING = 4
    };
    const float input_values[FULL_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        10.0f, 12.0f, 14.0f, 16.0f,
        14.0f, 16.0f, 18.0f, 20.0f,
        -8.0f, -4.0f,  0.0f,  4.0f,
        -4.0f,  0.0f,  4.0f,  8.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    };
    const float pooled_values[REDUCED_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        12.0f, 14.0f, 16.0f, 18.0f,
        -6.0f, -2.0f,  2.0f,  6.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    };
    const float processed_values[REDUCED_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        14.0f, 12.0f, 20.0f, 14.0f,
        -5.0f,  0.0f,  5.0f, 10.0f,
        30.0f, 31.0f, 32.0f, 33.0f
    };
    const float expanded_values[FULL_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        12.0f, 10.0f, 18.0f, 12.0f,
        16.0f, 14.0f, 22.0f, 16.0f,
        -7.0f, -2.0f,  3.0f,  8.0f,
        -3.0f,  2.0f,  7.0f, 12.0f,
        30.0f, 31.0f, 32.0f, 33.0f
    };
    const uint32_t pairs[REDUCED_ROWS * 2] = {
        0, 0, 1, 2, 3, 4, 5, 5
    };
    const uint32_t baseline_indices[REDUCED_ROWS] = {
        UINT32_MAX, 0, 1, UINT32_MAX
    };
    const uint32_t parents[FULL_ROWS] = {0, 1, 1, 2, 2, 3};
    uint16_t input_bf16[PADDING + FULL_ROWS * WIDTH];
    uint16_t processed_bf16[REDUCED_ROWS * WIDTH];
    uint16_t expected_pooled[REDUCED_ROWS * WIDTH];
    uint16_t expected_expanded[FULL_ROWS * WIDTH];
    for (size_t index = 0; index < PADDING; index++)
        input_bf16[index] = f32_to_bf16(-99.0f);
    for (size_t index = 0; index < FULL_ROWS * WIDTH; index++) {
        input_bf16[PADDING + index] = f32_to_bf16(input_values[index]);
        expected_expanded[index] = f32_to_bf16(expanded_values[index]);
    }
    for (size_t index = 0; index < REDUCED_ROWS * WIDTH; index++) {
        processed_bf16[index] = f32_to_bf16(processed_values[index]);
        expected_pooled[index] = f32_to_bf16(pooled_values[index]);
    }
    h3_gpu_tensor *input = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, input_bf16, PADDING + FULL_ROWS * WIDTH));
    h3_gpu_tensor *gpu_pairs = own(test, h3_gpu_tensor_from_u32(
        test->gpu, pairs, REDUCED_ROWS * 2));
    h3_gpu_tensor *gpu_baseline_indices = own(
        test, h3_gpu_tensor_from_u32(test->gpu, baseline_indices,
                                     REDUCED_ROWS));
    h3_gpu_tensor *pooled = fresh(
        test, (REDUCED_ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *original = fresh(
        test, PADDING + FULL_ROWS * WIDTH);
    h3_gpu_tensor *processed = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, processed_bf16, REDUCED_ROWS * WIDTH));
    h3_gpu_tensor *gpu_parents = own(test, h3_gpu_tensor_from_u32(
        test->gpu, parents, FULL_ROWS));
    h3_gpu_tensor *expanded = fresh(test, FULL_ROWS * WIDTH);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token-pool command stream");
    require_gpu(test, h3_gpu_token_pool_bf16(
        test->gpu, pooled, input, PADDING, original, PADDING,
        pooled, REDUCED_ROWS * WIDTH,
        gpu_baseline_indices, gpu_pairs, FULL_ROWS, REDUCED_ROWS,
        BASELINE_ROWS, WIDTH),
        "encode token pooling");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token-pool command stream");
    uint16_t got_pooled[REDUCED_ROWS * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                pooled, got_pooled, REDUCED_ROWS * WIDTH),
            "cannot read pooled tokens");
    require(memcmp(got_pooled, expected_pooled, sizeof(got_pooled)) == 0,
            "horizontal token pooling differs from exact BF16 reference");
    uint16_t got_baseline[(REDUCED_ROWS + BASELINE_ROWS) * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                pooled, got_baseline,
                (REDUCED_ROWS + BASELINE_ROWS) * WIDTH),
            "cannot read fused pair baselines");
    require(memcmp(got_baseline + REDUCED_ROWS * WIDTH,
                   expected_pooled + WIDTH,
                   BASELINE_ROWS * WIDTH * sizeof(*got_baseline)) == 0,
            "fused pooling wrote the wrong dense pair baselines");
    uint16_t got_original[PADDING + FULL_ROWS * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                original, got_original, PADDING + FULL_ROWS * WIDTH),
            "cannot read fused full-token snapshot");
    require(memcmp(got_original + PADDING, input_bf16 + PADDING,
                   FULL_ROWS * WIDTH * sizeof(*got_original)) == 0,
            "fused pooling wrote the wrong full-token snapshot");

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token-expand command stream");
    require_gpu(test, h3_gpu_token_expand_delta_bf16(
        test->gpu, expanded, original, PADDING, processed, pooled,
        REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_parents,
        FULL_ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH, 1, 1.0f),
        "encode token delta expansion");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token-expand command stream");
    uint16_t got_expanded[FULL_ROWS * WIDTH];
    require(h3_gpu_tensor_read_bf16(
                expanded, got_expanded, FULL_ROWS * WIDTH),
            "cannot read expanded tokens");
    require(memcmp(got_expanded, expected_expanded,
                   sizeof(got_expanded)) == 0,
            "token expansion lost the full-grid residual");

    const float norm_values[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float modulation_values[2 * 2 * WIDTH] = {
         0.25f, -0.5f, 0.75f, -1.0f,
         0.0f,   0.5f, 0.25f, -0.25f,
        -0.75f,  0.5f, 0.25f,  1.0f,
         0.25f,  0.0f, 0.75f, -0.5f
    };
    const uint32_t row_map[FULL_ROWS] = {0, 1, 0, 1, 0, 1};
    const uint32_t reduced_row_map[REDUCED_ROWS] = {0, 1, 0, 1};
    uint16_t norm_bf16[WIDTH];
    uint16_t modulation_bf16[2 * 2 * WIDTH];
    enum { HEAD_OUTPUT = 3 };
    uint16_t head_weight_bf16[HEAD_OUTPUT * WIDTH];
    uint16_t head_bias_bf16[HEAD_OUTPUT];
    for (size_t index = 0; index < WIDTH; index++)
        norm_bf16[index] = f32_to_bf16(norm_values[index]);
    for (size_t index = 0; index < 2 * 2 * WIDTH; index++)
        modulation_bf16[index] = f32_to_bf16(modulation_values[index]);
    for (size_t index = 0; index < HEAD_OUTPUT * WIDTH; index++)
        head_weight_bf16[index] = f32_to_bf16(
            (float)((int)(index % 7) - 3) * 0.125f);
    for (size_t index = 0; index < HEAD_OUTPUT; index++)
        head_bias_bf16[index] = f32_to_bf16((float)index * 0.0625f);
    h3_gpu_tensor *norm = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, norm_bf16, WIDTH));
    h3_gpu_tensor *modulation = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, modulation_bf16, 2 * 2 * WIDTH));
    h3_gpu_tensor *head_weight = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, head_weight_bf16, HEAD_OUTPUT * WIDTH));
    h3_gpu_tensor *head_bias = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, head_bias_bf16, HEAD_OUTPUT));
    h3_gpu_tensor *gpu_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, row_map, FULL_ROWS));
    h3_gpu_tensor *gpu_reduced_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, reduced_row_map, REDUCED_ROWS));
    h3_gpu_tensor *pooled_adaln_reference = fresh(
        test, REDUCED_ROWS * WIDTH);
    h3_gpu_tensor *fused_pooled = fresh(
        test, (REDUCED_ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *fused_pool_original = fresh(
        test, PADDING + FULL_ROWS * WIDTH);
    h3_gpu_tensor *fused_pooled_adaln = fresh(
        test, REDUCED_ROWS * WIDTH);
    h3_gpu_tensor *adaln_reference = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *fused_expanded = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *fused_adaln = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *gate_reference = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *gate_adaln_reference = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *fused_gate_residual = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *fused_gate_adaln = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *offset_slice = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *offset_adaln_reference = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *offset_adaln = fresh(test, FULL_ROWS * WIDTH);
    h3_gpu_tensor *head_reference = fresh(test, FULL_ROWS * HEAD_OUTPUT);
    h3_gpu_tensor *head_fused = fresh(test, FULL_ROWS * HEAD_OUTPUT);
    h3_gpu_tensor *head_inverse = own(
        test, h3_gpu_tensor_new_f32(test->gpu, FULL_ROWS));
    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin fused token AdaLN command stream");
    require_gpu(test, h3_gpu_adaln_bf16(
        test->gpu, pooled_adaln_reference, pooled, norm, modulation,
        gpu_reduced_row_map, REDUCED_ROWS, WIDTH, 2, 0, 1, 1e-5f),
        "encode reference pooled AdaLN");
    require_gpu(test, h3_gpu_token_pool_adaln_bf16(
        test->gpu, fused_pooled, fused_pooled_adaln, input, PADDING,
        fused_pool_original, PADDING, fused_pooled, REDUCED_ROWS * WIDTH,
        gpu_baseline_indices, gpu_pairs, norm, modulation,
        gpu_reduced_row_map, FULL_ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH,
        2, 0, 1, 1e-5f),
        "encode fused token pooling and AdaLN");
    require_gpu(test, h3_gpu_adaln_bf16(
        test->gpu, adaln_reference, expanded, norm, modulation, gpu_row_map,
        FULL_ROWS, WIDTH, 2, 0, 1, 1e-5f),
        "encode reference token AdaLN");
    require_gpu(test, h3_gpu_token_expand_adaln_bf16(
        test->gpu, fused_expanded, fused_adaln, original, PADDING,
        processed, pooled, REDUCED_ROWS * WIDTH, gpu_baseline_indices,
        gpu_parents, norm, modulation, gpu_row_map, FULL_ROWS, REDUCED_ROWS,
        BASELINE_ROWS, WIDTH, 1, 1.0f, 2, 0, 1, 1e-5f),
        "encode fused token expansion and AdaLN");
    require_gpu(test, h3_gpu_gate_bf16(
        test->gpu, gate_reference, expanded, fused_expanded, modulation,
        gpu_row_map, FULL_ROWS, WIDTH, 2, 0),
        "encode gate reference for fused AdaLN");
    require_gpu(test, h3_gpu_adaln_bf16(
        test->gpu, gate_adaln_reference, gate_reference, norm, modulation,
        gpu_row_map, FULL_ROWS, WIDTH, 2, 0, 1, 1e-5f),
        "encode gate AdaLN reference");
    require_gpu(test, h3_gpu_gate_adaln_bf16(
        test->gpu, fused_gate_residual, fused_gate_adaln, expanded,
        fused_expanded, norm, modulation, modulation, gpu_row_map,
        FULL_ROWS, WIDTH,
        2, 0, 0, 1, 1e-5f),
        "encode fused gate AdaLN");
    require_gpu(test, h3_gpu_copy_bf16(
        test->gpu, offset_slice, 0, input, PADDING, FULL_ROWS * WIDTH),
        "encode reference final slice");
    require_gpu(test, h3_gpu_adaln_bf16(
        test->gpu, offset_adaln_reference, offset_slice, norm, modulation,
        gpu_row_map, FULL_ROWS, WIDTH, 2, 0, 1, 1e-5f),
        "encode reference sliced AdaLN");
    require_gpu(test, h3_gpu_adaln_bf16_offset(
        test->gpu, offset_adaln, input, PADDING, norm, modulation,
        gpu_row_map, FULL_ROWS, WIDTH, 2, 0, 1, 1e-5f),
        "encode offset-bound AdaLN");
    require_gpu(test, h3_gpu_linear_bf16(
        test->gpu, head_reference, offset_adaln_reference, head_weight,
        head_bias, FULL_ROWS, WIDTH, HEAD_OUTPUT),
        "encode reference final head");
    require_gpu(test, h3_gpu_adaln_linear_bf16(
        test->gpu, head_fused, head_inverse, input, PADDING, norm, modulation,
        gpu_row_map, head_weight, head_bias, FULL_ROWS, WIDTH, HEAD_OUTPUT,
        2, 0, 1, 1e-5f),
        "encode fused final AdaLN/head");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit fused token AdaLN command stream");
    uint16_t got_fused_pooled[(REDUCED_ROWS + BASELINE_ROWS) * WIDTH];
    uint16_t got_fused_pool_original[PADDING + FULL_ROWS * WIDTH];
    uint16_t got_pooled_adaln_reference[REDUCED_ROWS * WIDTH];
    uint16_t got_fused_pooled_adaln[REDUCED_ROWS * WIDTH];
    uint16_t got_fused_expanded[FULL_ROWS * WIDTH];
    uint16_t got_adaln_reference[FULL_ROWS * WIDTH];
    uint16_t got_fused_adaln[FULL_ROWS * WIDTH];
    uint16_t got_gate_reference[FULL_ROWS * WIDTH];
    uint16_t got_gate_adaln_reference[FULL_ROWS * WIDTH];
    uint16_t got_fused_gate_residual[FULL_ROWS * WIDTH];
    uint16_t got_fused_gate_adaln[FULL_ROWS * WIDTH];
    uint16_t got_offset_adaln_reference[FULL_ROWS * WIDTH];
    uint16_t got_offset_adaln[FULL_ROWS * WIDTH];
    uint16_t got_head_reference[FULL_ROWS * HEAD_OUTPUT];
    uint16_t got_head_fused[FULL_ROWS * HEAD_OUTPUT];
    require(h3_gpu_tensor_read_bf16(
                fused_pooled, got_fused_pooled,
                (REDUCED_ROWS + BASELINE_ROWS) * WIDTH),
            "cannot read fused pooled residual and baselines");
    require(h3_gpu_tensor_read_bf16(
                fused_pool_original, got_fused_pool_original,
                PADDING + FULL_ROWS * WIDTH),
            "cannot read fused pooled full-token snapshot");
    require(h3_gpu_tensor_read_bf16(
                pooled_adaln_reference, got_pooled_adaln_reference,
                REDUCED_ROWS * WIDTH),
            "cannot read reference pooled AdaLN");
    require(h3_gpu_tensor_read_bf16(
                fused_pooled_adaln, got_fused_pooled_adaln,
                REDUCED_ROWS * WIDTH),
            "cannot read fused pooled AdaLN");
    require(h3_gpu_tensor_read_bf16(
                fused_expanded, got_fused_expanded, FULL_ROWS * WIDTH),
            "cannot read fused restored tokens");
    require(h3_gpu_tensor_read_bf16(
                adaln_reference, got_adaln_reference, FULL_ROWS * WIDTH),
            "cannot read reference restored AdaLN");
    require(h3_gpu_tensor_read_bf16(
                fused_adaln, got_fused_adaln, FULL_ROWS * WIDTH),
            "cannot read fused restored AdaLN");
    require(h3_gpu_tensor_read_bf16(
                gate_reference, got_gate_reference, FULL_ROWS * WIDTH),
            "cannot read gate reference");
    require(h3_gpu_tensor_read_bf16(
                gate_adaln_reference, got_gate_adaln_reference,
                FULL_ROWS * WIDTH),
            "cannot read gate AdaLN reference");
    require(h3_gpu_tensor_read_bf16(
                fused_gate_residual, got_fused_gate_residual,
                FULL_ROWS * WIDTH),
            "cannot read fused gated residual");
    require(h3_gpu_tensor_read_bf16(
                fused_gate_adaln, got_fused_gate_adaln, FULL_ROWS * WIDTH),
            "cannot read fused gate AdaLN");
    require(h3_gpu_tensor_read_bf16(
                offset_adaln_reference, got_offset_adaln_reference,
                FULL_ROWS * WIDTH),
            "cannot read reference sliced AdaLN");
    require(h3_gpu_tensor_read_bf16(
                offset_adaln, got_offset_adaln, FULL_ROWS * WIDTH),
            "cannot read offset-bound AdaLN");
    require(h3_gpu_tensor_read_bf16(
                head_reference, got_head_reference,
                FULL_ROWS * HEAD_OUTPUT),
            "cannot read reference final head");
    require(h3_gpu_tensor_read_bf16(
                head_fused, got_head_fused, FULL_ROWS * HEAD_OUTPUT),
            "cannot read fused final head");
    require(memcmp(got_fused_pooled, got_baseline,
                   sizeof(got_fused_pooled)) == 0,
            "fused token pool AdaLN changed residual or baselines");
    require(memcmp(got_fused_pool_original + PADDING,
                   got_original + PADDING,
                   FULL_ROWS * WIDTH * sizeof(*got_fused_pool_original)) == 0,
            "fused token pool AdaLN changed the full-token snapshot");
    require(memcmp(got_fused_pooled_adaln, got_pooled_adaln_reference,
                   sizeof(got_fused_pooled_adaln)) == 0,
            "fused token pool AdaLN differs from the exact two-kernel path");
    require(memcmp(got_fused_expanded, expected_expanded,
                   sizeof(got_fused_expanded)) == 0,
            "fused token AdaLN changed the restored residual");
    require(memcmp(got_fused_adaln, got_adaln_reference,
                   sizeof(got_fused_adaln)) == 0,
            "fused token AdaLN differs from the exact two-kernel path");
    require(memcmp(got_fused_gate_residual, got_gate_reference,
                   sizeof(got_fused_gate_residual)) == 0,
            "fused gate AdaLN changed the gated residual");
    require(memcmp(got_fused_gate_adaln, got_gate_adaln_reference,
                   sizeof(got_fused_gate_adaln)) == 0,
            "fused gate AdaLN differs from the exact two-kernel path");
    require(memcmp(got_offset_adaln, got_offset_adaln_reference,
                   sizeof(got_offset_adaln)) == 0,
            "offset-bound AdaLN differs from copy plus AdaLN");
    require(memcmp(got_head_fused, got_head_reference,
                   sizeof(got_head_fused)) == 0,
            "fused final AdaLN/head differs from the two-kernel path");
}

static double monotonic_seconds(void) {
    struct timespec value;
    require(clock_gettime(CLOCK_MONOTONIC, &value) == 0,
            "cannot read monotonic clock");
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void bench_patch_projection(test_context *test, uint32_t rows) {
    enum { INPUT_DIM = 96, OUTPUT_DIM = 5376, ITERATIONS = 20 };
    size_t input_count = (size_t)rows * INPUT_DIM;
    size_t weight_count = (size_t)OUTPUT_DIM * INPUT_DIM;
    size_t output_count = (size_t)rows * OUTPUT_DIM;
    require(output_count <= UINT32_MAX, "patch benchmark shape is too large");
    h3_gpu_tensor *input = own(
        test, h3_gpu_tensor_new_f32(test->gpu, input_count));
    h3_gpu_tensor *weight = own(
        test, h3_gpu_tensor_new_f32(test->gpu, weight_count));
    h3_gpu_tensor *bias = own(
        test, h3_gpu_tensor_new_f32(test->gpu, OUTPUT_DIM));
    h3_gpu_tensor *f32_output = own(
        test, h3_gpu_tensor_new_f32(test->gpu, output_count));
    h3_gpu_tensor *bf16_output = own(
        test, h3_gpu_tensor_new_bf16(test->gpu, output_count));
    size_t destination_offset = (size_t)80 * OUTPUT_DIM;
    h3_gpu_tensor *packed_output = own(
        test, h3_gpu_tensor_new_bf16(
            test->gpu, destination_offset + output_count));
    static const int fused_pattern[] = {0, 1, 1, 0, 0, 1, 1, 0};
    double separate_seconds = 0.0, fused_seconds = 0.0;
    unsigned separate_count = 0, fused_count = 0;
    for (size_t run = 0;
         run < sizeof(fused_pattern) / sizeof(*fused_pattern); run++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin patch projection benchmark");
        for (unsigned iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[run]) {
                require_gpu(test, h3_gpu_patch_linear_bf16(
                    test->gpu, bf16_output, input, weight, bias, rows,
                    INPUT_DIM, OUTPUT_DIM), "benchmark fused patch projection");
            } else {
                require_gpu(test, h3_gpu_linear_f32(
                    test->gpu, f32_output, input, weight, bias, rows,
                    INPUT_DIM, OUTPUT_DIM), "benchmark patch projection");
                require_gpu(test, h3_gpu_cast_f32_to_bf16(
                    test->gpu, bf16_output, f32_output,
                    (uint32_t)output_count), "benchmark patch cast");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit patch projection benchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[run]) {
            fused_seconds += elapsed;
            fused_count++;
        } else {
            separate_seconds += elapsed;
            separate_count++;
        }
    }
    double separate = separate_seconds /
        ((double)separate_count * (double)ITERATIONS);
    double fused = fused_seconds /
        ((double)fused_count * (double)ITERATIONS);
    printf("patch projection %u rows: separate %.3f ms, fused %.3f ms, "
           "ratio %.4f\n", rows, separate * 1000.0, fused * 1000.0,
           fused / separate);

    separate_seconds = 0.0;
    fused_seconds = 0.0;
    separate_count = 0;
    fused_count = 0;
    for (size_t run = 0;
         run < sizeof(fused_pattern) / sizeof(*fused_pattern); run++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin patch packing benchmark");
        for (unsigned iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[run]) {
                require_gpu(test, h3_gpu_patch_linear_bf16_offset(
                    test->gpu, packed_output, destination_offset, input, 0,
                    weight, bias, rows, INPUT_DIM, OUTPUT_DIM),
                            "benchmark directly packed projection");
            } else {
                require_gpu(test, h3_gpu_patch_linear_bf16(
                    test->gpu, bf16_output, input, weight, bias, rows,
                    INPUT_DIM, OUTPUT_DIM),
                            "benchmark temporary patch projection");
                require_gpu(test, h3_gpu_copy_bf16(
                    test->gpu, packed_output, destination_offset, bf16_output,
                    0, output_count), "benchmark patch packing blit");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit patch packing benchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[run]) {
            fused_seconds += elapsed;
            fused_count++;
        } else {
            separate_seconds += elapsed;
            separate_count++;
        }
    }
    separate = separate_seconds /
        ((double)separate_count * (double)ITERATIONS);
    fused = fused_seconds / ((double)fused_count * (double)ITERATIONS);
    printf("patch packing %u rows: temporary %.3f ms, direct %.3f ms, "
           "ratio %.4f\n", rows, separate * 1000.0, fused * 1000.0,
           fused / separate);
}

static void bench_gate_adaln_shape(test_context *test, uint32_t rows,
                                   const char *label) {
    enum { WIDTH = 5376, SLOTS = 6, ITERATIONS = 64 };
    size_t elements = (size_t)rows * WIDTH;
    uint32_t *row_map = calloc(rows, sizeof(*row_map));
    require(row_map != NULL, "gate AdaLN benchmark row-map allocation failed");
    h3_gpu_tensor *residual = fresh(test, elements);
    h3_gpu_tensor *branch = fresh(test, elements);
    h3_gpu_tensor *norm = fresh(test, WIDTH);
    h3_gpu_tensor *modulation = fresh(test, (size_t)SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, row_map, rows));
    h3_gpu_tensor *separate_residual = fresh(test, elements);
    h3_gpu_tensor *separate_output = fresh(test, elements);
    h3_gpu_tensor *fused_residual = fresh(test, elements);
    h3_gpu_tensor *fused_output = fresh(test, elements);
    free(row_map);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin gate AdaLN microbenchmark warmup");
    for (int warm = 0; warm < 16; warm++) {
        require_gpu(test, h3_gpu_gate_bf16(
            test->gpu, separate_residual, residual, branch, modulation,
            gpu_row_map, rows, WIDTH, SLOTS, 2),
            "encode gate AdaLN gate warmup");
        require_gpu(test, h3_gpu_adaln_bf16(
            test->gpu, separate_output, separate_residual, norm, modulation,
            gpu_row_map, rows, WIDTH, SLOTS, 3, 4, 1e-5f),
            "encode gate AdaLN norm warmup");
        require_gpu(test, h3_gpu_gate_adaln_bf16(
            test->gpu, fused_residual, fused_output, residual, branch, norm,
            modulation, modulation, gpu_row_map, rows, WIDTH, SLOTS,
            2, 3, 4, 1e-5f),
            "encode fused gate AdaLN warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit gate AdaLN microbenchmark warmup");

    static const int fused_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double separate_seconds = 0.0, fused_seconds = 0.0;
    int separate_count = 0, fused_count = 0;
    for (size_t sample = 0;
         sample < sizeof(fused_pattern) / sizeof(*fused_pattern); sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin gate AdaLN microbenchmark");
        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[sample]) {
                require_gpu(test, h3_gpu_gate_adaln_bf16(
                    test->gpu, fused_residual, fused_output, residual, branch,
                    norm, modulation, modulation, gpu_row_map, rows, WIDTH, SLOTS,
                    2, 3, 4, 1e-5f),
                    "encode fused gate AdaLN microbenchmark");
            } else {
                require_gpu(test, h3_gpu_gate_bf16(
                    test->gpu, separate_residual, residual, branch,
                    modulation, gpu_row_map, rows, WIDTH, SLOTS, 2),
                    "encode gate microbenchmark");
                require_gpu(test, h3_gpu_adaln_bf16(
                    test->gpu, separate_output, separate_residual, norm,
                    modulation, gpu_row_map, rows, WIDTH, SLOTS, 3, 4,
                    1e-5f), "encode gate AdaLN microbenchmark");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit gate AdaLN microbenchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[sample]) {
            fused_seconds += elapsed;
            fused_count++;
        } else {
            separate_seconds += elapsed;
            separate_count++;
        }
    }
    double separate_average = separate_seconds / separate_count /
                              (double)ITERATIONS;
    double fused_average = fused_seconds / fused_count / (double)ITERATIONS;
    printf("gate AdaLN %s separate %.6fs, fused %.6fs, ratio %.4f\n",
           label, separate_average, fused_average,
           fused_average / separate_average);
}

static void bench_final_slice_adaln(test_context *test, uint32_t rows) {
    enum {
        PREFIX_ROWS = 300, WIDTH = 5376, SLOTS = 2, ITERATIONS = 64
    };
    size_t elements = (size_t)rows * WIDTH;
    size_t prefix = (size_t)PREFIX_ROWS * WIDTH;
    uint32_t *row_map = calloc(rows, sizeof(*row_map));
    require(row_map != NULL,
            "final slice AdaLN benchmark row-map allocation failed");
    h3_gpu_tensor *source = fresh(test, prefix + elements);
    h3_gpu_tensor *slice = fresh(test, elements);
    h3_gpu_tensor *norm = fresh(test, WIDTH);
    h3_gpu_tensor *modulation = fresh(test, (size_t)SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, row_map, rows));
    h3_gpu_tensor *separate_output = fresh(test, elements);
    h3_gpu_tensor *offset_output = fresh(test, elements);
    free(row_map);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin final slice AdaLN warmup");
    for (int warm = 0; warm < 16; warm++) {
        require_gpu(test, h3_gpu_copy_bf16(
            test->gpu, slice, 0, source, prefix, elements),
            "encode final slice warmup");
        require_gpu(test, h3_gpu_adaln_bf16(
            test->gpu, separate_output, slice, norm, modulation, gpu_row_map,
            rows, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode separate final AdaLN warmup");
        require_gpu(test, h3_gpu_adaln_bf16_offset(
            test->gpu, offset_output, source, prefix, norm, modulation,
            gpu_row_map, rows, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode offset-bound final AdaLN warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit final slice AdaLN warmup");

    static const int offset_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double separate_seconds = 0.0, offset_seconds = 0.0;
    int separate_count = 0, offset_count = 0;
    for (size_t sample = 0;
         sample < sizeof(offset_pattern) / sizeof(*offset_pattern); sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin final slice AdaLN microbenchmark");
        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            if (offset_pattern[sample]) {
                require_gpu(test, h3_gpu_adaln_bf16_offset(
                    test->gpu, offset_output, source, prefix, norm,
                    modulation, gpu_row_map, rows, WIDTH, SLOTS,
                    0, 1, 1e-5f),
                    "encode offset-bound final AdaLN microbenchmark");
            } else {
                require_gpu(test, h3_gpu_copy_bf16(
                    test->gpu, slice, 0, source, prefix, elements),
                    "encode final slice microbenchmark");
                require_gpu(test, h3_gpu_adaln_bf16(
                    test->gpu, separate_output, slice, norm, modulation,
                    gpu_row_map, rows, WIDTH, SLOTS, 0, 1, 1e-5f),
                    "encode separate final AdaLN microbenchmark");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit final slice AdaLN microbenchmark");
        double elapsed = monotonic_seconds() - start;
        if (offset_pattern[sample]) {
            offset_seconds += elapsed;
            offset_count++;
        } else {
            separate_seconds += elapsed;
            separate_count++;
        }
    }
    double separate = separate_seconds / separate_count / (double)ITERATIONS;
    double offset = offset_seconds / offset_count / (double)ITERATIONS;
    printf("final video slice/AdaLN %u rows separate %.6fs, offset %.6fs, "
           "ratio %.4f\n", rows, separate, offset, offset / separate);
}

static void bench_final_head(test_context *test, uint32_t rows) {
    enum {
        PREFIX_ROWS = 300, WIDTH = 5376, OUTPUT_DIM = 24, SLOTS = 2,
        ITERATIONS = 64
    };
    size_t elements = (size_t)rows * WIDTH;
    size_t prefix = (size_t)PREFIX_ROWS * WIDTH;
    uint32_t *row_map = calloc(rows, sizeof(*row_map));
    require(row_map != NULL, "final head benchmark row-map allocation failed");
    h3_gpu_tensor *source = fresh(test, prefix + elements);
    h3_gpu_tensor *norm = fresh(test, WIDTH);
    h3_gpu_tensor *modulation = fresh(test, (size_t)SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, row_map, rows));
    h3_gpu_tensor *weight = fresh(test, (size_t)OUTPUT_DIM * WIDTH);
    h3_gpu_tensor *bias = fresh(test, OUTPUT_DIM);
    h3_gpu_tensor *separate_norm = fresh(test, elements);
    h3_gpu_tensor *separate_output = fresh(test, (size_t)rows * OUTPUT_DIM);
    h3_gpu_tensor *fused_output = fresh(test, (size_t)rows * OUTPUT_DIM);
    h3_gpu_tensor *inverse = own(
        test, h3_gpu_tensor_new_f32(test->gpu, rows));
    free(row_map);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin final head warmup");
    for (int warm = 0; warm < 16; warm++) {
        require_gpu(test, h3_gpu_adaln_bf16_offset(
            test->gpu, separate_norm, source, prefix, norm, modulation,
            gpu_row_map, rows, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode separate final head AdaLN warmup");
        require_gpu(test, h3_gpu_linear_bf16(
            test->gpu, separate_output, separate_norm, weight, bias,
            rows, WIDTH, OUTPUT_DIM),
            "encode separate final head linear warmup");
        require_gpu(test, h3_gpu_adaln_linear_bf16(
            test->gpu, fused_output, inverse, source, prefix, norm, modulation,
            gpu_row_map, weight, bias, rows, WIDTH, OUTPUT_DIM, SLOTS,
            0, 1, 1e-5f),
            "encode fused final head warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit final head warmup");

    static const int fused_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double separate_seconds = 0.0, fused_seconds = 0.0;
    int separate_count = 0, fused_count = 0;
    for (size_t sample = 0;
         sample < sizeof(fused_pattern) / sizeof(*fused_pattern); sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin final head microbenchmark");
        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[sample]) {
                require_gpu(test, h3_gpu_adaln_linear_bf16(
                    test->gpu, fused_output, inverse, source, prefix, norm,
                    modulation, gpu_row_map, weight, bias, rows, WIDTH,
                    OUTPUT_DIM, SLOTS, 0, 1, 1e-5f),
                    "encode fused final head microbenchmark");
            } else {
                require_gpu(test, h3_gpu_adaln_bf16_offset(
                    test->gpu, separate_norm, source, prefix, norm,
                    modulation, gpu_row_map, rows, WIDTH, SLOTS,
                    0, 1, 1e-5f),
                    "encode separate final head AdaLN microbenchmark");
                require_gpu(test, h3_gpu_linear_bf16(
                    test->gpu, separate_output, separate_norm, weight, bias,
                    rows, WIDTH, OUTPUT_DIM),
                    "encode separate final head linear microbenchmark");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit final head microbenchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[sample]) {
            fused_seconds += elapsed;
            fused_count++;
        } else {
            separate_seconds += elapsed;
            separate_count++;
        }
    }
    double separate = separate_seconds / separate_count / (double)ITERATIONS;
    double fused = fused_seconds / fused_count / (double)ITERATIONS;
    printf("final video AdaLN/head %u rows separate %.6fs, fused %.6fs, "
           "ratio %.4f\n", rows, separate, fused, fused / separate);
}

static void bench_token_pool_adaln(test_context *test) {
    enum {
        INPUT_ROWS = 2915, ROWS = 1550, PREFIX_ROWS = 80,
        BASELINE_ROWS = 1365, WIDTH = 5376, SLOTS = 6, ITERATIONS = 64
    };
    uint32_t *pairs = malloc((size_t)ROWS * 2 * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(ROWS * sizeof(*baseline_indices));
    uint32_t *row_map = calloc(ROWS, sizeof(*row_map));
    require(pairs && baseline_indices && row_map,
            "token pool AdaLN benchmark map allocation failed");
    for (uint32_t row = 0; row < PREFIX_ROWS; row++) {
        pairs[(size_t)row * 2] = row;
        pairs[(size_t)row * 2 + 1] = row;
        baseline_indices[row] = UINT32_MAX;
    }
    uint32_t baseline_row = 0;
    for (uint32_t local = 0; local < ROWS - PREFIX_ROWS; local++) {
        uint32_t spatial_row = local / 14;
        uint32_t column = local % 14;
        uint32_t first = PREFIX_ROWS + spatial_row * 27 + column * 2;
        uint32_t second = column < 13 ? first + 1 : first;
        uint32_t row = PREFIX_ROWS + local;
        pairs[(size_t)row * 2] = first;
        pairs[(size_t)row * 2 + 1] = second;
        baseline_indices[row] = column < 13 ? baseline_row++ : UINT32_MAX;
    }
    require(baseline_row == BASELINE_ROWS,
            "token pool AdaLN benchmark map is inconsistent");
    h3_gpu_tensor *input = fresh(test, (size_t)INPUT_ROWS * WIDTH);
    h3_gpu_tensor *separate_residual = fresh(
        test, (size_t)(ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *separate_output = fresh(test, (size_t)ROWS * WIDTH);
    h3_gpu_tensor *separate_original = fresh(
        test, (size_t)INPUT_ROWS * WIDTH);
    h3_gpu_tensor *fused_residual = fresh(
        test, (size_t)(ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *fused_output = fresh(test, (size_t)ROWS * WIDTH);
    h3_gpu_tensor *fused_original = fresh(
        test, (size_t)INPUT_ROWS * WIDTH);
    h3_gpu_tensor *norm = fresh(test, WIDTH);
    h3_gpu_tensor *modulation = fresh(test, (size_t)SLOTS * WIDTH);
    h3_gpu_tensor *gpu_pairs = own(test, h3_gpu_tensor_from_u32(
        test->gpu, pairs, (size_t)ROWS * 2));
    h3_gpu_tensor *gpu_baseline_indices = own(test,
        h3_gpu_tensor_from_u32(test->gpu, baseline_indices, ROWS));
    h3_gpu_tensor *gpu_row_map = own(test, h3_gpu_tensor_from_u32(
        test->gpu, row_map, ROWS));
    free(pairs);
    free(baseline_indices);
    free(row_map);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token pool AdaLN microbenchmark warmup");
    for (int warm = 0; warm < 16; warm++) {
        require_gpu(test, h3_gpu_token_pool_bf16(
            test->gpu, separate_residual, input, 0, separate_original, 0,
            separate_residual, (size_t)ROWS * WIDTH, gpu_baseline_indices,
            gpu_pairs, INPUT_ROWS, ROWS, BASELINE_ROWS, WIDTH),
            "encode token pool warmup");
        require_gpu(test, h3_gpu_adaln_bf16(
            test->gpu, separate_output, separate_residual, norm, modulation,
            gpu_row_map, ROWS, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode pooled AdaLN warmup");
        require_gpu(test, h3_gpu_token_pool_adaln_bf16(
            test->gpu, fused_residual, fused_output, input, 0,
            fused_original, 0, fused_residual, (size_t)ROWS * WIDTH,
            gpu_baseline_indices, gpu_pairs, norm, modulation, gpu_row_map,
            INPUT_ROWS, ROWS, BASELINE_ROWS, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode fused token pool AdaLN warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token pool AdaLN microbenchmark warmup");

    static const int fused_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double separate_seconds = 0.0, fused_seconds = 0.0;
    int separate_count = 0, fused_count = 0;
    for (size_t sample = 0;
         sample < sizeof(fused_pattern) / sizeof(*fused_pattern); sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin token pool AdaLN microbenchmark");
        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[sample]) {
                require_gpu(test, h3_gpu_token_pool_adaln_bf16(
                    test->gpu, fused_residual, fused_output, input, 0,
                    fused_original, 0, fused_residual, (size_t)ROWS * WIDTH,
                    gpu_baseline_indices, gpu_pairs, norm, modulation,
                    gpu_row_map, INPUT_ROWS, ROWS, BASELINE_ROWS, WIDTH,
                    SLOTS, 0, 1, 1e-5f),
                    "encode fused token pool AdaLN microbenchmark");
            } else {
                require_gpu(test, h3_gpu_token_pool_bf16(
                    test->gpu, separate_residual, input, 0,
                    separate_original, 0, separate_residual,
                    (size_t)ROWS * WIDTH, gpu_baseline_indices, gpu_pairs,
                    INPUT_ROWS, ROWS, BASELINE_ROWS, WIDTH),
                    "encode token pool microbenchmark");
                require_gpu(test, h3_gpu_adaln_bf16(
                    test->gpu, separate_output, separate_residual, norm,
                    modulation, gpu_row_map, ROWS, WIDTH, SLOTS, 0, 1,
                    1e-5f), "encode pooled AdaLN microbenchmark");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit token pool AdaLN microbenchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[sample]) {
            fused_seconds += elapsed;
            fused_count++;
            printf("  fused token pool AdaLN %.6fs\n",
                   elapsed / (double)ITERATIONS);
        } else {
            separate_seconds += elapsed;
            separate_count++;
            printf("  separate token pool AdaLN %.6fs\n",
                   elapsed / (double)ITERATIONS);
        }
    }
    double separate_average = separate_seconds / separate_count /
                              (double)ITERATIONS;
    double fused_average = fused_seconds / fused_count / (double)ITERATIONS;
    printf("token pool AdaLN microbenchmark separate %.6fs, fused %.6fs, "
           "ratio %.4f\n", separate_average, fused_average,
           fused_average / separate_average);
}

static void bench_token_expand_adaln(test_context *test) {
    enum {
        ROWS = 2915, REDUCED_ROWS = 1550, PREFIX_ROWS = 80,
        BASELINE_ROWS = 1365, WIDTH = 5376, SLOTS = 6, ITERATIONS = 64
    };
    uint32_t *baseline_indices = malloc(
        REDUCED_ROWS * sizeof(*baseline_indices));
    uint32_t *parents = malloc(ROWS * sizeof(*parents));
    uint32_t *row_map = calloc(ROWS, sizeof(*row_map));
    require(baseline_indices && parents && row_map,
            "token AdaLN benchmark map allocation failed");
    for (uint32_t row = 0; row < PREFIX_ROWS; row++) {
        baseline_indices[row] = UINT32_MAX;
        parents[row] = row;
    }
    uint32_t baseline_row = 0;
    for (uint32_t local = 0; local < ROWS - PREFIX_ROWS; local++) {
        uint32_t spatial_row = local / 27;
        uint32_t column = local % 27;
        parents[PREFIX_ROWS + local] = PREFIX_ROWS + spatial_row * 14 +
                                      column / 2;
    }
    for (uint32_t local = 0; local < REDUCED_ROWS - PREFIX_ROWS; local++) {
        uint32_t column = local % 14;
        baseline_indices[PREFIX_ROWS + local] = column < 13 ?
            baseline_row++ : UINT32_MAX;
    }
    require(baseline_row == BASELINE_ROWS,
            "token AdaLN benchmark map is inconsistent");
    h3_gpu_tensor *original = fresh(test, (size_t)ROWS * WIDTH);
    h3_gpu_tensor *reduced = fresh(test,
        (size_t)(REDUCED_ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *residual = fresh(test, (size_t)ROWS * WIDTH);
    h3_gpu_tensor *output = fresh(test, (size_t)ROWS * WIDTH);
    h3_gpu_tensor *norm = fresh(test, WIDTH);
    h3_gpu_tensor *modulation = fresh(test, (size_t)SLOTS * WIDTH);
    h3_gpu_tensor *gpu_baseline_indices = own(test,
        h3_gpu_tensor_from_u32(test->gpu, baseline_indices, REDUCED_ROWS));
    h3_gpu_tensor *gpu_parents = own(test,
        h3_gpu_tensor_from_u32(test->gpu, parents, ROWS));
    h3_gpu_tensor *gpu_row_map = own(test,
        h3_gpu_tensor_from_u32(test->gpu, row_map, ROWS));
    free(baseline_indices);
    free(parents);
    free(row_map);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin token AdaLN microbenchmark warmup");
    for (int warm = 0; warm < 16; warm++) {
        require_gpu(test, h3_gpu_token_expand_delta_bf16(
            test->gpu, residual, original, 0, reduced, reduced,
            (size_t)REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_parents,
            ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH, PREFIX_ROWS, 1.0f),
            "encode token expansion warmup");
        require_gpu(test, h3_gpu_adaln_bf16(
            test->gpu, output, residual, norm, modulation, gpu_row_map,
            ROWS, WIDTH, SLOTS, 0, 1, 1e-5f),
            "encode token AdaLN warmup");
        require_gpu(test, h3_gpu_token_expand_adaln_bf16(
            test->gpu, residual, output, original, 0, reduced, reduced,
            (size_t)REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_parents,
            norm, modulation, gpu_row_map, ROWS, REDUCED_ROWS, BASELINE_ROWS,
            WIDTH, PREFIX_ROWS, 1.0f, SLOTS, 0, 1, 1e-5f),
            "encode fused token AdaLN warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit token AdaLN microbenchmark warmup");

    static const int fused_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double old_seconds = 0.0, fused_seconds = 0.0;
    int old_count = 0, fused_count = 0;
    for (size_t sample = 0;
         sample < sizeof(fused_pattern) / sizeof(*fused_pattern); sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu),
                    "begin token AdaLN microbenchmark");
        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            if (fused_pattern[sample]) {
                require_gpu(test, h3_gpu_token_expand_adaln_bf16(
                    test->gpu, residual, output, original, 0, reduced,
                    reduced, (size_t)REDUCED_ROWS * WIDTH,
                    gpu_baseline_indices, gpu_parents, norm, modulation,
                    gpu_row_map, ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH,
                    PREFIX_ROWS, 1.0f, SLOTS, 0, 1, 1e-5f),
                    "encode fused token AdaLN microbenchmark");
            } else {
                require_gpu(test, h3_gpu_token_expand_delta_bf16(
                    test->gpu, residual, original, 0, reduced, reduced,
                    (size_t)REDUCED_ROWS * WIDTH, gpu_baseline_indices,
                    gpu_parents, ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH,
                    PREFIX_ROWS, 1.0f),
                    "encode token expansion microbenchmark");
                require_gpu(test, h3_gpu_adaln_bf16(
                    test->gpu, output, residual, norm, modulation,
                    gpu_row_map, ROWS, WIDTH, SLOTS, 0, 1, 1e-5f),
                    "encode token AdaLN microbenchmark");
            }
        }
        require_gpu(test, h3_gpu_submit(test->gpu),
                    "submit token AdaLN microbenchmark");
        double elapsed = monotonic_seconds() - start;
        if (fused_pattern[sample]) {
            fused_seconds += elapsed;
            fused_count++;
            printf("  fused token AdaLN %.6fs\n",
                   elapsed / (double)ITERATIONS);
        } else {
            old_seconds += elapsed;
            old_count++;
            printf("  separate token AdaLN %.6fs\n",
                   elapsed / (double)ITERATIONS);
        }
    }
    double old_average = old_seconds / old_count / (double)ITERATIONS;
    double fused_average = fused_seconds / fused_count / (double)ITERATIONS;
    printf("token AdaLN microbenchmark separate %.6fs, fused %.6fs, "
           "ratio %.4f\n", old_average, fused_average,
           fused_average / old_average);
}

static void bench_h3_sdpa(test_context *test) {
    enum { MAX_SEQUENCE = 1550, H3_HEADS = 56, DIMENSION = 128 };
    size_t elements = (size_t)MAX_SEQUENCE * H3_HEADS * DIMENSION;
    h3_gpu_tensor *query = fresh(test, elements);
    h3_gpu_tensor *key = fresh(test, elements);
    h3_gpu_tensor *value = fresh(test, elements);
    h3_gpu_tensor *output = fresh(test, elements);
    const float scale = 1.0f / sqrtf((float)DIMENSION);
    const uint32_t shapes[] = {976, 1550};
    require_gpu(test, h3_gpu_begin(test->gpu), "begin H3 SDPA warmup");
    for (int warm = 0; warm < 8; warm++) {
        for (size_t index = 0; index < sizeof(shapes) / sizeof(*shapes);
             index++)
            require_gpu(test, h3_gpu_sdpa_bf16(
                test->gpu, output, query, key, value, shapes[index], H3_HEADS,
                DIMENSION, scale), "encode H3 SDPA warmup");
    }
    require_gpu(test, h3_gpu_submit(test->gpu), "submit H3 SDPA warmup");

    static const uint32_t pattern[] = {
        976, 1550, 1550, 976, 1550, 976,
        976, 1550, 976, 1550, 1550, 976
    };
    double small_seconds = 0.0, large_seconds = 0.0;
    double small_encode = 0.0, large_encode = 0.0;
    int small_count = 0, large_count = 0;
    for (size_t sample = 0; sample < sizeof(pattern) / sizeof(*pattern);
         sample++) {
        double start = monotonic_seconds();
        require_gpu(test, h3_gpu_begin(test->gpu), "begin H3 SDPA benchmark");
        double encode_start = monotonic_seconds();
        require_gpu(test, h3_gpu_sdpa_bf16(
            test->gpu, output, query, key, value, pattern[sample], H3_HEADS,
            DIMENSION, scale), "encode H3 SDPA benchmark");
        double encode = monotonic_seconds() - encode_start;
        require_gpu(test, h3_gpu_submit(test->gpu), "submit H3 SDPA benchmark");
        double elapsed = monotonic_seconds() - start;
        if (pattern[sample] == shapes[0]) {
            small_seconds += elapsed;
            small_encode += encode;
            small_count++;
        } else {
            large_seconds += elapsed;
            large_encode += encode;
            large_count++;
        }
        printf("  H3 SDPA %u tokens %.6fs encode %.6fs\n",
               pattern[sample], elapsed, encode);
    }
    printf("H3 SDPA MPSGraph 976 %.6fs encode %.6fs, 1550 %.6fs "
           "encode %.6fs\n", small_seconds / small_count,
           small_encode / small_count, large_seconds / large_count,
           large_encode / large_count);
}

static void test_euler_update(test_context *test) {
    float sample_values[] = {-7.0f, -8.0f, 10.0f, 20.0f,
                             30.0f, 40.0f, -9.0f};
    const float last_values[] = {1.0f, -2.0f, 0.5f, 4.0f};
    const float previous_values[] = {0.5f, -1.0f, 0.25f, 5.0f};
    uint16_t last_bf16[4], previous_bf16[4];
    for (size_t index = 0; index < 4; index++) {
        last_bf16[index] = f32_to_bf16(last_values[index]);
        previous_bf16[index] = f32_to_bf16(previous_values[index]);
    }
    h3_gpu_tensor *sample = own(test, h3_gpu_tensor_from_f32(
        test->gpu, sample_values, sizeof(sample_values) / sizeof(*sample_values)));
    h3_gpu_tensor *last = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, last_bf16, sizeof(last_bf16) / sizeof(*last_bf16)));
    h3_gpu_tensor *previous = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, previous_bf16,
        sizeof(previous_bf16) / sizeof(*previous_bf16)));
    require_gpu(test, h3_gpu_begin(test->gpu), "begin Euler update");
    require_gpu(test, h3_gpu_euler_bf16(test->gpu, sample, 2, last, previous,
                                        4, 0.25f, 0.5f),
                "encode Euler update");
    require_gpu(test, h3_gpu_submit(test->gpu), "submit Euler update");
    float got[7];
    require(h3_gpu_tensor_read_f32(sample, got, 7),
            "cannot read Euler-updated sample");
    const float expected[] = {-7.0f, -8.0f, 10.3125f, 19.375f,
                              30.15625f, 40.875f, -9.0f};
    require(memcmp(got, expected, sizeof(expected)) == 0,
            "fused Euler update differs from exact reference");
    float range[4];
    require(h3_gpu_tensor_read_f32_range(sample, 2, range, 4),
            "cannot read Euler sample range");
    require(memcmp(range, expected + 2, sizeof(range)) == 0,
            "F32 range read returned the wrong offset");
}

static double compare(test_context *test, const char *name,
                      h3_gpu_tensor *got_tensor, size_t elements,
                      double *maximum_absolute) {
    uint16_t *got = malloc(elements * sizeof(*got));
    uint16_t *want = load(test, name, H3_DTYPE_BF16, elements);
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read completed BF16 tensor");
    double error = 0.0;
    double scale = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double got_value = bf16_to_f32(got[index]);
        double want_value = bf16_to_f32(want[index]);
        double delta = fabs(got_value - want_value);
        double magnitude = fabs(want_value);
        if (delta > error) error = delta;
        if (magnitude > scale) scale = magnitude;
    }
    free(got);
    free(want);
    *maximum_absolute = error;
    return error / (scale > 1e-12 ? scale : 1e-12);
}

static void require_same_bf16(h3_gpu_tensor *left, h3_gpu_tensor *right,
                              size_t elements, const char *message) {
    uint16_t *a = malloc(elements * sizeof(*a));
    uint16_t *b = malloc(elements * sizeof(*b));
    require(a != NULL && b != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(left, a, elements),
            "cannot read first BF16 tensor");
    require(h3_gpu_tensor_read_bf16(right, b, elements),
            "cannot read second BF16 tensor");
    require(memcmp(a, b, elements * sizeof(*a)) == 0, message);
    free(a);
    free(b);
}

static void test_wide_token_adaln(test_context *test) {
    enum {
        ROWS = 3, REDUCED_ROWS = 2, BASELINE_ROWS = 1,
        WIDTH = 5376, SLOTS = 2
    };
    size_t full_elements = (size_t)ROWS * WIDTH;
    size_t reduced_elements = (size_t)(REDUCED_ROWS + BASELINE_ROWS) * WIDTH;
    uint16_t *original_values = malloc(full_elements * sizeof(*original_values));
    uint16_t *reduced_values = malloc(
        reduced_elements * sizeof(*reduced_values));
    uint16_t *norm_values = malloc(WIDTH * sizeof(*norm_values));
    uint16_t *modulation_values = malloc(
        (size_t)SLOTS * WIDTH * sizeof(*modulation_values));
    require(original_values && reduced_values && norm_values &&
            modulation_values, "wide token AdaLN host allocation failed");
    for (size_t index = 0; index < full_elements; index++)
        original_values[index] = f32_to_bf16(
            (float)((int)(index % 97) - 48) / 16.0f);
    for (size_t index = 0; index < reduced_elements; index++)
        reduced_values[index] = f32_to_bf16(
            (float)((int)(index % 89) - 44) / 32.0f);
    for (size_t column = 0; column < WIDTH; column++) {
        norm_values[column] = f32_to_bf16(
            0.5f + (float)(column % 11) / 16.0f);
        modulation_values[column] = f32_to_bf16(
            (float)((int)(column % 7) - 3) / 32.0f);
        modulation_values[WIDTH + column] = f32_to_bf16(
            (float)((int)(column % 5) - 2) / 64.0f);
    }
    const uint32_t baseline_indices[REDUCED_ROWS] = {UINT32_MAX, 0};
    const uint32_t parents[ROWS] = {0, 1, 1};
    const uint32_t row_map[ROWS] = {0, 0, 0};
    h3_gpu_tensor *original = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, original_values, full_elements));
    h3_gpu_tensor *reduced = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, reduced_values, reduced_elements));
    h3_gpu_tensor *norm = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, norm_values, WIDTH));
    h3_gpu_tensor *modulation = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, modulation_values, (size_t)SLOTS * WIDTH));
    h3_gpu_tensor *gpu_baseline_indices = own(test,
        h3_gpu_tensor_from_u32(test->gpu, baseline_indices, REDUCED_ROWS));
    h3_gpu_tensor *gpu_parents = own(test,
        h3_gpu_tensor_from_u32(test->gpu, parents, ROWS));
    h3_gpu_tensor *gpu_row_map = own(test,
        h3_gpu_tensor_from_u32(test->gpu, row_map, ROWS));
    h3_gpu_tensor *reference_residual = fresh(test, full_elements);
    h3_gpu_tensor *reference_output = fresh(test, full_elements);
    h3_gpu_tensor *fused_residual = fresh(test, full_elements);
    h3_gpu_tensor *fused_output = fresh(test, full_elements);
    free(original_values);
    free(reduced_values);
    free(norm_values);
    free(modulation_values);

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin wide token AdaLN parity");
    require_gpu(test, h3_gpu_token_expand_delta_bf16(
        test->gpu, reference_residual, original, 0, reduced, reduced,
        (size_t)REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_parents,
        ROWS, REDUCED_ROWS, BASELINE_ROWS, WIDTH, 1, 1.0f),
        "encode wide token expansion reference");
    require_gpu(test, h3_gpu_adaln_bf16(
        test->gpu, reference_output, reference_residual, norm, modulation,
        gpu_row_map, ROWS, WIDTH, SLOTS, 0, 1, 1e-5f),
        "encode wide token AdaLN reference");
    require_gpu(test, h3_gpu_token_expand_adaln_bf16(
        test->gpu, fused_residual, fused_output, original, 0, reduced,
        reduced, (size_t)REDUCED_ROWS * WIDTH, gpu_baseline_indices,
        gpu_parents, norm, modulation, gpu_row_map, ROWS, REDUCED_ROWS,
        BASELINE_ROWS, WIDTH, 1, 1.0f, SLOTS, 0, 1, 1e-5f),
        "encode wide fused token AdaLN");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit wide token AdaLN parity");
    require_same_bf16(reference_residual, fused_residual, full_elements,
                      "wide fused token residual differs");
    require_same_bf16(reference_output, fused_output, full_elements,
                      "wide fused token AdaLN differs");
}

static void cleanup(test_context *test) {
    for (size_t index = 0; index < test->owned_count; index++) {
        h3_gpu_tensor_free(test->owned[index]);
    }
    h3_gpu_free(test->gpu);
    h3_st_free_header(&test->fixture);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "misc/fixtures/h3_dit_bf16.safetensors";
    test_context test;
    memset(&test, 0, sizeof(test));
    char error[512];
    if (!h3_st_read_header(path, &test.fixture, error, sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        return 1;
    }
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) {
        fprintf(stderr, "FAIL tests/test_bf16.c: %s\n", error);
        h3_st_free_header(&test.fixture);
        return 1;
    }
    if (getenv("H3_BENCH_TOKEN_ADALN")) {
        bench_gate_adaln_shape(&test, 1550, "1550 rows");
        bench_gate_adaln_shape(&test, 2915, "2915 rows");
        bench_token_pool_adaln(&test);
        bench_token_expand_adaln(&test);
        cleanup(&test);
        return 0;
    }
    const char *final_rows = getenv("H3_BENCH_FINAL_SLICE_ADALN");
    if (final_rows) {
        char *end = NULL;
        unsigned long parsed = strtoul(final_rows, &end, 10);
        if (end == final_rows || *end || !parsed || parsed > UINT32_MAX)
            die("H3_BENCH_FINAL_SLICE_ADALN must be a row count");
        bench_final_slice_adaln(&test, parsed == 1 ? 1792 : (uint32_t)parsed);
        cleanup(&test);
        return 0;
    }
    const char *final_head_rows = getenv("H3_BENCH_FINAL_HEAD");
    if (final_head_rows) {
        char *end = NULL;
        unsigned long parsed = strtoul(final_head_rows, &end, 10);
        if (end == final_head_rows || *end || !parsed || parsed > UINT32_MAX)
            die("H3_BENCH_FINAL_HEAD must be a row count");
        bench_final_head(&test, parsed == 1 ? 1792 : (uint32_t)parsed);
        cleanup(&test);
        return 0;
    }
    const char *patch_rows = getenv("H3_BENCH_PATCH_PROJECTION");
    if (patch_rows) {
        char *end = NULL;
        unsigned long parsed = strtoul(patch_rows, &end, 10);
        if (end == patch_rows || *end || !parsed || parsed > UINT32_MAX)
            die("H3_BENCH_PATCH_PROJECTION must be a row count");
        bench_patch_projection(&test,
            parsed == 1 ? 2835 : (uint32_t)parsed);
        cleanup(&test);
        return 0;
    }
    if (getenv("H3_BENCH_SDPA")) {
        bench_h3_sdpa(&test);
        cleanup(&test);
        return 0;
    }
    test_euler_update(&test);
    test_token_reduction_kernels(&test);
    test_wide_token_adaln(&test);

    {
        enum {
            PATCH_ROWS = 16, PATCH_IN = 32, PATCH_OUT = 5376,
            PACKED_ROWS = 24, FIRST_ROWS = 7, SECOND_DESTINATION = 12
        };
        size_t input_count = PATCH_ROWS * PATCH_IN;
        size_t weight_count = PATCH_OUT * PATCH_IN;
        size_t output_count = PATCH_ROWS * PATCH_OUT;
        float *input = malloc(input_count * sizeof(*input));
        float *weight = malloc(weight_count * sizeof(*weight));
        float *bias = malloc(PATCH_OUT * sizeof(*bias));
        float *scalar = malloc(output_count * sizeof(*scalar));
        float *tiled = malloc(output_count * sizeof(*tiled));
        uint16_t *legacy_bf16 = malloc(output_count * sizeof(*legacy_bf16));
        uint16_t *fused_bf16 = malloc(output_count * sizeof(*fused_bf16));
        uint16_t *packed_bf16 = malloc(
            (size_t)PACKED_ROWS * PATCH_OUT * sizeof(*packed_bf16));
        uint16_t *mapped_bf16 = malloc(
            (size_t)PACKED_ROWS * PATCH_OUT * sizeof(*mapped_bf16));
        require(input && weight && bias && scalar && tiled && legacy_bf16 &&
                fused_bf16 && packed_bf16 && mapped_bf16,
                "patch tile host allocation failed");
        for (size_t index = 0; index < input_count; index++)
            input[index] = (float)((int)(index % 17) - 8) * 0.03125f;
        for (size_t index = 0; index < weight_count; index++)
            weight[index] = (float)((int)(index % 23) - 11) * 0.0078125f;
        for (size_t index = 0; index < PATCH_OUT; index++)
            bias[index] = (float)((int)(index % 7) - 3) * 0.015625f;
        h3_gpu_tensor *gpu_input = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, input, input_count));
        h3_gpu_tensor *gpu_weight = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, weight, weight_count));
        h3_gpu_tensor *gpu_bias = own(
            &test, h3_gpu_tensor_from_f32(test.gpu, bias, PATCH_OUT));
        h3_gpu_tensor *scalar_output = own(
            &test, h3_gpu_tensor_new_f32(test.gpu, output_count));
        h3_gpu_tensor *tiled_output = own(
            &test, h3_gpu_tensor_new_f32(test.gpu, output_count));
        h3_gpu_tensor *legacy_bf16_output = own(
            &test, h3_gpu_tensor_new_bf16(test.gpu, output_count));
        h3_gpu_tensor *fused_bf16_output = own(
            &test, h3_gpu_tensor_new_bf16(test.gpu, output_count));
        h3_gpu_tensor *packed_bf16_output = own(
            &test, h3_gpu_tensor_new_bf16(
                test.gpu, (size_t)PACKED_ROWS * PATCH_OUT));
        h3_gpu_tensor *mapped_bf16_output = own(
            &test, h3_gpu_tensor_new_bf16(
                test.gpu, (size_t)PACKED_ROWS * PATCH_OUT));
        uint32_t packed_rows[PATCH_ROWS];
        for (uint32_t row = 0; row < PATCH_ROWS; row++)
            packed_rows[row] = row < FIRST_ROWS ? 3 + row :
                SECOND_DESTINATION + row - FIRST_ROWS;
        h3_gpu_tensor *gpu_packed_rows = own(
            &test, h3_gpu_tensor_from_u32(test.gpu, packed_rows, PATCH_ROWS));
        setenv("H3_SCALAR_PATCH", "1", 1);
        require_gpu(&test, h3_gpu_begin(test.gpu),
                    "begin scalar patch stream");
        require_gpu(&test, h3_gpu_linear_f32(
            test.gpu, scalar_output, gpu_input, gpu_weight, gpu_bias,
            PATCH_ROWS, PATCH_IN, PATCH_OUT), "scalar patch linear");
        require_gpu(&test, h3_gpu_submit(test.gpu),
                    "submit scalar patch stream");
        unsetenv("H3_SCALAR_PATCH");
        require_gpu(&test, h3_gpu_begin(test.gpu),
                    "begin tiled patch stream");
        require_gpu(&test, h3_gpu_linear_f32(
            test.gpu, tiled_output, gpu_input, gpu_weight, gpu_bias,
            PATCH_ROWS, PATCH_IN, PATCH_OUT), "tiled patch linear");
        require_gpu(&test, h3_gpu_submit(test.gpu),
                    "submit tiled patch stream");
        require(h3_gpu_tensor_read_f32(scalar_output, scalar, output_count),
                "cannot read scalar patch output");
        require(h3_gpu_tensor_read_f32(tiled_output, tiled, output_count),
                "cannot read tiled patch output");
        require(memcmp(scalar, tiled, output_count * sizeof(*scalar)) == 0,
                "tiled patch output differs from scalar F32");
        require_gpu(&test, h3_gpu_begin(test.gpu),
                    "begin fused patch/cast stream");
        require_gpu(&test, h3_gpu_cast_f32_to_bf16(
            test.gpu, legacy_bf16_output, tiled_output,
            (uint32_t)output_count),
                    "standalone patch BF16 cast");
        require_gpu(&test, h3_gpu_patch_linear_bf16(
            test.gpu, fused_bf16_output, gpu_input, gpu_weight, gpu_bias,
            PATCH_ROWS, PATCH_IN, PATCH_OUT), "fused patch BF16 linear");
        require_gpu(&test, h3_gpu_patch_linear_bf16_offset(
            test.gpu, packed_bf16_output, (size_t)3 * PATCH_OUT,
            gpu_input, 0, gpu_weight, gpu_bias, FIRST_ROWS, PATCH_IN,
            PATCH_OUT), "first packed patch segment");
        require_gpu(&test, h3_gpu_patch_linear_bf16_offset(
            test.gpu, packed_bf16_output,
            (size_t)SECOND_DESTINATION * PATCH_OUT, gpu_input,
            (size_t)FIRST_ROWS * PATCH_IN, gpu_weight, gpu_bias,
            PATCH_ROWS - FIRST_ROWS, PATCH_IN, PATCH_OUT),
                    "second packed patch segment");
        require_gpu(&test, h3_gpu_patch_linear_bf16_map(
            test.gpu, mapped_bf16_output, gpu_input, gpu_weight, gpu_bias,
            gpu_packed_rows, PACKED_ROWS, PATCH_ROWS, PATCH_IN, PATCH_OUT),
                    "mapped patch segments");
        require_gpu(&test, h3_gpu_submit(test.gpu),
                    "submit fused patch/cast stream");
        require(h3_gpu_tensor_read_bf16(
                    legacy_bf16_output, legacy_bf16, output_count),
                "cannot read standalone patch BF16 output");
        require(h3_gpu_tensor_read_bf16(
                    fused_bf16_output, fused_bf16, output_count),
                "cannot read fused patch BF16 output");
        require(h3_gpu_tensor_read_bf16(
                    packed_bf16_output, packed_bf16,
                    (size_t)PACKED_ROWS * PATCH_OUT),
                "cannot read packed patch BF16 output");
        require(h3_gpu_tensor_read_bf16(
                    mapped_bf16_output, mapped_bf16,
                    (size_t)PACKED_ROWS * PATCH_OUT),
                "cannot read mapped patch BF16 output");
        require(memcmp(legacy_bf16, fused_bf16,
                       output_count * sizeof(*legacy_bf16)) == 0,
                "fused patch projection differs from linear plus cast");
        require(memcmp(legacy_bf16,
                       packed_bf16 + (size_t)3 * PATCH_OUT,
                       (size_t)FIRST_ROWS * PATCH_OUT * sizeof(*legacy_bf16)) == 0,
                "first offset patch segment differs");
        require(memcmp(legacy_bf16 + (size_t)FIRST_ROWS * PATCH_OUT,
                       packed_bf16 + (size_t)SECOND_DESTINATION * PATCH_OUT,
                       (size_t)(PATCH_ROWS - FIRST_ROWS) * PATCH_OUT *
                           sizeof(*legacy_bf16)) == 0,
                "second offset patch segment differs");
        require(memcmp(packed_bf16 + (size_t)3 * PATCH_OUT,
                       mapped_bf16 + (size_t)3 * PATCH_OUT,
                       (size_t)FIRST_ROWS * PATCH_OUT * sizeof(*packed_bf16)) == 0,
                "first mapped patch segment differs");
        require(memcmp(packed_bf16 +
                           (size_t)SECOND_DESTINATION * PATCH_OUT,
                       mapped_bf16 +
                           (size_t)SECOND_DESTINATION * PATCH_OUT,
                       (size_t)(PATCH_ROWS - FIRST_ROWS) * PATCH_OUT *
                           sizeof(*packed_bf16)) == 0,
                "second mapped patch segment differs");
        free(input); free(weight); free(bias); free(scalar); free(tiled);
        free(legacy_bf16); free(fused_bf16); free(packed_bf16);
        free(mapped_bf16);
    }
    h3_gpu_stats setup_stats;
    require(h3_gpu_get_stats(test.gpu, &setup_stats),
            "cannot read patch-tile counters");

    h3_gpu_tensor *h_in = upload(&test, "x.h_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *attn_in = upload(&test, "x.attn_in", SEQUENCE * HIDDEN);
    h3_gpu_tensor *t_emb = upload(&test, "x.t_emb", T_ROWS * T_DIM);
    h3_gpu_tensor *rope_cos = upload(&test, "x.rope_cos", SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = upload(&test, "x.rope_sin", SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *norm1 = upload(&test, "norm1.weight", HIDDEN);
    h3_gpu_tensor *norm2 = upload(&test, "norm2.weight", HIDDEN);
    h3_gpu_tensor *adaln_w = upload(&test, "adaln_proj.linear.weight",
                                    MODALITIES * MODULATION_SLOTS * HIDDEN * T_DIM);
    h3_gpu_tensor *adaln_b = upload(&test, "adaln_proj.linear.bias",
                                    MODALITIES * MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *qkv_w = upload(&test, "attn.qkv_proj.weight",
                                  INNER * 3 * HIDDEN);
    h3_gpu_tensor *q_norm = upload(&test, "attn.q_norm.weight", HEAD_DIM);
    h3_gpu_tensor *k_norm = upload(&test, "attn.k_norm.weight", HEAD_DIM);
    h3_gpu_tensor *out_w = upload(&test, "attn.out_proj.weight", HIDDEN * INNER);
    h3_gpu_tensor *fc1_w = upload(&test, "mlp.fc1.weight", FFN * 2 * HIDDEN);
    h3_gpu_tensor *fc2_w = upload(&test, "mlp.fc2.weight", HIDDEN * FFN);

    int32_t *runs = load(&test, "x.runs", H3_DTYPE_I32, 12);
    uint32_t row_map[SEQUENCE];
    for (size_t index = 0; index < SEQUENCE; index++) row_map[index] = UINT32_MAX;
    for (size_t run = 0; run < 4; run++) {
        int32_t start = runs[run * 3];
        int32_t stop = runs[run * 3 + 1];
        int32_t row = runs[run * 3 + 2];
        require(start >= 0 && stop >= start && stop <= SEQUENCE,
                "invalid run bounds");
        require(row >= 0 && row < T_ROWS * MODALITIES,
                "invalid modulation row");
        for (int32_t index = start; index < stop; index++) {
            require(row_map[index] == UINT32_MAX, "overlapping runs");
            row_map[index] = (uint32_t)row;
        }
    }
    free(runs);
    for (size_t index = 0; index < SEQUENCE; index++) {
        require(row_map[index] != UINT32_MAX, "runs do not cover sequence");
    }
    h3_gpu_tensor *gpu_rows = own(&test, h3_gpu_tensor_from_u32(test.gpu,
                                                                row_map,
                                                                SEQUENCE));

    h3_gpu_tensor *plain_qkv = fresh(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *plain_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_v = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *oracle_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *oracle_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *oracle_v = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_sdpa = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *plain_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *plain_fc1 = fresh(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *plain_swiglu = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *plain_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *fused_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *t_silu = fresh(&test, T_ROWS * T_DIM);
    h3_gpu_tensor *modulation = fresh(&test, T_ROWS * MODALITIES *
                                             MODULATION_SLOTS * HIDDEN);
    h3_gpu_tensor *mod_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_qkv = fresh(&test, SEQUENCE * INNER * 3);
    h3_gpu_tensor *full_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_v = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_sdpa = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *full_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *after_attn = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *mod_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *full_fc1 = fresh(&test, SEQUENCE * FFN * 2);
    h3_gpu_tensor *full_swiglu = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *full_mlp = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *h_out = fresh(&test, SEQUENCE * HIDDEN);

    require_gpu(&test, h3_gpu_begin(test.gpu), "begin command stream");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_qkv, attn_in, qkv_w,
                                           NULL, SEQUENCE, HIDDEN, INNER * 3),
                "plain QKV");
    require_gpu(&test, h3_gpu_qkv_rope_bf16(test.gpu, plain_q, plain_k, plain_v,
                                             plain_qkv, q_norm, k_norm, rope_cos,
                                             rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                             ROPE_HALF, 1e-5f), "plain RoPE");
    require_gpu(&test, h3_gpu_sdpa_bf16(test.gpu, plain_sdpa, plain_q, plain_k,
                                        plain_v, SEQUENCE, HEADS, HEAD_DIM,
                                        1.0f / sqrtf((float)HEAD_DIM)),
                "plain attention");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_attn, plain_sdpa,
                                           out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "plain attention output");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_fc1, attn_in, fc1_w,
                                           NULL, SEQUENCE, HIDDEN, FFN * 2),
                "plain MLP input");
    require_gpu(&test, h3_gpu_swiglu_bf16(test.gpu, plain_swiglu, plain_fc1,
                                           SEQUENCE, FFN), "plain SwiGLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, plain_mlp, plain_swiglu,
                                           fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "plain MLP output");
    require_gpu(&test, h3_gpu_mlp_bf16(test.gpu, fused_mlp, attn_in, fc1_w,
                                        fc2_w, SEQUENCE, HIDDEN, FFN, HIDDEN),
                "fused MLP");

    require_gpu(&test, h3_gpu_silu_bf16(test.gpu, t_silu, t_emb, T_ROWS * T_DIM),
                "AdaLN SiLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, modulation, t_silu, adaln_w,
                                           adaln_b, T_ROWS, T_DIM,
                                           MODALITIES * MODULATION_SLOTS * HIDDEN),
                "AdaLN projection");
    require_gpu(&test, h3_gpu_adaln_bf16(test.gpu, mod_attn, h_in, norm1,
                                          modulation, gpu_rows, SEQUENCE, HIDDEN,
                                          MODULATION_SLOTS, 0, 1, 1e-5f),
                "attention AdaLN");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_qkv, mod_attn, qkv_w,
                                           NULL, SEQUENCE, HIDDEN, INNER * 3),
                "full QKV");
    require_gpu(&test, h3_gpu_qkv_rope_bf16(test.gpu, full_q, full_k, full_v,
                                             full_qkv, q_norm, k_norm, rope_cos,
                                             rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                             ROPE_HALF, 1e-5f), "full RoPE");
    require_gpu(&test, h3_gpu_sdpa_bf16(test.gpu, full_sdpa, full_q, full_k,
                                        full_v, SEQUENCE, HEADS, HEAD_DIM,
                                        1.0f / sqrtf((float)HEAD_DIM)),
                "full attention");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_attn, full_sdpa,
                                           out_w, NULL, SEQUENCE, INNER, HIDDEN),
                "full attention output");
    require_gpu(&test, h3_gpu_continue(test.gpu),
                "continue split command stream");
    require_gpu(&test, h3_gpu_gate_bf16(test.gpu, after_attn, h_in, full_attn,
                                         modulation, gpu_rows, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 2), "attention gate");
    require_gpu(&test, h3_gpu_adaln_bf16(test.gpu, mod_mlp, after_attn, norm2,
                                          modulation, gpu_rows, SEQUENCE, HIDDEN,
                                          MODULATION_SLOTS, 3, 4, 1e-5f),
                "MLP AdaLN");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_fc1, mod_mlp, fc1_w,
                                           NULL, SEQUENCE, HIDDEN, FFN * 2),
                "full MLP input");
    require_gpu(&test, h3_gpu_swiglu_bf16(test.gpu, full_swiglu, full_fc1,
                                           SEQUENCE, FFN), "full SwiGLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, full_mlp, full_swiglu,
                                           fc2_w, NULL, SEQUENCE, FFN, HIDDEN),
                "full MLP output");
    require_gpu(&test, h3_gpu_gate_bf16(test.gpu, h_out, after_attn, full_mlp,
                                         modulation, gpu_rows, SEQUENCE, HIDDEN,
                                         MODULATION_SLOTS, 5), "MLP gate");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit command stream");

    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "cannot read Metal counters");

    /* The cooperative production mapping must retain the scalar kernel's
     * BF16 boundary exactly, not merely stay within the MLX error tolerance. */
    setenv("H3_DISABLE_COOP_QKV", "1", 1);
    require_gpu(&test, h3_gpu_begin(test.gpu), "begin scalar QKV oracle");
    require_gpu(&test, h3_gpu_qkv_rope_bf16(
        test.gpu, oracle_q, oracle_k, oracle_v, plain_qkv, q_norm, k_norm,
        rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "scalar QKV oracle");
    unsetenv("H3_DISABLE_COOP_QKV");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit scalar QKV oracle");
    require_same_bf16(plain_q, oracle_q, SEQUENCE * INNER,
                      "cooperative query differs from scalar oracle");
    require_same_bf16(plain_k, oracle_k, SEQUENCE * INNER,
                      "cooperative key differs from scalar oracle");
    require_same_bf16(plain_v, oracle_v, SEQUENCE * INNER,
                      "cooperative value differs from scalar oracle");

    /* The released H3 checkpoint emits QKV interleaved per head. Verify the
     * production deinterleaver against the conventional fixture layout. */
    size_t qkv_elements = SEQUENCE * INNER * 3;
    uint16_t *plain_qkv_host = malloc(qkv_elements * sizeof(*plain_qkv_host));
    uint16_t *grouped_qkv_host = malloc(qkv_elements * sizeof(*grouped_qkv_host));
    require(plain_qkv_host != NULL && grouped_qkv_host != NULL,
            "grouped QKV allocation failed");
    require(h3_gpu_tensor_read_bf16(plain_qkv, plain_qkv_host, qkv_elements),
            "cannot read conventional QKV");
    for (size_t row = 0; row < SEQUENCE; row++) {
        size_t row_base = row * INNER * 3;
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t kind = 0; kind < 3; kind++) {
                memcpy(grouped_qkv_host + row_base +
                           (head * 3 + kind) * HEAD_DIM,
                       plain_qkv_host + row_base + kind * INNER +
                           head * HEAD_DIM,
                       HEAD_DIM * sizeof(*plain_qkv_host));
            }
        }
    }
    h3_gpu_tensor *grouped_qkv = own(
        &test, h3_gpu_tensor_from_bf16(test.gpu, grouped_qkv_host, qkv_elements));
    h3_gpu_tensor *grouped_q = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *grouped_k = fresh(&test, SEQUENCE * INNER);
    h3_gpu_tensor *grouped_v = fresh(&test, SEQUENCE * INNER);
    free(plain_qkv_host);
    free(grouped_qkv_host);
    require_gpu(&test, h3_gpu_begin(test.gpu), "begin grouped QKV stream");
    require_gpu(&test, h3_gpu_grouped_qkv_rope_bf16(
        test.gpu, grouped_q, grouped_k, grouped_v, grouped_qkv, q_norm, k_norm,
        rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "grouped QKV RoPE");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit grouped QKV stream");
    require_same_bf16(plain_q, grouped_q, SEQUENCE * INNER,
                      "grouped query deinterleave mismatch");
    require_same_bf16(plain_k, grouped_k, SEQUENCE * INNER,
                      "grouped key deinterleave mismatch");
    require_same_bf16(plain_v, grouped_v, SEQUENCE * INNER,
                      "grouped value deinterleave mismatch");

    double attn_abs, mlp_abs, block_abs;
    double attn_rel = compare(&test, "x.attn_out", plain_attn,
                              SEQUENCE * HIDDEN, &attn_abs);
    double mlp_rel = compare(&test, "x.mlp_out", plain_mlp,
                             SEQUENCE * HIDDEN, &mlp_abs);
    double fused_mlp_abs;
    double fused_mlp_rel = compare(&test, "x.mlp_out", fused_mlp,
                                   SEQUENCE * HIDDEN, &fused_mlp_abs);
    double block_rel = compare(&test, "x.h_out", h_out,
                               SEQUENCE * HIDDEN, &block_abs);
    printf("BF16 Metal/MLX parity: attention rel %.3g abs %.3g; "
           "MLP rel %.3g abs %.3g; fused MLP rel %.3g abs %.3g; "
           "block rel %.3g abs %.3g\n", attn_rel, attn_abs, mlp_rel,
           mlp_abs, fused_mlp_rel, fused_mlp_abs, block_rel, block_abs);
    require(attn_rel < 1e-2, "BF16 attention exceeds MLX error bound");
    require(mlp_rel < 1e-2, "BF16 MLP exceeds MLX error bound");
    require(fused_mlp_rel < 1e-2,
            "fused BF16 MLP exceeds MLX error bound");
    require(block_rel < 1e-2, "BF16 block exceeds MLX error bound");
    printf("BF16 dispatches: %llu MPS linear, %llu MPS SDPA, %llu direct; "
           "%.2f MiB allocated\n",
           (unsigned long long)stats.mps_linear_dispatches,
           (unsigned long long)stats.mps_sdpa_dispatches,
           (unsigned long long)stats.direct_dispatches,
           (double)stats.allocated_bytes / (1024.0 * 1024.0));
    require(stats.mps_linear_dispatches == 6,
            "wide BF16 linears did not use the cached MPSGraph path");
    require(stats.mps_sdpa_dispatches == 2, "BF16 SDPA counter mismatch");
    require(stats.submissions == setup_stats.submissions + 2,
            "split block did not use two command submissions");
    cleanup(&test);
    puts("ok: portable BF16 Metal block matches the MLX BF16 oracle");
    return 0;
}
