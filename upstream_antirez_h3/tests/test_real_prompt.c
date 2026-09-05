#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_prompt.c: %s\n", message);
    exit(1);
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (result) snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 5 == 0) {
        fprintf(stderr, "Qwen command encoding: %d/%d layers\n", completed,
                total);
    }
}

static uint64_t hash_bf16(const uint16_t *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        hash ^= values[index] & 255u;
        hash *= UINT64_C(1099511628211);
        hash ^= values[index] >> 8;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void compare_golden(const char *path, const uint32_t *ids,
                           size_t token_count,
                           const h3_text_embedding *embedding) {
    h3_st_header header;
    char error[512];
    if (!h3_st_read_header(path, &header, error, sizeof(error))) fail(error);
    const h3_st_tensor *ids_tensor = h3_st_find(&header, "x.ids");
    const h3_st_tensor *output_tensor = h3_st_find(&header, "x.output");
    size_t elements = token_count * embedding->width;
    if (!ids_tensor || ids_tensor->dtype != H3_DTYPE_I32 ||
        h3_st_tensor_elements(ids_tensor) != (uint64_t)token_count ||
        !output_tensor || output_tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(output_tensor) != (uint64_t)elements) {
        h3_st_free_header(&header);
        fail("real-prompt MLX fixture has the wrong schema");
    }
    int32_t *expected_ids = malloc(token_count * sizeof(*expected_ids));
    uint16_t *expected = malloc(elements * sizeof(*expected));
    if (!expected_ids || !expected) fail("comparison allocation failed");
    if (!h3_st_read_data(&header, ids_tensor, expected_ids,
                         token_count * sizeof(*expected_ids), error,
                         sizeof(error)) ||
        !h3_st_read_data(&header, output_tensor, expected,
                         elements * sizeof(*expected), error, sizeof(error))) {
        fail(error);
    }
    for (size_t index = 0; index < token_count; index++) {
        if (expected_ids[index] < 0 || ids[index] != (uint32_t)expected_ids[index]) {
            fail("native and MLX prompt token IDs differ");
        }
    }
    double maximum_error = 0.0;
    double maximum_value = 0.0;
    double squared_error = 0.0;
    double squared_value = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double got = bf16_to_f32(embedding->values[index]);
        double want = bf16_to_f32(expected[index]);
        double delta = got - want;
        if (fabs(delta) > maximum_error) maximum_error = fabs(delta);
        if (fabs(want) > maximum_value) maximum_value = fabs(want);
        squared_error += delta * delta;
        squared_value += want * want;
    }
    double relative_max = maximum_error / (maximum_value > 1e-12 ?
                                             maximum_value : 1e-12);
    double relative_l2 = sqrt(squared_error / (squared_value > 1e-24 ?
                                                squared_value : 1e-24));
    printf("MLX layer-50 parity: relative-max %.6g, absolute-max %.6g, "
           "relative-L2 %.6g\n", relative_max, maximum_error, relative_l2);
    if (relative_l2 >= 0.05 || relative_max >= 0.1) {
        fail("released Qwen output exceeds the MLX error bound");
    }
    free(expected_ids);
    free(expected);
    h3_st_free_header(&header);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL_ROOT [MLX_GOLDEN] [PROMPT]\n", argv[0]);
        return 2;
    }
    const char *prompt = argc >= 4 ? argv[3] :
        "A red fox walking through snow";
    const char *golden = argc >= 3 ? argv[2] : NULL;
    char *tokenizer_path = path_join(argv[1], "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = path_join(argv[1], "FL2VA/text_encoder");
    if (!tokenizer_path || !weights_path) fail("path allocation failed");
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                 sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                             error, sizeof(error))) fail(error);
    printf("prompt: %s\ntokens: %zu", prompt, token_count);
    for (size_t index = 0; index < token_count; index++) {
        printf(" %u", ids[index]);
    }
    putchar('\n');

    h3_text_embedding embedding;
    double start = seconds();
    if (!h3_text_encode_bf16(weights_path, "h3_shaders.metal", ids,
                              token_count, progress, NULL, &embedding,
                              error, sizeof(error))) fail(error);
    double elapsed = seconds() - start;
    size_t elements = embedding.tokens * embedding.width;
    printf("native layer-50: %zux%zu BF16, hash %016llx, %.3f seconds\n",
           embedding.tokens, embedding.width,
           (unsigned long long)hash_bf16(embedding.values, elements), elapsed);
    printf("GPU accounting: %.3f GiB allocated, %llu MPS linears, "
           "%llu direct dispatches, %llu submission, %.3f GPU seconds\n",
           (double)embedding.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           (unsigned long long)embedding.gpu_stats.mps_linear_dispatches,
           (unsigned long long)embedding.gpu_stats.direct_dispatches,
           (unsigned long long)embedding.gpu_stats.submissions,
           embedding.gpu_stats.gpu_seconds);
    if (embedding.tokens != token_count ||
        embedding.width != H3_TEXT_HIDDEN_SIZE ||
        embedding.gpu_stats.submissions != 51) {
        fail("invalid released Qwen execution metadata");
    }
    if (golden) compare_golden(golden, ids, token_count, &embedding);

    h3_text_embedding_free(&embedding);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    puts("ok: released Qwen prompt encoder completed");
    return 0;
}
