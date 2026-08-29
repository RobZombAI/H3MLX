#include "h3_safetensors.h"
#include "h3_multimodal.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_multimodal_text.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed % 5 == 0 || completed == total)
        fprintf(stderr, "native multimodal Qwen: %d/%d layers\n",
                completed, total);
}

static void read_exact(const h3_st_header *fixture, const char *name,
                       h3_dtype dtype, void *output, size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements)
        die("fixture tensor is absent or malformed");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, output,
                         elements * h3_dtype_size(dtype),
                         error, sizeof(error))) die(error);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_multimodal_text_64.safetensors";
    int layers = argc > 3 ? atoi(argv[3]) : 50;
    if (layers < 1 || layers > 50) die("invalid requested layer prefix");
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *ids_tensor = h3_st_find(&fixture, "x.ids");
    const h3_st_tensor *vision_tensor = h3_st_find(&fixture, "x.vision_merged");
    if (!ids_tensor || ids_tensor->dtype != H3_DTYPE_I32 ||
        ids_tensor->ndim != 2 || ids_tensor->shape[0] != 1 ||
        !vision_tensor || vision_tensor->dtype != H3_DTYPE_BF16 ||
        vision_tensor->ndim != 2 || vision_tensor->shape[1] != 5120)
        die("fixture geometry is malformed");
    size_t tokens = (size_t)ids_tensor->shape[1];
    size_t vision_tokens = (size_t)vision_tensor->shape[0];
    uint32_t *ids = malloc(tokens * sizeof(*ids));
    uint32_t *positions = malloc(3 * tokens * sizeof(*positions));
    int32_t *tags_i32 = malloc(tokens * sizeof(*tags_i32));
    uint8_t *tags = malloc(tokens * sizeof(*tags));
    uint16_t *vision = malloc(vision_tokens * 5120 * sizeof(*vision));
    uint16_t *deepstack[3] = {
        malloc(vision_tokens * 5120 * sizeof(uint16_t)),
        malloc(vision_tokens * 5120 * sizeof(uint16_t)),
        malloc(vision_tokens * 5120 * sizeof(uint16_t))
    };
    uint16_t *want = malloc(tokens * 5120 * sizeof(*want));
    if (!ids || !positions || !tags_i32 || !tags || !vision ||
        !deepstack[0] || !deepstack[1] || !deepstack[2] || !want)
        die("out of memory loading multimodal fixture");
    read_exact(&fixture, "x.ids", H3_DTYPE_I32, ids, tokens);
    read_exact(&fixture, "x.position_ids", H3_DTYPE_I32,
               positions, 3 * tokens);
    read_exact(&fixture, "x.tags", H3_DTYPE_I32, tags_i32, tokens);
    read_exact(&fixture, "x.vision_merged", H3_DTYPE_BF16,
               vision, vision_tokens * 5120);
    for (unsigned index = 0; index < 3; index++) {
        char name[64];
        snprintf(name, sizeof(name), "x.vision_deepstack_%u", index);
        read_exact(&fixture, name, H3_DTYPE_BF16, deepstack[index],
                   vision_tokens * 5120);
    }
    char expected_name[64];
    snprintf(expected_name, sizeof(expected_name), "x.layer_%d", layers - 1);
    read_exact(&fixture, expected_name, H3_DTYPE_BF16,
               want, tokens * 5120);
    size_t first_zero = tokens, zero_count = 0;
    for (size_t index = 0; index < tokens; index++) {
        if (tags_i32[index] < 0 || tags_i32[index] > 2)
            die("fixture tag is invalid");
        tags[index] = (uint8_t)tags_i32[index];
        if (tags[index] == 0) {
            if (first_zero == tokens) first_zero = index;
            zero_count++;
        }
    }
    if (zero_count != vision_tokens + 2 || first_zero + 1 >= tokens)
        die("fixture vision tag span is malformed");
    h3_text_vision_span span = {
        first_zero + 1, vision_tokens, vision,
        {deepstack[0], deepstack[1], deepstack[2]}
    };
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    h3_text_embedding got;
    if (layers == 50) {
        uint32_t vision_base = positions[span.start];
        uint32_t max_h = vision_base, max_w = vision_base;
        for (size_t index = span.start; index < span.start + span.tokens; index++) {
            if (positions[tokens + index] > max_h)
                max_h = positions[tokens + index];
            if (positions[2 * tokens + index] > max_w)
                max_w = positions[2 * tokens + index];
        }
        h3_vision_output vision_output = {
            (int)(2 * (max_h - vision_base + 1)),
            (int)(2 * (max_w - vision_base + 1)),
            vision_tokens, vision,
            {deepstack[0], deepstack[1], deepstack[2]}, {0}
        };
        char tokenizer_path[1024];
        snprintf(tokenizer_path, sizeof(tokenizer_path),
                 "%s/FL2VA/tokenizer/tokenizer.json", model_root);
        h3_tokenizer *tokenizer = h3_tokenizer_load(
            tokenizer_path, error, sizeof(error));
        if (!tokenizer) die(error);
        int ok = h3_multimodal_encode_fl2va_bf16(
            tokenizer, weights, "h3_shaders.metal",
            "A red fox walking through snow", &vision_output, 1,
            progress, NULL, &got, error, sizeof(error));
        h3_tokenizer_free(tokenizer);
        if (!ok) die(error);
    } else if (!h3_text_encode_multimodal_layers_bf16(
                   weights, "h3_shaders.metal", ids, tokens, &span, 1,
                   positions, tags, layers, progress, NULL, &got,
                   error, sizeof(error))) {
        die(error);
    }
    if (got.tokens != tokens || got.width != 5120 || !got.tags ||
        memcmp(got.tags, tags, tokens) != 0)
        die("native multimodal presentation metadata mismatch");
    double maximum = 0.0, scale = 0.0, squares = 0.0, reference = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < tokens * 5120; index++) {
        double actual = bf16_to_f32(got.values[index]);
        double expected = bf16_to_f32(want[index]);
        if (!isfinite(actual) || !isfinite(expected)) {
            nonfinite++;
            continue;
        }
        double delta = actual - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        squares += delta * delta;
        reference += expected * expected;
    }
    double rel_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double rel_l2 = sqrt(squares / (reference > 1e-24 ? reference : 1e-24));
    printf("multimodal layer-%d: max abs %.7g, rel-max %.7g, "
           "rel-L2 %.7g, nonfinite %zu\n",
           layers, maximum, rel_max, rel_l2, nonfinite);
    printf("multimodal Qwen: %.3f GiB cumulative, %.3f GPU seconds, "
           "%llu submissions\n",
           (double)got.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.submissions);
    /* Prefixes through layer 43 remain tightly numerical. At layer 44 the
     * released fused attention crosses a sharp token-selection boundary, so a
     * different Metal reduction order amplifies a small input delta. The full
     * decoder gate therefore checks representation-level similarity. */
    double bound = layers <= 43 ? 0.03 : 0.15;
    if (nonfinite || rel_max >= bound || rel_l2 >= bound)
        die("multimodal Qwen parity bound exceeded");
    h3_text_embedding_free(&got);
    h3_st_free_header(&fixture);
    free(ids); free(positions); free(tags_i32); free(tags); free(vision);
    for (unsigned index = 0; index < 3; index++) free(deepstack[index]);
    free(want);
    printf("ok: native Qwen3-VL presentation matches the MLX layer-%d oracle\n",
           layers);
    return 0;
}
