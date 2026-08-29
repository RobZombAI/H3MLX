#include "h3_multimodal.h"
#include "h3_safetensors.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_ref_video_text.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
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
        "misc/fixtures/h3_real_ref_video_text_64.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *ids_tensor = h3_st_find(&fixture, "x.ids");
    const h3_st_tensor *vision_tensor = h3_st_find(
        &fixture, "x.vision_merged");
    if (!ids_tensor || ids_tensor->dtype != H3_DTYPE_I32 ||
        ids_tensor->ndim != 2 || ids_tensor->shape[0] != 1 ||
        !vision_tensor || vision_tensor->dtype != H3_DTYPE_BF16 ||
        vision_tensor->ndim != 2 || vision_tensor->shape[1] != 5120)
        die("fixture geometry is malformed");
    size_t tokens = (size_t)ids_tensor->shape[1];
    size_t vision_tokens = (size_t)vision_tensor->shape[0];
    uint32_t *positions = malloc(3 * tokens * sizeof(*positions));
    int32_t *tags_i32 = malloc(tokens * sizeof(*tags_i32));
    uint16_t *vision = malloc(vision_tokens * 5120 * sizeof(*vision));
    uint16_t *deepstack[3] = {
        malloc(vision_tokens * 5120 * sizeof(uint16_t)),
        malloc(vision_tokens * 5120 * sizeof(uint16_t)),
        malloc(vision_tokens * 5120 * sizeof(uint16_t))
    };
    uint16_t *want = malloc(tokens * 5120 * sizeof(*want));
    if (!positions || !tags_i32 || !vision || !deepstack[0] ||
        !deepstack[1] || !deepstack[2] || !want)
        die("out of memory loading fixture");
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
    read_exact(&fixture, "x.output", H3_DTYPE_BF16,
               want, tokens * 5120);
    size_t start = tokens;
    for (size_t index = 0; index < tokens; index++)
        if (tags_i32[index] == 0) { start = index + 1; break; }
    if (start >= tokens) die("fixture has no vision span");
    uint32_t base = positions[start], max_h = base, max_w = base;
    for (size_t index = start; index < start + vision_tokens; index++) {
        if (positions[tokens + index] > max_h) max_h = positions[tokens + index];
        if (positions[2 * tokens + index] > max_w)
            max_w = positions[2 * tokens + index];
    }
    h3_vision_output visual = {
        (int)(2 * (max_h - base + 1)), (int)(2 * (max_w - base + 1)),
        vision_tokens, vision, {deepstack[0], deepstack[1], deepstack[2]}, {0}
    };
    double timestamp = 0.25;
    h3_reference_presentation reference = {
        H3_PRESENTATION_VIDEO, 1, &visual, 1, &timestamp};
    char tokenizer_path[1024], weights[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path),
             "%s/Ref2VA/tokenizer/tokenizer.json", model_root);
    snprintf(weights, sizeof(weights), "%s/Ref2VA/text_encoder", model_root);
    h3_tokenizer *tokenizer = h3_tokenizer_load(
        tokenizer_path, error, sizeof(error));
    if (!tokenizer) die(error);
    h3_text_embedding got;
    int ok = h3_multimodal_encode_ref2va_bf16(
        tokenizer, weights, "h3_shaders.metal",
        "A red fox walking through snow", &reference, 1,
        NULL, NULL, &got, error, sizeof(error));
    h3_tokenizer_free(tokenizer);
    if (!ok) die(error);
    if (got.tokens != tokens || got.width != 5120 || !got.tags)
        die("native Ref2VA presentation geometry mismatch");
    for (size_t index = 0; index < tokens; index++)
        if (got.tags[index] != (uint8_t)tags_i32[index])
            die("native Ref2VA presentation tags mismatch");
    double maximum = 0.0, scale = 0.0, squares = 0.0, norm = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < tokens * 5120; index++) {
        double actual = bf16_to_f32(got.values[index]);
        double expected = bf16_to_f32(want[index]);
        if (!isfinite(actual) || !isfinite(expected)) { nonfinite++; continue; }
        double delta = actual - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        squares += delta * delta;
        norm += expected * expected;
    }
    double rel_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double rel_l2 = sqrt(squares / (norm > 1e-24 ? norm : 1e-24));
    printf("Ref2VA video+audio presentation: rel-max %.7g, rel-L2 %.7g, "
           "nonfinite %zu\n", rel_max, rel_l2, nonfinite);
    if (nonfinite || rel_max >= 0.15 || rel_l2 >= 0.15)
        die("Ref2VA video presentation parity bound exceeded");
    h3_text_embedding_free(&got);
    h3_st_free_header(&fixture);
    free(positions); free(tags_i32); free(vision); free(want);
    for (unsigned index = 0; index < 3; index++) free(deepstack[index]);
    puts("ok: native Ref2VA video+audio presentation matches the MLX oracle");
    return 0;
}
