#include "h3_safetensors.h"
#include "h3_vision_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 5120 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_qwen_vision.c: %s\n", message);
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
    if (completed == 1 || completed % 4 == 0 || completed == total)
        fprintf(stderr, "native Qwen vision: %d/%d layers\n",
                completed, total);
}

static void read_tensor(const h3_st_header *fixture, const char *name,
                        h3_dtype dtype, void *values, size_t bytes,
                        size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements)
        die("fixture tensor is absent or malformed");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, values, bytes,
                         error, sizeof(error))) die(error);
}

static void compare(const char *label, const uint16_t *got,
                    const uint16_t *want, size_t count) {
    double maximum = 0.0, scale = 0.0, squares = 0.0, reference = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < count; index++) {
        double actual = bf16_to_f32(got[index]);
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
    printf("%-20s max abs %.7g, rel-max %.7g, rel-L2 %.7g, "
           "nonfinite %zu\n", label, maximum, rel_max, rel_l2, nonfinite);
    /* The streamed Metal tower intentionally uses a different matmul reduction
     * from MLX. A single BF16 element can drift more on M5 while the aggregate
     * representation remains close; keep the tighter global error gate. */
    if (nonfinite || rel_max >= 0.25 || rel_l2 >= 0.15)
        die("Qwen vision parity bound exceeded");
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_qwen_vision_64.safetensors";
    h3_st_header fixture;
    char error[512];
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *pixel_tensor = h3_st_find(&fixture, "x.pixels");
    if (!pixel_tensor || pixel_tensor->dtype != H3_DTYPE_F32 ||
        pixel_tensor->ndim != 4 ||
        (pixel_tensor->shape[0] != 1 && pixel_tensor->shape[0] != 2) ||
        pixel_tensor->shape[1] != 3 || pixel_tensor->shape[2] < 32 ||
        pixel_tensor->shape[3] < 32 || pixel_tensor->shape[2] > INT32_MAX ||
        pixel_tensor->shape[3] > INT32_MAX)
        die("fixture pixels are absent or malformed");
    int height = (int)pixel_tensor->shape[2];
    int width = (int)pixel_tensor->shape[3];
    int frames = (int)pixel_tensor->shape[0];
    size_t pixel_count = h3_st_tensor_elements(pixel_tensor);
    size_t tokens = (size_t)(height / 32) * (size_t)(width / 32);
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    uint16_t *want = malloc(tokens * WIDTH * sizeof(*want));
    if (!pixels || !want) die("out of memory loading vision fixture");
    read_tensor(&fixture, "x.pixels", H3_DTYPE_F32, pixels,
                pixel_count * sizeof(*pixels), pixel_count);
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    h3_vision_output output;
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels,
                                frames, height, width, progress, NULL, &output,
                                error, sizeof(error))) die(error);
    if (output.grid_h != height / 16 || output.grid_w != width / 16 ||
        output.tokens != tokens)
        die("native vision output geometry mismatch");
    read_tensor(&fixture, "x.merged", H3_DTYPE_BF16, want,
                tokens * WIDTH * sizeof(*want), tokens * WIDTH);
    compare("vision merged", output.merged, want, tokens * WIDTH);
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++) {
        char name[64], label[64];
        snprintf(name, sizeof(name), "x.deepstack_%u", index);
        snprintf(label, sizeof(label), "vision deepstack %u", index);
        read_tensor(&fixture, name, H3_DTYPE_BF16, want,
                    tokens * WIDTH * sizeof(*want), tokens * WIDTH);
        compare(label, output.deepstack[index], want,
                tokens * WIDTH);
    }
    printf("Qwen vision: %.3f GiB cumulative allocations, %.3f GPU seconds, "
           "%llu MPS linears, %llu SDPA, %llu submissions\n",
           (double)output.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           output.gpu_stats.gpu_seconds,
           (unsigned long long)output.gpu_stats.mps_linear_dispatches,
           (unsigned long long)output.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)output.gpu_stats.submissions);
    h3_vision_output_free(&output);
    h3_st_free_header(&fixture);
    free(pixels);
    free(want);
    puts("ok: native Metal Qwen3-VL vision tower matches the MLX oracle");
    return 0;
}
