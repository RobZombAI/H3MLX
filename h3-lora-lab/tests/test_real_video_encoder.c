#include "h3_safetensors.h"
#include "h3_host.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_video_encoder.c: %s\n", message);
    exit(1);
}

static int axis_tiles(int extent) {
    if (extent <= 256) return 1;
    int count = (extent + 255) / 256;
    while (256 * count - 64 * (count - 1) < extent) count++;
    return count;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "native visual encoder tile: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_video_encoder_256.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *pixel_tensor = h3_st_find(&fixture, "x.pixels");
    const h3_st_tensor *latent_tensor = h3_st_find(&fixture, "x.latent");
    if (!pixel_tensor || pixel_tensor->dtype != H3_DTYPE_F32 ||
        (pixel_tensor->ndim != 4 && pixel_tensor->ndim != 5) ||
        pixel_tensor->shape[0] != 1 ||
        pixel_tensor->shape[1] != 3 || !latent_tensor ||
        latent_tensor->dtype != H3_DTYPE_F32 || latent_tensor->ndim != 5 ||
        latent_tensor->shape[0] != 1 || latent_tensor->shape[1] != 24 ||
        latent_tensor->shape[3] * 16 !=
            pixel_tensor->shape[pixel_tensor->ndim - 2] ||
        latent_tensor->shape[4] * 16 !=
            pixel_tensor->shape[pixel_tensor->ndim - 1])
        die("malformed visual encoder fixture");
    int frames = pixel_tensor->ndim == 5 ? (int)pixel_tensor->shape[2] : 1;
    int height = (int)pixel_tensor->shape[pixel_tensor->ndim - 2];
    int width = (int)pixel_tensor->shape[pixel_tensor->ndim - 1];
    if (latent_tensor->shape[2] != (uint64_t)h3_video_encoder_latent_t(frames))
        die("visual encoder fixture has unexpected temporal compression");
    size_t pixel_count = h3_st_tensor_elements(pixel_tensor);
    size_t latent_count = h3_st_tensor_elements(latent_tensor);
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    float *want = malloc(latent_count * sizeof(*want));
    if (!pixels || !want) die("out of memory loading visual encoder fixture");
    if (!h3_st_read_data(&fixture, pixel_tensor, pixels,
                         pixel_count * sizeof(*pixels), error, sizeof(error)) ||
        !h3_st_read_data(&fixture, latent_tensor, want,
                         latent_count * sizeof(*want), error, sizeof(error)))
        die(error);
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    h3_video_latent got;
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, frames,
                             height, width,
                             progress, NULL, &got, error, sizeof(error)))
        die(error);
    if (got.time != h3_video_encoder_latent_t(frames) ||
        got.height != height / 16 ||
        got.width != width / 16)
        die("native visual encoder returned the wrong shape");
    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < latent_count; index++) {
        if (!isfinite(got.values[index]))
            die("native visual encoder returned non-finite values");
        double delta = (double)got.values[index] - want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    printf("visual encoder: max abs %.7g, relative L2 %.7g\n",
           maximum, relative_l2);
    printf("visual encoder: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu submissions\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (maximum >= 2e-3 || relative_l2 >= 0.02)
        die("native visual encoder exceeds MLX parity bound");
    int tiles = axis_tiles(height) * axis_tiles(width);
    if (got.gpu_stats.mps_conv_dispatches != (uint64_t)(34 * tiles) ||
        got.gpu_stats.submissions != (uint64_t)(18 * tiles))
        die("native visual encoder dispatch structure changed unexpectedly");
    h3_video_latent_free(&got);
    h3_st_free_header(&fixture);
    free(pixels);
    free(want);
    puts("ok: native Metal visual encoder matches the MLX anchor latent");
    return 0;
}
