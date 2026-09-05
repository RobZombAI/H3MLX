#include "h3_audio_vae.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    LATENT_LENGTH = 37,
    LATENT_COUNT = 32 * 2 * LATENT_LENGTH,
    SAMPLES = LATENT_LENGTH * 800,
    WAVEFORM_COUNT = 2 * SAMPLES
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_audio_vae.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "native AudioVAE stage: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_audio_vae_37.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *latent_tensor = h3_st_find(&fixture, "x.latent");
    const h3_st_tensor *waveform_tensor = h3_st_find(&fixture, "x.waveform");
    if (!latent_tensor || latent_tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(latent_tensor) != LATENT_COUNT ||
        !waveform_tensor || waveform_tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(waveform_tensor) != WAVEFORM_COUNT)
        die("malformed AudioVAE fixture");
    float *latent = malloc(LATENT_COUNT * sizeof(*latent));
    float *want = malloc(WAVEFORM_COUNT * sizeof(*want));
    if (!latent || !want) die("out of memory loading AudioVAE fixture");
    if (!h3_st_read_data(&fixture, latent_tensor, latent,
                         LATENT_COUNT * sizeof(*latent), error, sizeof(error)) ||
        !h3_st_read_data(&fixture, waveform_tensor, want,
                         WAVEFORM_COUNT * sizeof(*want), error, sizeof(error)))
        die(error);

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root);
    h3_audio_waveform got;
    if (!h3_audio_vae_decode(weights, "h3_shaders.metal", latent,
                             LATENT_LENGTH, progress, NULL, &got,
                             error, sizeof(error))) die(error);
    if (got.channels != 2 || got.samples != SAMPLES ||
        got.sample_rate != 32000) die("native AudioVAE returned the wrong shape");

    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < WAVEFORM_COUNT; index++) {
        if (!isfinite(got.pcm[index])) die("native AudioVAE returned non-finite PCM");
        double delta = (double)got.pcm[index] - want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    printf("AudioVAE waveform: max abs %.7g, relative L2 %.7g\n",
           maximum, relative_l2);
    printf("AudioVAE: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu submissions\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (maximum >= 1e-3 || relative_l2 >= 0.05)
        die("native AudioVAE exceeds MLX parity bound");
    if (got.gpu_stats.mps_conv_dispatches != 136 ||
        got.gpu_stats.submissions != 16)
        die("native AudioVAE dispatch structure changed unexpectedly");

    h3_audio_waveform_free(&got);
    free(latent);
    free(want);
    h3_st_free_header(&fixture);
    puts("ok: native Metal AudioVAE matches the corrected MLX waveform");
    return 0;
}
