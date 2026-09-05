#include "h3_audio_vae.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_audio_encoder.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "native audio encoder stage: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_audio_encoder_64000.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *waveform_tensor = h3_st_find(&fixture, "x.waveform");
    const h3_st_tensor *latent_tensor = h3_st_find(&fixture, "x.latent");
    if (!waveform_tensor || waveform_tensor->dtype != H3_DTYPE_F32 ||
        waveform_tensor->ndim != 3 || waveform_tensor->shape[0] != 1 ||
        waveform_tensor->shape[1] != 2 || !latent_tensor ||
        latent_tensor->dtype != H3_DTYPE_F32 || latent_tensor->ndim != 4 ||
        latent_tensor->shape[0] != 1 || latent_tensor->shape[1] != 32 ||
        latent_tensor->shape[2] != 2 ||
        latent_tensor->shape[3] != (waveform_tensor->shape[2] + 799) / 800)
        die("malformed AudioVAE encoder fixture");
    size_t waveform_count = h3_st_tensor_elements(waveform_tensor);
    size_t latent_count = h3_st_tensor_elements(latent_tensor);
    float *waveform = malloc(waveform_count * sizeof(*waveform));
    float *want = malloc(latent_count * sizeof(*want));
    if (!waveform || !want) die("out of memory loading audio encoder fixture");
    if (!h3_st_read_data(&fixture, waveform_tensor, waveform,
                         waveform_count * sizeof(*waveform), error,
                         sizeof(error)) ||
        !h3_st_read_data(&fixture, latent_tensor, want,
                         latent_count * sizeof(*want), error, sizeof(error)))
        die(error);

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root);
    h3_audio_latent got;
    if (!h3_audio_vae_encode(weights, "h3_shaders.metal", waveform,
                             (int)waveform_tensor->shape[2], progress, NULL,
                             &got, error, sizeof(error))) die(error);
    if (got.channels != 32 || got.stereo != 2 ||
        got.length != (int)latent_tensor->shape[3])
        die("native audio encoder returned the wrong shape");

    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < latent_count; index++) {
        if (!isfinite(got.values[index]))
            die("native audio encoder returned non-finite values");
        double delta = (double)got.values[index] - want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    printf("audio encoder: max abs %.7g, relative L2 %.7g\n",
           maximum, relative_l2);
    printf("audio encoder: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu MPS attention, %llu submissions\n",
           (double)got.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (maximum >= 2e-3 || relative_l2 >= 0.02)
        die("native audio encoder exceeds MLX parity bound");
    if (got.gpu_stats.mps_conv_dispatches != 38 ||
        got.gpu_stats.mps_sdpa_dispatches != 1 ||
        got.gpu_stats.submissions != 26)
        die("native audio encoder dispatch structure changed unexpectedly");

    h3_audio_latent_free(&got);
    h3_st_free_header(&fixture);
    free(waveform);
    free(want);
    puts("ok: native Metal audio encoder matches the MLX posterior mean");
    return 0;
}
