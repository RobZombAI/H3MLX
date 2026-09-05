#include "h3_safetensors.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FRAMES = 22,
    HEIGHT = 256,
    WIDTH = 256,
    LATENT_T = 7,
    LATENT_H = 16,
    LATENT_W = 16,
    LATENT_COUNT = 24 * LATENT_T * LATENT_H * LATENT_W,
    FRAME_COUNT = 3 * FRAMES * HEIGHT * WIDTH
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_semantic_vae.c: %s\n", message);
    exit(1);
}

static float *load_f32(const h3_st_header *header, const char *name,
                       size_t count) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != (uint64_t)count)
        die("semantic VAE fixture has the wrong schema");
    float *values = malloc(count * sizeof(*values));
    if (!values) die("out of memory loading semantic VAE fixture");
    char error[512];
    if (!h3_st_read_data(header, tensor, values, count * sizeof(*values),
                         error, sizeof(error))) die(error);
    return values;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 5 == 0)
        fprintf(stderr, "semantic VAE: %d/%d blocks\n", completed, total);
}

static void test_resident_preview(const char *model_root) {
    enum {
        TEST_T = 12,
        TEST_FRAMES = 39,
        TEST_LATENTS = 24 * TEST_T * LATENT_H * LATENT_W,
        TEST_PIXELS = 3 * TEST_FRAMES * HEIGHT * WIDTH
    };
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(
            "misc/fixtures/h3_real_video_vae_256x256x39_f32.safetensors",
            &fixture, error, sizeof(error))) die(error);
    float *latent = load_f32(&fixture, "x.latent", TEST_LATENTS);
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    h3_video_frames ordinary;
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent,
                             TEST_T, LATENT_H, LATENT_W, progress, NULL,
                             &ordinary, error, sizeof(error))) die(error);
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        weights, "h3_shaders.metal", LATENT_H, LATENT_W,
        progress, NULL, error, sizeof(error));
    if (!decoder) die(error);
    h3_video_frames preview;
    int frame_index = -1;
    if (!h3_video_vae_decoder_preview(
            decoder, latent, TEST_T, &preview, &frame_index,
            error, sizeof(error))) die(error);
    if (preview.frames != 1 || preview.height != HEIGHT ||
        preview.width != WIDTH || frame_index < 0 ||
        frame_index >= TEST_FRAMES) {
        die("resident VAE preview returned the wrong shape or frame");
    }
    size_t frame_elements = (size_t)HEIGHT * WIDTH * 3;
    if (memcmp(preview.rgb,
               ordinary.rgb + (size_t)frame_index * frame_elements,
               frame_elements * sizeof(*preview.rgb)))
        die("resident VAE preview differs from the complete decoder frame");
    h3_video_frames_free(&preview);
    h3_video_frames resident;
    if (!h3_video_vae_decoder_decode(
            decoder, latent, TEST_T, &resident,
            error, sizeof(error))) die(error);
    if (resident.frames != ordinary.frames ||
        resident.height != ordinary.height ||
        resident.width != ordinary.width ||
        memcmp(resident.rgb, ordinary.rgb,
               (size_t)TEST_PIXELS * sizeof(*resident.rgb)))
        die("resident final VAE decode differs from the ordinary path");
    h3_video_frames_free(&resident);
    h3_video_frames_free(&ordinary);
    h3_video_vae_decoder_free(decoder);
    h3_st_free_header(&fixture);
    free(latent);
    puts("ok: resident VAE preview and final decode are byte-identical");
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--resident-preview")) {
        test_resident_preview(argc > 2 ? argv[2] : "MiniMax-H3");
        return 0;
    }
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_semantic_256x22_seed42.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    float *latent = load_f32(&fixture, "x.video_final", LATENT_COUNT);
    float *want = load_f32(&fixture, "x.frames", FRAME_COUNT);
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    h3_video_frames got;
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent,
                             LATENT_T, LATENT_H, LATENT_W, progress, NULL,
                             &got, error, sizeof(error))) die(error);
    if (got.frames != FRAMES || got.height != HEIGHT || got.width != WIDTH)
        die("semantic VAE returned the wrong shape");
    double maximum = 0.0, scale = 0.0, square_error = 0.0, square_value = 0.0;
    for (int frame = 0; frame < FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                for (int channel = 0; channel < 3; channel++) {
                    size_t native = (((size_t)frame * HEIGHT + (size_t)y) *
                        WIDTH + (size_t)x) * 3 + (size_t)channel;
                    size_t mlx = ((((size_t)channel * FRAMES + (size_t)frame) *
                        HEIGHT + (size_t)y) * WIDTH + (size_t)x);
                    double delta = (double)got.rgb[native] - (double)want[mlx];
                    if (fabs(delta) > maximum) maximum = fabs(delta);
                    if (fabs((double)want[mlx]) > scale)
                        scale = fabs((double)want[mlx]);
                    square_error += delta * delta;
                    square_value += (double)want[mlx] * (double)want[mlx];
                }
    double relative_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double relative_l2 = sqrt(square_error /
        (square_value > 1e-24 ? square_value : 1e-24));
    printf("256x256x22 decoded RGB: rel-max %.6g abs %.6g rel-L2 %.6g\n",
           relative_max, maximum, relative_l2);
    if (relative_max >= 0.05 || relative_l2 >= 0.05)
        die("native 22-frame VAE diverges from MLX");
    printf("semantic VAE: %.3f GiB cumulative, %.3f GPU seconds, "
           "%llu submissions\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.submissions);
    h3_video_frames_free(&got);
    if (argc > 3 && !strcmp(argv[3], "--tiled-smoke")) {
        enum { TILED_W = 18, TILED_COUNT = 24 * LATENT_T * LATENT_H * TILED_W };
        float *tiled_latent = malloc(TILED_COUNT * sizeof(*tiled_latent));
        if (!tiled_latent) die("out of memory making tiled VAE input");
        for (int channel = 0; channel < 24; channel++)
            for (int time = 0; time < LATENT_T; time++)
                for (int y = 0; y < LATENT_H; y++)
                    for (int x = 0; x < TILED_W; x++) {
                        size_t source = (((size_t)channel * LATENT_T +
                            (size_t)time) * LATENT_H + (size_t)y) * LATENT_W +
                            (size_t)(x < LATENT_W ? x : LATENT_W - 1);
                        size_t destination = (((size_t)channel * LATENT_T +
                            (size_t)time) * LATENT_H + (size_t)y) * TILED_W +
                            (size_t)x;
                        tiled_latent[destination] = latent[source];
                    }
        if (!h3_video_vae_decode(weights, "h3_shaders.metal", tiled_latent,
                                 LATENT_T, LATENT_H, TILED_W, progress, NULL,
                                 &got, error, sizeof(error))) die(error);
        if (got.frames != FRAMES || got.height != HEIGHT || got.width != 288)
            die("tiled VAE smoke test returned the wrong shape");
        size_t tiled_pixels = (size_t)got.frames * (size_t)got.height *
                              (size_t)got.width * 3;
        for (size_t index = 0; index < tiled_pixels; index++)
            if (!isfinite(got.rgb[index])) die("tiled VAE produced non-finite RGB");
        h3_st_header tiled_fixture;
        if (!h3_st_read_header(
                "misc/fixtures/h3_real_video_vae_256x288_f32.safetensors",
                &tiled_fixture, error, sizeof(error))) die(error);
        float *tiled_want = load_f32(&tiled_fixture, "x.frames", tiled_pixels);
        double tiled_error = 0.0, tiled_value = 0.0;
        for (int frame = 0; frame < FRAMES; frame++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < 288; x++)
                    for (int channel = 0; channel < 3; channel++) {
                        size_t native = (((size_t)frame * HEIGHT + (size_t)y) *
                            288 + (size_t)x) * 3 + (size_t)channel;
                        size_t mlx = ((((size_t)channel * FRAMES +
                            (size_t)frame) * HEIGHT + (size_t)y) * 288 +
                            (size_t)x);
                        double delta = (double)got.rgb[native] - tiled_want[mlx];
                        tiled_error += delta * delta;
                        tiled_value += (double)tiled_want[mlx] * tiled_want[mlx];
                    }
        double tiled_l2 = sqrt(tiled_error /
            (tiled_value > 1e-24 ? tiled_value : 1e-24));
        printf("256x288 tiled VAE: %.3f GiB cumulative, %.3f GPU seconds, "
               "%llu submissions, rel-L2 %.6g\n",
               (double)got.gpu_stats.allocated_bytes /
                   (1024.0 * 1024.0 * 1024.0),
               got.gpu_stats.gpu_seconds,
               (unsigned long long)got.gpu_stats.submissions, tiled_l2);
        if (got.gpu_stats.submissions != 2 || tiled_l2 >= 0.05)
            die("tiled VAE did not use one submission per tile");
        free(tiled_want);
        h3_st_free_header(&tiled_fixture);
        h3_video_frames_free(&got);
        free(tiled_latent);
    }
    if (argc > 3 && !strcmp(argv[3], "--chunked-smoke")) {
        enum {
            CHUNKED_T = 12,
            CHUNKED_FRAMES = 39,
            CHUNKED_LATENTS = 24 * CHUNKED_T * LATENT_H * LATENT_W,
            CHUNKED_PIXELS = 3 * CHUNKED_FRAMES * HEIGHT * WIDTH
        };
        h3_st_header chunked_fixture;
        if (!h3_st_read_header(
                "misc/fixtures/h3_real_video_vae_256x256x39_f32.safetensors",
                &chunked_fixture, error, sizeof(error))) die(error);
        float *chunked_latent = load_f32(&chunked_fixture, "x.latent",
                                         CHUNKED_LATENTS);
        float *chunked_want = load_f32(&chunked_fixture, "x.frames",
                                       CHUNKED_PIXELS);
        if (!h3_video_vae_decode(weights, "h3_shaders.metal", chunked_latent,
                                 CHUNKED_T, LATENT_H, LATENT_W, progress, NULL,
                                 &got, error, sizeof(error))) die(error);
        if (got.frames != CHUNKED_FRAMES || got.height != HEIGHT ||
            got.width != WIDTH) die("chunked VAE returned the wrong shape");
        double chunked_error = 0.0, chunked_value = 0.0;
        for (int frame = 0; frame < CHUNKED_FRAMES; frame++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    for (int channel = 0; channel < 3; channel++) {
                        size_t native = (((size_t)frame * HEIGHT + (size_t)y) *
                            WIDTH + (size_t)x) * 3 + (size_t)channel;
                        size_t mlx = ((((size_t)channel * CHUNKED_FRAMES +
                            (size_t)frame) * HEIGHT + (size_t)y) * WIDTH +
                            (size_t)x);
                        double delta = (double)got.rgb[native] - chunked_want[mlx];
                        chunked_error += delta * delta;
                        chunked_value += (double)chunked_want[mlx] *
                                         chunked_want[mlx];
                    }
        double chunked_l2 = sqrt(chunked_error /
            (chunked_value > 1e-24 ? chunked_value : 1e-24));
        printf("256x256x39 chunked VAE: %.3f GPU seconds, %llu submissions, "
               "rel-L2 %.6g\n", got.gpu_stats.gpu_seconds,
               (unsigned long long)got.gpu_stats.submissions, chunked_l2);
        if (got.gpu_stats.submissions != 2 || chunked_l2 >= 0.05)
            die("chunked VAE diverges from MLX");
        h3_video_frames_free(&got);
        h3_st_free_header(&chunked_fixture);
        free(chunked_latent);
        free(chunked_want);
    }
    h3_st_free_header(&fixture);
    free(latent);
    free(want);
    puts("ok: native Metal matches MLX for the genuine 22-frame VAE chunk");
    return 0;
}
