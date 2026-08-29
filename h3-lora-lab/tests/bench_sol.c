#include "h3_dit.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_ROWS 64
#define TEXT_WIDTH 5120
#define LATENT_T 7
#define LATENT_H 30
#define LATENT_W 54
#define AUDIO_T 37
#define STEPS 6

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "/Users/robzomb/h3-models/MiniMax-H3";
    printf("===============================================================\n");
    printf("     Sol-Engine Inference Acceleration Performance Benchmark     \n");
    printf("===============================================================\n");
    printf("Model Directory: %s\n", model_dir);
    printf("Canvas Shape:    T=%d, H=%d, W=%d (Tokens: %d)\n",
           LATENT_T, LATENT_H, LATENT_W, LATENT_T * (LATENT_H / 2) * (LATENT_W / 2));
    printf("Denoise Steps:   %d\n\n", STEPS);

    char error[512] = {0};
    char dit_path[512];
    snprintf(dit_path, sizeof(dit_path), "%s/FL2VA/transformer", model_dir);

    // Setup layout
    h3_layout_spec spec = {
        .text_len = TEXT_ROWS,
        .latent_t = LATENT_T,
        .latent_h = LATENT_H,
        .latent_w = LATENT_W,
        .audio_t = AUDIO_T,
        .frame_count = 56,
        .keyframes = NULL,
        .keyframe_count = 0,
        .references = NULL,
        .reference_count = 0
    };
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "Failed to build layout: %s\n", error);
        return 1;
    }

    h3_sigma_schedule sigmas;
    h3_serving_schedule_build(STEPS, &sigmas);

    uint16_t *text_embedding = calloc((size_t)TEXT_ROWS * TEXT_WIDTH, sizeof(uint16_t));
    uint8_t *text_tags = calloc((size_t)TEXT_ROWS, sizeof(uint8_t));
    h3_text_embedding text = {
        .tokens = TEXT_ROWS,
        .width = TEXT_WIDTH,
        .values = text_embedding,
        .tags = text_tags
    };

    printf("Loading DiT Model on Apple Silicon Metal...\n");
    double load_start = get_time_sec();
    h3_dit *dit = h3_dit_load_t2va(
        dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 0, 1.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        NULL, NULL, error, sizeof(error));

    if (!dit) {
        fprintf(stderr, "Failed to load DiT: %s\n", error);
        free(text_embedding);
        free(text_tags);
        h3_layout_free(&layout);
        return 1;
    }
    printf("Model loaded in %.2f seconds.\n\n", get_time_sec() - load_start);

    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);

    float *video_init = malloc(video_count * sizeof(float));
    float *audio_init = malloc(audio_count * sizeof(float));
    float *video_out = malloc(video_count * sizeof(float));
    float *audio_out = malloc(audio_count * sizeof(float));

    for (size_t i = 0; i < video_count; i++) video_init[i] = sinf((float)i * 0.01f);
    for (size_t i = 0; i < audio_count; i++) audio_init[i] = cosf((float)i * 0.01f);

    // -------------------------------------------------------------
    // RUN 1: Baseline (Standard Dense SDPA, No Caching)
    // -------------------------------------------------------------
    printf("--- [1] Baseline Run (Standard Dense SDPA, No Caching) ---\n");
    h3_dit_enable_sol_attn(dit, 0, 10.0f, 32);
    h3_dit_enable_sol_cache(dit, 0, 0.0f);
    memcpy(video_out, video_init, video_count * sizeof(float));
    memcpy(audio_out, audio_init, audio_count * sizeof(float));

    double t0 = get_time_sec();
    int ok = h3_dit_denoise_euler(dit, video_out, audio_out, 1, NULL, NULL, error, sizeof(error));
    double baseline_time = get_time_sec() - t0;
    if (!ok) { fprintf(stderr, "Baseline failed: %s\n", error); return 1; }
    printf("  Baseline Execution Time: %.3f s (%.2f ms/step)\n\n",
           baseline_time, (baseline_time * 1000.0) / (double)STEPS);

    // -------------------------------------------------------------
    // RUN 2: Sol Adaptive Caching (tau=0.08)
    // -------------------------------------------------------------
    printf("--- [2] Sol Adaptive Caching (Threshold tau=0.08) ---\n");
    h3_dit_enable_sol_attn(dit, 0, 10.0f, 32);
    h3_dit_enable_sol_cache(dit, 1, 0.08f);
    memcpy(video_out, video_init, video_count * sizeof(float));
    memcpy(audio_out, audio_init, audio_count * sizeof(float));

    t0 = get_time_sec();
    ok = h3_dit_denoise_sol_adaptive(dit, video_out, audio_out, 1, 0.08f, NULL, NULL, NULL, NULL, error, sizeof(error));
    double sol_cache_time = get_time_sec() - t0;
    if (!ok) { fprintf(stderr, "Sol Cache failed: %s\n", error); return 1; }

    h3_sol_stats sstats1 = {0};
    h3_dit_get_sol_stats(dit, &sstats1);
    printf("  Sol Cache Execution Time: %.3f s (%.2f ms/step)\n",
           sol_cache_time, (sol_cache_time * 1000.0) / (double)STEPS);
    printf("  Speedup vs Baseline:      %.2fx faster\n", baseline_time / sol_cache_time);
    printf("  Cached/Skipped Steps:     %llu of %llu (%.1f%%)\n\n",
           (unsigned long long)sstats1.cached_steps,
           (unsigned long long)sstats1.total_steps,
           sstats1.total_steps ? (100.0 * (double)sstats1.cached_steps / (double)sstats1.total_steps) : 0.0);

    // -------------------------------------------------------------
    // RUN 3: Sol Adaptive Caching Aggressive (tau=0.12)
    // -------------------------------------------------------------
    printf("--- [3] Sol Adaptive Caching Aggressive (Threshold tau=0.12) ---\n");
    h3_dit_enable_sol_attn(dit, 0, 10.0f, 32);
    h3_dit_enable_sol_cache(dit, 1, 0.12f);
    memcpy(video_out, video_init, video_count * sizeof(float));
    memcpy(audio_out, audio_init, audio_count * sizeof(float));

    t0 = get_time_sec();
    ok = h3_dit_denoise_sol_adaptive(dit, video_out, audio_out, 1, 0.12f, NULL, NULL, NULL, NULL, error, sizeof(error));
    double sol_aggr_time = get_time_sec() - t0;
    if (!ok) { fprintf(stderr, "Sol Aggressive failed: %s\n", error); return 1; }

    h3_sol_stats sstats2 = {0};
    h3_dit_get_sol_stats(dit, &sstats2);
    printf("  Sol Aggr Execution Time:  %.3f s (%.2f ms/step)\n",
           sol_aggr_time, (sol_aggr_time * 1000.0) / (double)STEPS);
    printf("  Speedup vs Baseline:      %.2fx faster\n", baseline_time / sol_aggr_time);
    printf("  Cached/Skipped Steps:     %llu of %llu (%.1f%%)\n",
           (unsigned long long)sstats2.cached_steps,
           (unsigned long long)sstats2.total_steps,
           sstats2.total_steps ? (100.0 * (double)sstats2.cached_steps / (double)sstats2.total_steps) : 0.0);
    printf("===============================================================\n");

    h3_dit_free(dit);
    free(text_embedding);
    free(text_tags);
    h3_layout_free(&layout);
    free(video_init);
    free(audio_init);
    free(video_out);
    free(audio_out);

    return 0;
}
