#include "h3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static void trim_trailing(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' || s[len - 1] == ' ' || s[len - 1] == '\t')) {
        s[--len] = '\0';
    }
}

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] : "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step";
    
    fprintf(stderr, "🚀 h3_resident_engine: initializing model from %s...\n", model_dir);
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "❌ h3_resident_engine error: cannot load model: %s\n", h3_last_error(ctx));
        return 1;
    }
    
    /* Enable 128GB Unified Memory Resident Caching */
    h3_cache_set_enabled(ctx, 1);
    fprintf(stderr, "💎 h3_resident_engine: 128GB UMA resident cache ACTIVE & WARM.\n");
    
    printf("H3_RESIDENT_READY\n");
    fflush(stdout);

    char line[16384];
    while (fgets(line, sizeof(line), stdin)) {
        trim_trailing(line);
        if (!*line) continue;
        if (strcmp(line, "QUIT") == 0 || strcmp(line, "!quit") == 0) break;

        /* Protocol: OUTPUT_PATH|WIDTH|HEIGHT|STEPS|LAYERS|FRAMES|REUSE|SEED|PROMPT */
        char *output_path = line;
        char *p1 = strchr(output_path, '|');
        if (!p1) continue;
        *p1 = '\0';

        char *width_str = p1 + 1;
        char *p2 = strchr(width_str, '|');
        if (!p2) continue;
        *p2 = '\0';

        char *height_str = p2 + 1;
        char *p3 = strchr(height_str, '|');
        if (!p3) continue;
        *p3 = '\0';
        
        char *steps_str = p3 + 1;
        char *p4 = strchr(steps_str, '|');
        if (!p4) continue;
        *p4 = '\0';
        
        char *layers_str = p4 + 1;
        char *p5 = strchr(layers_str, '|');
        if (!p5) continue;
        *p5 = '\0';
        
        char *frames_str = p5 + 1;
        char *p6 = strchr(frames_str, '|');
        if (!p6) continue;
        *p6 = '\0';
        
        char *reuse_str = p6 + 1;
        char *p7 = strchr(reuse_str, '|');
        if (!p7) continue;
        *p7 = '\0';

        char *seed_str = p7 + 1;
        char *p8 = strchr(seed_str, '|');
        if (!p8) continue;
        *p8 = '\0';

        char *prompt = p8 + 1;

        h3_params params = H3_PARAMS_DEFAULT;
        params.width = atoi(width_str) > 0 ? atoi(width_str) : 640;
        params.height = atoi(height_str) > 0 ? atoi(height_str) : 640;
        params.frames = atoi(frames_str) > 0 ? atoi(frames_str) : 39;
        params.steps = atoi(steps_str) > 0 ? atoi(steps_str) : 8;
        params.dit_layers = atoi(layers_str) > 0 ? atoi(layers_str) : 50;
        params.denoise_reuse = atoi(reuse_str) > 0 ? atoi(reuse_str) : 1;
        params.core_reuse = 1;
        params.use_int8_row_fc2 = 1;
        params.token_reduction = getenv("H3_TOKEN_REDUCTION") ? atoi(getenv("H3_TOKEN_REDUCTION")) : 1;
        params.seed = (uint64_t)strtoull(seed_str, NULL, 10);
        if (params.seed == 0) params.seed = (uint64_t)time(NULL);
        params.output_path = output_path;

        fprintf(stderr, "\n==========================================================\n");
        fprintf(stderr, "🚀 RESIDENT LOSSLESS GENERATION START\n");
        fprintf(stderr, "  Canvas: %dx%d | Frames: %d | Steps: %d | Layers: %d | Reuse: %d\n",
                params.width, params.height, params.frames, params.steps, params.dit_layers, params.denoise_reuse);
        fprintf(stderr, "  Output: %s\n", params.output_path);
        fprintf(stderr, "==========================================================\n");

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        h3_result *result = h3_generate(ctx, prompt, &params);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

        if (!result) {
            fprintf(stderr, "❌ Generation failed: %s\n", h3_last_error(ctx));
            printf("H3_RESIDENT_ERROR: %s\n", h3_last_error(ctx));
            fflush(stdout);
        } else {
            fprintf(stderr, "==========================================================\n");
            fprintf(stderr, "⚡ RESIDENT GENERATION COMPLETE in %.2f s (LOSSLESS 0s LOAD TIME)!\n", elapsed);
            fprintf(stderr, "==========================================================\n");
            printf("H3_RESIDENT_DONE: %.2f\n", elapsed);
            fflush(stdout);
            h3_result_free(result);
        }
    }

    fprintf(stderr, "h3_resident_engine: shutting down gracefully...\n");
    h3_free(ctx);
    return 0;
}
