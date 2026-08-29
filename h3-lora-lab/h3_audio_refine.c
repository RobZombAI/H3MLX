#include "h3.h"
#include "h3_audio_vae.h"
#include "h3_dit.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * H3 AudioRefine Engine (Inspired by Adudeguyman ComfyUI-H3-AudioRefine)
 * 
 * Takes a converged video latent and refines the audio latent using a masked
 * Euler flow schedule, polishing audio dynamics, bandoneón/violin timbres,
 * and room acoustics without modifying video tokens.
 */

int h3_audio_refine_pass(h3_dit *dit,
                         const float *clean_video_latent,
                         float *audio_latent,
                         int audio_steps,
                         float denoise_strength,
                         uint64_t seed,
                         char *error, size_t error_size) {
    if (!dit || !clean_video_latent || !audio_latent || audio_steps < 1) {
        if (error && error_size) snprintf(error, error_size, "invalid AudioRefine parameters");
        return 0;
    }

    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);

    /* Allocate scratch velocity buffers */
    float *video_vel = malloc(video_count * sizeof(float));
    float *audio_vel = malloc(audio_count * sizeof(float));
    float *video_frozen = malloc(video_count * sizeof(float));
    if (!video_vel || !audio_vel || !video_frozen) {
        free(video_vel); free(audio_vel); free(video_frozen);
        if (error && error_size) snprintf(error, error_size, "out of memory in AudioRefine pass");
        return 0;
    }

    /* Freeze video latent */
    memcpy(video_frozen, clean_video_latent, video_count * sizeof(float));

    /* If denoise_strength < 1.0, blend audio with initial noise */
    if (denoise_strength > 0.0f && denoise_strength < 1.0f) {
        h3_rng rng;
        h3_rng_seed(&rng, seed ? seed + 999 : 42);
        for (size_t i = 0; i < audio_count; i++) {
            float noise = h3_rng_normal(&rng);
            audio_latent[i] = (1.0f - denoise_strength) * audio_latent[i] + denoise_strength * noise;
        }
    }

    fprintf(stderr, "h3: executing AudioRefine masked pass (%d steps, strength=%.2f)...\n",
            audio_steps, denoise_strength);

    /* Audio Refinement Euler Iterations */
    for (int step = 0; step < audio_steps; step++) {
        /* Forward pass with video frozen and audio active */
        int ok = h3_dit_forward(dit, step, video_frozen, audio_latent,
                                video_vel, audio_vel, error, error_size);
        if (!ok) {
            free(video_vel); free(audio_vel); free(video_frozen);
            return 0;
        }

        /* Update ONLY the audio latent stream; keep video bit-identical */
        float dt = 1.0f / (float)audio_steps;
        for (size_t i = 0; i < audio_count; i++) {
            audio_latent[i] += audio_vel[i] * dt;
        }
    }

    free(video_vel);
    free(audio_vel);
    free(video_frozen);

    fprintf(stderr, "h3: AudioRefine pass completed successfully.\n");
    return 1;
}
