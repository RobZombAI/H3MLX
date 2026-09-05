#ifndef H3_AUDIO_VAE_H
#define H3_AUDIO_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int channels;
    int samples;
    int sample_rate;
    /* Channel-major F32: [channels,samples], clipped to [-1,1]. */
    float *pcm;
    h3_gpu_stats gpu_stats;
} h3_audio_waveform;

typedef struct {
    int channels;
    int stereo;
    int length;
    /* Channel-major F32: [32,2,length], normalized posterior means. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_audio_latent;

typedef void (*h3_audio_vae_progress)(int completed_stages, int total_stages,
                                      void *opaque);

/* Decode normalized H3 audio latents in [32,2,T] channel/stereo/time order.
 * The released mono BigVGAN decoder folds stereo into the batch dimension. */
int h3_audio_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_length,
                        h3_audio_vae_progress progress, void *progress_opaque,
                        h3_audio_waveform *output,
                        char *error, size_t error_size);
/* Encode channel-major 32 kHz stereo PCM [2,samples]. The input is zero-padded
 * on the right to the next 800-sample latent boundary. */
int h3_audio_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pcm, int samples,
                        h3_audio_vae_progress progress, void *progress_opaque,
                        h3_audio_latent *output,
                        char *error, size_t error_size);
void h3_audio_waveform_free(h3_audio_waveform *waveform);
void h3_audio_latent_free(h3_audio_latent *latent);

#endif
