#include "h3_audio_vae.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 32,
    LATENT_DIM = 2048,
    DECODER_DIM = 1024,
    STEREO = 2,
    STAGES = 7,
    RESBLOCKS = 3,
    RESIDUAL_PAIRS = 3,
    FILTER_SIZE = 12,
    SAMPLE_RATE = 32000,
    HOP_LENGTH = 800
};

static const uint32_t upsample_rates[STAGES] = {5, 5, 2, 2, 2, 2, 2};
static const uint32_t upsample_kernels[STAGES] = {9, 9, 4, 4, 4, 4, 4};
static const uint32_t residual_kernels[RESBLOCKS] = {3, 7, 11};
static const uint32_t residual_dilations[RESIDUAL_PAIRS] = {1, 3, 5};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *vector;
    h3_gpu_tensor *magnitude;
    h3_gpu_tensor *bias;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t padding;
    uint32_t dilation;
    uint32_t stride;
    int transpose;
} audio_conv;

typedef struct {
    h3_gpu_tensor *alpha;
    h3_gpu_tensor *beta;
} audio_activation;

typedef struct {
    audio_activation activations[RESIDUAL_PAIRS * 2];
    audio_conv convs1[RESIDUAL_PAIRS];
    audio_conv convs2[RESIDUAL_PAIRS];
} audio_resblock;

typedef struct {
    audio_conv upsample;
    audio_resblock blocks[RESBLOCKS];
} audio_stage;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_gpu_tensor *upsample_filter;
    h3_gpu_tensor *downsample_filter;
    h3_gpu_tensor *hidden;
    uint32_t length;
} audio_context;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(audio_context *audio, int ok, char *error,
                  size_t error_size, const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(audio->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void free_conv(audio_conv *conv) {
    free_tensor(&conv->weight);
    free_tensor(&conv->vector);
    free_tensor(&conv->magnitude);
    free_tensor(&conv->bias);
    memset(conv, 0, sizeof(*conv));
}

static void free_activation(audio_activation *activation) {
    free_tensor(&activation->alpha);
    free_tensor(&activation->beta);
}

static void free_resblock(audio_resblock *block) {
    for (int index = 0; index < RESIDUAL_PAIRS * 2; index++)
        free_activation(&block->activations[index]);
    for (int index = 0; index < RESIDUAL_PAIRS; index++) {
        free_conv(&block->convs1[index]);
        free_conv(&block->convs2[index]);
    }
}

static void free_stage(audio_stage *stage) {
    free_conv(&stage->upsample);
    for (int index = 0; index < RESBLOCKS; index++)
        free_resblock(&stage->blocks[index]);
}

static void cleanup(audio_context *audio) {
    if (!audio) return;
    free_tensor(&audio->upsample_filter);
    free_tensor(&audio->downsample_filter);
    free_tensor(&audio->hidden);
    h3_weight_store_free(audio->weights);
    h3_gpu_free(audio->gpu);
    memset(audio, 0, sizeof(*audio));
}

static h3_gpu_tensor *load_f32(audio_context *audio, const char *name,
                               int ndim, const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(audio->weights, audio->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(audio_context *audio, const char *name,
                         uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(audio, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f3(audio_context *audio, const char *name,
                         uint64_t first, uint64_t second, uint64_t third,
                         char *error, size_t error_size) {
    uint64_t shape[] = {first, second, third};
    return load_f32(audio, name, 3, shape, error, error_size);
}

static h3_gpu_tensor *f2(audio_context *audio, const char *name,
                         uint64_t rows, uint64_t columns, char *error,
                         size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_f32(audio, name, 2, shape, error, error_size);
}

static int load_plain_conv(audio_context *audio, audio_conv *conv,
                           const char *prefix, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t padding, uint32_t dilation, int has_bias,
                           char *error, size_t error_size) {
    char name[192];
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->stride = 1;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f3(audio, name, output_channels, input_channels, kernel,
                      error, error_size);
    if (!conv->weight) return 0;
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        conv->bias = f1(audio, name, output_channels, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_normalized_conv(audio_context *audio, audio_conv *conv,
                                const char *prefix,
                                uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t padding, uint32_t dilation,
                                uint32_t stride, int transpose, int has_bias,
                                char *error, size_t error_size) {
    char name[192];
    uint32_t outer = transpose ? input_channels : output_channels;
    uint32_t inner_channels = transpose ? output_channels : input_channels;
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->stride = stride;
    conv->transpose = transpose;
    snprintf(name, sizeof(name), "%s.weight_v", prefix);
    conv->vector = f3(audio, name, outer, inner_channels, kernel,
                      error, error_size);
    if (!conv->vector) return 0;
    snprintf(name, sizeof(name), "%s.weight_g", prefix);
    conv->magnitude = f3(audio, name, outer, 1, 1, error, error_size);
    if (!conv->magnitude) return 0;
    size_t elements = (size_t)outer * inner_channels * kernel;
    conv->weight = h3_gpu_tensor_new_f32(audio->gpu, elements);
    if (!conv->weight) {
        fail(error, error_size, "cannot allocate normalized %s weight: %s",
             prefix, h3_gpu_error(audio->gpu));
        return 0;
    }
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        conv->bias = f1(audio, name, output_channels, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int normalize_conv(audio_context *audio, audio_conv *conv,
                          char *error, size_t error_size) {
    uint32_t outer = conv->transpose ? conv->input_channels :
                                      conv->output_channels;
    uint32_t inner_channels = conv->transpose ? conv->output_channels :
                                               conv->input_channels;
    uint64_t inner = (uint64_t)inner_channels * conv->kernel;
    if (inner > UINT32_MAX) {
        fail(error, error_size, "audio convolution weight is too large");
        return 0;
    }
    return gpu_op(audio, h3_gpu_weight_norm_f32(
        audio->gpu, conv->weight, conv->vector, conv->magnitude, outer,
        (uint32_t)inner), error, error_size, "AudioVAE weight normalization");
}

static void retire_normalization_inputs(audio_conv *conv) {
    free_tensor(&conv->vector);
    free_tensor(&conv->magnitude);
}

static int load_activation(audio_context *audio,
                           audio_activation *activation, const char *prefix,
                           uint32_t channels, char *error, size_t error_size) {
    char name[224];
    snprintf(name, sizeof(name), "%s.act.alpha", prefix);
    activation->alpha = f1(audio, name, channels, error, error_size);
    if (!activation->alpha) return 0;
    snprintf(name, sizeof(name), "%s.act.beta", prefix);
    activation->beta = f1(audio, name, channels, error, error_size);
    return activation->beta != NULL;
}

static int load_filters(audio_context *audio, char *error,
                        size_t error_size) {
    audio->upsample_filter = f3(
        audio, "decoder.activation_post.upsample.filter", 1, 1, FILTER_SIZE,
        error, error_size);
    audio->downsample_filter = f3(
        audio, "decoder.activation_post.downsample.lowpass.filter", 1, 1,
        FILTER_SIZE, error, error_size);
    return audio->upsample_filter && audio->downsample_filter;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "audio VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "audio VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "audio VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "audio VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving audio VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open audio VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid audio VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read audio VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size,
                 "audio VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static int prepare_input(audio_context *audio, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t elements = (size_t)STEREO * audio->length * LATENT_CHANNELS;
    float *rows = malloc(elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory transposing audio latent");
        return 0;
    }
    size_t destination = 0;
    for (int stereo = 0; stereo < STEREO; stereo++)
        for (uint32_t time = 0; time < audio->length; time++)
            for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                size_t source = ((size_t)channel * STEREO + (size_t)stereo) *
                                audio->length + time;
                rows[destination++] = input[source] * deviation[channel] +
                                      mean[channel];
            }
    h3_gpu_tensor *latent = h3_gpu_tensor_from_f32(audio->gpu, rows, elements);
    free(rows);
    if (!latent) {
        fail(error, error_size, "cannot allocate audio latent: %s",
             h3_gpu_error(audio->gpu));
        return 0;
    }

    audio_conv projection = {0}, pre = {0};
    h3_gpu_tensor *projected = NULL, *hidden = NULL;
    int ok = load_plain_conv(audio, &projection, "dec_in_proj",
                             LATENT_CHANNELS, LATENT_DIM, 1, 0, 1, 1,
                             error, error_size) &&
             load_normalized_conv(audio, &pre, "decoder.conv_pre",
                                  LATENT_DIM, DECODER_DIM, 7, 3, 1, 1, 0, 1,
                                  error, error_size);
    if (!ok) goto done;
    projected = h3_gpu_tensor_new_f32(
        audio->gpu, (size_t)STEREO * audio->length * LATENT_DIM);
    hidden = h3_gpu_tensor_new_f32(
        audio->gpu, (size_t)STEREO * audio->length * DECODER_DIM);
    if (!projected || !hidden) {
        fail(error, error_size, "cannot allocate AudioVAE input activations: %s",
             h3_gpu_error(audio->gpu));
        ok = 0;
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE input") &&
         normalize_conv(audio, &pre, error, error_size) &&
         gpu_op(audio, h3_gpu_conv1d_f32(
             audio->gpu, projected, latent, projection.weight, projection.bias,
             STEREO, audio->length, LATENT_CHANNELS, LATENT_DIM, 1, 0, 1),
             error, error_size, "AudioVAE input projection") &&
         gpu_op(audio, h3_gpu_conv1d_f32(
             audio->gpu, hidden, projected, pre.weight, pre.bias, STEREO,
             audio->length, LATENT_DIM, DECODER_DIM, 7, 3, 1), error,
             error_size, "AudioVAE pre convolution") &&
         gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE input");
    if (ok) {
        audio->hidden = hidden;
        hidden = NULL;
    }

done:
    h3_gpu_tensor_free(latent);
    h3_gpu_tensor_free(projected);
    h3_gpu_tensor_free(hidden);
    free_conv(&projection);
    free_conv(&pre);
    return ok;
}

static int load_stage(audio_context *audio, audio_stage *stage, int index,
                      char *error, size_t error_size) {
    uint32_t input_channels = DECODER_DIM >> index;
    uint32_t output_channels = DECODER_DIM >> (index + 1);
    uint32_t rate = upsample_rates[index];
    uint32_t kernel = upsample_kernels[index];
    uint32_t padding = (kernel - rate) / 2;
    char prefix[192];
    snprintf(prefix, sizeof(prefix), "decoder.ups.%d.0", index);
    if (!load_normalized_conv(audio, &stage->upsample, prefix,
                              input_channels, output_channels, kernel, padding,
                              1, rate, 1, 1, error, error_size)) return 0;

    for (int block_index = 0; block_index < RESBLOCKS; block_index++) {
        int global = index * RESBLOCKS + block_index;
        audio_resblock *block = &stage->blocks[block_index];
        uint32_t residual_kernel = residual_kernels[block_index];
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            for (int which = 0; which < 2; which++) {
                int activation = pair * 2 + which;
                snprintf(prefix, sizeof(prefix),
                         "decoder.resblocks.%d.activations.%d", global,
                         activation);
                if (!load_activation(audio, &block->activations[activation],
                                     prefix, output_channels, error,
                                     error_size)) return 0;
            }
            uint32_t dilation = residual_dilations[pair];
            uint32_t dilated_padding =
                dilation * (residual_kernel - 1) / 2;
            snprintf(prefix, sizeof(prefix),
                     "decoder.resblocks.%d.convs1.%d", global, pair);
            if (!load_normalized_conv(
                    audio, &block->convs1[pair], prefix, output_channels,
                    output_channels, residual_kernel, dilated_padding,
                    dilation, 1, 0, 1, error, error_size)) return 0;
            snprintf(prefix, sizeof(prefix),
                     "decoder.resblocks.%d.convs2.%d", global, pair);
            if (!load_normalized_conv(
                    audio, &block->convs2[pair], prefix, output_channels,
                    output_channels, residual_kernel,
                    (residual_kernel - 1) / 2, 1, 1, 0, 1, error,
                    error_size)) return 0;
        }
    }
    return 1;
}

static int normalize_stage(audio_context *audio, audio_stage *stage,
                           char *error, size_t error_size) {
    if (!gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE stage normalization") ||
        !normalize_conv(audio, &stage->upsample, error, error_size)) return 0;
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++)
            if (!normalize_conv(audio, &stage->blocks[block].convs1[pair],
                                error, error_size) ||
                !normalize_conv(audio, &stage->blocks[block].convs2[pair],
                                error, error_size)) return 0;
    if (!gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE stage normalization")) return 0;
    retire_normalization_inputs(&stage->upsample);
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            retire_normalization_inputs(&stage->blocks[block].convs1[pair]);
            retire_normalization_inputs(&stage->blocks[block].convs2[pair]);
        }
    return 1;
}

static int run_activation(audio_context *audio, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const audio_activation *activation,
                          uint32_t length, uint32_t channels, char *error,
                          size_t error_size) {
    return gpu_op(audio, h3_gpu_alias_free_snake_f32(
        audio->gpu, output, input, activation->alpha, activation->beta,
        audio->upsample_filter, audio->downsample_filter, STEREO, length,
        channels), error, error_size, "AudioVAE alias-free SnakeBeta");
}

static int run_conv(audio_context *audio, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, const audio_conv *conv,
                    uint32_t length, char *error, size_t error_size) {
    int ok = conv->transpose ? h3_gpu_conv_transpose1d_f32(
        audio->gpu, output, input, conv->weight, conv->bias, STEREO, length,
        conv->input_channels, conv->output_channels, conv->kernel, conv->stride,
        conv->padding) : h3_gpu_conv1d_f32(
        audio->gpu, output, input, conv->weight, conv->bias, STEREO, length,
        conv->input_channels, conv->output_channels, conv->kernel, conv->padding,
        conv->dilation);
    return gpu_op(audio, ok, error, error_size,
                  conv->transpose ? "AudioVAE transposed convolution" :
                                    "AudioVAE residual convolution");
}

static int run_stage(audio_context *audio, audio_stage *stage, int index,
                     char *error, size_t error_size) {
    uint32_t input_length = audio->length;
    uint32_t channels = DECODER_DIM >> (index + 1);
    uint32_t kernel = upsample_kernels[index];
    uint32_t stride = upsample_rates[index];
    uint32_t padding = (kernel - stride) / 2;
    uint64_t expanded = (uint64_t)(input_length - 1) * stride + kernel -
                        2 * padding;
    if (expanded > UINT32_MAX) {
        fail(error, error_size, "AudioVAE stage length overflows");
        return 0;
    }
    uint32_t output_length = (uint32_t)expanded;
    uint64_t elements64 = (uint64_t)STEREO * output_length * channels;
    if (elements64 > UINT32_MAX || elements64 > SIZE_MAX) {
        fail(error, error_size, "AudioVAE stage activation is too large");
        return 0;
    }
    size_t elements = (size_t)elements64;
    h3_gpu_tensor *upsampled = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *sum = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *work = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(audio->gpu, elements);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_f32(audio->gpu, elements);
    int ok = upsampled && sum && work && activated && branch;
    if (!ok) {
        fail(error, error_size, "cannot allocate AudioVAE stage activations: %s",
             h3_gpu_error(audio->gpu));
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE stage") &&
         run_conv(audio, upsampled, audio->hidden, &stage->upsample,
                  input_length, error, error_size);
    for (int block_index = 0; ok && block_index < RESBLOCKS; block_index++) {
        audio_resblock *block = &stage->blocks[block_index];
        h3_gpu_tensor *target = block_index == 0 ? sum : work;
        ok = gpu_op(audio, h3_gpu_copy_f32(audio->gpu, target, 0, upsampled, 0,
                                           elements), error, error_size,
                    "copy AudioVAE residual input");
        for (int pair = 0; ok && pair < RESIDUAL_PAIRS; pair++) {
            ok = run_activation(audio, activated, target,
                                &block->activations[pair * 2], output_length,
                                channels, error, error_size) &&
                 run_conv(audio, branch, activated, &block->convs1[pair],
                          output_length, error, error_size) &&
                 run_activation(audio, activated, branch,
                                &block->activations[pair * 2 + 1],
                                output_length, channels, error, error_size) &&
                 run_conv(audio, branch, activated, &block->convs2[pair],
                          output_length, error, error_size) &&
                 gpu_op(audio, h3_gpu_add_scaled_f32(
                     audio->gpu, target, target, branch, 1.0f, 1.0f,
                     (uint32_t)elements), error, error_size,
                     "AudioVAE residual add");
        }
        if (ok && block_index == 1)
            ok = gpu_op(audio, h3_gpu_add_scaled_f32(
                audio->gpu, sum, sum, work, 1.0f, 1.0f,
                (uint32_t)elements), error, error_size,
                "AudioVAE resblock accumulation");
        if (ok && block_index == 2)
            ok = gpu_op(audio, h3_gpu_add_scaled_f32(
                audio->gpu, sum, sum, work, 1.0f / 3.0f, 1.0f / 3.0f,
                (uint32_t)elements), error, error_size,
                "AudioVAE resblock average");
    }
    if (ok) ok = gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                        "submit AudioVAE stage");
    if (ok) {
        free_tensor(&audio->hidden);
        audio->hidden = sum;
        sum = NULL;
        audio->length = output_length;
    }

done:
    h3_gpu_tensor_free(upsampled);
    h3_gpu_tensor_free(sum);
    h3_gpu_tensor_free(work);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(branch);
    return ok;
}

static int decode_output(audio_context *audio, h3_audio_waveform *output,
                         char *error, size_t error_size) {
    audio_activation activation = {0};
    audio_conv post = {0};
    h3_gpu_tensor *activated = NULL, *waveform = NULL;
    size_t hidden_elements = (size_t)STEREO * audio->length * 8;
    size_t waveform_elements = (size_t)STEREO * audio->length;
    int ok = load_activation(audio, &activation, "decoder.activation_post", 8,
                             error, error_size) &&
             load_normalized_conv(audio, &post, "decoder.conv_post", 8, 1, 7,
                                  3, 1, 1, 0, 0, error, error_size);
    if (!ok) goto done;
    activated = h3_gpu_tensor_new_f32(audio->gpu, hidden_elements);
    waveform = h3_gpu_tensor_new_f32(audio->gpu, waveform_elements);
    if (!activated || !waveform || waveform_elements > UINT32_MAX) {
        fail(error, error_size, "cannot allocate AudioVAE output: %s",
             h3_gpu_error(audio->gpu));
        ok = 0;
        goto done;
    }
    ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                "begin AudioVAE output") &&
         normalize_conv(audio, &post, error, error_size) &&
         run_activation(audio, activated, audio->hidden, &activation,
                        audio->length, 8, error, error_size) &&
         run_conv(audio, waveform, activated, &post, audio->length,
                  error, error_size) &&
         gpu_op(audio, h3_gpu_clip_f32(audio->gpu, waveform, waveform,
                                       (uint32_t)waveform_elements,
                                       -1.0f, 1.0f), error, error_size,
                "AudioVAE output clip") &&
         gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
                "submit AudioVAE output");
    if (!ok) goto done;
    output->pcm = malloc(waveform_elements * sizeof(*output->pcm));
    if (!output->pcm || !h3_gpu_tensor_read_f32(
            waveform, output->pcm, waveform_elements)) {
        fail(error, error_size, "cannot read AudioVAE waveform");
        free(output->pcm);
        output->pcm = NULL;
        ok = 0;
        goto done;
    }
    output->channels = STEREO;
    output->samples = (int)audio->length;
    output->sample_rate = SAMPLE_RATE;

done:
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(waveform);
    free_activation(&activation);
    free_conv(&post);
    return ok;
}

int h3_audio_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_length,
                        h3_audio_vae_progress progress, void *progress_opaque,
                        h3_audio_waveform *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path || !normalized_latent || !output ||
        latent_length < 1 || latent_length > INT32_MAX / HOP_LENGTH) {
        fail(error, error_size, "invalid AudioVAE decode arguments");
        return 0;
    }
    audio_context audio = {0};
    audio.length = (uint32_t)latent_length;
    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, mean, deviation,
                                   error, error_size)) return 0;
    audio.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!audio.gpu) return 0;
    h3_gpu_profile_set_label(audio.gpu, "audio VAE decoder");
    audio.weights = h3_weight_store_open(weight_directory, error, error_size);
    int ok = audio.weights && load_filters(&audio, error, error_size) &&
             prepare_input(&audio, normalized_latent, mean, deviation,
                           error, error_size);
    for (int index = 0; ok && index < STAGES; index++) {
        audio_stage stage = {0};
        ok = load_stage(&audio, &stage, index, error, error_size) &&
             normalize_stage(&audio, &stage, error, error_size) &&
             run_stage(&audio, &stage, index, error, error_size);
        free_stage(&stage);
        if (ok && progress) progress(index + 1, STAGES, progress_opaque);
    }
    if (ok) ok = audio.length == (uint32_t)latent_length * HOP_LENGTH;
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "AudioVAE produced the wrong sample count");
    if (ok) ok = decode_output(&audio, output, error, error_size);
    if (ok && !h3_gpu_get_stats(audio.gpu, &output->gpu_stats)) {
        fail(error, error_size, "cannot read AudioVAE GPU statistics");
        ok = 0;
    }
    cleanup(&audio);
    if (!ok) h3_audio_waveform_free(output);
    return ok;
}

void h3_audio_waveform_free(h3_audio_waveform *waveform) {
    if (!waveform) return;
    free(waveform->pcm);
    memset(waveform, 0, sizeof(*waveform));
}

enum {
    ENCODER_STAGES = 5,
    ENCODER_RESIDUALS = 3,
    ENCODER_HEADS = 8,
    ENCODER_DIM = 64
};

static const uint32_t encoder_strides[ENCODER_STAGES] = {2, 4, 4, 5, 5};
static const uint32_t encoder_dilations[ENCODER_RESIDUALS] = {1, 3, 9};

static void encoder_trace(audio_context *audio, const char *label,
                          const h3_gpu_tensor *tensor, size_t elements) {
    (void)audio;
    if (!getenv("H3_AUDIO_ENCODER_TRACE") || !tensor || !elements) return;
    float *values = malloc(elements * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_f32(tensor, values, elements)) {
        free(values);
        return;
    }
    double sum = 0.0, square = 0.0;
    for (size_t index = 0; index < elements; index++) {
        sum += values[index];
        square += (double)values[index] * values[index];
    }
    fprintf(stderr, "audio trace %-12s n=%zu mean=%.9g rms=%.9g first=",
            label, elements, sum / (double)elements,
            sqrt(square / (double)elements));
    for (size_t index = 0; index < (elements < 6 ? elements : 6); index++)
        fprintf(stderr, " % .7g", values[index]);
    fputc('\n', stderr);
    free(values);
}

static int encoder_conv_length(uint32_t length, uint32_t kernel,
                               uint32_t stride, uint32_t padding,
                               uint32_t dilation, uint32_t *result) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    uint64_t padded = (uint64_t)length + 2 * padding;
    if (!result || !stride || padded < effective) return 0;
    uint64_t output = (padded - effective) / stride + 1;
    if (!output || output > UINT32_MAX) return 0;
    *result = (uint32_t)output;
    return 1;
}

static h3_gpu_tensor *encoder_alpha(audio_context *audio, const char *name,
                                    uint32_t channels, char *error,
                                    size_t error_size) {
    return f3(audio, name, 1, channels, 1, error, error_size);
}

static int encoder_initial(audio_context *audio, const float *pcm, int samples,
                           char *error, size_t error_size) {
    uint32_t padded = (uint32_t)(((uint64_t)(uint32_t)samples +
                                  HOP_LENGTH - 1) / HOP_LENGTH * HOP_LENGTH);
    size_t input_elements = (size_t)STEREO * padded;
    float *input_rows = calloc(input_elements, sizeof(*input_rows));
    if (!input_rows) {
        fail(error, error_size, "out of memory padding reference audio");
        return 0;
    }
    for (int channel = 0; channel < STEREO; channel++)
        memcpy(input_rows + (size_t)channel * padded,
               pcm + (size_t)channel * (size_t)samples,
               (size_t)samples * sizeof(*pcm));
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        audio->gpu, input_rows, input_elements);
    free(input_rows);
    audio_conv conv = {0};
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        audio->gpu, (size_t)STEREO * padded * ENCODER_DIM);
    int ok = input && hidden && load_normalized_conv(
        audio, &conv, "encoder.block.0", 1, ENCODER_DIM, 7, 3, 1, 1, 0, 1,
        error, error_size);
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "cannot allocate AudioVAE encoder input: %s",
             h3_gpu_error(audio->gpu));
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE encoder input") &&
        normalize_conv(audio, &conv, error, error_size) &&
        gpu_op(audio, h3_gpu_conv1d_f32(
            audio->gpu, hidden, input, conv.weight, conv.bias, STEREO, padded,
            1, ENCODER_DIM, 7, 3, 1), error, error_size,
            "AudioVAE encoder input convolution") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE encoder input");
    if (ok) {
        audio->hidden = hidden;
        audio->length = padded;
        hidden = NULL;
        encoder_trace(audio, "initial", audio->hidden,
                      (size_t)STEREO * padded * ENCODER_DIM);
    }
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(hidden);
    free_conv(&conv);
    return ok;
}

static int encoder_residual(audio_context *audio, int stage, int residual,
                            uint32_t channels, char *error,
                            size_t error_size) {
    char name[224];
    h3_gpu_tensor *alpha1 = NULL, *alpha2 = NULL;
    h3_gpu_tensor *activated = NULL, *branch = NULL;
    audio_conv conv1 = {0}, conv2 = {0};
    uint32_t dilation = encoder_dilations[residual];
    snprintf(name, sizeof(name),
             "encoder.block.%d.block.%d.block.0.alpha", stage, residual);
    alpha1 = encoder_alpha(audio, name, channels, error, error_size);
    snprintf(name, sizeof(name),
             "encoder.block.%d.block.%d.block.1", stage, residual);
    int ok = alpha1 && load_normalized_conv(
        audio, &conv1, name, channels, channels, 7, 3 * dilation, dilation,
        1, 0, 1, error, error_size);
    snprintf(name, sizeof(name),
             "encoder.block.%d.block.%d.block.2.alpha", stage, residual);
    if (ok) alpha2 = encoder_alpha(audio, name, channels, error, error_size);
    snprintf(name, sizeof(name),
             "encoder.block.%d.block.%d.block.3", stage, residual);
    if (ok) ok = alpha2 && load_normalized_conv(
        audio, &conv2, name, channels, channels, 1, 0, 1, 1, 0, 1,
        error, error_size);
    size_t elements = (size_t)STEREO * audio->length * channels;
    if (ok) {
        activated = h3_gpu_tensor_new_f32(audio->gpu, elements);
        branch = h3_gpu_tensor_new_f32(audio->gpu, elements);
        if (!activated || !branch) {
            fail(error, error_size,
                 "cannot allocate AudioVAE encoder residual activations: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE encoder residual") &&
        normalize_conv(audio, &conv1, error, error_size) &&
        normalize_conv(audio, &conv2, error, error_size) &&
        gpu_op(audio, h3_gpu_snake1d_f32(
            audio->gpu, activated, audio->hidden, alpha1, STEREO,
            audio->length, channels), error, error_size,
            "AudioVAE encoder Snake1d") &&
        gpu_op(audio, h3_gpu_conv1d_f32(
            audio->gpu, branch, activated, conv1.weight, conv1.bias, STEREO,
            audio->length, channels, channels, 7, 3 * dilation, dilation),
            error, error_size, "AudioVAE encoder dilated convolution") &&
        gpu_op(audio, h3_gpu_snake1d_f32(
            audio->gpu, activated, branch, alpha2, STEREO, audio->length,
            channels), error, error_size, "AudioVAE encoder Snake1d") &&
        gpu_op(audio, h3_gpu_conv1d_f32(
            audio->gpu, branch, activated, conv2.weight, conv2.bias, STEREO,
            audio->length, channels, channels, 1, 0, 1), error, error_size,
            "AudioVAE encoder pointwise convolution") &&
        gpu_op(audio, h3_gpu_add_scaled_f32(
            audio->gpu, audio->hidden, audio->hidden, branch, 1.0f, 1.0f,
            (uint32_t)elements), error, error_size,
            "AudioVAE encoder residual add") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE encoder residual");
    h3_gpu_tensor_free(alpha1);
    h3_gpu_tensor_free(alpha2);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(branch);
    free_conv(&conv1);
    free_conv(&conv2);
    return ok;
}

static int encoder_downsample(audio_context *audio, int stage,
                              uint32_t channels, char *error,
                              size_t error_size) {
    char name[192];
    h3_gpu_tensor *alpha = NULL, *activated = NULL, *down = NULL;
    audio_conv conv = {0};
    uint32_t stride = encoder_strides[stage - 1];
    uint32_t kernel = stride * 2;
    uint32_t padding = (stride + 1) / 2;
    uint32_t output_length = 0;
    snprintf(name, sizeof(name), "encoder.block.%d.block.3.alpha", stage);
    alpha = encoder_alpha(audio, name, channels, error, error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.4", stage);
    int ok = alpha && load_normalized_conv(
        audio, &conv, name, channels, channels * 2, kernel, padding, 1,
        stride, 0, 1, error, error_size) &&
        encoder_conv_length(audio->length, kernel, stride, padding, 1,
                            &output_length);
    size_t input_elements = (size_t)STEREO * audio->length * channels;
    size_t output_elements = (size_t)STEREO * output_length * channels * 2;
    if (ok) {
        activated = h3_gpu_tensor_new_f32(audio->gpu, input_elements);
        down = h3_gpu_tensor_new_f32(audio->gpu, output_elements);
        if (!activated || !down) {
            fail(error, error_size,
                 "cannot allocate AudioVAE encoder downsample: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE encoder downsample") &&
        normalize_conv(audio, &conv, error, error_size) &&
        gpu_op(audio, h3_gpu_snake1d_f32(
            audio->gpu, activated, audio->hidden, alpha, STEREO,
            audio->length, channels), error, error_size,
            "AudioVAE encoder downsample Snake1d") &&
        gpu_op(audio, h3_gpu_conv1d_stride_f32(
            audio->gpu, down, activated, conv.weight, conv.bias, STEREO,
            audio->length, channels, channels * 2, kernel, stride, padding, 1),
            error, error_size, "AudioVAE encoder strided convolution") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE encoder downsample");
    if (ok) {
        free_tensor(&audio->hidden);
        audio->hidden = down;
        down = NULL;
        audio->length = output_length;
        char label[32];
        snprintf(label, sizeof(label), "down%d", stage);
        encoder_trace(audio, label, audio->hidden, output_elements);
    }
    h3_gpu_tensor_free(alpha);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(down);
    free_conv(&conv);
    return ok;
}

static int encoder_final_conv(audio_context *audio, char *error,
                              size_t error_size) {
    h3_gpu_tensor *alpha = encoder_alpha(
        audio, "encoder.block.6.alpha", LATENT_DIM, error, error_size);
    audio_conv conv = {0};
    h3_gpu_tensor *activated = NULL, *hidden = NULL;
    int ok = alpha && load_normalized_conv(
        audio, &conv, "encoder.block.7", LATENT_DIM, LATENT_DIM, 3, 1, 1,
        1, 0, 1, error, error_size);
    size_t elements = (size_t)STEREO * audio->length * LATENT_DIM;
    if (ok) {
        activated = h3_gpu_tensor_new_f32(audio->gpu, elements);
        hidden = h3_gpu_tensor_new_f32(audio->gpu, elements);
        if (!activated || !hidden) {
            fail(error, error_size,
                 "cannot allocate AudioVAE encoder final activations: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE encoder final convolution") &&
        normalize_conv(audio, &conv, error, error_size) &&
        gpu_op(audio, h3_gpu_snake1d_f32(
            audio->gpu, activated, audio->hidden, alpha, STEREO,
            audio->length, LATENT_DIM), error, error_size,
            "AudioVAE encoder final Snake1d") &&
        gpu_op(audio, h3_gpu_conv1d_f32(
            audio->gpu, hidden, activated, conv.weight, conv.bias, STEREO,
            audio->length, LATENT_DIM, LATENT_DIM, 3, 1, 1), error,
            error_size, "AudioVAE encoder final convolution") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE encoder final convolution");
    if (ok) {
        free_tensor(&audio->hidden);
        audio->hidden = hidden;
        hidden = NULL;
        encoder_trace(audio, "finalconv", audio->hidden, elements);
    }
    h3_gpu_tensor_free(alpha);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(hidden);
    free_conv(&conv);
    return ok;
}

static int load_norm(audio_context *audio, const char *prefix, uint32_t width,
                     h3_gpu_tensor **weight, h3_gpu_tensor **bias,
                     char *error, size_t error_size) {
    char name[160];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    *weight = f1(audio, name, width, error, error_size);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    *bias = f1(audio, name, width, error, error_size);
    return *weight && *bias;
}

static int load_linear(audio_context *audio, const char *prefix,
                       uint32_t input_dim, uint32_t output_dim,
                       h3_gpu_tensor **weight, h3_gpu_tensor **bias,
                       int has_bias, char *error, size_t error_size) {
    char name[160];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    *weight = f2(audio, name, output_dim, input_dim, error, error_size);
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        *bias = f1(audio, name, output_dim, error, error_size);
    }
    return *weight && (!has_bias || *bias);
}

static int encoder_projection_branch(audio_context *audio,
                                     h3_gpu_tensor **base,
                                     char *error, size_t error_size) {
    h3_gpu_tensor *norm_w = NULL, *norm_b = NULL;
    h3_gpu_tensor *proj_w = NULL, *proj_b = NULL;
    h3_gpu_tensor *normalized = NULL, *projected = NULL;
    uint32_t rows = STEREO * audio->length;
    int ok = load_norm(audio, "pre_block.norm3", LATENT_DIM,
                       &norm_w, &norm_b, error, error_size) &&
        load_linear(audio, "pre_block.proj", LATENT_DIM, LATENT_CHANNELS,
                    &proj_w, &proj_b, 1, error, error_size);
    if (ok) {
        normalized = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_DIM);
        projected = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        if (!normalized || !projected) {
            fail(error, error_size,
                 "cannot allocate AudioVAE projection branch: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE projection branch") &&
        gpu_op(audio, h3_gpu_layer_norm_f32(
            audio->gpu, normalized, audio->hidden, norm_w, norm_b, rows,
            LATENT_DIM, 1e-5f), error, error_size,
            "AudioVAE projection LayerNorm") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, projected, normalized, proj_w, proj_b, rows,
            LATENT_DIM, LATENT_CHANNELS), error, error_size,
            "AudioVAE projection linear") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE projection branch");
    if (ok) {
        *base = projected;
        projected = NULL;
        encoder_trace(audio, "base", *base,
                      (size_t)rows * LATENT_CHANNELS);
    }
    h3_gpu_tensor_free(norm_w); h3_gpu_tensor_free(norm_b);
    h3_gpu_tensor_free(proj_w); h3_gpu_tensor_free(proj_b);
    h3_gpu_tensor_free(normalized); h3_gpu_tensor_free(projected);
    return ok;
}

static int encoder_attention_branch(audio_context *audio,
                                    h3_gpu_tensor *base,
                                    char *error, size_t error_size) {
    h3_gpu_tensor *norm_w = NULL, *norm_b = NULL, *qkv_w = NULL;
    h3_gpu_tensor *unused_bias = NULL, *q_bias = NULL, *k_bias = NULL;
    h3_gpu_tensor *v_bias = NULL, *proj_w = NULL, *proj_b = NULL;
    h3_gpu_tensor *normalized = NULL, *qkv = NULL, *query = NULL;
    h3_gpu_tensor *key = NULL, *value = NULL, *attended = NULL;
    h3_gpu_tensor *pooled = NULL, *projected = NULL;
    uint32_t rows = STEREO * audio->length;
    size_t attention_elements = (size_t)rows * LATENT_DIM;
    int ok = load_norm(audio, "pre_block.norm1", LATENT_DIM,
                       &norm_w, &norm_b, error, error_size) &&
        load_linear(audio, "pre_block.attn.qkv", LATENT_DIM, LATENT_DIM * 3,
                    &qkv_w, &unused_bias, 0, error, error_size);
    if (ok) q_bias = f1(audio, "pre_block.attn.q_bias", LATENT_DIM,
                        error, error_size);
    if (ok) k_bias = f1(audio, "pre_block.attn.zero_k_bias", LATENT_DIM,
                        error, error_size);
    if (ok) v_bias = f1(audio, "pre_block.attn.v_bias", LATENT_DIM,
                        error, error_size);
    if (ok) ok = q_bias && k_bias && v_bias && load_linear(
        audio, "pre_block.attn.proj", LATENT_CHANNELS, LATENT_CHANNELS,
        &proj_w, &proj_b, 1, error, error_size);
    if (ok) {
        normalized = h3_gpu_tensor_new_f32(audio->gpu, attention_elements);
        qkv = h3_gpu_tensor_new_f32(audio->gpu, attention_elements * 3);
        query = h3_gpu_tensor_new_f32(audio->gpu, attention_elements);
        key = h3_gpu_tensor_new_f32(audio->gpu, attention_elements);
        value = h3_gpu_tensor_new_f32(audio->gpu, attention_elements);
        attended = h3_gpu_tensor_new_f32(audio->gpu, attention_elements);
        pooled = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        projected = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        if (!normalized || !qkv || !query || !key || !value || !attended ||
            !pooled || !projected) {
            fail(error, error_size,
                 "cannot allocate AudioVAE attention branch: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE attention branch") &&
        gpu_op(audio, h3_gpu_layer_norm_f32(
            audio->gpu, normalized, audio->hidden, norm_w, norm_b, rows,
            LATENT_DIM, 1e-5f), error, error_size,
            "AudioVAE attention LayerNorm") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, qkv, normalized, qkv_w, NULL, rows, LATENT_DIM,
            LATENT_DIM * 3), error, error_size, "AudioVAE attention QKV") &&
        gpu_op(audio, h3_gpu_audio_qkv_split_f32(
            audio->gpu, query, key, value, qkv, q_bias, k_bias, v_bias,
            STEREO, audio->length, ENCODER_HEADS,
            LATENT_DIM / ENCODER_HEADS), error, error_size,
            "AudioVAE attention QKV split") &&
        gpu_op(audio, h3_gpu_sdpa_causal_f32(
            audio->gpu, attended, query, key, value, STEREO, audio->length,
            ENCODER_HEADS, LATENT_DIM / ENCODER_HEADS,
            1.0f / sqrtf((float)(LATENT_DIM / ENCODER_HEADS))),
            error, error_size, "AudioVAE causal attention") &&
        gpu_op(audio, h3_gpu_audio_attention_pool_f32(
            audio->gpu, pooled, attended, STEREO, audio->length,
            ENCODER_HEADS, LATENT_DIM / ENCODER_HEADS, LATENT_CHANNELS),
            error, error_size, "AudioVAE attention pooling") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, projected, pooled, proj_w, proj_b, rows,
            LATENT_CHANNELS, LATENT_CHANNELS), error, error_size,
            "AudioVAE attention output projection") &&
        gpu_op(audio, h3_gpu_add_scaled_f32(
            audio->gpu, base, base, projected, 1.0f, 1.0f,
            rows * LATENT_CHANNELS), error, error_size,
            "AudioVAE projection branch add") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE attention branch");
    if (ok) encoder_trace(audio, "base+attn", base,
                          (size_t)rows * LATENT_CHANNELS);
    h3_gpu_tensor_free(norm_w); h3_gpu_tensor_free(norm_b);
    h3_gpu_tensor_free(qkv_w); h3_gpu_tensor_free(q_bias);
    h3_gpu_tensor_free(k_bias); h3_gpu_tensor_free(v_bias);
    h3_gpu_tensor_free(proj_w); h3_gpu_tensor_free(proj_b);
    h3_gpu_tensor_free(normalized); h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value); h3_gpu_tensor_free(attended);
    h3_gpu_tensor_free(pooled); h3_gpu_tensor_free(projected);
    return ok;
}

static int encoder_mlp(audio_context *audio, h3_gpu_tensor *base,
                       char *error, size_t error_size) {
    h3_gpu_tensor *n2w = NULL, *n2b = NULL, *nmw = NULL, *nmb = NULL;
    h3_gpu_tensor *w0 = NULL, *b0 = NULL, *w1 = NULL, *b1 = NULL;
    h3_gpu_tensor *w2 = NULL, *b2 = NULL;
    h3_gpu_tensor *norm2 = NULL, *normm = NULL, *gate = NULL, *linear = NULL;
    h3_gpu_tensor *geglu = NULL, *branch = NULL;
    uint32_t rows = STEREO * audio->length;
    int ok = load_norm(audio, "pre_block.norm2", LATENT_CHANNELS,
                       &n2w, &n2b, error, error_size) &&
        load_norm(audio, "pre_block.mlp.norm", LATENT_CHANNELS,
                  &nmw, &nmb, error, error_size) &&
        load_linear(audio, "pre_block.mlp.w0", LATENT_CHANNELS,
                    LATENT_CHANNELS * 2, &w0, &b0, 1, error, error_size) &&
        load_linear(audio, "pre_block.mlp.w1", LATENT_CHANNELS,
                    LATENT_CHANNELS * 2, &w1, &b1, 1, error, error_size) &&
        load_linear(audio, "pre_block.mlp.w2", LATENT_CHANNELS * 2,
                    LATENT_CHANNELS, &w2, &b2, 1, error, error_size);
    if (ok) {
        norm2 = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        normm = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        gate = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS * 2);
        linear = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS * 2);
        geglu = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS * 2);
        branch = h3_gpu_tensor_new_f32(
            audio->gpu, (size_t)rows * LATENT_CHANNELS);
        if (!norm2 || !normm || !gate || !linear || !geglu || !branch) {
            fail(error, error_size, "cannot allocate AudioVAE MLP: %s",
                 h3_gpu_error(audio->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE MLP") &&
        gpu_op(audio, h3_gpu_layer_norm_f32(
            audio->gpu, norm2, base, n2w, n2b, rows, LATENT_CHANNELS, 1e-5f),
            error, error_size, "AudioVAE pre-MLP LayerNorm") &&
        gpu_op(audio, h3_gpu_layer_norm_f32(
            audio->gpu, normm, norm2, nmw, nmb, rows, LATENT_CHANNELS, 1e-5f),
            error, error_size, "AudioVAE MLP LayerNorm") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, gate, normm, w0, b0, rows, LATENT_CHANNELS,
            LATENT_CHANNELS * 2), error, error_size, "AudioVAE MLP gate") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, linear, normm, w1, b1, rows, LATENT_CHANNELS,
            LATENT_CHANNELS * 2), error, error_size, "AudioVAE MLP linear") &&
        gpu_op(audio, h3_gpu_geglu_f32(
            audio->gpu, geglu, gate, linear, rows * LATENT_CHANNELS * 2),
            error, error_size, "AudioVAE GeGLU") &&
        gpu_op(audio, h3_gpu_linear_f32(
            audio->gpu, branch, geglu, w2, b2, rows, LATENT_CHANNELS * 2,
            LATENT_CHANNELS), error, error_size, "AudioVAE MLP output") &&
        gpu_op(audio, h3_gpu_add_scaled_f32(
            audio->gpu, base, base, branch, 1.0f, 1.0f,
            rows * LATENT_CHANNELS), error, error_size,
            "AudioVAE MLP residual") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE MLP");
    if (ok) encoder_trace(audio, "base+mlp", base,
                          (size_t)rows * LATENT_CHANNELS);
    h3_gpu_tensor_free(n2w); h3_gpu_tensor_free(n2b);
    h3_gpu_tensor_free(nmw); h3_gpu_tensor_free(nmb);
    h3_gpu_tensor_free(w0); h3_gpu_tensor_free(b0);
    h3_gpu_tensor_free(w1); h3_gpu_tensor_free(b1);
    h3_gpu_tensor_free(w2); h3_gpu_tensor_free(b2);
    h3_gpu_tensor_free(norm2); h3_gpu_tensor_free(normm);
    h3_gpu_tensor_free(gate); h3_gpu_tensor_free(linear);
    h3_gpu_tensor_free(geglu); h3_gpu_tensor_free(branch);
    return ok;
}

static int encoder_output(audio_context *audio, h3_gpu_tensor *base,
                          const float *mean, const float *deviation,
                          h3_audio_latent *output,
                          char *error, size_t error_size) {
    audio_conv conv = {0};
    size_t elements = (size_t)STEREO * audio->length * LATENT_CHANNELS;
    h3_gpu_tensor *latent = h3_gpu_tensor_new_f32(audio->gpu, elements);
    int ok = latent && load_plain_conv(
        audio, &conv, "mean_proj", LATENT_CHANNELS, LATENT_CHANNELS, 1, 0, 1,
        1, error, error_size);
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "cannot allocate AudioVAE latent output: %s",
             h3_gpu_error(audio->gpu));
    if (ok) ok = gpu_op(audio, h3_gpu_begin(audio->gpu), error, error_size,
                        "begin AudioVAE mean projection") &&
        gpu_op(audio, h3_gpu_conv1d_f32(
            audio->gpu, latent, base, conv.weight, conv.bias, STEREO,
            audio->length, LATENT_CHANNELS, LATENT_CHANNELS, 1, 0, 1),
            error, error_size, "AudioVAE mean projection") &&
        gpu_op(audio, h3_gpu_submit(audio->gpu), error, error_size,
               "submit AudioVAE mean projection");
    float *rows = NULL;
    if (ok) {
        rows = malloc(elements * sizeof(*rows));
        output->values = malloc(elements * sizeof(*output->values));
        if (!rows || !output->values ||
            !h3_gpu_tensor_read_f32(latent, rows, elements)) {
            fail(error, error_size, "cannot read AudioVAE encoded latent");
            ok = 0;
        }
    }
    if (ok) {
        for (int channel = 0; channel < LATENT_CHANNELS; channel++)
            for (int stereo = 0; stereo < STEREO; stereo++)
                for (uint32_t time = 0; time < audio->length; time++) {
                    size_t source = ((size_t)stereo * audio->length + time) *
                                    LATENT_CHANNELS + (size_t)channel;
                    size_t destination = ((size_t)channel * STEREO +
                                          (size_t)stereo) * audio->length + time;
                    output->values[destination] =
                        (rows[source] - mean[channel]) / deviation[channel];
                }
        output->channels = LATENT_CHANNELS;
        output->stereo = STEREO;
        output->length = (int)audio->length;
    }
    free(rows);
    h3_gpu_tensor_free(latent);
    free_conv(&conv);
    return ok;
}

int h3_audio_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pcm, int samples,
                        h3_audio_vae_progress progress, void *progress_opaque,
                        h3_audio_latent *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path || !pcm || !output || samples < 1 ||
        samples > SAMPLE_RATE * 15) {
        fail(error, error_size, "invalid AudioVAE encode arguments");
        return 0;
    }
    audio_context audio = {0};
    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, mean, deviation,
                                   error, error_size)) return 0;
    audio.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!audio.gpu) return 0;
    h3_gpu_profile_set_label(audio.gpu, "audio VAE encoder");
    audio.weights = h3_weight_store_open(weight_directory, error, error_size);
    int ok = audio.weights && encoder_initial(&audio, pcm, samples,
                                               error, error_size);
    uint32_t channels = ENCODER_DIM;
    for (int stage = 1; ok && stage <= ENCODER_STAGES; stage++) {
        for (int residual = 0; ok && residual < ENCODER_RESIDUALS; residual++)
            ok = encoder_residual(&audio, stage, residual, channels,
                                  error, error_size);
        if (ok) ok = encoder_downsample(&audio, stage, channels,
                                        error, error_size);
        channels *= 2;
        if (ok && progress) progress(stage, ENCODER_STAGES + 2,
                                     progress_opaque);
    }
    if (ok && (channels != LATENT_DIM ||
               audio.length != (uint32_t)((samples + HOP_LENGTH - 1) /
                                           HOP_LENGTH))) {
        fail(error, error_size, "AudioVAE encoder produced invalid geometry");
        ok = 0;
    }
    if (ok) ok = encoder_final_conv(&audio, error, error_size);
    if (ok && progress) progress(ENCODER_STAGES + 1, ENCODER_STAGES + 2,
                                 progress_opaque);
    h3_gpu_tensor *base = NULL;
    if (ok) ok = encoder_projection_branch(&audio, &base,
                                            error, error_size) &&
                 encoder_attention_branch(&audio, base,
                                           error, error_size) &&
                 encoder_mlp(&audio, base, error, error_size) &&
                 encoder_output(&audio, base, mean, deviation, output,
                                error, error_size);
    h3_gpu_tensor_free(base);
    if (ok && !h3_gpu_get_stats(audio.gpu, &output->gpu_stats)) {
        fail(error, error_size, "cannot read AudioVAE encoder GPU statistics");
        ok = 0;
    }
    if (ok && progress) progress(ENCODER_STAGES + 2, ENCODER_STAGES + 2,
                                 progress_opaque);
    cleanup(&audio);
    if (!ok) h3_audio_latent_free(output);
    return ok;
}

void h3_audio_latent_free(h3_audio_latent *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}
