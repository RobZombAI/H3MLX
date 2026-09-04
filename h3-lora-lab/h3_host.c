#include "h3_host.h"

#include <Accelerate/Accelerate.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int h3_frame_per_token[5] = {1, 4, 4, 4, 4};
static const double h3_frame_rescale = 5.0 / 3.0;

int h3_align_frame_count(int requested) {
    int value = requested < 5 ? 5 : requested;
    int remainder = (value - 5) % 17;
    if (remainder < 0) remainder += 17;
    if (remainder != 0) value += 17 - remainder;
    return value;
}

int h3_video_latent_t(int frame_count) {
    if (frame_count <= 5) return 2;
    return ((frame_count - 5) / 17) * 5 + 2;
}

int h3_video_encoder_latent_t(int frame_count) {
    return frame_count > 0 ? (frame_count + 3) / 4 : 0;
}

h3_temporal_shape h3_temporal(int requested_frames) {
    h3_temporal_shape result;
    result.frame_count = h3_align_frame_count(requested_frames);
    result.video_t = h3_video_latent_t(result.frame_count);
    result.audio_t = (int)llround((double)result.frame_count *
                                  H3_AUDIO_LATENT_FPS / H3_FPS);
    return result;
}

void h3_latent_canvas(int width, int height, int *latent_w, int *latent_h) {
    if (latent_w) *latent_w = width / H3_VAE_SPATIAL_RATIO;
    if (latent_h) *latent_h = height / H3_VAE_SPATIAL_RATIO;
}

int h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h) {
    if (width <= 0 || height <= 0 || !adapted_w || !adapted_h) return 0;
    double ratio = (double)width / (double)height;
    double nominal_w;
    double nominal_h;
    if (ratio >= 1.0) {
        nominal_w = 768.0 * ratio;
        nominal_h = 768.0;
    } else {
        nominal_w = 768.0;
        nominal_h = 768.0 / ratio;
    }
    double pixels = nominal_w * nominal_h;
    if (pixels > H3_MAX_PIXELS) {
        double scale = sqrt((double)H3_MAX_PIXELS / pixels);
        nominal_w *= scale;
        nominal_h *= scale;
    }
    int out_w = (int)(nearbyint(nominal_w / H3_CANVAS_MULTIPLE) *
                      H3_CANVAS_MULTIPLE);
    int out_h = (int)(nearbyint(nominal_h / H3_CANVAS_MULTIPLE) *
                      H3_CANVAS_MULTIPLE);
    *adapted_w = out_w < H3_CANVAS_MULTIPLE ? H3_CANVAS_MULTIPLE : out_w;
    *adapted_h = out_h < H3_CANVAS_MULTIPLE ? H3_CANVAS_MULTIPLE : out_h;
    return 1;
}

int h3_reference_image_canvas(int width, int height,
                              int target_width, int target_height,
                              int max_short_edge,
                              int *adapted_w, int *adapted_h) {
    if (width < 1 || height < 1 || target_width < 1 || target_height < 1 ||
        max_short_edge < 0 || !adapted_w || !adapted_h) return 0;
    double scale;
    if (max_short_edge) {
        int short_edge = width < height ? width : height;
        scale = fmin(1.0, (double)max_short_edge / (double)short_edge);
    } else {
        double target_area = (double)target_width * (double)target_height;
        double source_area = (double)width * (double)height;
        scale = fmin(1.0, sqrt(target_area / source_area));
    }
    double out_w = nearbyint((double)width * scale / H3_CANVAS_MULTIPLE) *
                   H3_CANVAS_MULTIPLE;
    double out_h = nearbyint((double)height * scale / H3_CANVAS_MULTIPLE) *
                   H3_CANVAS_MULTIPLE;
    if (out_w < H3_CANVAS_MULTIPLE) out_w = H3_CANVAS_MULTIPLE;
    if (out_h < H3_CANVAS_MULTIPLE) out_h = H3_CANVAS_MULTIPLE;
    if (out_w > INT_MAX || out_h > INT_MAX) return 0;
    *adapted_w = (int)out_w;
    *adapted_h = (int)out_h;
    return 1;
}

int h3_reference_video_canvas(int width, int height,
                              int *adapted_w, int *adapted_h) {
    if (width < 1 || height < 1 || !adapted_w || !adapted_h ||
        !h3_adapt_canvas(width, height, adapted_w, adapted_h)) return 0;
    double source_area = (double)width * (double)height;
    double target_area = (double)*adapted_w * (double)*adapted_h;
    if (source_area < target_area) {
        int out_w = (int)(nearbyint((double)width / H3_CANVAS_MULTIPLE) *
                          H3_CANVAS_MULTIPLE);
        int out_h = (int)(nearbyint((double)height / H3_CANVAS_MULTIPLE) *
                          H3_CANVAS_MULTIPLE);
        *adapted_w = out_w < H3_CANVAS_MULTIPLE ? H3_CANVAS_MULTIPLE : out_w;
        *adapted_h = out_h < H3_CANVAS_MULTIPLE ? H3_CANVAS_MULTIPLE : out_h;
    }
    return 1;
}

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift) {
    double base = sigma / (from_shift + sigma * (1.0 - from_shift));
    return to_shift * base / (1.0 + (to_shift - 1.0) * base);
}

double h3_time_shift_slope(double sigma, double from_shift, double to_shift) {
    double base = sigma / (from_shift + sigma * (1.0 - from_shift));
    double a = 1.0 + (from_shift - 1.0) * base;
    double b = 1.0 + (to_shift - 1.0) * base;
    return to_shift * a * a / (from_shift * b * b);
}

float h3_get_video_shift(void) {
    const char *env = getenv("H3_VIDEO_SHIFT");
    if (env && *env) {
        float val = (float)atof(env);
        if (val > 0.0f) return val;
    }
    return (float)H3_VIDEO_SIGMA_SHIFT;
}

float h3_get_audio_shift(void) {
    const char *env = getenv("H3_AUDIO_SHIFT");
    if (env && *env) {
        float val = (float)atof(env);
        if (val > 0.0f) return val;
    }
    return (float)H3_AUDIO_SIGMA_SHIFT;
}

static float h3_shifted_sigma(int index, int steps, float shift) {
    if (steps <= 0) return 0.0f;
    float u = (float)index / (float)steps;
    const char *cheb_env = getenv("H3_CHEBYSHEV_WARP");
    const char *frontier_env = getenv("H3_FRONTIER");
    int cheb_warp = (cheb_env && (*cheb_env == '1' || *cheb_env == 'y' || *cheb_env == 't'));
    if (frontier_env && atoi(frontier_env) >= 10) cheb_warp = 1;

    if (cheb_warp) {
        /* Frontier Level 10: Curvature-Adaptive Dual-Cusp Chebyshev Time-Warping (CACFM)
         * Clusters ODE integration points at boundary cusps t in [1.0, 0.85] and t <= 0.15 */
        float gamma = 1.15f;
        float alpha = 0.85f;
        float u_pow = powf(u, gamma);
        float cheb = 0.5f * (1.0f - cosf((float)M_PI * u_pow));
        float t_val = 1.0f - powf(fmaxf(0.0f, fminf(1.0f, cheb)), alpha);
        float base = fmaxf(0.0f, fminf(1.0f, t_val));
        return shift * base / (1.0f + (shift - 1.0f) * base);
    }

    const char *warp_env = getenv("H3_WARP_GAMMA");
    const char *facial_env = getenv("H3_FACIAL_WARP");
    float gamma = (warp_env && *warp_env) ? (float)atof(warp_env) : (1.20f + 0.15f * (1.0f - u));
    int facial_warp = (facial_env && (*facial_env == '1' || *facial_env == 'y' || *facial_env == 't'));
    float base;
    if ((gamma > 0.01f && fabsf(gamma - 1.0f) > 0.001f) || facial_warp) {
        /* Curvature-Aware Cosine-Power schedule warping with Facial Transition Window Retardation */
        float u_warped = (gamma > 0.01f && fabsf(gamma - 1.0f) > 0.001f) ? powf(u, gamma) : u;
        /* Gaussian retardation kernel around critical facial semantic boundary (u ~ 0.50) */
        float retardation = 0.08f * sinf((float)M_PI * 2.0f * u) * expf(-powf(u - 0.50f, 2.0f) / (2.0f * 0.16f * 0.16f));
        float u_focal = fmaxf(0.0f, fminf(1.0f, u_warped - retardation));
        base = cosf((float)M_PI * 0.5f * u_focal);
        base = fmaxf(0.0f, fminf(1.0f, base));
    } else {
        int base_index = (index * 1000) / steps;
        base = (float)(1000 - base_index) / 1000.0f;
    }
    return shift * base / (1.0f + (shift - 1.0f) * base);
}

int h3_schedule_build(int steps, h3_sigma_schedule *schedule) {
    if (!schedule || steps < 1 || steps > H3_MAX_STEPS) return 0;
    memset(schedule, 0, sizeof(*schedule));
    schedule->steps = steps;
    float vshift = h3_get_video_shift();
    float ashift = h3_get_audio_shift();
    for (int index = 0; index < steps; index++) {
        schedule->video[index] = h3_shifted_sigma(index, steps, vshift);
        schedule->audio[index] = h3_shifted_sigma(index, steps, ashift);
        if (index > 0) {
            if (schedule->video[index] >= schedule->video[index - 1]) {
                schedule->video[index] = schedule->video[index - 1] * 0.999f;
            }
            if (schedule->audio[index] >= schedule->audio[index - 1]) {
                schedule->audio[index] = schedule->audio[index - 1] * 0.999f;
            }
        }
    }
    schedule->video[steps] = 0.0f;
    schedule->audio[steps] = 0.0f;
    return 1;
}

int h3_serving_schedule_build(int evaluations, h3_sigma_schedule *schedule) {
    if (!schedule || evaluations < 2 || evaluations > H3_MAX_STEPS) return 0;
    memset(schedule, 0, sizeof(*schedule));
    schedule->steps = evaluations;
    float vshift = h3_get_video_shift();
    float ashift = h3_get_audio_shift();
    for (int index = 0; index <= evaluations; index++) {
        schedule->video[index] = h3_shifted_sigma(index, evaluations, vshift);
        schedule->audio[index] = h3_shifted_sigma(index, evaluations, ashift);
        if (index > 0) {
            if (schedule->video[index] >= schedule->video[index - 1]) {
                schedule->video[index] = schedule->video[index - 1] * 0.999f;
            }
            if (schedule->audio[index] >= schedule->audio[index - 1]) {
                schedule->audio[index] = schedule->audio[index - 1] * 0.999f;
            }
        }
    }
    schedule->video[evaluations] = 0.0f;
    schedule->audio[evaluations] = 0.0f;
    return 1;
}

typedef struct {
    h3_layout *layout;
    size_t position_capacity;
    size_t segment_capacity;
    char *error;
    size_t error_size;
} h3_layout_builder;

static int h3_builder_fail(h3_layout_builder *builder, const char *message) {
    if (builder->error && builder->error_size) {
        snprintf(builder->error, builder->error_size, "%s", message);
    }
    return 0;
}

static int h3_reserve_positions(h3_layout_builder *builder, size_t add) {
    h3_layout *layout = builder->layout;
    if (add > SIZE_MAX - layout->seq_len) {
        return h3_builder_fail(builder, "layout row count overflow");
    }
    size_t wanted = layout->seq_len + add;
    if (wanted <= builder->position_capacity) return 1;
    size_t capacity = builder->position_capacity ? builder->position_capacity : 256;
    while (capacity < wanted) {
        if (capacity > SIZE_MAX / 2) {
            capacity = wanted;
            break;
        }
        capacity *= 2;
    }
    h3_position *positions = realloc(layout->positions,
                                     capacity * sizeof(*positions));
    if (!positions) return h3_builder_fail(builder, "out of memory for positions");
    layout->positions = positions;
    builder->position_capacity = capacity;
    return 1;
}

static int h3_reserve_segment(h3_layout_builder *builder) {
    h3_layout *layout = builder->layout;
    if (layout->segment_count < builder->segment_capacity) return 1;
    size_t capacity = builder->segment_capacity ? builder->segment_capacity * 2 : 16;
    h3_segment *segments = realloc(layout->segments,
                                   capacity * sizeof(*segments));
    if (!segments) return h3_builder_fail(builder, "out of memory for segments");
    layout->segments = segments;
    builder->segment_capacity = capacity;
    return 1;
}

static int h3_emit(h3_layout_builder *builder, h3_segment_kind kind,
                   const h3_position *positions, size_t count) {
    h3_layout *layout = builder->layout;
    if (!h3_reserve_positions(builder, count) || !h3_reserve_segment(builder)) {
        return 0;
    }
    size_t start = layout->seq_len;
    if (count) memcpy(layout->positions + start, positions, count * sizeof(*positions));
    layout->seq_len += count;
    layout->segments[layout->segment_count++] =
        (h3_segment){start, layout->seq_len, kind};
    return 1;
}

static int h3_frame_grid(int latent_h, int latent_w, h3_position **positions,
                         size_t *count, double **w_axis, size_t *w_count) {
    if (latent_h < 2 || latent_w < 2 || latent_h % 2 || latent_w % 2) return 0;
    size_t nh = (size_t)latent_h / 2;
    size_t nw = (size_t)latent_w / 2;
    if (nh > SIZE_MAX / nw) return 0;
    size_t total = nh * nw;
    h3_position *grid = malloc(total * sizeof(*grid));
    double *widths = malloc(nw * sizeof(*widths));
    if (!grid || !widths) {
        free(grid);
        free(widths);
        return 0;
    }
    double area = sqrt((double)latent_h * (double)latent_w);
    double ratio_h = latent_h / area;
    double ratio_w = latent_w / area;
    double step_h = ratio_h / (double)nh;
    double step_w = ratio_w / (double)nw;
    double base_h = (1.0 - ratio_h) / 2.0;
    double base_w = (1.0 - ratio_w) / 2.0;
    for (size_t column = 0; column < nw; column++) {
        widths[column] = ((double)column * step_w + base_w) * 32.0;
    }
    size_t offset = 0;
    for (size_t row = 0; row < nh; row++) {
        double hh = ((double)row * step_h + base_h) * 32.0;
        for (size_t column = 0; column < nw; column++) {
            grid[offset++] = (h3_position){0.0, hh, widths[column]};
        }
    }
    *positions = grid;
    *count = total;
    *w_axis = widths;
    *w_count = nw;
    return 1;
}

static double h3_video_span_sum(int latent_t) {
    double sum = 0.0;
    for (int index = 0; index < latent_t; index++) {
        sum += h3_frame_rescale * h3_frame_per_token[index % 5];
    }
    return sum;
}

static h3_position *h3_audio_grid(double cursor, int audio_t,
                                  double low, double high, size_t *count) {
    if (audio_t < 0 || (size_t)audio_t > SIZE_MAX / (2 * sizeof(h3_position))) {
        return NULL;
    }
    size_t rows = (size_t)audio_t * 2;
    h3_position *grid = malloc((rows ? rows : 1) * sizeof(*grid));
    if (!grid) return NULL;
    for (int index = 0; index < audio_t; index++) {
        grid[index] = (h3_position){cursor + index, 0.0, low};
        grid[(size_t)audio_t + (size_t)index] =
            (h3_position){cursor + index, 0.0, high};
    }
    *count = rows;
    return grid;
}

static h3_position *h3_video_grid(double cursor, int latent_t,
                                  const h3_position *frame, size_t frame_rows,
                                  size_t *count) {
    if (latent_t < 0 || (size_t)latent_t > SIZE_MAX / frame_rows) return NULL;
    size_t rows = (size_t)latent_t * frame_rows;
    h3_position *grid = malloc((rows ? rows : 1) * sizeof(*grid));
    if (!grid) return NULL;
    size_t offset = 0;
    double time = cursor;
    for (int index = 0; index < latent_t; index++) {
        for (size_t spatial = 0; spatial < frame_rows; spatial++) {
            grid[offset++] = (h3_position){time, frame[spatial].h, frame[spatial].w};
        }
        time += h3_frame_rescale * h3_frame_per_token[index % 5];
    }
    *count = rows;
    return grid;
}

int h3_layout_build(const h3_layout_spec *spec, h3_layout *layout,
                    char *error, size_t error_size) {
    if (!spec || !layout) return 0;
    memset(layout, 0, sizeof(*layout));
    if (error && error_size) error[0] = '\0';
    h3_layout_builder builder = {layout, 0, 0, error, error_size};
    if (spec->text_len < 1 || spec->latent_t < 1 || spec->audio_t < 0 ||
        spec->frame_count < 5) {
        return h3_builder_fail(&builder, "invalid target layout dimensions");
    }
    if (spec->keyframe_count && spec->reference_count) {
        return h3_builder_fail(&builder, "keyframes and references are mutually exclusive");
    }

    h3_position *frame = NULL;
    double *w_axis = NULL;
    size_t frame_rows = 0;
    size_t w_count = 0;
    if (!h3_frame_grid(spec->latent_h, spec->latent_w, &frame, &frame_rows,
                       &w_axis, &w_count)) {
        return h3_builder_fail(&builder, "invalid latent canvas or out of memory");
    }

    h3_position *text = malloc((size_t)spec->text_len * sizeof(*text));
    if (!text) goto oom;
    for (int index = 0; index < spec->text_len; index++) {
        text[index] = (h3_position){(double)index, 0.0, 0.0};
    }
    if (!h3_emit(&builder, H3_SEG_TEXT, text, (size_t)spec->text_len)) {
        free(text);
        goto fail;
    }
    free(text);

    double cursor = (double)spec->text_len;
    for (size_t index = 0; index < spec->keyframe_count; index++) {
        double condition_time;
        if (spec->keyframes[index] == 0) {
            condition_time = (double)spec->text_len;
        } else if (spec->keyframes[index] == spec->frame_count - 1) {
            condition_time = (double)spec->text_len +
                             h3_video_span_sum(spec->latent_t) - h3_frame_rescale;
        } else {
            h3_builder_fail(&builder, "only first and last keyframes are valid");
            goto fail;
        }
        for (size_t row = 0; row < frame_rows; row++) frame[row].t = condition_time;
        if (!h3_emit(&builder, H3_SEG_COND, frame, frame_rows)) goto fail;
        layout->img_cond_rows += frame_rows;
    }

    for (size_t index = 0; index < spec->reference_count; index++) {
        const h3_layout_ref *reference = &spec->references[index];
        if (reference->kind == H3_LAYOUT_REF_IMAGE) {
            h3_position *ref_frame = NULL;
            double *ref_w = NULL;
            size_t ref_rows = 0;
            size_t ref_w_count = 0;
            if (!h3_frame_grid(reference->latent_h, reference->latent_w,
                               &ref_frame, &ref_rows, &ref_w, &ref_w_count)) goto oom;
            for (size_t row = 0; row < ref_rows; row++) ref_frame[row].t = cursor;
            int ok = h3_emit(&builder, H3_SEG_REF_IMAGE, ref_frame, ref_rows);
            free(ref_frame);
            free(ref_w);
            if (!ok) goto fail;
            layout->img_cond_rows += ref_rows;
            cursor += 1.0;
        } else if (reference->kind == H3_LAYOUT_REF_AUDIO) {
            size_t rows = 0;
            h3_position *audio = h3_audio_grid(cursor, reference->audio_t,
                                               w_axis[0], w_axis[w_count - 1], &rows);
            if (!audio) goto oom;
            int ok = 1;
            if (rows) ok = h3_emit(&builder, H3_SEG_REF_AUDIO, audio, rows);
            free(audio);
            if (!ok) goto fail;
            layout->audio_cond_rows += rows;
            cursor += reference->audio_t;
        } else if (reference->kind == H3_LAYOUT_REF_VIDEO) {
            h3_position *ref_frame = NULL;
            double *ref_w = NULL;
            size_t ref_rows = 0;
            size_t ref_w_count = 0;
            if (!h3_frame_grid(reference->latent_h, reference->latent_w,
                               &ref_frame, &ref_rows, &ref_w, &ref_w_count)) goto oom;
            if (reference->audio_t > 0) {
                size_t audio_rows = 0;
                h3_position *audio = h3_audio_grid(cursor, reference->audio_t,
                                                   ref_w[0], ref_w[ref_w_count - 1],
                                                   &audio_rows);
                if (!audio) {
                    free(ref_frame);
                    free(ref_w);
                    goto oom;
                }
                int ok = h3_emit(&builder, H3_SEG_REF_AUDIO, audio, audio_rows);
                free(audio);
                if (!ok) {
                    free(ref_frame);
                    free(ref_w);
                    goto fail;
                }
                layout->audio_cond_rows += audio_rows;
            }
            size_t video_rows = 0;
            h3_position *video = h3_video_grid(cursor, reference->latent_t,
                                               ref_frame, ref_rows, &video_rows);
            free(ref_frame);
            free(ref_w);
            if (!video) goto oom;
            int ok = h3_emit(&builder, H3_SEG_REF_IMAGE, video, video_rows);
            free(video);
            if (!ok) goto fail;
            layout->img_cond_rows += video_rows;
            double video_span = h3_video_span_sum(reference->latent_t);
            cursor += fmax((double)reference->audio_t, video_span);
        } else {
            h3_builder_fail(&builder, "unknown reference layout type");
            goto fail;
        }
    }

    size_t audio_rows = 0;
    h3_position *audio = h3_audio_grid(cursor, spec->audio_t,
                                       w_axis[0], w_axis[w_count - 1], &audio_rows);
    if (!audio) goto oom;
    if (!h3_emit(&builder, H3_SEG_AUDIO, audio, audio_rows)) {
        free(audio);
        goto fail;
    }
    free(audio);

    size_t video_rows = 0;
    h3_position *video = h3_video_grid(cursor, spec->latent_t,
                                       frame, frame_rows, &video_rows);
    if (!video) goto oom;
    if (!h3_emit(&builder, H3_SEG_VIDEO, video, video_rows)) {
        free(video);
        goto fail;
    }
    free(video);

    layout->img_target_rows = video_rows;
    layout->audio_target_rows = audio_rows;
    layout->signature[0] = spec->text_len;
    layout->signature[1] = spec->latent_t;
    layout->signature[2] = spec->latent_h;
    layout->signature[3] = spec->latent_w;
    layout->signature[4] = spec->audio_t;
    free(frame);
    free(w_axis);
    return 1;

oom:
    h3_builder_fail(&builder, "out of memory while building layout");
fail:
    free(frame);
    free(w_axis);
    h3_layout_free(layout);
    return 0;
}

void h3_layout_free(h3_layout *layout) {
    if (!layout) return;
    free(layout->segments);
    free(layout->positions);
    memset(layout, 0, sizeof(*layout));
}

const char *h3_segment_name(h3_segment_kind kind) {
    switch (kind) {
        case H3_SEG_TEXT: return "text";
        case H3_SEG_COND: return "cond";
        case H3_SEG_REF_IMAGE: return "ref_img";
        case H3_SEG_REF_AUDIO: return "ref_audio";
        case H3_SEG_AUDIO: return "audio";
        case H3_SEG_VIDEO: return "video";
    }
    return "unknown";
}

uint32_t h3_rng_u32(h3_rng *rng) {
    uint64_t old_state = rng->state;
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    uint32_t shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    return (shifted >> rotation) | (shifted << ((-(int32_t)rotation) & 31));
}

void h3_rng_seed(h3_rng *rng, uint64_t seed) {
    memset(rng, 0, sizeof(*rng));
    rng->increment = (seed << 1u) | 1u;
    (void)h3_rng_u32(rng);
    rng->state += seed ^ UINT64_C(0x9e3779b97f4a7c15);
    (void)h3_rng_u32(rng);
}

float h3_rng_normal(h3_rng *rng) {
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    double u1 = ((double)h3_rng_u32(rng) + 1.0) / 4294967297.0;
    double u2 = ((double)h3_rng_u32(rng) + 0.5) / 4294967296.0;
    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * 3.14159265358979323846 * u2;
    rng->spare = (float)(radius * sin(angle));
    rng->has_spare = 1;
    return (float)(radius * cos(angle));
}

void h3_rng_fill_normal(h3_rng *rng, float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        values[index] = h3_rng_normal(rng);
    }
}

int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output) {
    if (output) *output = NULL;
    if (!input || !output || frames < 1 || input_width < 1 ||
        input_height < 1 || output_width < 1 || output_height < 1)
        return 0;
    size_t input_area = (size_t)input_width * (size_t)input_height;
    size_t output_area = (size_t)output_width * (size_t)output_height;
    if (input_area > SIZE_MAX / 3 || output_area > SIZE_MAX / 3 ||
        (size_t)frames > SIZE_MAX / (output_area * 3)) return 0;
    size_t output_bytes = (size_t)frames * output_area * 3;
    uint8_t *pixels = malloc(output_bytes);
    if (!pixels) return 0;
    if (input_width == output_width && input_height == output_height) {
        if ((size_t)frames > SIZE_MAX / (input_area * 3)) {
            free(pixels);
            return 0;
        }
        memcpy(pixels, input, (size_t)frames * input_area * 3);
        *output = pixels;
        return 1;
    }
    if (input_area > SIZE_MAX / 4 || output_area > SIZE_MAX / 4) {
        free(pixels);
        return 0;
    }
    uint8_t *source_argb = malloc(input_area * 4);
    uint8_t *output_argb = malloc(output_area * 4);
    if (!source_argb || !output_argb) {
        free(source_argb); free(output_argb); free(pixels);
        return 0;
    }
    size_t input_frame_bytes = input_area * 3;
    size_t output_frame_bytes = output_area * 3;
    vImage_Buffer source_buffer = {
        source_argb, (vImagePixelCount)input_height,
        (vImagePixelCount)input_width, (size_t)input_width * 4
    };
    vImage_Buffer output_buffer = {
        output_argb, (vImagePixelCount)output_height,
        (vImagePixelCount)output_width, (size_t)output_width * 4
    };
    for (int frame = 0; frame < frames; frame++) {
        const uint8_t *source_frame = input + (size_t)frame * input_frame_bytes;
        uint8_t *output_frame = pixels + (size_t)frame * output_frame_bytes;
        for (size_t pixel = 0; pixel < input_area; pixel++) {
            source_argb[4 * pixel] = 255;
            source_argb[4 * pixel + 1] = source_frame[3 * pixel];
            source_argb[4 * pixel + 2] = source_frame[3 * pixel + 1];
            source_argb[4 * pixel + 3] = source_frame[3 * pixel + 2];
        }
        if (vImageScale_ARGB8888(
                &source_buffer, &output_buffer, NULL,
                kvImageHighQualityResampling | kvImageEdgeExtend) !=
            kvImageNoError) {
            free(source_argb); free(output_argb); free(pixels);
            return 0;
        }
        for (size_t pixel = 0; pixel < output_area; pixel++) {
            output_frame[3 * pixel] = output_argb[4 * pixel + 1];
            output_frame[3 * pixel + 1] = output_argb[4 * pixel + 2];
            output_frame[3 * pixel + 2] = output_argb[4 * pixel + 3];
        }
    }
    free(source_argb); free(output_argb);
    *output = pixels;
    return 1;
}

static double h3_phi1(double value) {
    return expm1(value) / value;
}

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count,
                const float *sigmas, int step, int total_steps) {
    if (!output || !sample || !denoised || !sigmas || step < 0 ||
        total_steps < 1 || step >= total_steps) return 0;
    double sigma = sigmas[step];
    double next = sigmas[step + 1];
    if (!(sigma > next && next >= 0.0)) return 0;
    if (!old_denoised || next == 0.0) {
        double delta = next - sigma;
        for (size_t index = 0; index < count; index++) {
            double derivative = ((double)sample[index] - denoised[index]) / sigma;
            output[index] = (float)((double)sample[index] + derivative * delta);
        }
        return 1;
    }
    if (step == 0) return 0;
    double t = -log(sigma);
    double t_next = -log(next);
    double t_previous = -log(sigmas[step - 1]);
    double h = t_next - t;
    double c2 = (t_previous - t) / h;
    double phi1 = h3_phi1(-h);
    double phi2 = (phi1 - 1.0) / -h;
    double b1 = phi1 - phi2 / c2;
    double b2 = phi2 / c2;
    double decay = exp(-h);
    for (size_t index = 0; index < count; index++) {
        output[index] = (float)(decay * sample[index] + h *
            (b1 * denoised[index] + b2 * old_denoised[index]));
    }
    return 1;
}

int h3_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next) {
    if (!sample || !velocity || !isfinite(sigma) || !isfinite(sigma_next) ||
        !(sigma > sigma_next) || sigma_next < 0.0f) return 0;
    float delta = sigma - sigma_next;
    for (size_t index = 0; index < count; index++)
        sample[index] += delta * velocity[index];
    return 1;
}

int h3_ab3_assc_velocity_step(float *sample, const float *velocity,
                              const float *prev_velocity, const float *prev2_velocity,
                              size_t count, float sigma, float sigma_next) {
    if (!sample || !velocity || !isfinite(sigma) || !isfinite(sigma_next) ||
        !(sigma > sigma_next) || sigma_next < 0.0f) return 0;
    float delta = sigma - sigma_next;
    if (!prev_velocity || !prev2_velocity) {
        for (size_t index = 0; index < count; index++)
            sample[index] += delta * velocity[index];
        return 1;
    }
    const float c0 = (23.0f / 12.0f) * 0.985f;
    const float c1 = (16.0f / 12.0f) * 0.985f;
    const float c2 = (5.0f / 12.0f) * 0.985f;
    for (size_t index = 0; index < count; index++) {
        float v_eff = c0 * velocity[index] - c1 * prev_velocity[index] + c2 * prev2_velocity[index];
        sample[index] += delta * v_eff;
    }
    return 1;
}

int h3_dpm2m_velocity_step(float *sample, const float *velocity, const float *prev_velocity,
                           size_t count, float sigma, float sigma_next, float sigma_prev) {
    if (!sample || !velocity || !isfinite(sigma) || !isfinite(sigma_next) ||
        !(sigma > sigma_next) || sigma_next < 0.0f) return 0;
    float delta = sigma - sigma_next;
    if (!prev_velocity || !isfinite(sigma_prev) || !(sigma_prev > sigma)) {
        /* Exact 1st-order Euler fallback for step 0 or when no previous velocity is available */
        for (size_t index = 0; index < count; index++)
            sample[index] += delta * velocity[index];
        return 1;
    }
    /* DPM-Solver++ 2M: Generalized variable-step Adams-Bashforth 2nd order */
    float h_curr = delta;
    float h_prev = sigma_prev - sigma;
    float r = (h_prev > 1e-7f) ? (h_curr / h_prev) : 1.0f;
    float b1 = 1.0f + 0.5f * r;
    float b2 = 0.5f * r;
    for (size_t index = 0; index < count; index++) {
        sample[index] += h_curr * (b1 * velocity[index] - b2 * prev_velocity[index]);
    }
    return 1;
}

int h3_dpm3m_velocity_step(float *sample, const float *velocity,
                           const float *prev_velocity, const float *prev2_velocity,
                           size_t count, float sigma, float sigma_next,
                           float sigma_prev, float sigma_prev2) {
    if (!sample || !velocity || !isfinite(sigma) || !isfinite(sigma_next) ||
        !(sigma > sigma_next) || sigma_next < 0.0f) return 0;
    float h_k = sigma - sigma_next;
    if (!prev_velocity || !isfinite(sigma_prev) || !(sigma_prev > sigma)) {
        /* Order 1: Euler fallback for step 0 */
        for (size_t index = 0; index < count; index++)
            sample[index] += h_k * velocity[index];
        return 1;
    }
    float h_prev1 = sigma_prev - sigma;
    if (!prev2_velocity || !isfinite(sigma_prev2) || !(sigma_prev2 > sigma_prev) || h_prev1 <= 1e-7f) {
        /* Order 2: DPM++ 2M fallback for step 1 */
        float r = h_k / h_prev1;
        float b1 = 1.0f + 0.5f * r;
        float b2 = 0.5f * r;
        for (size_t index = 0; index < count; index++)
            sample[index] += h_k * (b1 * velocity[index] - b2 * prev_velocity[index]);
        return 1;
    }
    /* Order 3: DPM-Solver++ 3M (Adams-Bashforth 3rd order with Newton divided differences) */
    float h_prev2 = sigma_prev2 - sigma_prev;
    if (h_prev2 <= 1e-7f) {
        float r = h_k / h_prev1;
        float b1 = 1.0f + 0.5f * r;
        float b2 = 0.5f * r;
        for (size_t index = 0; index < count; index++)
            sample[index] += h_k * (b1 * velocity[index] - b2 * prev_velocity[index]);
        return 1;
    }
    float inv_h1 = 1.0f / h_prev1;
    float inv_h2 = 1.0f / h_prev2;
    float inv_h12 = 1.0f / (h_prev1 + h_prev2);
    float int_d1 = 0.5f * h_k * h_k;
    float int_d2 = (1.0f / 3.0f) * h_k * h_k * h_k + 0.5f * h_k * h_k * h_prev1;

    for (size_t index = 0; index < count; index++) {
        float v0 = velocity[index];
        float v1 = prev_velocity[index];
        float v2 = prev2_velocity[index];
        float d1 = (v0 - v1) * inv_h1;
        float d1_prev = (v1 - v2) * inv_h2;
        float d2 = (d1 - d1_prev) * inv_h12;
        float euler = h_k * v0;
        float corr = int_d1 * d1 + int_d2 * d2;
        float max_corr = 0.55f * fabsf(euler) + 0.005f;
        if (corr > max_corr) corr = max_corr;
        else if (corr < -max_corr) corr = -max_corr;
        sample[index] += euler + corr;
    }
    return 1;
}

int h3_symplectic_flow_normalize(float *latent, size_t count, float sigma_next) {
    if (!latent || count == 0 || sigma_next < 0.0f) return 0;
    /* Soft energy regularization on boundary flow to preserve dynamic range */
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum_sq += (double)latent[i] * (double)latent[i];
    }
    float mean_energy = (float)(sum_sq / (double)count);
    if (mean_energy > 1e-6f && mean_energy < 100.0f) {
        float target_scale = 1.0f / sqrtf(mean_energy);
        float lambda = 0.02f * (1.0f - fminf(1.0f, sigma_next));
        float blend_scale = (1.0f - lambda) + lambda * target_scale;
        for (size_t i = 0; i < count; i++) {
            latent[i] *= blend_scale;
        }
    }
    return 1;
}

/*
 * Frontier Level 6: FreqFlow Dynamic High-Frequency Spectral Velocity Boost
 * Operates on [VIDEO_CHANNELS, time, height, width].
 * Selectively boosts high-frequency spatial gradients in late ODE steps (sigma <= 0.35).
 */
int h3_freqflow_velocity_boost(float *velocity, int channels, int time,
                              int height, int width, float sigma, float strength) {
    if (!velocity || channels <= 0 || time <= 0 || height < 3 || width < 3 ||
        sigma > 0.35f || strength <= 0.001f) return 1;

    /* Dynamic late-step spectral scaling: smoothly grows as sigma -> 0 */
    float alpha = strength * (1.0f - (sigma / 0.35f));
    size_t plane_size = (size_t)height * (size_t)width;
    size_t total_planes = (size_t)channels * (size_t)time;

    for (size_t p = 0; p < total_planes; p++) {
        float *plane = velocity + p * plane_size;
        for (int y = 1; y + 1 < height; y++) {
            size_t row_curr = (size_t)y * (size_t)width;
            size_t row_prev = (size_t)(y - 1) * (size_t)width;
            size_t row_next = (size_t)(y + 1) * (size_t)width;
            for (int x = 1; x + 1 < width; x++) {
                float center = plane[row_curr + x];
                float up     = plane[row_prev + x];
                float down   = plane[row_next + x];
                float left   = plane[row_curr + (x - 1)];
                float right  = plane[row_curr + (x + 1)];

                float lap = 4.0f * center - (up + down + left + right);
                float d_up = fabsf(center - up);
                float d_dn = fabsf(center - down);
                float d_lf = fabsf(center - left);
                float d_rt = fabsf(center - right);
                float max_grad = fmaxf(fmaxf(d_up, d_dn), fmaxf(d_lf, d_rt));
                float limited_lap = copysignf(fminf(fabsf(lap), 1.5f * max_grad), lap);

                plane[row_curr + x] += alpha * limited_lap;
            }
        }
    }
    return 1;
}

/*
 * Frontier Level 7: 2D Spatial Super-Nyquist Pre-VAE Phase Alignment
 * Operates on [VIDEO_CHANNELS, time, height, width] right before VAE decode.
 * Pre-compensates the 3D VAE spatial upsampling and Conv3D low-pass blurring.
 */
int h3_spatial_phase_align(float *video, int channels, int time,
                           int height, int width, float gamma) {
    if (!video || channels <= 0 || time <= 0 || height < 3 || width < 3 || gamma <= 0.001f)
        return 1;

    size_t plane_size = (size_t)height * (size_t)width;
    size_t total_planes = (size_t)channels * (size_t)time;

    float *temp_plane = malloc(plane_size * sizeof(float));
    if (!temp_plane) return 0;

    for (size_t p = 0; p < total_planes; p++) {
        float *plane = video + p * plane_size;
        memcpy(temp_plane, plane, plane_size * sizeof(float));

        for (int y = 1; y + 1 < height; y++) {
            size_t row_curr = (size_t)y * (size_t)width;
            size_t row_prev = (size_t)(y - 1) * (size_t)width;
            size_t row_next = (size_t)(y + 1) * (size_t)width;
            for (int x = 1; x + 1 < width; x++) {
                float center = temp_plane[row_curr + x];
                float up     = temp_plane[row_prev + x];
                float down   = temp_plane[row_next + x];
                float left   = temp_plane[row_curr + (x - 1)];
                float right  = temp_plane[row_curr + (x + 1)];

                float dx1 = center - left;
                float dx2 = right - center;
                float dy1 = center - up;
                float dy2 = down - center;

                /* 2D TVD-Minmod Spatial Limiter to prevent edge ringing */
                float lap_x = (dx1 * dx2 > 0.0f) ? copysignf(fminf(fabsf(dx2 - dx1), fminf(fabsf(dx1), fabsf(dx2))), dx2 - dx1) : 0.0f;
                float lap_y = (dy1 * dy2 > 0.0f) ? copysignf(fminf(fabsf(dy2 - dy1), fminf(fabsf(dy1), fabsf(dy2))), dy2 - dy1) : 0.0f;
                float limited_lap = lap_x + lap_y;

                /* Pre-compensate the 3D VAE spatial low-pass filter */
                plane[row_curr + x] = center - gamma * limited_lap;
            }
        }
    }
    free(temp_plane);
    return 1;
}

/*
 * Frontier Level 8: Temporal Block-Tridiagonal Momentum Regularization (TFM)
 * Operates on [channels, time, height, width] velocity field during ODE sampling.
 * TVD-Minmod flux-limited coupling along timeline tau eliminates foot sliding & locomotion jitter.
 */
int h3_tfm_temporal_momentum_regularize(float *velocity, int channels, int time,
                                       int height, int width, float lambda_tau) {
    if (!velocity || channels <= 0 || time <= 2 || height <= 0 || width <= 0 || lambda_tau <= 0.0001f)
        return 1;

    size_t plane_size = (size_t)height * (size_t)width;
    size_t chan_elements = (size_t)time * plane_size;
    float *chan_buf = malloc(chan_elements * sizeof(float));
    if (!chan_buf) return 0;

    for (int c = 0; c < channels; c++) {
        float *chan_ptr = velocity + (size_t)c * chan_elements;
        memcpy(chan_buf, chan_ptr, chan_elements * sizeof(float));

        /* Internal temporal frames: 1 .. time - 2 */
        for (int t = 1; t + 1 < time; t++) {
            const float *prev_plane = chan_buf + (size_t)(t - 1) * plane_size;
            const float *curr_plane = chan_buf + (size_t)t * plane_size;
            const float *next_plane = chan_buf + (size_t)(t + 1) * plane_size;
            float *out_plane = chan_ptr + (size_t)t * plane_size;

            for (size_t i = 0; i < plane_size; i++) {
                float v_prev = prev_plane[i];
                float v_curr = curr_plane[i];
                float v_next = next_plane[i];

                float d_prev = v_curr - v_prev;
                float d_next = v_next - v_curr;

                /* TVD-Minmod flux limiter: couples monotonic acceleration trajectories */
                if (d_prev * d_next > 0.0f) {
                    float minmod = copysignf(fminf(fabsf(d_prev), fabsf(d_next)), d_next);
                    out_plane[i] = v_curr + lambda_tau * minmod;
                }
            }
        }
    }
    free(chan_buf);
    return 1;
}

/*
 * Frontier Level 11: Pre-VAE Spectral Eigen-Clamping & High-Frequency Noise Limiting
 * Operates on [VIDEO_CHANNELS, time, height, width] latents right before VAE decode.
 * Clamps out-of-distribution 2D Laplacian spatial energy spikes, preventing transposed Conv3D scintillation.
 */
int h3_spectral_eigen_clamp(float *video, int channels, int time,
                            int height, int width, float threshold) {
    if (!video || channels <= 0 || time <= 0 || height < 3 || width < 3 || threshold <= 0.0001f)
        return 1;

    size_t plane_size = (size_t)height * (size_t)width;
    size_t total_planes = (size_t)channels * (size_t)time;

    for (size_t p = 0; p < total_planes; p++) {
        float *plane = video + p * plane_size;
        for (int y = 1; y + 1 < height; y++) {
            size_t row_curr = (size_t)y * (size_t)width;
            size_t row_prev = (size_t)(y - 1) * (size_t)width;
            size_t row_next = (size_t)(y + 1) * (size_t)width;
            for (int x = 1; x + 1 < width; x++) {
                float center = plane[row_curr + x];
                float up     = plane[row_prev + x];
                float down   = plane[row_next + x];
                float left   = plane[row_curr + (x - 1)];
                float right  = plane[row_curr + (x + 1)];

                /* Discrete Laplacian */
                float lap = 4.0f * center - (up + down + left + right);
                float local_mag = 0.2f * (fabsf(center) + fabsf(up) + fabsf(down) + fabsf(left) + fabsf(right)) + 1e-4f;
                float rel_energy = fabsf(lap) / local_mag;

                /* Energy clamp: if relative Laplacian spike exceeds threshold ratio */
                float clamp_ratio = 2.4f / (1.0f + threshold);
                if (rel_energy > clamp_ratio) {
                    float excess = rel_energy - clamp_ratio;
                    float damping = 0.25f * (1.0f - expf(-excess * 0.5f));
                    plane[row_curr + x] = center - damping * lap;
                }
            }
        }
    }
    return 1;
}

