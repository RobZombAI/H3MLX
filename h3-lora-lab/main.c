#include "h3.h"
#include "h3_cli.h"
#include "h3_host.h"
#include "h3_terminal.h"
#include "h3_daemon.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR [options]              # interactive\n"
        "       %s -d MODEL_DIR -p PROMPT [-o OUTPUT] [options]\n"
        "       %s --daemon [SOCKET_PATH] -d MODEL_DIR [--warmup] # start resident daemon (0s load)\n"
        "       %s --client [SOCKET_PATH] -p PROMPT [-o OUTPUT]   # instant generation via daemon\n"
        "       %s -d MODEL_DIR --info\n\n"
        "Resident Daemon Options (0.00s DiT Load):\n"
        "      --daemon [PATH]    Start background resident engine in 128GB UMA\n"
        "      --client [PATH]    Send generation request to active resident engine\n"
        "      --socket PATH      Specify Unix Domain Socket path (default: /tmp/h3_resident.sock)\n"
        "      --warmup           Pre-warm DiT tensors and Metal buffers on daemon startup\n"
        "      --quit-daemon      Signal active resident daemon to shut down cleanly\n\n",
        program, program, program, program, program);
    fprintf(stderr,
        "Standard Options:\n"
        "  -d, --model-dir PATH   MiniMax-H3 local directory\n"
        "  -p, --prompt TEXT      Raw H3 prompt\n"
        "  -o, --output PATH      Output MP4 (default: outputs/h3.mp4)\n"
        "      --mode MODE        Engine mode: canonical (antirez pure) or boosted (H3XML)\n"
        "      --canonical        Force antirez canonical reference mode (Euler, float, no-reuse)\n"
        "      --boosted          Force H3XML frontier mode (Metal 4 NAX, AB-3, token-reduction)\n"
        "      --int8, --w8a8     Enable Metal 4 NAX Row-Major INT8 dynamic FC2 quantization\n"
        "      --solver SOLVER    ODE flow solver: euler, ab3, dpm3m\n"
        "      --width N          Output width (default: 864)\n"
        "      --height N         Output height (default: 480)\n"
        "      --render-width N   Lower internal model width (optional)\n"
        "      --render-height N  Lower internal model height (optional)\n"
        "      --frames N         Requested frames (default: 56)\n"
        "      --seconds N        Requested duration at 24 fps (instead of --frames)\n"
        "      --steps N          Denoising passes (default: 20)\n"
        "      --reuse N          Denoiser reuse: 1 close, 2 fast, 3 aggressive\n"
        "      --layers N         DiT blocks: 50 exact, 45 fast, 40 aggressive\n"
        "      --core-reuse N     Core refresh: 1 exact, 4 fast, 6 aggressive\n"
        "      --token-reduction  Pair video tokens in middle DiT blocks\n"
        "      --ssd-streaming    Stream original BF16 DiT layers from SSD\n"
        "      --use-int8-row-fc2 Faster one-scale int8 FC2 (M5)\n"
        "      --use-reference-rope  Disable native 256 RoPE adaptation\n"
        "      --use-slower-bf16-mlp  Force close-reference BF16/MPS MLP\n"
        "      --use-slower-bf16-qkv  Force close-reference BF16 QKV\n"
        "      --use-slower-bf16-attention-output  Force BF16 attention output\n"
        "      --use-slower-row-major-attention-output  Restore SDPA transpose\n"
        "      --use-slower-unfused-int8-inputs  Keep standalone quantizers\n"
        "      --use-slower-unfused-qkv-rope  Keep separate Q/K norm/RoPE\n"
        "      --use-slower-scalar-qkv-rms  Force scalar Q/K RMS loads\n"
        "      --use-slower-uncached-int8-scales  Reread projection scales\n"
        "      --use-slower-dynamic-fc1-k  Use runtime-bound FC1 K loop\n"
        "      --use-slower-grouped-quantizer  Force 256-thread FC2 quantizer\n"
        "      --sol-attn         Enable Sol-Attn dynamic block-sparse attention\n"
        "      --sol-attn-thresh N  Sol-Attn relative pruning threshold (default: 10.0)\n"
        "      --sol-attn-block N   Sol-Attn block size in tokens (default: 32)\n"
        "      --sol-cache        Enable Sol-Engine adaptive velocity/step caching\n"
        "      --sol-cache-thresh N Velocity delta threshold for skipping (default: 0.08)\n"
        "      --sol-draft-refine Enable 2-stage Draft & Refine Super Acceleration\n"
        "      --sol-draft-steps N  Number of draft steps for Draft & Refine\n"
        "      --sol-stats        Print Sol-Engine acceleration summary after generation\n"
        "      --ngram           Enable N-Gram speculative patch drafting and VAE tile cache\n"
        "      --ngram-thresh N  Cosine acceptance threshold (default: 0.985)\n"
        "      --nax-st          Enable NAX-Spatiotemporal Multimodal Attention for long video\n"
        "      --nax-chunk N     Frames per local temporal chunk (default: 4)\n"
        "      --nax-stride N    Keyframe anchor stride in frames (default: 4)\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame conditioning image\n"
        "      --last-frame PATH  Last-frame conditioning image\n"
        "      --ref-image PATH    Append an ordered Ref2VA image\n"
        "      --ref-image-size S  Image sizing: match (default) or max\n"
        "      --ref-video PATH    Append video, including embedded audio\n"
        "      --ref-silent-video PATH  Append video without its audio\n"
        "      --ref-video-audio VIDEO AUDIO  Append video + soundtrack\n"
        "      --ref-audio PATH    Append an ordered standalone audio clip\n"
        "      --frames-dir PATH  Write generated frames as PPM files\n"
        "      --show             Display a frame after every denoising step (M5)\n"
        "      --zoom N           Terminal image zoom (default: 2 for Retina)\n"
        "      --profile          Print per-phase Metal timing and allocation data\n"
        "      --info             Inspect model/device without mapping weights\n"
        "  -h, --help             Show this help\n");
}

static int parse_int(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)parsed;
}

static int frames_from_seconds(const char *value) {
    char *end = NULL;
    errno = 0;
    double seconds = strtod(value, &end);
    double frames = seconds * (double)H3_FPS;
    if (errno || !end || *end || !isfinite(seconds) || seconds <= 0.0 ||
        !isfinite(frames) || frames > (double)INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    long long rounded = llround(frames);
    if (rounded < 1 || rounded > INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    return (int)rounded;
}

static uint64_t parse_u64(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

static h3_reference *append_reference(h3_reference references[12],
                                      size_t *count) {
    if (*count >= 12) {
        fprintf(stderr, "h3: Ref2VA supports at most 12 references\n");
        exit(2);
    }
    h3_reference *reference = &references[(*count)++];
    memset(reference, 0, sizeof(*reference));
    return reference;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_component(const char *label, const h3_component_info *item) {
    printf("  %-18s %2zu files  %4zu tensors  %7.3f GiB\n",
           label, item->files, item->tensors, gib(item->tensor_bytes));
}

static void print_info(const h3_ctx *ctx) {
    const h3_device_info *device = h3_device(ctx);
    const h3_model_info *model = h3_model(ctx);
    printf("h3-metal %s\n", H3_VERSION);
    printf("Device: %s (%s)\n", device->name, device->architecture);
    printf("  physical memory       %.1f GiB\n", gib(device->physical_memory));
    printf("  recommended GPU set   %.1f GiB\n", gib(device->recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(device->max_buffer_length));
    printf("  Apple GPU family      %d\n", device->apple_gpu_family);
    printf("  Metal 4               %s\n", device->metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", device->unified_memory ? "yes" : "no");
    printf("Native checkpoint inventory (header-only):\n");
    print_component("Qwen3-VL encoder", &model->text_encoder);
    print_component("FL2VA DiT", &model->fl2va_transformer);
    print_component("Ref2VA DiT", &model->ref2va_transformer);
    print_component("video VAE", &model->video_vae);
    print_component("audio VAE", &model->audio_vae);
}

#include <time.h>

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
    double phase_start_time;
    h3_terminal_protocol terminal;
    int display_failed;
    const char *frames_dir;
    int frame_write_failed;
} cli_state;

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    cli_state *state = opaque;
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return 0;
    
    double now = cli_now_sec();
    
    // New phase started:
    if (strcmp(state->phase, phase)) {
        if (state->active && state->phase[0]) {
            double elapsed = now - state->phase_start_time;
            fprintf(stderr, "\r\033[K  \033[1;32m✓\033[0m \033[1m%-24s\033[0m [\033[32m100%%\033[0m] (\033[36m%.2fs\033[0m)\n",
                    state->phase, elapsed);
        }
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
        state->phase_start_time = now;
        state->active = 1;
    }
    
    state->completed = completed;
    state->total = total;
    
    int pct = total > 0 ? (completed * 100) / total : 100;
    if (pct > 100) pct = 100;
    
    // 20-character visual progress bar:
    char bar[25];
    int filled = (pct * 20) / 100;
    for (int i = 0; i < 20; i++) {
        bar[i] = (i < filled) ? '=' : ' ';
    }
    bar[20] = '\0';
    
    double elapsed = now - state->phase_start_time;
    
    if (completed >= total && total > 0) {
        fprintf(stderr, "\r\033[K  \033[1;32m✓\033[0m \033[1m%-24s\033[0m [\033[32m100%%\033[0m] (\033[36m%.2fs\033[0m)\n",
                phase, elapsed);
        state->active = 0;
    } else {
        fprintf(stderr, "\r\033[K  \033[1;33m⏳\033[0m \033[1m%-24s\033[0m [\033[1;34m%s\033[0m] \033[1;33m%3d%%\033[0m (%d/%d) \033[90m[%.1fs]\033[0m",
                phase, bar, pct, completed, total, elapsed);
    }
    fflush(stderr);
    return 0;
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    cli_state *state = opaque;
    int preview = frame->denoise_step >= 0;
    if (!preview && state->frames_dir && !state->frame_write_failed) {
        char path[1024];
        int length = snprintf(path, sizeof(path), "%s/frame-%04d.ppm",
                              state->frames_dir, frame->frame_index);
        FILE *output = length > 0 && (size_t)length < sizeof(path) ?
            fopen(path, "wb") : NULL;
        if (!output ||
            fprintf(output, "P6\n%d %d\n255\n", frame->width,
                    frame->height) < 0) {
            fprintf(stderr, "h3: cannot write frame %d to %s\n",
                    frame->frame_index, state->frames_dir);
            if (output) fclose(output);
            state->frame_write_failed = 1;
        } else {
            size_t row_bytes = (size_t)frame->width * 3;
            for (int row = 0; row < frame->height; row++) {
                if (fwrite(frame->rgb + (size_t)row * frame->stride, 1,
                           row_bytes, output) != row_bytes) {
                    state->frame_write_failed = 1;
                    break;
                }
            }
            if (fclose(output) != 0) state->frame_write_failed = 1;
            if (state->frame_write_failed)
                fprintf(stderr, "h3: incomplete frame %d in %s\n",
                        frame->frame_index, state->frames_dir);
        }
    }
    if (state->frame_write_failed) return 1;
    if (state->display_failed || state->terminal == H3_TERM_NONE) return 0;
    if (state->active) {
        fputc('\n', stderr);
        state->active = 0;
    }
    if (preview)
        fprintf(stderr,
                "h3: denoise preview %d/%d, video frame %d/%d via %s\n",
                frame->denoise_step + 1, frame->denoise_steps,
                frame->frame_index + 1, frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    else
        fprintf(stderr, "h3: frame %d/%d via %s\n", frame->frame_index + 1,
                frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3: terminal display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_RENDER_WIDTH, OPT_RENDER_HEIGHT,
           OPT_FRAMES, OPT_SECONDS, OPT_STEPS, OPT_REUSE,
           OPT_LAYERS,
           OPT_CORE_REUSE,
           OPT_TOKEN_REDUCTION,
           OPT_SSD_STREAMING,
           OPT_USE_INT8_ROW_FC2,
           OPT_USE_REFERENCE_ROPE,
           OPT_USE_SLOWER_BF16_MLP,
           OPT_USE_SLOWER_BF16_QKV,
           OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_UNFUSED_INT8_INPUTS,
           OPT_USE_SLOWER_UNFUSED_QKV_ROPE,
           OPT_USE_SLOWER_SCALAR_QKV_RMS,
           OPT_USE_SLOWER_UNCACHED_INT8_SCALES,
           OPT_USE_SLOWER_DYNAMIC_FC1_K,
           OPT_USE_SLOWER_GROUPED_QUANTIZER,
           OPT_SOL_ATTN,
           OPT_SOL_ATTN_THRESH,
           OPT_SOL_ATTN_BLOCK,
           OPT_SOL_CACHE,
           OPT_SOL_CACHE_THRESH,
           OPT_SOL_DRAFT_REFINE,
           OPT_SOL_DRAFT_STEPS,
           OPT_SOL_STATS,
           OPT_SEED,
           OPT_FIRST, OPT_LAST, OPT_REF_IMAGE, OPT_REF_IMAGE_SIZE,
           OPT_REF_VIDEO, OPT_REF_SILENT_VIDEO, OPT_REF_VIDEO_AUDIO,
            OPT_REF_AUDIO, OPT_SPEECH_AUDIO, OPT_MASTER_10BIT,
            OPT_DETAILER_2K, OPT_FLUID_60FPS,
            OPT_NGRAM, OPT_NGRAM_THRESH,
            OPT_NAX_ST, OPT_NAX_CHUNK, OPT_NAX_STRIDE,
            OPT_FRAMES_DIR, OPT_SHOW, OPT_ZOOM,
            OPT_DAEMON, OPT_CLIENT, OPT_SOCKET, OPT_WARMUP, OPT_QUIT_DAEMON,
            OPT_INT8, OPT_MODE, OPT_SOLVER, OPT_CANONICAL, OPT_BOOSTED,
            OPT_PROFILE, OPT_INFO };
    static const struct option options[] = {
        {"model-dir", required_argument, NULL, 'd'},
        {"prompt", required_argument, NULL, 'p'},
        {"output", required_argument, NULL, 'o'},
        {"mode", required_argument, NULL, OPT_MODE},
        {"solver", required_argument, NULL, OPT_SOLVER},
        {"canonical", no_argument, NULL, OPT_CANONICAL},
        {"boosted", no_argument, NULL, OPT_BOOSTED},
        {"int8", no_argument, NULL, OPT_INT8},
        {"w8a8", no_argument, NULL, OPT_INT8},
        {"daemon", optional_argument, NULL, OPT_DAEMON},
        {"server", optional_argument, NULL, OPT_DAEMON},
        {"client", optional_argument, NULL, OPT_CLIENT},
        {"connect", optional_argument, NULL, OPT_CLIENT},
        {"socket", required_argument, NULL, OPT_SOCKET},
        {"warmup", no_argument, NULL, OPT_WARMUP},
        {"quit-daemon", no_argument, NULL, OPT_QUIT_DAEMON},
        {"width", required_argument, NULL, OPT_WIDTH},
        {"height", required_argument, NULL, OPT_HEIGHT},
        {"render-width", required_argument, NULL, OPT_RENDER_WIDTH},
        {"render-height", required_argument, NULL, OPT_RENDER_HEIGHT},
        {"frames", required_argument, NULL, OPT_FRAMES},
        {"seconds", required_argument, NULL, OPT_SECONDS},
        {"steps", required_argument, NULL, OPT_STEPS},
        {"reuse", required_argument, NULL, OPT_REUSE},
        {"layers", required_argument, NULL, OPT_LAYERS},
        {"core-reuse", required_argument, NULL, OPT_CORE_REUSE},
        {"token-reduction", no_argument, NULL, OPT_TOKEN_REDUCTION},
        {"ssd-streaming", no_argument, NULL, OPT_SSD_STREAMING},
        {"use-int8-row-fc2", no_argument, NULL, OPT_USE_INT8_ROW_FC2},
        {"use-reference-rope", no_argument, NULL, OPT_USE_REFERENCE_ROPE},
        {"use-slower-bf16-mlp", no_argument, NULL,
         OPT_USE_SLOWER_BF16_MLP},
        {"use-slower-bf16-qkv", no_argument, NULL,
         OPT_USE_SLOWER_BF16_QKV},
        {"use-slower-bf16-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT},
        {"use-slower-row-major-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT},
        {"use-slower-unfused-int8-inputs", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_INT8_INPUTS},
        {"use-slower-unfused-qkv-rope", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_QKV_ROPE},
        {"use-slower-scalar-qkv-rms", no_argument, NULL,
         OPT_USE_SLOWER_SCALAR_QKV_RMS},
        {"use-slower-uncached-int8-scales", no_argument, NULL,
         OPT_USE_SLOWER_UNCACHED_INT8_SCALES},
        {"use-slower-dynamic-fc1-k", no_argument, NULL,
         OPT_USE_SLOWER_DYNAMIC_FC1_K},
        {"use-slower-grouped-quantizer", no_argument, NULL,
         OPT_USE_SLOWER_GROUPED_QUANTIZER},
        {"sol-attn", no_argument, NULL, OPT_SOL_ATTN},
        {"sol-attn-thresh", required_argument, NULL, OPT_SOL_ATTN_THRESH},
        {"sol-attn-threshold", required_argument, NULL, OPT_SOL_ATTN_THRESH},
        {"sol-attn-block", required_argument, NULL, OPT_SOL_ATTN_BLOCK},
        {"sol-attn-block-size", required_argument, NULL, OPT_SOL_ATTN_BLOCK},
        {"sol-cache", no_argument, NULL, OPT_SOL_CACHE},
        {"sol-cache-thresh", required_argument, NULL, OPT_SOL_CACHE_THRESH},
        {"sol-draft-refine", no_argument, NULL, OPT_SOL_DRAFT_REFINE},
        {"sol-draft-steps", required_argument, NULL, OPT_SOL_DRAFT_STEPS},
        {"sol-stats", no_argument, NULL, OPT_SOL_STATS},
        {"seed", required_argument, NULL, OPT_SEED},
        {"first-frame", required_argument, NULL, OPT_FIRST},
        {"last-frame", required_argument, NULL, OPT_LAST},
        {"ref-image", required_argument, NULL, OPT_REF_IMAGE},
        {"ref-image-size", required_argument, NULL, OPT_REF_IMAGE_SIZE},
        {"ref-video", required_argument, NULL, OPT_REF_VIDEO},
        {"ref-silent-video", required_argument, NULL, OPT_REF_SILENT_VIDEO},
        {"ref-video-audio", required_argument, NULL, OPT_REF_VIDEO_AUDIO},
        {"ref-audio", required_argument, NULL, OPT_REF_AUDIO},
        {"speech-audio", required_argument, NULL, OPT_SPEECH_AUDIO},
        {"master-10bit", no_argument, NULL, OPT_MASTER_10BIT},
        {"detailer-2k", no_argument, NULL, OPT_DETAILER_2K},
        {"master-2k", no_argument, NULL, OPT_DETAILER_2K},
        {"fluid-60fps", no_argument, NULL, OPT_FLUID_60FPS},
        {"fps60", no_argument, NULL, OPT_FLUID_60FPS},
        {"ngram", no_argument, NULL, OPT_NGRAM},
        {"ngram-threshold", required_argument, NULL, OPT_NGRAM_THRESH},
        {"ngram-thresh", required_argument, NULL, OPT_NGRAM_THRESH},
        {"nax-st", no_argument, NULL, OPT_NAX_ST},
        {"nax-chunk", required_argument, NULL, OPT_NAX_CHUNK},
        {"nax-stride", required_argument, NULL, OPT_NAX_STRIDE},
        {"frames-dir", required_argument, NULL, OPT_FRAMES_DIR},
        {"show", no_argument, NULL, OPT_SHOW},
        {"zoom", required_argument, NULL, OPT_ZOOM},
        {"profile", no_argument, NULL, OPT_PROFILE},
        {"info", no_argument, NULL, OPT_INFO},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = "outputs/h3.mp4";
    h3_params params = H3_PARAMS_DEFAULT;
    h3_reference references[12];
    size_t reference_count = 0;
    cli_state cli = {{0}, 0, -1, -1, 0.0, H3_TERM_NONE, 0, NULL, 0};
    int show = 0;
    int profile = 0;
    int info = 0;
    int frames_given = 0;
    int seconds_given = 0;
    int seed_given = 0;
    const char *daemon_socket = NULL;
    int is_daemon = 0;
    int is_client = 0;
    int warmup = 0;
    int quit_daemon = 0;
    int option;
    while ((option = getopt_long(argc, argv, "d:p:o:h", options, NULL)) != -1) {
        switch (option) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'h': usage(argv[0]); return 0;
            case OPT_WIDTH: params.width = parse_int(optarg, "width"); break;
            case OPT_HEIGHT: params.height = parse_int(optarg, "height"); break;
            case OPT_RENDER_WIDTH:
                params.render_width = parse_int(optarg, "render width");
                break;
            case OPT_RENDER_HEIGHT:
                params.render_height = parse_int(optarg, "render height");
                break;
            case OPT_FRAMES:
                params.frames = parse_int(optarg, "frames");
                frames_given = 1;
                break;
            case OPT_SECONDS:
                params.frames = frames_from_seconds(optarg);
                seconds_given = 1;
                break;
            case OPT_STEPS: params.steps = parse_int(optarg, "steps"); break;
            case OPT_REUSE:
                params.denoise_reuse = parse_int(optarg, "reuse");
                if (params.denoise_reuse <= 0) params.denoise_reuse = 1;
                break;
            case OPT_LAYERS:
                params.dit_layers = parse_int(optarg, "layers");
                break;
            case OPT_CORE_REUSE:
                params.core_reuse = parse_int(optarg, "core reuse");
                break;
            case OPT_TOKEN_REDUCTION: params.token_reduction = 1; break;
            case OPT_SSD_STREAMING: params.ssd_streaming = 1; break;
            case OPT_INT8:
            case OPT_USE_INT8_ROW_FC2:
                params.use_int8_row_fc2 = 1;
                break;
            case OPT_CANONICAL:
                params.use_int8_row_fc2 = 0;
                params.token_reduction = 0;
                params.denoise_reuse = 1;
                params.sol_attn = 0;
                params.sol_cache = 0;
                break;
            case OPT_BOOSTED:
                params.use_int8_row_fc2 = 1;
                params.token_reduction = 1;
                break;
            case OPT_MODE:
                if (!strcmp(optarg, "canonical") || !strcmp(optarg, "antirez") || !strcmp(optarg, "pure")) {
                    params.use_int8_row_fc2 = 0;
                    params.token_reduction = 0;
                    params.denoise_reuse = 1;
                    params.sol_attn = 0;
                    params.sol_cache = 0;
                } else if (!strcmp(optarg, "boosted") || !strcmp(optarg, "h3xml") || !strcmp(optarg, "robzomb")) {
                    params.use_int8_row_fc2 = 1;
                    params.token_reduction = 1;
                }
                break;
            case OPT_SOLVER:
                if (!strcmp(optarg, "dpm3m") || !strcmp(optarg, "ab3")) {
                    setenv("H3_SOLVER", "dpm3m", 1);
                    setenv("H3_CPU_SAMPLER", "1", 1);
                } else if (!strcmp(optarg, "euler")) {
                    setenv("H3_SOLVER", "euler", 1);
                    setenv("H3_CPU_SAMPLER", "1", 1);
                }
                break;
            case OPT_USE_REFERENCE_ROPE:
                params.use_reference_rope = 1;
                break;
            case OPT_USE_SLOWER_BF16_MLP:
                params.use_slower_bf16_mlp = 1;
                break;
            case OPT_USE_SLOWER_BF16_QKV:
                params.use_slower_bf16_qkv = 1;
                break;
            case OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT:
                params.use_slower_bf16_attention_output = 1;
                break;
            case OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT:
                params.use_slower_row_major_attention_output = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_INT8_INPUTS:
                params.use_slower_unfused_int8_inputs = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_QKV_ROPE:
                params.use_slower_unfused_qkv_rope = 1;
                break;
            case OPT_USE_SLOWER_SCALAR_QKV_RMS:
                params.use_slower_scalar_qkv_rms = 1;
                break;
            case OPT_USE_SLOWER_UNCACHED_INT8_SCALES:
                params.use_slower_uncached_int8_scales = 1;
                break;
            case OPT_USE_SLOWER_DYNAMIC_FC1_K:
                params.use_slower_dynamic_fc1_k = 1;
                break;
            case OPT_USE_SLOWER_GROUPED_QUANTIZER:
                params.use_slower_grouped_quantizer = 1;
                break;
            case OPT_SOL_ATTN:
                params.sol_attn = 1;
                break;
            case OPT_SOL_ATTN_THRESH:
                params.sol_attn_threshold = strtof(optarg, NULL);
                params.sol_attn = 1;
                break;
            case OPT_SOL_ATTN_BLOCK:
                params.sol_attn_block_size = (uint32_t)parse_int(optarg, "sol-attn-block");
                params.sol_attn = 1;
                break;
            case OPT_SOL_CACHE:
                params.sol_cache = 1;
                break;
            case OPT_SOL_CACHE_THRESH:
                params.sol_cache_thresh = strtof(optarg, NULL);
                params.sol_cache = 1;
                break;
            case OPT_SOL_DRAFT_REFINE:
                params.sol_draft_refine = 1;
                params.sol_attn = 1;
                params.sol_cache = 1;
                break;
            case OPT_SOL_DRAFT_STEPS:
                params.sol_draft_steps = parse_int(optarg, "sol-draft-steps");
                params.sol_draft_refine = 1;
                params.sol_attn = 1;
                params.sol_cache = 1;
                break;
            case OPT_SOL_STATS:
                params.sol_stats = 1;
                break;
            case OPT_NGRAM:
                params.ngram = 1;
                break;
            case OPT_NGRAM_THRESH:
                params.ngram_threshold = strtof(optarg, NULL);
                params.ngram = 1;
                break;
            case OPT_NAX_ST:
                params.nax_st = 1;
                break;
            case OPT_NAX_CHUNK:
                params.nax_st_chunk_frames = (uint32_t)parse_int(optarg, "nax-chunk");
                params.nax_st = 1;
                break;
            case OPT_NAX_STRIDE:
                params.nax_st_keyframe_stride = (uint32_t)parse_int(optarg, "nax-stride");
                params.nax_st = 1;
                break;
            case OPT_SEED:
                params.seed = parse_u64(optarg, "seed");
                seed_given = 1;
                break;
            case OPT_FIRST: params.first_frame = optarg; break;
            case OPT_LAST: params.last_frame = optarg; break;
            case OPT_REF_IMAGE: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_IMAGE;
                reference->path = optarg;
                break;
            }
            case OPT_REF_IMAGE_SIZE:
                if (!strcmp(optarg, "match"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
                else if (!strcmp(optarg, "max"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MAX;
                else {
                    fprintf(stderr,
                        "h3: --ref-image-size must be match or max\n");
                    return 2;
                }
                break;
            case OPT_REF_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 1;
                break;
            }
            case OPT_REF_SILENT_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 0;
                break;
            }
            case OPT_REF_VIDEO_AUDIO: {
                if (optind >= argc) {
                    fprintf(stderr,
                        "h3: --ref-video-audio requires VIDEO and AUDIO\n");
                    return 2;
                }
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO_AUDIO;
                reference->path = optarg;
                reference->audio_path = argv[optind++];
                break;
            }
            case OPT_REF_AUDIO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_AUDIO;
                reference->path = optarg;
                break;
            }
            case OPT_SPEECH_AUDIO: params.speech_audio = optarg; break;
            case OPT_MASTER_10BIT: params.master_10bit = 1; break;
            case OPT_DETAILER_2K: params.detailer_2k = 1; params.master_10bit = 1; break;
            case OPT_FLUID_60FPS: params.fluid_60fps = 1; break;
            case OPT_FRAMES_DIR: cli.frames_dir = optarg; break;
            case OPT_SHOW: show = 1; break;
            case OPT_ZOOM:
                if (!h3_terminal_set_zoom(parse_int(optarg, "zoom"))) {
                    fprintf(stderr, "h3: --zoom must be at least 1\n");
                    return 2;
                }
                break;
            case OPT_DAEMON:
                is_daemon = 1;
                if (optarg && *optarg) daemon_socket = optarg;
                break;
            case OPT_CLIENT:
                is_client = 1;
                if (optarg && *optarg) daemon_socket = optarg;
                break;
            case OPT_SOCKET:
                daemon_socket = optarg;
                break;
            case OPT_WARMUP:
                warmup = 1;
                break;
            case OPT_QUIT_DAEMON:
                quit_daemon = 1;
                break;
            case OPT_PROFILE: profile = 1; break;
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    if (is_client || quit_daemon) {
        if (!daemon_socket) daemon_socket = H3_DEFAULT_SOCKET_PATH;
        if (quit_daemon) {
            return h3_client_generate(daemon_socket, NULL, NULL, NULL, NULL, 1);
        }
        if (!prompt) {
            fprintf(stderr, "h3: --prompt is required in client mode\n");
            return 2;
        }
        params.output_path = output;
        return h3_client_generate(daemon_socket, prompt, &params, cli_progress, &cli, 0);
    }
    if (is_daemon) {
        if (!model_dir) {
            usage(argv[0]);
            return 2;
        }
        if (!daemon_socket) daemon_socket = H3_DEFAULT_SOCKET_PATH;
        h3_ctx *ctx = h3_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "h3: cannot load model for resident daemon: %s\n", h3_last_error(NULL));
            return 1;
        }
        int status = h3_daemon_run(ctx, daemon_socket,
                                   warmup ? params.width : 0,
                                   warmup ? params.height : 0,
                                   warmup ? params.steps : 0,
                                   warmup ? params.dit_layers : 0);
        h3_free(ctx);
        return status;
    }
    if (!model_dir) {
        usage(argv[0]);
        return 2;
    }
    if (frames_given && seconds_given) {
        fprintf(stderr, "h3: --seconds and --frames are mutually exclusive\n");
        return 2;
    }
    if (prompt && params.steps >= 2 && params.steps <= 7 &&
        params.denoise_reuse > 1) {
        fprintf(stderr,
            "h3: warning: --reuse with only %d denoising steps leaves very "
            "few fresh model evaluations\n", params.steps);
    }
    params.references = references;
    params.reference_count = reference_count;
    if (cli.frames_dir && mkdir(cli.frames_dir, 0755) != 0 &&
        errno != EEXIST) {
        fprintf(stderr, "h3: cannot create frames directory %s: %s\n",
                cli.frames_dir, strerror(errno));
        return 1;
    }
    if (profile) setenv("H3_PROFILE", "1", 1);
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    if (info) print_info(ctx);
    if (prompt) {
        params.output_path = output;
        params.on_progress = cli_progress;
        params.callback_opaque = &cli;
        if (cli.frames_dir) params.on_frame = cli_frame;
        if (show) {
            cli.terminal = h3_terminal_detect();
            if (cli.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3: graphical output uses %s\n",
                        h3_terminal_protocol_name(cli.terminal));
                params.on_frame = cli_frame;
                params.preview_denoise = 1;
            }
        }
        h3_result *result = h3_generate(ctx, prompt, &params);
        if (!result) {
            if (cli.active) fputc('\n', stderr);
            fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            return 1;
        }
        h3_result_free(result);
        if (output && *output) fprintf(stderr, "h3: wrote %s\n", output);
        if (cli.frames_dir)
            fprintf(stderr, "h3: wrote frames to %s\n", cli.frames_dir);
    } else if (!info) {
        int cli_status = h3_cli_run(ctx, model_dir, &params, show, seed_given);
        h3_free(ctx);
        return cli_status;
    }
    h3_free(ctx);
    return 0;
}
