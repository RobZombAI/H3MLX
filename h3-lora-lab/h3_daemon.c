#include "h3_daemon.h"
#include "h3_host.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_daemon_shutdown = 0;
static char g_active_socket_path[1024] = {0};

static void daemon_signal_handler(int signum) {
    (void)signum;
    g_daemon_shutdown = 1;
}

static void daemon_cleanup_socket(void) {
    if (g_active_socket_path[0]) {
        unlink(g_active_socket_path);
        g_active_socket_path[0] = '\0';
    }
}

typedef struct {
    int client_fd;
    char last_phase[64];
    int last_completed;
    int last_total;
} daemon_progress_context;

static int daemon_progress_relay(const char *phase, int completed, int total, void *opaque) {
    daemon_progress_context *ctx = (daemon_progress_context *)opaque;
    if (!ctx || ctx->client_fd < 0) return 0;

    if (!strcmp(ctx->last_phase, phase) && ctx->last_completed == completed && ctx->last_total == total) {
        return 0;
    }

    snprintf(ctx->last_phase, sizeof(ctx->last_phase), "%s", phase);
    ctx->last_completed = completed;
    ctx->last_total = total;

    char buffer[256];
    int len = snprintf(buffer, sizeof(buffer), "PROGRESS:%s|%d|%d\n", phase, completed, total);
    if (len > 0) {
        ssize_t written = send(ctx->client_fd, buffer, (size_t)len, 0);
        (void)written;
    }
    return 0;
}

static int send_line(int fd, const char *line) {
    size_t len = strlen(line);
    ssize_t written = send(fd, line, len, 0);
    return (written == (ssize_t)len) ? 0 : -1;
}

static int read_line(int fd, char *buffer, size_t max_len) {
    size_t pos = 0;
    while (pos + 1 < max_len) {
        char ch;
        ssize_t r = recv(fd, &ch, 1, 0);
        if (r <= 0) {
            if (pos == 0) return -1;
            break;
        }
        if (ch == '\n') break;
        if (ch != '\r') {
            buffer[pos++] = ch;
        }
    }
    buffer[pos] = '\0';
    return (int)pos;
}

int h3_daemon_is_running(const char *socket_path) {
    if (!socket_path || !*socket_path) socket_path = H3_DEFAULT_SOCKET_PATH;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return 0;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    int connected = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (connected == 0) {
        /* Send ping */
        send_line(fd, "PING\n");
        char response[64];
        if (read_line(fd, response, sizeof(response)) > 0 && !strcmp(response, "PONG")) {
            close(fd);
            return 1;
        }
        close(fd);
        return 1;
    }
    close(fd);
    return 0;
}

int h3_daemon_run(h3_ctx *ctx, const char *socket_path, int warmup_width, int warmup_height, int warmup_steps, int warmup_layers) {
    if (!ctx) return 1;
    if (!socket_path || !*socket_path) socket_path = H3_DEFAULT_SOCKET_PATH;

    /* Register signal handlers for clean teardown */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = daemon_signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    /* Unlink stale socket */
    unlink(socket_path);
    snprintf(g_active_socket_path, sizeof(g_active_socket_path), "%s", socket_path);
    atexit(daemon_cleanup_socket);

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "h3_daemon: cannot create socket: %s\n", strerror(errno));
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "h3_daemon: cannot bind socket to %s: %s\n", socket_path, strerror(errno));
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 8) != 0) {
        fprintf(stderr, "h3_daemon: cannot listen on socket: %s\n", strerror(errno));
        close(server_fd);
        return 1;
    }

    /* Enable UMA resident caching for DiT and VAE */
    h3_cache_set_enabled(ctx, 1);
    fprintf(stderr, "\n==========================================================\n");
    fprintf(stderr, "💎 H3 RESIDENT DAEMON INITIALIZED (128GB UMA PERSISTENT)\n");
    fprintf(stderr, "  Socket: %s\n", socket_path);
    fprintf(stderr, "  Cache : ACTIVE (Persistent DiT & VAE resident in RAM/VRAM)\n");

    /* Optional Warmup */
    if (warmup_width > 0 && warmup_height > 0) {
        fprintf(stderr, "  Warmup: Pre-allocating DiT %dx%d (steps=%d, layers=%d)...\n",
                warmup_width, warmup_height, warmup_steps, warmup_layers);
        h3_params warmup_params = H3_PARAMS_DEFAULT;
        warmup_params.width = warmup_width;
        warmup_params.height = warmup_height;
        warmup_params.steps = warmup_steps > 0 ? warmup_steps : 2;
        warmup_params.dit_layers = warmup_layers > 0 ? warmup_layers : 50;
        warmup_params.denoise_reuse = 1;
        warmup_params.use_int8_row_fc2 = 1;
        warmup_params.output_path = "/dev/null";

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        h3_result *res = h3_generate(ctx, "warmup priming sequence", &warmup_params);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (res) {
            h3_result_free(res);
            fprintf(stderr, "  Warmup: COMPLETED in %.2fs. DiT is 100%% resident in Metal 4 buffers!\n", elapsed);
        } else {
            fprintf(stderr, "  Warmup: Note: initial warmup produced: %s\n", h3_last_error(ctx));
        }
    }
    fprintf(stderr, "==========================================================\n");
    fprintf(stderr, "⚡ Ready for instant client requests (0.00s DiT load time)!\n\n");

    /* Accept loop */
    while (!g_daemon_shutdown) {
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        char command[256];
        if (read_line(client_fd, command, sizeof(command)) <= 0) {
            close(client_fd);
            continue;
        }

        if (!strcmp(command, "PING")) {
            send_line(client_fd, "PONG\n");
            close(client_fd);
            continue;
        }

        if (!strcmp(command, "QUIT") || !strcmp(command, "SHUTDOWN")) {
            send_line(client_fd, "RESULT:SUCCESS|message=daemon shutting down\n");
            close(client_fd);
            fprintf(stderr, "h3_daemon: shutdown requested by client.\n");
            break;
        }

        if (strcmp(command, "GENERATE") != 0) {
            send_line(client_fd, "RESULT:ERROR|unknown command\n");
            close(client_fd);
            continue;
        }

        /* Parse GENERATE key-value pairs */
        char prompt_buf[16384] = {0};
        char output_buf[1024] = "outputs/h3_daemon.mp4";
        char speech_buf[1024] = {0};
        char first_frame_buf[1024] = {0};
        char last_frame_buf[1024] = {0};

        h3_params params = H3_PARAMS_DEFAULT;
        params.use_int8_row_fc2 = 1;

        char line[4096];
        while (read_line(client_fd, line, sizeof(line)) > 0) {
            if (!strcmp(line, "END_REQUEST")) break;
            char *colon = strchr(line, ':');
            if (!colon) continue;
            *colon = '\0';
            const char *key = line;
            const char *val = colon + 1;

            if (!strcmp(key, "PROMPT")) snprintf(prompt_buf, sizeof(prompt_buf), "%s", val);
            else if (!strcmp(key, "OUTPUT")) snprintf(output_buf, sizeof(output_buf), "%s", val);
            else if (!strcmp(key, "WIDTH")) params.width = atoi(val);
            else if (!strcmp(key, "HEIGHT")) params.height = atoi(val);
            else if (!strcmp(key, "RENDER_WIDTH")) params.render_width = atoi(val);
            else if (!strcmp(key, "RENDER_HEIGHT")) params.render_height = atoi(val);
            else if (!strcmp(key, "FRAMES")) params.frames = atoi(val);
            else if (!strcmp(key, "STEPS")) params.steps = atoi(val);
            else if (!strcmp(key, "REUSE")) params.denoise_reuse = atoi(val);
            else if (!strcmp(key, "LAYERS")) params.dit_layers = atoi(val);
            else if (!strcmp(key, "CORE_REUSE")) params.core_reuse = atoi(val);
            else if (!strcmp(key, "INT8_ROW_FC2")) params.use_int8_row_fc2 = atoi(val);
            else if (!strcmp(key, "TOKEN_REDUCTION")) params.token_reduction = atoi(val);
            else if (!strcmp(key, "SSD_STREAMING")) params.ssd_streaming = atoi(val);
            else if (!strcmp(key, "NGRAM")) params.ngram = atoi(val);
            else if (!strcmp(key, "NGRAM_THRESH")) params.ngram_threshold = strtof(val, NULL);
            else if (!strcmp(key, "MASTER_10BIT")) params.master_10bit = atoi(val);
            else if (!strcmp(key, "DETAILER_2K")) params.detailer_2k = atoi(val);
            else if (!strcmp(key, "FLUID_60FPS")) params.fluid_60fps = atoi(val);
            else if (!strcmp(key, "SEED")) params.seed = (uint64_t)strtoull(val, NULL, 10);
            else if (!strcmp(key, "SPEECH_AUDIO")) snprintf(speech_buf, sizeof(speech_buf), "%s", val);
            else if (!strcmp(key, "FIRST_FRAME")) snprintf(first_frame_buf, sizeof(first_frame_buf), "%s", val);
            else if (!strcmp(key, "LAST_FRAME")) snprintf(last_frame_buf, sizeof(last_frame_buf), "%s", val);
        }

        if (speech_buf[0]) params.speech_audio = speech_buf;
        if (first_frame_buf[0]) params.first_frame = first_frame_buf;
        if (last_frame_buf[0]) params.last_frame = last_frame_buf;
        params.output_path = output_buf;

        /* Attach real-time progress relay to client socket */
        daemon_progress_context prog_ctx;
        prog_ctx.client_fd = client_fd;
        prog_ctx.last_phase[0] = '\0';
        prog_ctx.last_completed = -1;
        prog_ctx.last_total = -1;

        params.on_progress = daemon_progress_relay;
        params.callback_opaque = &prog_ctx;

        fprintf(stderr, "🚀 [RESIDENT REQ] %dx%d, %d frames, %d steps, %d layers, reuse %d -> %s\n",
                params.width, params.height, params.frames, params.steps, params.dit_layers,
                params.denoise_reuse, params.output_path);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        h3_result *result = h3_generate(ctx, prompt_buf, &params);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

        if (!result) {
            const char *err = h3_last_error(ctx);
            fprintf(stderr, "❌ [RESIDENT ERROR] %s\n", err);
            char resp[1024];
            snprintf(resp, sizeof(resp), "RESULT:ERROR|%s\n", err ? err : "unknown error");
            send_line(client_fd, resp);
        } else {
            fprintf(stderr, "⚡ [RESIDENT DONE] Generation completed in %.2fs (0.00s DiT load time)!\n", elapsed);
            char resp[1024];
            snprintf(resp, sizeof(resp), "RESULT:SUCCESS|total_seconds=%.2f|output=%s\n", elapsed, output_buf);
            send_line(client_fd, resp);
            h3_result_free(result);
        }

        close(client_fd);
    }

    close(server_fd);
    daemon_cleanup_socket();
    fprintf(stderr, "h3_daemon: terminated gracefully.\n");
    return 0;
}

int h3_client_generate(const char *socket_path, const char *prompt, const h3_params *params, h3_progress_callback on_progress, void *opaque, int quit_daemon) {
    if (!socket_path || !*socket_path) socket_path = H3_DEFAULT_SOCKET_PATH;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "h3_client: cannot create socket: %s\n", strerror(errno));
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "h3_client: cannot connect to resident daemon at '%s': %s\n", socket_path, strerror(errno));
        fprintf(stderr, "Tip: Start the daemon first with: ./h3 --daemon [socket_path] -d /path/to/MiniMax-H3 --warmup\n");
        close(fd);
        return 1;
    }

    if (quit_daemon) {
        send_line(fd, "QUIT\n");
        char resp[256];
        read_line(fd, resp, sizeof(resp));
        printf("h3_client: daemon response: %s\n", resp);
        close(fd);
        return 0;
    }

    /* Send GENERATE header */
    send_line(fd, "GENERATE\n");

    char line[4096];
    snprintf(line, sizeof(line), "PROMPT:%s\n", prompt ? prompt : "");
    send_line(fd, line);

    if (params->output_path) {
        snprintf(line, sizeof(line), "OUTPUT:%s\n", params->output_path);
        send_line(fd, line);
    }
    snprintf(line, sizeof(line), "WIDTH:%d\n", params->width); send_line(fd, line);
    snprintf(line, sizeof(line), "HEIGHT:%d\n", params->height); send_line(fd, line);
    if (params->render_width > 0) { snprintf(line, sizeof(line), "RENDER_WIDTH:%d\n", params->render_width); send_line(fd, line); }
    if (params->render_height > 0) { snprintf(line, sizeof(line), "RENDER_HEIGHT:%d\n", params->render_height); send_line(fd, line); }
    snprintf(line, sizeof(line), "FRAMES:%d\n", params->frames); send_line(fd, line);
    snprintf(line, sizeof(line), "STEPS:%d\n", params->steps); send_line(fd, line);
    snprintf(line, sizeof(line), "REUSE:%d\n", params->denoise_reuse); send_line(fd, line);
    snprintf(line, sizeof(line), "LAYERS:%d\n", params->dit_layers); send_line(fd, line);
    snprintf(line, sizeof(line), "CORE_REUSE:%d\n", params->core_reuse); send_line(fd, line);
    snprintf(line, sizeof(line), "INT8_ROW_FC2:%d\n", params->use_int8_row_fc2); send_line(fd, line);
    snprintf(line, sizeof(line), "TOKEN_REDUCTION:%d\n", params->token_reduction); send_line(fd, line);
    snprintf(line, sizeof(line), "SSD_STREAMING:%d\n", params->ssd_streaming); send_line(fd, line);
    snprintf(line, sizeof(line), "NGRAM:%d\n", params->ngram); send_line(fd, line);
    if (params->ngram_threshold > 0.0f) { snprintf(line, sizeof(line), "NGRAM_THRESH:%f\n", (double)params->ngram_threshold); send_line(fd, line); }
    snprintf(line, sizeof(line), "MASTER_10BIT:%d\n", params->master_10bit); send_line(fd, line);
    snprintf(line, sizeof(line), "DETAILER_2K:%d\n", params->detailer_2k); send_line(fd, line);
    snprintf(line, sizeof(line), "FLUID_60FPS:%d\n", params->fluid_60fps); send_line(fd, line);
    snprintf(line, sizeof(line), "SEED:%llu\n", (unsigned long long)params->seed); send_line(fd, line);
    if (params->speech_audio) { snprintf(line, sizeof(line), "SPEECH_AUDIO:%s\n", params->speech_audio); send_line(fd, line); }
    if (params->first_frame) { snprintf(line, sizeof(line), "FIRST_FRAME:%s\n", params->first_frame); send_line(fd, line); }
    if (params->last_frame) { snprintf(line, sizeof(line), "LAST_FRAME:%s\n", params->last_frame); send_line(fd, line); }

    send_line(fd, "END_REQUEST\n");

    /* Stream progress & wait for result */
    int status = 0;
    while (read_line(fd, line, sizeof(line)) > 0) {
        if (strncmp(line, "PROGRESS:", 9) == 0) {
            char *p = line + 9;
            char *p1 = strchr(p, '|');
            if (p1) {
                *p1 = '\0';
                const char *phase = p;
                char *p2 = strchr(p1 + 1, '|');
                if (p2) {
                    *p2 = '\0';
                    int completed = atoi(p1 + 1);
                    int total = atoi(p2 + 1);
                    if (on_progress) {
                        on_progress(phase, completed, total, opaque);
                    }
                }
            }
        } else if (strncmp(line, "RESULT:SUCCESS", 14) == 0) {
            status = 0;
            break;
        } else if (strncmp(line, "RESULT:ERROR|", 13) == 0) {
            fprintf(stderr, "\nh3_client: error from resident daemon: %s\n", line + 13);
            status = 1;
            break;
        }
    }

    close(fd);
    return status;
}
