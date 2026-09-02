#ifndef H3_DAEMON_H
#define H3_DAEMON_H

#include "h3.h"

#define H3_DEFAULT_SOCKET_PATH "/tmp/h3_resident.sock"

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the resident daemon listening on socket_path until killed or shutdown requested.
 * Returns 0 on clean exit, non-zero on error. */
int h3_daemon_run(h3_ctx *ctx, const char *socket_path, int warmup_width, int warmup_height, int warmup_steps, int warmup_layers);

/* Connects as a client to a running daemon, sends generation parameters, streams progress, and waits for completion.
 * If quit_daemon != 0, requests the daemon to shut down cleanly.
 * Returns 0 on success, non-zero on error. */
int h3_client_generate(const char *socket_path, const char *prompt, const h3_params *params, h3_progress_callback on_progress, void *opaque, int quit_daemon);

/* Checks if a resident daemon is currently listening on the socket */
int h3_daemon_is_running(const char *socket_path);

#ifdef __cplusplus
}
#endif

#endif /* H3_DAEMON_H */
