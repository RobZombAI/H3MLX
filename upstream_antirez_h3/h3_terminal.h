#ifndef H3_TERMINAL_H
#define H3_TERMINAL_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_TERM_NONE = 0,
    H3_TERM_KITTY,
    H3_TERM_ITERM2
} h3_terminal_protocol;

h3_terminal_protocol h3_terminal_detect(void);
const char *h3_terminal_protocol_name(h3_terminal_protocol protocol);

/* Terminal display dimensions default to 2x for macOS Retina displays. */
int h3_terminal_set_zoom(int zoom);
int h3_terminal_display_dimensions(int width, int height,
                                   int *display_width, int *display_height);

int h3_terminal_display_rgb24(h3_terminal_protocol protocol,
                              const uint8_t *rgb, int width, int height,
                              int stride, char *error, size_t error_size);

#endif
