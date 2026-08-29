#ifndef H3_CLI_H
#define H3_CLI_H

#include "h3.h"

/* Run the interactive prompt. The supplied parameters become session defaults.
 * seed_was_given preserves Iris's random-by-default interactive behavior while
 * honoring an explicit command-line --seed. */
int h3_cli_run(h3_ctx *ctx, const char *model_dir,
               const h3_params *initial, int show,
               int seed_was_given);

#endif
